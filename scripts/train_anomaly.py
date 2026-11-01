"""Train one anomaly model on one category and write raw results (GPU notebook step).

Writes, under ``results/<model>/<category>/``:
  scores.csv          raw score of every validation and test image
  scores_<cond>.csv   raw test scores under each perturbation (brightness, blur, noise)
  pixel_metrics.json  pixel AU-PRO on the test set
  run_info.json       versions, hardware, timings, settings
  overlays/           failure gallery: true positives, false negatives, false positives
and adds the threshold to ``results/thresholds.json``.

Example:
    python scripts/train_anomaly.py \
        --config configs/fabric.yaml --model configs/models/patchcore.yaml
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import anomalib
import numpy as np
import torch
from lightning.pytorch import seed_everything

from vqi.anomaly import (
    PerturbImage,
    Predictions,
    build_datamodule,
    build_engine,
    build_model,
    pixel_aupro,
    predict,
)
from vqi.config import Config, load_config
from vqi.overlay import save_overlay
from vqi.perturb import perturbations
from vqi.threshold import save_threshold, threshold_record


def relative_to_category(path: str, category: str) -> str:
    """Path from the category folder on, so no local machine paths end up in results."""
    parts = Path(path).parts
    start = len(parts) - 1 - parts[::-1].index(category)
    return "/".join(parts[start:])


def write_scores(path: Path, category: str, splits: dict[str, Predictions]) -> None:
    """One CSV row per image: split, image, label, raw score."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image", "label", "score"])
        for split, pred in splits.items():
            for image, label, score in zip(pred.image_paths, pred.labels, pred.scores, strict=True):
                writer.writerow(
                    [split, relative_to_category(image, category), int(label), f"{score:.6f}"]
                )


def save_gallery(
    out_dir: Path, category: str, test: Predictions, threshold: float, vmin: float, k: int
) -> list[dict]:
    """Save overlays: k most confident catches, k worst misses, k worst false alarms."""
    rejected = test.scores > threshold
    order = np.argsort(test.scores)
    groups = {
        "tp": [i for i in order[::-1] if test.labels[i] == 1 and rejected[i]][:k],
        "fn": [i for i in order if test.labels[i] == 1 and not rejected[i]][:k],
        "fp": [i for i in order[::-1] if test.labels[i] == 0 and rejected[i]][:k],
    }
    index = []
    for group, indices in groups.items():
        for i in indices:
            # e.g. "test_cut_000": defect folders reuse file names, so keep the folder in the name
            name = Path(relative_to_category(test.image_paths[i], category)).with_suffix("")
            name = "_".join(name.parts[1:])
            file = out_dir / group / f"{name}.jpg"
            caption = (
                f"{group.upper()}  score {test.scores[i]:.3f}  threshold {threshold:.3f}  {name}"
            )
            save_overlay(
                file,
                test.image_paths[i],
                test.anomaly_maps[i].numpy(),
                vmin=vmin,
                vmax=threshold,
                caption=caption,
                mask_path=test.mask_paths[i],
            )
            index.append(
                {
                    "group": group,
                    "file": f"overlays/{group}/{name}.jpg",
                    "score": float(test.scores[i]),
                }
            )
    return index


def git_commit() -> str:
    """Current commit hash, so every result can be traced to the code that made it."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def run(config: Config) -> None:
    """Fit, score validation and test images, set the threshold, save everything."""
    seed_everything(config.seed, workers=True)
    category = config.dataset.category
    key = f"{config.model.key}/{category}"
    out_dir = config.results_dir / config.model.key / category

    datamodule = build_datamodule(config)
    model = build_model(config)
    engine = build_engine(config, Path("runs") / key)

    start = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    train_seconds = time.perf_counter() - start

    val = predict(engine, model, datamodule.val_dataloader())
    test = predict(engine, model, datamodule.test_dataloader())
    if val.labels.any():
        raise RuntimeError(
            "Validation split contains defective images; threshold would leak labels"
        )

    record = threshold_record(val.scores, config.threshold.target_false_alarm_rate)
    save_threshold(config.results_dir / "thresholds.json", key, record)
    write_scores(out_dir / "scores.csv", category, {"validation": val, "test": test})

    # Robustness: same model and threshold, test images perturbed before resizing.
    for name, perturbation in perturbations(config.robustness, config.seed).items():
        perturbed = build_datamodule(config, test_augmentations=PerturbImage(perturbation))
        result = predict(engine, model, datamodule=perturbed)
        write_scores(out_dir / f"scores_{name}.csv", category, {"test": result})

    pixel = {f"pixel_aupro_{limit}": pixel_aupro(test, limit) for limit in (0.3, 0.05)}
    (out_dir / "pixel_metrics.json").write_text(json.dumps(pixel, indent=2) + "\n")

    vmin = float(val.anomaly_maps.median())  # typical pixel score on good parts
    gallery = save_gallery(
        out_dir / "overlays",
        category,
        test,
        record["threshold"],
        vmin,
        config.evaluation.overlays_per_group,
    )
    (out_dir / "overlays.json").write_text(json.dumps(gallery, indent=2) + "\n")

    run_info = {
        "git_commit": git_commit(),
        "anomalib": anomalib.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_seconds": round(train_seconds, 1),
        "n_train_images": len(datamodule.train_data),
        "n_validation_images": len(val.scores),
        "n_test_images": len(test.scores),
        "image_size": config.model.image_size,
        "model_params": config.model.params,
        "trainer_args": config.model.trainer,
        "seed": config.seed,
        "overlay_scale": {"vmin": vmin, "vmax": record["threshold"]},
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    n_val = record["n_validation_images"]
    print(f"{key}: threshold {record['threshold']:.4f} from {n_val} good validation images")


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="category config")
    parser.add_argument("--model", required=True, type=Path, help="configs/models/<model>.yaml")
    args = parser.parse_args()
    run(load_config(args.config, args.model))


if __name__ == "__main__":
    main()
