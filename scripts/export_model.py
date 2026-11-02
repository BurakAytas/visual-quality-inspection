"""Export the anomaly model to ONNX and OpenVINO, then check accuracy and CPU speed.

For the PyTorch model (baseline) and each exported format:
  1. score the good validation images and set that format's own threshold (same policy);
  2. score and time the test images on CPU, one image at a time;
  3. report image AUROC, recall and false-alarm rate with 95% bootstrap intervals.

The exported graphs resize without antialiasing (ONNX cannot export it), so their scores
differ slightly from training; that is why every format gets its own threshold.

Writes ``results/export/<model>/<category>/``: export.json, export.md, scores_<format>.csv,
and adds ``<model>/<category>/<format>`` thresholds to ``results/thresholds.json``.
Exported model files go to ``exports/`` (git-ignored).

Example:
    python scripts/export_model.py --config configs/leather.yaml --model <model.yaml>
    (with <model.yaml> = configs/models/patchcore.yaml)
"""

import argparse
import csv
import json
import os
import platform
import subprocess
from pathlib import Path

import anomalib
import numpy as np
import onnxruntime
import openvino
import torch
from anomalib.data.utils import read_image
from lightning.pytorch import seed_everything

from vqi.anomaly import build_datamodule, build_engine, build_model
from vqi.config import Config, load_config
from vqi.export import export_all, model_size_mb, scorer_for, torch_scorer
from vqi.latency import timed_map
from vqi.metrics import physical_part, summarize
from vqi.threshold import save_threshold, threshold_record

METRICS = ("image_auroc", "recall", "false_alarm_rate")


def load_image(path: str) -> np.ndarray:
    """Decode an image to (H, W, C) float32 in [0, 1] (not timed)."""
    return read_image(path, as_tensor=False).astype(np.float32)


def cpu_name() -> str:
    """Human-readable CPU model, so latency numbers can be compared fairly."""
    if platform.system() == "Linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def environment() -> dict:
    """Where and with what the numbers were measured."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "anomalib": anomalib.__version__,
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "openvino": openvino.__version__,
        "git_commit": commit.stdout.strip() or "unknown",
    }


def write_scores(path: Path, rows: list[tuple[str, str, int, float]]) -> None:
    """One CSV row per image: split, image file name, label, raw score."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image", "label", "score"])
        writer.writerows((s, Path(p).name, y, f"{v:.6f}") for s, p, y, v in rows)


def markdown_table(formats: dict) -> str:
    """Format x size x latency x accuracy."""
    lines = [
        "| Format | Size (MB) | Median ms/image | p95 ms/image | Image AUROC | Recall "
        "| False-alarm rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, f in formats.items():
        size = "-" if f["size_mb"] is None else f"{f['size_mb']:.0f}"
        auroc, recall, far = (f["image"][m] for m in METRICS)
        lines.append(
            f"| {name} | {size} | {f['latency']['median_ms']:.1f} | {f['latency']['p95_ms']:.1f} "
            f"| {auroc['value']:.3f} | {recall['value']:.3f} | {far['value']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def run(config: Config, warmup: int) -> None:
    """Fit, export, then score and time every format on CPU."""
    seed_everything(config.seed, workers=True)
    key = f"{config.model.key}/{config.dataset.category}"
    out_dir = config.results_dir / "export" / key
    out_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(config)
    model = build_model(config)
    engine = build_engine(config, Path("runs") / "export" / key)
    engine.fit(model=model, datamodule=datamodule)
    paths = export_all(engine, model, datamodule, Path("exports") / key)

    val = datamodule.val_data.samples
    test = datamodule.test_data.samples
    if val.label_index.any():
        raise RuntimeError("Validation split contains defective images")
    labels = test.label_index.to_numpy().astype(int)
    groups = np.array([physical_part(p) for p in test.image_path])

    # The PyTorch baseline goes last: torch_scorer moves the model to the CPU.
    scorers = {name: (scorer_for(name, path), path) for name, path in paths.items()}
    scorers = {"torch_fp32": (torch_scorer(model), None), **scorers}

    formats = {}
    for name, (scorer, path) in scorers.items():
        val_scores, _ = timed_map(scorer, val.image_path, load=load_image, warmup=0)
        record = threshold_record(np.array(val_scores), config.threshold.target_false_alarm_rate)
        save_threshold(config.results_dir / "thresholds.json", f"{key}/{name}", record)

        test_scores, latency = timed_map(scorer, test.image_path, load=load_image, warmup=warmup)
        image = summarize(
            np.array(test_scores),
            labels,
            record["threshold"],
            config.evaluation.fixed_false_alarm_rate,
            config.evaluation.bootstrap_samples,
            config.seed,
            groups=groups,
            names=METRICS,
        )
        rows = [("validation", p, 0, s) for p, s in zip(val.image_path, val_scores, strict=True)]
        rows += [
            ("test", p, int(y), s)
            for p, y, s in zip(test.image_path, labels, test_scores, strict=True)
        ]
        write_scores(out_dir / f"scores_{name}.csv", rows)
        formats[name] = {
            "size_mb": None if path is None else model_size_mb(path),
            "threshold": record["threshold"],
            "n_validation_images": record["n_validation_images"],
            "image": {m: image[m] for m in METRICS},
            "latency": latency,
        }
        auroc = image["image_auroc"]["value"]
        print(f"{key} {name}: {latency['median_ms']:.1f} ms/image, AUROC {auroc:.3f}")

    result = {"environment": environment(), "formats": formats}
    (out_dir / "export.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out_dir / "export.md").write_text(markdown_table(formats), encoding="utf-8")


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="category config")
    parser.add_argument("--model", required=True, type=Path, help="configs/models/<model>.yaml")
    parser.add_argument("--warmup", type=int, default=10, help="untimed runs before timing")
    args = parser.parse_args()
    run(load_config(args.config, args.model), args.warmup)


if __name__ == "__main__":
    main()
