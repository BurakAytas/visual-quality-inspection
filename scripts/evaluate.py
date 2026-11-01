"""Compute test metrics from saved raw scores (runs on CPU, no model needed).

Reads ``results/<model>/<category>/scores.csv``, the threshold from
``results/thresholds.json`` and the pixel metrics, then writes
``results/metrics.json``, ``results/metrics.md`` and one score plot per category.

Example:
    python scripts/evaluate.py configs/fabric.yaml configs/leather.yaml configs/carpet.yaml
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vqi.config import Config, load_config
from vqi.metrics import summarize
from vqi.threshold import load_threshold

# Categorical colours (fixed order) and text colour, from a colour-blind-checked palette.
COLOURS = {"validation good": "#2a78d6", "test good": "#1baf7a", "test defective": "#eb6834"}
INK = "#3b3b3b"

# MVTec AD 2 photographs each test part under several lights: "000_regular.png",
# "000_overexposed.png", "000_shift_2.png", ... These are one part, not six samples.
LIGHTING_SUFFIX = re.compile(r"_(regular|overexposed|underexposed|shift_\d+)$")


def physical_part(image: str) -> str:
    """Identify the physical part an image shows, e.g. ``fabric/test_public/bad/000``."""
    path = Path(image)
    return str(path.parent / LIGHTING_SUFFIX.sub("", path.stem))


def read_scores(path: Path) -> dict[str, np.ndarray]:
    """Split the scores CSV into validation scores, test scores and test labels."""
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    val = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"]
    return {
        "val_scores": np.array([float(r["score"]) for r in val]),
        "test_scores": np.array([float(r["score"]) for r in test]),
        "test_labels": np.array([int(r["label"]) for r in test]),
        "test_parts": np.array([physical_part(r["image"]) for r in test]),
    }


def plot_scores(path: Path, title: str, scores: dict[str, np.ndarray], threshold: float) -> None:
    """Overlaid score histograms of the three image groups with the threshold line."""
    groups = {
        "validation good": scores["val_scores"],
        "test good": scores["test_scores"][scores["test_labels"] == 0],
        "test defective": scores["test_scores"][scores["test_labels"] == 1],
    }
    bins = np.histogram_bin_edges(np.concatenate(list(groups.values())), bins=40)
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=120)
    for name, values in groups.items():
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=2,
            color=COLOURS[name],
            label=f"{name} (n={values.size})",
            density=True,
        )
    ax.axvline(threshold, color=INK, linestyle="--", linewidth=1.5)
    ax.annotate(
        "threshold",
        (threshold, 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -12),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )
    ax.set(title=title, xlabel="raw anomaly score", ylabel="density")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate(config: Config) -> tuple[str, dict]:
    """Metrics for one model and category."""
    key = f"{config.model.key}/{config.dataset.category}"
    out_dir = config.results_dir / config.model.key / config.dataset.category
    scores = read_scores(out_dir / "scores.csv")
    threshold = load_threshold(config.results_dir / "thresholds.json", key)
    image = summarize(
        scores["test_scores"],
        scores["test_labels"],
        threshold,
        config.evaluation.fixed_false_alarm_rate,
        config.evaluation.bootstrap_samples,
        config.seed,
        groups=scores["test_parts"],
    )
    pixel = json.loads((out_dir / "pixel_metrics.json").read_text(encoding="utf-8"))
    plot_scores(out_dir / "score_distribution.png", key, scores, threshold)
    return key, {
        "dataset": config.dataset.name,
        "threshold": threshold,
        "n_validation_images": int(scores["val_scores"].size),
        "image": image,
        "pixel": pixel,
    }


def fmt(metric: dict) -> str:
    """Value with its 95% CI, e.g. ``0.912 [0.871, 0.950]``."""
    low, high = metric["ci95"]
    return f"{metric['value']:.3f} [{low:.3f}, {high:.3f}]"


def markdown_table(results: dict) -> str:
    """Metrics table with 95% bootstrap CIs."""
    lines = [
        "| Model / category | Image AUROC | F1 | Precision | Recall | False-alarm rate | Miss rate "
        "| Miss rate @ fixed FAR | AU-PRO 0.3 | AU-PRO 0.05 | Val good | Test good / defective "
        "| Test parts |",
        "|---" * 13 + "|",
    ]
    for key, r in sorted(results.items()):
        m = r["image"]
        lines.append(
            f"| {key} | {fmt(m['image_auroc'])} | {fmt(m['f1'])} | {fmt(m['precision'])} "
            f"| {fmt(m['recall'])} | {fmt(m['false_alarm_rate'])} | {fmt(m['miss_rate'])} "
            f"| {fmt(m['miss_rate_at_fixed_far'])} | {r['pixel']['pixel_aupro_0.3']:.3f} "
            f"| {r['pixel']['pixel_aupro_0.05']:.3f} | {r['n_validation_images']} "
            f"| {m['n_test_good']} / {m['n_test_defective']} | {m['n_test_parts']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Evaluate every given config and merge into results/metrics.json."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="+", type=Path)
    configs = [load_config(p) for p in parser.parse_args().configs]

    results_dir = configs[0].results_dir
    metrics_path = results_dir / "metrics.json"
    results = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    for config in configs:
        key, result = evaluate(config)
        results[key] = result
        print(f"{key}: image AUROC {fmt(result['image']['image_auroc'])}")

    metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (results_dir / "metrics.md").write_text(markdown_table(results), encoding="utf-8")


if __name__ == "__main__":
    main()
