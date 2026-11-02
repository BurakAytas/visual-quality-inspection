"""Image-level metrics on raw anomaly scores, with bootstrap confidence intervals.

Labels: 0 = good part, 1 = defective part. A part is rejected (NOK) if ``score > threshold``.
"""

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vqi.threshold import threshold_for_false_alarm_rate

MetricFn = Callable[[np.ndarray, np.ndarray], float]

# MVTec AD 2 photographs each test part under several lights: "000_regular.png",
# "000_overexposed.png", "000_shift_2.png", ... These are one part, not six samples.
LIGHTING_SUFFIX = re.compile(r"_(regular|overexposed|underexposed|shift_\d+)$")


def physical_part(image: str) -> str:
    """Identify the physical part an image shows, e.g. ``fabric/test_public/bad/000``."""
    path = Path(image)
    return str(path.parent / LIGHTING_SUFFIX.sub("", path.stem))


def operating_point(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """Counts and rates at one fixed threshold."""
    rejected = scores > threshold
    defective = labels == 1
    tp = int(np.sum(rejected & defective))
    fp = int(np.sum(rejected & ~defective))
    fn = int(np.sum(~rejected & defective))
    tn = int(np.sum(~rejected & ~defective))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarm_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "miss_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
    }


def miss_rate_at_false_alarm_rate(
    scores: np.ndarray, labels: np.ndarray, false_alarm_rate: float
) -> float:
    """Share of defects missed when the threshold gives ``false_alarm_rate`` on these good images.

    This threshold is read from the evaluated set itself, so it is a model-comparison
    metric (like TPR at a fixed FPR), not a threshold that could be deployed.
    """
    threshold = threshold_for_false_alarm_rate(scores[labels == 0], false_alarm_rate)
    return float((scores[labels == 1] <= threshold).mean())


def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: MetricFn,
    n_samples: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[float, float]:
    """95% percentile interval of ``metric`` over stratified bootstrap resamples.

    Good and defective parts are resampled separately so every resample keeps the
    original class balance. ``groups`` marks images of the same physical part (for
    example one part photographed under several lights); such images are drawn
    together, because they are not independent samples. Thresholds stay fixed;
    only the test parts vary.
    """
    groups = np.arange(len(scores)) if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    good = [g for g, idx in members.items() if labels[idx[0]] == 0]
    bad = [g for g, idx in members.items() if labels[idx[0]] == 1]
    values = []
    for _ in range(n_samples):
        drawn = [*rng.choice(good, size=len(good)), *rng.choice(bad, size=len(bad))]
        idx = np.concatenate([members[g] for g in drawn])
        values.append(metric(scores[idx], labels[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def summarize(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    fixed_false_alarm_rate: float,
    bootstrap_samples: int,
    seed: int,
    groups: np.ndarray | None = None,
    names: tuple[str, ...] | None = None,
) -> dict:
    """Image-level test metrics: value and 95% CI for each, plus raw counts.

    ``groups`` (optional) gives the physical part of each image; see ``bootstrap_ci``.
    ``names`` limits which metrics get a value and CI (all by default).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain both good (0) and defective (1) images")
    groups = np.arange(len(scores)) if groups is None else np.asarray(groups)
    for g in np.unique(groups):
        if len(set(labels[groups == g])) != 1:
            raise ValueError(f"group {g!r} mixes good and defective images")

    def at_threshold(name: str) -> MetricFn:
        return lambda s, y: operating_point(s, y, threshold)[name]

    metrics: dict[str, MetricFn] = {
        "image_auroc": lambda s, y: float(roc_auc_score(y, s)),
        "f1": at_threshold("f1"),
        "precision": at_threshold("precision"),
        "recall": at_threshold("recall"),
        "false_alarm_rate": at_threshold("false_alarm_rate"),
        "miss_rate": at_threshold("miss_rate"),
        "miss_rate_at_fixed_far": lambda s, y: miss_rate_at_false_alarm_rate(
            s, y, fixed_false_alarm_rate
        ),
    }
    summary: dict = {
        name: {
            "value": fn(scores, labels),
            "ci95": bootstrap_ci(scores, labels, fn, bootstrap_samples, seed, groups),
        }
        for name, fn in metrics.items()
        if names is None or name in names
    }
    point = operating_point(scores, labels, threshold)
    summary["counts"] = {k: point[k] for k in ("tp", "fp", "fn", "tn")}
    summary["n_test_good"] = int((labels == 0).sum())
    summary["n_test_defective"] = int((labels == 1).sum())
    summary["n_test_parts"] = len(np.unique(groups))
    summary["fixed_false_alarm_rate"] = fixed_false_alarm_rate
    return summary
