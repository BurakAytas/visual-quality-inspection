"""Image-level metrics on raw anomaly scores, with bootstrap confidence intervals.

Labels: 0 = good part, 1 = defective part. A part is rejected (NOK) if ``score > threshold``.
"""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from vqi.threshold import threshold_for_false_alarm_rate

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def operating_point(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """Counts and rates at one fixed threshold."""
    predicted = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
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
) -> tuple[float, float]:
    """95% percentile interval of ``metric`` over stratified bootstrap resamples.

    Good and defective images are resampled separately so every resample keeps
    the original class counts. Thresholds stay fixed; only the test images vary.
    """
    rng = np.random.default_rng(seed)
    good = np.flatnonzero(labels == 0)
    bad = np.flatnonzero(labels == 1)
    values = []
    for _ in range(n_samples):
        idx = np.concatenate([rng.choice(good, size=good.size), rng.choice(bad, size=bad.size)])
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
) -> dict:
    """All image-level test metrics: value and 95% CI for each, plus raw counts."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain both good (0) and defective (1) images")

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
            "ci95": bootstrap_ci(scores, labels, fn, bootstrap_samples, seed),
        }
        for name, fn in metrics.items()
    }
    point = operating_point(scores, labels, threshold)
    summary["counts"] = {k: point[k] for k in ("tp", "fp", "fn", "tn")}
    summary["n_test_good"] = int((labels == 0).sum())
    summary["n_test_defective"] = int((labels == 1).sum())
    summary["fixed_false_alarm_rate"] = fixed_false_alarm_rate
    return summary
