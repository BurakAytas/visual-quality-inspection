"""Decision threshold chosen from good validation images only.

On a real line, defect samples are rare, so the threshold cannot depend on them.
We pick the score that only ``target_false_alarm_rate`` of good parts exceed.
The function takes scores only: it has no way to see a label, so it cannot leak
test or defect information.
"""

import json
from pathlib import Path

import numpy as np


def threshold_for_false_alarm_rate(
    normal_scores: np.ndarray, target_false_alarm_rate: float
) -> float:
    """Return the score above which a part is rejected (NOK if ``score > threshold``).

    With n good images we allow k = floor(rate * n) of them above the threshold, so the
    threshold is the (k+1)-th highest score: the lowest observed score that keeps the
    false-alarm rate on these images at or below the target. This is the
    ``1 - rate`` quantile. With fewer than 1/rate images k is 0 and the threshold is
    the maximum normal score; the false-alarm rate cannot be resolved below 1/n.
    """
    scores = np.asarray(normal_scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("normal_scores must be a non-empty 1-D array")
    if not np.isfinite(scores).all():
        raise ValueError("normal_scores contains NaN or inf")
    if not 0 < target_false_alarm_rate < 1:
        raise ValueError("target_false_alarm_rate must be between 0 and 1")
    allowed_false_alarms = int(np.floor(target_false_alarm_rate * scores.size))
    return float(np.sort(scores)[scores.size - 1 - allowed_false_alarms])


def review_band(
    normal_scores: np.ndarray, threshold: float, review_rate: float
) -> tuple[float, float]:
    """Scores in ``(lower, upper]`` go to an operator; the decision itself is unchanged.

    ``lower`` is the score that only ``review_rate`` of the good validation images exceed
    (the same quantile rule as the threshold), so about that share of good parts is
    reviewed. The band is mirrored above the threshold, because a part just above it is
    as uncertain as one just below.
    """
    if not 0 < review_rate < 1:
        raise ValueError("review_rate must be between 0 and 1")
    lower = threshold_for_false_alarm_rate(normal_scores, review_rate)
    if lower > threshold:
        raise ValueError("review_rate must be larger than the threshold's false-alarm rate")
    return lower, threshold + (threshold - lower)


def threshold_record(normal_scores: np.ndarray, target_false_alarm_rate: float) -> dict:
    """Threshold plus the facts needed to audit it later."""
    scores = np.asarray(normal_scores, dtype=float)
    threshold = threshold_for_false_alarm_rate(scores, target_false_alarm_rate)
    return {
        "threshold": threshold,
        "quantile": 1 - target_false_alarm_rate,
        "target_false_alarm_rate": target_false_alarm_rate,
        "n_validation_images": int(scores.size),
        "validation_false_alarm_rate": float((scores > threshold).mean()),
        "score": "raw anomaly score (anomalib normalization disabled)",
    }


def save_threshold(path: Path, key: str, record: dict) -> None:
    """Add or replace ``record`` under ``key`` in a JSON file shared by all categories."""
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[key] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_threshold(path: Path, key: str) -> float:
    """Read the threshold stored under ``key``."""
    return float(json.loads(path.read_text(encoding="utf-8"))[key]["threshold"])
