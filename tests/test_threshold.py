import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from vqi.threshold import (
    load_threshold,
    save_threshold,
    threshold_for_false_alarm_rate,
    threshold_record,
)


def test_threshold_function_cannot_receive_labels() -> None:
    # The policy must not see defect labels: its only inputs are scores and the target rate.
    params = list(inspect.signature(threshold_for_false_alarm_rate).parameters)
    assert params == ["normal_scores", "target_false_alarm_rate"]
    with pytest.raises(TypeError):
        threshold_for_false_alarm_rate(np.ones(10), 0.1, labels=np.zeros(10))  # type: ignore[call-arg]


def test_threshold_is_an_observed_score_with_at_most_target_false_alarms() -> None:
    scores = np.arange(1000, dtype=float)
    threshold = threshold_for_false_alarm_rate(scores, 0.01)
    assert threshold in scores
    assert (scores > threshold).mean() <= 0.01
    assert (scores >= threshold).mean() > 0.01  # one step lower would exceed the target


def test_threshold_generalises_to_new_good_images() -> None:
    rng = np.random.default_rng(0)
    threshold = threshold_for_false_alarm_rate(rng.normal(size=20_000), 0.01)
    new_good = rng.normal(size=200_000)
    assert 0.007 < (new_good > threshold).mean() < 0.013


def test_too_few_images_gives_the_maximum_score() -> None:
    scores = np.array([0.1, 0.5, 0.3])
    assert threshold_for_false_alarm_rate(scores, 0.01) == 0.5


@pytest.mark.parametrize(
    ("scores", "rate"),
    [(np.array([]), 0.01), (np.array([1.0, np.nan]), 0.01), (np.ones(5), 0.0), (np.ones(5), 1.0)],
    ids=["empty", "nan", "rate-zero", "rate-one"],
)
def test_invalid_input_is_rejected(scores: np.ndarray, rate: float) -> None:
    with pytest.raises(ValueError):
        threshold_for_false_alarm_rate(scores, rate)


def test_record_is_saved_with_audit_fields_and_keeps_other_keys(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    save_threshold(path, "padim/carpet", threshold_record(np.arange(10.0), 0.1))
    save_threshold(path, "patchcore/fabric", threshold_record(np.arange(200.0), 0.01))

    data = json.loads(path.read_text())
    assert set(data) == {"padim/carpet", "patchcore/fabric"}
    record = data["patchcore/fabric"]
    assert record["n_validation_images"] == 200
    assert record["quantile"] == pytest.approx(0.99)
    assert record["validation_false_alarm_rate"] <= 0.01
    assert load_threshold(path, "patchcore/fabric") == record["threshold"] == 197.0
