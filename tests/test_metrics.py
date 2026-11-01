import numpy as np
import pytest

from vqi.metrics import bootstrap_ci, miss_rate_at_false_alarm_rate, operating_point, summarize

# 4 good parts, 4 defective parts; threshold 0.5 rejects scores above it.
SCORES = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.6, 0.8, 0.9])
LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 1])


def test_operating_point_counts_and_rates() -> None:
    point = operating_point(SCORES, LABELS, 0.5)
    assert (point["tp"], point["fp"], point["fn"], point["tn"]) == (3, 1, 1, 3)
    assert point["precision"] == pytest.approx(0.75)
    assert point["recall"] == pytest.approx(0.75)
    assert point["f1"] == pytest.approx(0.75)
    assert point["false_alarm_rate"] == pytest.approx(0.25)
    assert point["miss_rate"] == pytest.approx(0.25)


def test_score_equal_to_threshold_is_accepted() -> None:
    assert operating_point(np.array([0.5, 0.9]), np.array([1, 1]), 0.5)["fn"] == 1


def test_no_rejections_gives_zero_precision_not_an_error() -> None:
    assert operating_point(SCORES, LABELS, 10.0)["precision"] == 0.0


def test_miss_rate_at_fixed_false_alarm_rate() -> None:
    # 1% false alarms on 4 good parts -> threshold is the highest good score (0.7),
    # so defects scoring 0.4 and 0.6 are missed.
    assert miss_rate_at_false_alarm_rate(SCORES, LABELS, 0.01) == pytest.approx(0.5)


def test_bootstrap_ci_brackets_the_value_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    labels = np.repeat([0, 1], 100)
    scores = rng.normal(size=200) + labels
    recall = lambda s, y: operating_point(s, y, 0.5)["recall"]  # noqa: E731
    low, high = bootstrap_ci(scores, labels, recall, 500, seed=3)
    assert low < recall(scores, labels) < high
    assert bootstrap_ci(scores, labels, recall, 500, seed=3) == (low, high)


def test_perfect_separation_has_a_zero_width_auroc_interval() -> None:
    summary = summarize(SCORES * 0 + LABELS, LABELS, 0.5, 0.01, 200, seed=0)
    assert summary["image_auroc"]["value"] == 1.0
    assert summary["image_auroc"]["ci95"] == (1.0, 1.0)


def test_summary_contents() -> None:
    summary = summarize(SCORES, LABELS, 0.5, 0.01, 200, seed=0)
    # 14 of the 16 (defective, good) pairs are ranked correctly; 0.7 beats 0.4 and 0.6.
    assert summary["image_auroc"]["value"] == pytest.approx(14 / 16)
    assert summary["counts"] == {"tp": 3, "fp": 1, "fn": 1, "tn": 3}
    assert (summary["n_test_good"], summary["n_test_defective"]) == (4, 4)


def test_summary_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        summarize(SCORES, np.zeros(8, dtype=int), 0.5, 0.01, 10, seed=0)


def test_grouped_bootstrap_is_wider_when_images_repeat_one_part() -> None:
    # 10 good + 10 defective parts, each photographed 6 times with the same score.
    rng = np.random.default_rng(0)
    part_scores = np.concatenate([rng.normal(0, 1, 10), rng.normal(1, 1, 10)])
    part_labels = np.repeat([0, 1], 10)
    scores = np.repeat(part_scores, 6)
    labels = np.repeat(part_labels, 6)
    groups = np.repeat(np.arange(20), 6)
    recall = lambda s, y: operating_point(s, y, 0.5)["recall"]  # noqa: E731

    naive = bootstrap_ci(scores, labels, recall, 500, seed=0)
    grouped = bootstrap_ci(scores, labels, recall, 500, seed=0, groups=groups)
    assert grouped[1] - grouped[0] > 1.5 * (naive[1] - naive[0])


def test_summary_counts_parts_and_rejects_mixed_groups() -> None:
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    summary = summarize(SCORES, LABELS, 0.5, 0.01, 50, seed=0, groups=groups)
    assert summary["n_test_parts"] == 4
    with pytest.raises(ValueError, match="mixes"):
        summarize(SCORES, LABELS, 0.5, 0.01, 50, seed=0, groups=np.zeros(8))
