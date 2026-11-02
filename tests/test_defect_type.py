"""Tests for the defect-type crop, features, metrics and OpenVINO classifier."""

import json
from pathlib import Path

import numpy as np
import pytest

from vqi.defect_type import (
    HOG_CELL,
    HOG_ORIENTATIONS,
    HOG_SIDE,
    DefectClassifier,
    classification_summary,
    crop_at_peak,
    hog_features,
    macro_f1,
)


def test_crop_at_peak_scales_map_to_image() -> None:
    """A peak in map cell (2, 5) of an 8x8 map over a 64 px image centres the crop there."""
    image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)[..., None].repeat(3, -1)
    anomaly_map = np.zeros((8, 8))
    anomaly_map[2, 5] = 1.0
    crop = crop_at_peak(image, anomaly_map, side=16)
    # Cell centre (20, 44) in image pixels, so rows 12..27 and columns 36..51.
    assert crop.shape == (16, 16, 3)
    np.testing.assert_array_equal(crop, image[12:28, 36:52])


def test_crop_at_peak_stays_inside_border() -> None:
    """A peak in the last corner cell gives the bottom-right crop, not a short one."""
    image = np.random.default_rng(0).random((64, 48, 3))
    anomaly_map = np.zeros((8, 6))
    anomaly_map[-1, -1] = 1.0
    crop = crop_at_peak(image, anomaly_map, side=20)
    np.testing.assert_array_equal(crop, image[-20:, -20:])


def test_crop_at_peak_rejects_crop_larger_than_image() -> None:
    """The crop cannot be larger than the image."""
    with pytest.raises(ValueError, match="exceeds"):
        crop_at_peak(np.zeros((32, 64, 3)), np.zeros((4, 4)), side=40)


def test_hog_features_have_fixed_length() -> None:
    """Any crop size gives the same, non-empty descriptor."""
    pytest.importorskip("skimage")
    cells = HOG_SIDE // HOG_CELL
    rng = np.random.default_rng(1)
    pattern = rng.random((96, 96)) > 0.5
    features = hog_features(np.stack([pattern] * 3, -1).astype(np.float32))
    assert features.shape == ((cells - 1) ** 2 * 4 * HOG_ORIENTATIONS,)
    assert np.abs(features).sum() > 0
    larger = hog_features(rng.random((300, 300, 3)).astype(np.float32))
    assert larger.shape == features.shape


def test_macro_f1_weights_classes_equally() -> None:
    """One class all right, one all wrong: macro F1 is the mean of the per-class F1."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    # Class 0: precision 4/6, recall 1, F1 0.8. Class 1: F1 0. Class 2 is absent: F1 0.
    assert macro_f1(y_true, y_pred, 2) == pytest.approx(0.4)
    assert macro_f1(y_true, y_pred, 3) == pytest.approx(0.8 / 3)


def test_classification_summary_counts_and_intervals() -> None:
    """Per-class counts, confusion rows and an interval around the point value."""
    classes = ["cut", "fold", "poke"]
    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 2, 2, 2, 2, 0])
    summary = classification_summary(y_true, y_pred, classes, n_bootstrap=200, seed=0)
    assert summary["accuracy"]["value"] == pytest.approx(0.6)
    low, high = summary["accuracy"]["ci95"]
    assert low < 0.6 < high
    assert summary["per_class"]["cut"] == {"n": 3, "correct": 2, "recall": pytest.approx(2 / 3)}
    assert summary["per_class"]["fold"] == {"n": 2, "correct": 0, "recall": 0.0}
    assert summary["per_class"]["poke"] == {"n": 5, "correct": 4, "recall": pytest.approx(0.8)}
    assert summary["confusion"]["rows_true_cols_predicted"] == [[2, 1, 0], [0, 0, 2], [1, 0, 4]]
    again = classification_summary(y_true, y_pred, classes, n_bootstrap=200, seed=0)
    assert again["macro_f1"]["ci95"] == summary["macro_f1"]["ci95"]


def mean_colour_model(out_dir: Path, classes: list[str]) -> Path:
    """OpenVINO model whose logits are 10x the mean of each RGB channel, with classes.json."""
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops

    crop = ops.parameter([1, 3, 224, 224], np.float32, name="crop")
    mean = ops.reduce_mean(crop, ops.constant(np.array([2, 3])), keep_dims=False)
    logits = ops.multiply(mean, ops.constant(np.float32(10.0)))
    model_file = out_dir / "model.xml"
    ov.save_model(ov.Model([logits], [crop]), model_file)
    (out_dir / "classes.json").write_text(json.dumps(classes), encoding="utf-8")
    return model_file


def test_defect_classifier_names_the_top_class(tmp_path: Path) -> None:
    """A green crop of any size is the second class, with the softmax probability."""
    classifier = DefectClassifier(mean_colour_model(tmp_path, ["red", "green", "blue"]))
    crop = np.zeros((300, 260, 3), dtype=np.float32)
    crop[..., 1] = 1.0
    result = classifier(crop)
    assert result.name == "green"
    # Logits 0, 10, 0.
    assert result.probability == pytest.approx(np.exp(10) / (np.exp(10) + 2), rel=1e-4)
