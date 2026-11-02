"""Defect type of a rejected part: crop at the anomaly peak, then classify the crop.

The anomaly model says where something is wrong; the classifier only names it. Crops are
taken where the anomaly map peaks, in training as on the line, so the classifier never
relies on ground-truth masks it would not have in operation (D54).

Everything here runs without torch: the API classifies with the exported OpenVINO model.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from sklearn.metrics import confusion_matrix, f1_score

# Crop side in pixels of the full camera image; 256 of 1024 px for leather. A crop centred
# on the map peak touched the defect mask in every leather test image (D54).
CROP_SIDE = 256
# Classifier input side; the crop is resized to it.
INPUT_SIDE = 224
# HOG baseline: grey crop of this side, 16 px cells, 2x2-cell blocks, 9 orientations.
HOG_SIDE = 128
HOG_CELL = 16
HOG_ORIENTATIONS = 9


def crop_at_peak(image: np.ndarray, anomaly_map: np.ndarray, side: int = CROP_SIDE) -> np.ndarray:
    """Square ``side`` crop of ``image`` centred on the anomaly-map maximum.

    The map may be smaller than the image (PatchCore's is 256 px for a 1024 px input); the
    peak is scaled to image pixels. Near a border the crop is shifted to stay inside.
    """
    height, width = image.shape[:2]
    if side > min(height, width):
        raise ValueError(f"crop side {side} exceeds the image {width}x{height}")
    row, col = np.unravel_index(int(np.argmax(anomaly_map)), anomaly_map.shape)
    center_y = (row + 0.5) * height / anomaly_map.shape[0]
    center_x = (col + 0.5) * width / anomaly_map.shape[1]
    top = int(np.clip(round(center_y - side / 2), 0, height - side))
    left = int(np.clip(round(center_x - side / 2), 0, width - side))
    return image[top : top + side, left : left + side]


def hog_features(crop: np.ndarray) -> np.ndarray:
    """HOG descriptor of an RGB float crop in [0, 1], on its grey version (texture only).

    scikit-image (training extra), because OpenCV 5 no longer ships ``HOGDescriptor`` (D55).
    """
    from skimage.feature import hog

    grey = cv2.cvtColor(crop.astype(np.float32), cv2.COLOR_RGB2GRAY)
    grey = cv2.resize(grey, (HOG_SIDE, HOG_SIDE), interpolation=cv2.INTER_AREA)
    return hog(
        grey,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=(HOG_CELL, HOG_CELL),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of parts given their true defect type."""
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """F1 averaged over classes, each class weighted equally."""
    return float(
        f1_score(y_true, y_pred, labels=range(n_classes), average="macro", zero_division=0)
    )


def classification_summary(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], n_bootstrap: int, seed: int
) -> dict:
    """Accuracy and macro F1 with 95% bootstrap intervals, recall per class, confusion matrix.

    Each image is one physical part (leather), so images are resampled one by one. Every
    class keeps its count ``n`` next to its recall: with about 18 parts per class one error
    moves a recall by about 5 points.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    metrics = {
        "accuracy": accuracy,
        "macro_f1": lambda t, p: macro_f1(t, p, len(classes)),
    }
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(y_true), len(y_true)) for _ in range(n_bootstrap)]
    summary = {}
    for name, fn in metrics.items():
        values = [fn(y_true[idx], y_pred[idx]) for idx in draws]
        low, high = np.percentile(values, [2.5, 97.5])
        summary[name] = {"value": fn(y_true, y_pred), "ci95": [float(low), float(high)]}
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    summary["per_class"] = {
        name: {
            "n": int(matrix[i].sum()),
            "correct": int(matrix[i, i]),
            "recall": float(matrix[i, i] / matrix[i].sum()),
        }
        for i, name in enumerate(classes)
    }
    summary["confusion"] = {"classes": classes, "rows_true_cols_predicted": matrix.tolist()}
    return summary


@dataclass(frozen=True)
class DefectType:
    """Predicted defect type of one part and the classifier's probability for it."""

    name: str
    probability: float


class DefectClassifier:
    """Exported crop classifier, run with the plain OpenVINO runtime on the CPU.

    The graph takes a [1, 3, INPUT_SIDE, INPUT_SIDE] RGB crop in [0, 1] (normalisation is
    inside) and returns one logit per class; the class names are in ``classes.json``.
    """

    def __init__(self, model_file: Path) -> None:
        """Compile the model and read the class names saved next to it."""
        self.compiled = ov.Core().compile_model(str(model_file), "CPU")
        self.classes = json.loads((model_file.parent / "classes.json").read_text("utf-8"))

    def __call__(self, crop: np.ndarray) -> DefectType:
        """Classify one RGB float crop of any size."""
        resized = cv2.resize(crop, (INPUT_SIDE, INPUT_SIDE), interpolation=cv2.INTER_AREA)
        batch = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32)
        logits = np.ravel(self.compiled(batch)[0]).astype(np.float64)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        best = int(np.argmax(probabilities))
        return DefectType(self.classes[best], float(probabilities[best]))
