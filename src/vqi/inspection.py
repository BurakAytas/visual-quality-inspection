"""Inspect one image: exported model, threshold decision, review flag and heatmap.

The API runs the exported OpenVINO model with the plain ``openvino`` runtime instead of
anomalib's ``OpenVINOInferencer``, which imports torch and anomalib; the API image then
needs neither (D49). The golden-set test checks that both give the same scores.
"""

import csv
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from vqi.config import ServeConfig
from vqi.defect_type import CROP_SIDE, DefectClassifier, DefectType, crop_at_peak
from vqi.imageio import resize_to
from vqi.overlay import MAX_SIDE, heatmap_overlay
from vqi.threshold import load_threshold, review_band

# One (H, W, 3) RGB float32 image in [0, 1] at the model's input size -> (raw score, map).
Model = Callable[[np.ndarray], tuple[float, np.ndarray]]
# One RGB float32 crop of any size -> its defect type.
Classifier = Callable[[np.ndarray], DefectType]


class OpenVINOModel:
    """Exported anomalib model, run with the plain OpenVINO runtime on the CPU."""

    def __init__(self, model_file: Path) -> None:
        """Compile the model; its input is a fixed [1, 3, H, W] image (D47)."""
        self.compiled = ov.Core().compile_model(str(model_file), "CPU")
        _, _, height, width = self.compiled.input(0).shape
        self.input_size = (int(height), int(width))

    def __call__(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        """Raw anomaly score and (h, w) anomaly map, as anomalib's exported graph gives them."""
        batch = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)
        outputs = self.compiled(batch)
        score = float(np.ravel(outputs["pred_score"])[0])
        anomaly_map = np.asarray(outputs["anomaly_map"], dtype=np.float32).squeeze()
        return score, anomaly_map


@dataclass(frozen=True)
class Inspection:
    """Result for one image."""

    score: float
    decision: str  # "OK" or "NOK"
    needs_review: bool
    latency_ms: float  # resize, model and classifier; decoding the upload is not included
    heatmap_png: bytes
    defect_type: DefectType | None = None  # NOK parts only, when a classifier is served


def decide(score: float, threshold: float) -> str:
    """NOK if the score is above the threshold, as in every evaluation (D12)."""
    return "NOK" if score > threshold else "OK"


class Inspector:
    """Applies the model, the threshold and the review band to one image."""

    def __init__(
        self,
        model: Model,
        input_size: tuple[int, int],
        threshold: float,
        band: tuple[float, float],
        heatmap_vmin: float,
        model_version: str,
        classifier: Classifier | None = None,
    ) -> None:
        """Store the parts; ``build_inspector`` reads them from a serve config."""
        self.model = model
        self.input_size = input_size
        self.threshold = threshold
        self.band = band
        self.heatmap_vmin = heatmap_vmin
        self.model_version = model_version
        self.classifier = classifier

    def inspect(self, image: np.ndarray) -> Inspection:
        """Score an (H, W, 3) RGB float32 image of any size."""
        start = time.perf_counter()
        resized = resize_to(image, self.input_size)
        score, anomaly_map = self.model(resized)
        decision = decide(score, self.threshold)
        defect_type = None
        if decision == "NOK" and self.classifier is not None:
            # Crop the original pixels where the map peaks (D54); small images crop less.
            side = min(CROP_SIDE, *image.shape[:2])
            defect_type = self.classifier(crop_at_peak(image, anomaly_map, side))
        latency_ms = (time.perf_counter() - start) * 1000
        return Inspection(
            score=score,
            decision=decision,
            needs_review=self.band[0] < score <= self.band[1],
            latency_ms=latency_ms,
            heatmap_png=self.heatmap_png(resized, anomaly_map),
            defect_type=defect_type,
        )

    def heatmap_png(self, image: np.ndarray, anomaly_map: np.ndarray) -> bytes:
        """Overlay coloured from the median good validation score up to the threshold."""
        image_bgr = cv2.cvtColor((image * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        overlay = heatmap_overlay(image_bgr, anomaly_map, self.heatmap_vmin, self.threshold)
        scale = MAX_SIDE / max(overlay.shape[:2])
        if scale < 1:
            overlay = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return cv2.imencode(".png", overlay)[1].tobytes()


def read_validation_scores(path: Path) -> np.ndarray:
    """Good validation scores from a ``scores_<format>.csv`` written by export_model.py."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == "validation"]
    if not rows or any(row["label"] != "0" for row in rows):
        raise ValueError(f"{path} must hold good validation images only")
    return np.array([float(row["score"]) for row in rows])


def git_commit() -> str:
    """Commit of the running code: ``VQI_GIT_COMMIT`` (set in Docker), else git, else unknown."""
    if os.environ.get("VQI_GIT_COMMIT"):
        return os.environ["VQI_GIT_COMMIT"]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def model_version(model_format: str, commit: str, threshold: float) -> str:
    """Format, code commit and threshold: a change in any of them changes the decisions."""
    return f"{model_format}@{commit}+thr={threshold:.6g}"


def build_inspector(config: ServeConfig, commit: str) -> Inspector:
    """Load the model, its threshold and its validation scores named in ``config``."""
    threshold = load_threshold(config.thresholds_file, config.threshold_key)
    good_scores = read_validation_scores(config.validation_scores)
    model = OpenVINOModel(config.model_file)
    return Inspector(
        model=model,
        input_size=model.input_size,
        threshold=threshold,
        band=review_band(good_scores, threshold, config.review_rate),
        heatmap_vmin=float(np.median(good_scores)),
        model_version=model_version(config.model_format, commit, threshold),
        classifier=DefectClassifier(config.classifier_file) if config.classifier_file else None,
    )
