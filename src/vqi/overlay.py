"""Anomaly-map overlays for the failure gallery and, later, the API heatmap."""

from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 512  # saved overlays are downscaled to keep the repository small


def heatmap_overlay(
    image_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    vmin: float,
    vmax: float,
    gt_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Blend an anomaly map onto an image.

    The map is resized to the image and scaled from ``vmin`` (fully transparent) to
    ``vmax`` (strongest colour). Using fixed values per model keeps colours comparable
    between images. A ground-truth mask, if given, is drawn as a green outline.
    """
    height, width = image_bgr.shape[:2]
    resized = cv2.resize(anomaly_map.astype(np.float32), (width, height), cv2.INTER_LINEAR)
    strength = np.clip((resized - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    colour = cv2.applyColorMap((strength * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (0.6 * strength)[..., None]  # low scores leave the image visible
    out = (image_bgr * (1 - alpha) + colour * alpha).astype(np.uint8)
    if gt_mask is not None:
        mask = cv2.resize(gt_mask.astype(np.uint8), (width, height), cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 0), max(2, width // 300))
    return out


def save_overlay(
    out_path: Path,
    image_path: str,
    anomaly_map: np.ndarray,
    vmin: float,
    vmax: float,
    caption: str,
    mask_path: str | None = None,
) -> None:
    """Read an image (and its mask), draw the overlay with a caption bar and save it as JPEG."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0 if mask_path else None
    overlay = heatmap_overlay(image, anomaly_map, vmin, vmax, mask)

    scale = MAX_SIDE / max(overlay.shape[:2])
    if scale < 1:
        overlay = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    bar = np.full((28, overlay.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, caption, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([bar, overlay]), [cv2.IMWRITE_JPEG_QUALITY, 85])
