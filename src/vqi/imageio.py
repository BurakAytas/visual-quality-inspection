"""Load images the same way everywhere: export benchmark, API and tests (no torch needed)."""

from pathlib import Path

import cv2
import numpy as np


def load_rgb(path: str | Path) -> np.ndarray:
    """Read an image file as (H, W, 3) RGB float32 in [0, 1]."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return decode_to_rgb(image)


def decode_bytes(data: bytes) -> np.ndarray:
    """Decode an encoded image (PNG, JPEG, ...) to (H, W, 3) RGB float32 in [0, 1]."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not a readable image")
    return decode_to_rgb(image)


def decode_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """OpenCV BGR uint8 to RGB float32 in [0, 1]."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def resize_to(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to (height, width) if needed; the exported models take one fixed size."""
    height, width = size
    if image.shape[:2] == (height, width):
        return image
    shrinking = image.shape[0] > height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)
