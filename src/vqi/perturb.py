"""Fixed image perturbations for robustness tests: brightness, blur and noise.

Images are float arrays in [0, 1], shape (H, W, C). Every perturbation is deterministic:
the noise for an image is seeded from the global seed and the image's own pixels, so the
result does not depend on loading order or the number of data-loader workers.
"""

import zlib
from collections.abc import Callable

import cv2
import numpy as np

from vqi.config import RobustnessConfig

Perturbation = Callable[[np.ndarray], np.ndarray]


def brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale all pixel values (a darker or brighter scene), clipped to [0, 1]."""
    return np.clip(image * factor, 0.0, 1.0)


def blur(image: np.ndarray, sigma_fraction: float) -> np.ndarray:
    """Gaussian blur (an out-of-focus camera) with sigma relative to the image width."""
    sigma = sigma_fraction * image.shape[1]
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


def noise(image: np.ndarray, std: float, seed: int) -> np.ndarray:
    """Add Gaussian sensor noise, seeded by ``seed`` and the image content."""
    image_seed = zlib.crc32(np.ascontiguousarray(image, dtype=np.float32).tobytes()) ^ seed
    rng = np.random.default_rng(image_seed)
    return np.clip(image + rng.normal(0.0, std, image.shape), 0.0, 1.0).astype(image.dtype)


def perturbations(config: RobustnessConfig, seed: int) -> dict[str, Perturbation]:
    """All test conditions except the clean one, keyed by a short name for result files."""
    conditions: dict[str, Perturbation] = {
        f"brightness_{factor:g}": lambda img, f=factor: brightness(img, f)
        for factor in config.brightness_factors
    }
    conditions["blur"] = lambda img: blur(img, config.blur_sigma_fraction)
    conditions["noise"] = lambda img: noise(img, config.noise_std, seed)
    return conditions
