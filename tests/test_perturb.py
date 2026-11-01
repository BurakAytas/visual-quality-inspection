import numpy as np
import pytest

from vqi.config import RobustnessConfig
from vqi.perturb import blur, brightness, noise, perturbations

RNG = np.random.default_rng(0)
IMAGE = RNG.uniform(0.2, 0.8, (64, 80, 3)).astype(np.float32)


def test_brightness_scales_and_clips() -> None:
    assert np.allclose(brightness(IMAGE, 0.5), IMAGE * 0.5)
    assert brightness(IMAGE, 3.0).max() == 1.0


def test_blur_smooths_but_keeps_mean() -> None:
    blurred = blur(IMAGE, 0.05)
    assert blurred.shape == IMAGE.shape
    assert blurred.std() < 0.5 * IMAGE.std()
    assert blurred.mean() == pytest.approx(IMAGE.mean(), abs=0.01)


def test_noise_is_reproducible_per_image_and_seed() -> None:
    first = noise(IMAGE, 0.05, seed=42)
    assert np.array_equal(first, noise(IMAGE.copy(), 0.05, seed=42))
    assert not np.array_equal(first, noise(IMAGE, 0.05, seed=43))
    assert (first - IMAGE).std() == pytest.approx(0.05, rel=0.1)


def test_noise_differs_between_images() -> None:
    other = IMAGE[::-1].copy()
    assert not np.allclose(noise(IMAGE, 0.05, 1) - IMAGE, noise(other, 0.05, 1) - other)


def test_perturbation_names_follow_the_config() -> None:
    config = RobustnessConfig(brightness_factors=[0.7, 1.3])
    conditions = perturbations(config, seed=0)
    assert list(conditions) == ["brightness_0.7", "brightness_1.3", "blur", "noise"]
    assert np.allclose(conditions["brightness_1.3"](IMAGE), brightness(IMAGE, 1.3))
