from pathlib import Path

import cv2
import numpy as np
import pytest

from vqi.imageio import decode_bytes, load_rgb, resize_to


def test_load_rgb_returns_rgb_floats(tmp_path: Path) -> None:
    bgr = np.zeros((4, 6, 3), np.uint8)
    bgr[..., 0] = 255  # pure blue in OpenCV's BGR order
    path = tmp_path / "blue.png"
    cv2.imwrite(str(path), bgr)
    image = load_rgb(path)
    assert image.shape == (4, 6, 3) and image.dtype == np.float32
    assert image[0, 0].tolist() == [0.0, 0.0, 1.0]  # blue is the last channel in RGB


def test_decode_bytes_matches_load_rgb(tmp_path: Path) -> None:
    path = tmp_path / "x.png"
    cv2.imwrite(str(path), np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8))
    assert np.array_equal(decode_bytes(path.read_bytes()), load_rgb(path))


def test_bad_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_rgb(tmp_path / "missing.png")
    with pytest.raises(ValueError):
        decode_bytes(b"not an image")


def test_resize_to_is_a_no_op_at_the_right_size() -> None:
    image = np.random.default_rng(0).random((10, 20, 3)).astype(np.float32)
    assert resize_to(image, (10, 20)) is image
    assert resize_to(image, (5, 8)).shape == (5, 8, 3)
    assert resize_to(image, (40, 60)).shape == (40, 60, 3)
