from pathlib import Path

import cv2
import numpy as np

from vqi.overlay import MAX_SIDE, heatmap_overlay, save_overlay

IMAGE = np.full((100, 120, 3), 128, np.uint8)


def test_scores_at_or_below_vmin_leave_the_image_unchanged() -> None:
    out = heatmap_overlay(IMAGE, np.zeros((10, 12)), vmin=0.0, vmax=1.0)
    assert np.array_equal(out, IMAGE)


def test_only_the_anomalous_region_is_coloured() -> None:
    anomaly_map = np.zeros((100, 120))
    anomaly_map[:, 60:] = 1.0
    out = heatmap_overlay(IMAGE, anomaly_map, vmin=0.0, vmax=1.0)
    assert np.array_equal(out[:, :50], IMAGE[:, :50])
    assert not np.array_equal(out[:, 70:], IMAGE[:, 70:])


def test_ground_truth_mask_is_outlined_in_green() -> None:
    mask = np.zeros((100, 120), bool)
    mask[30:60, 30:60] = True
    out = heatmap_overlay(IMAGE, np.zeros((100, 120)), 0.0, 1.0, gt_mask=mask)
    green = (out[..., 1] == 255) & (out[..., 0] == 0) & (out[..., 2] == 0)
    assert green[30:60, 30:60].any() and not green[:20].any()


def test_save_overlay_downscales_and_adds_caption(tmp_path: Path) -> None:
    image_path = tmp_path / "big.png"
    cv2.imwrite(str(image_path), np.zeros((1000, 2000, 3), np.uint8))
    out_path = tmp_path / "out" / "fn" / "x.jpg"
    save_overlay(out_path, str(image_path), np.ones((8, 8)), 0.0, 1.0, "FN score 1.0")
    saved = cv2.imread(str(out_path))
    assert saved.shape[1] == MAX_SIDE
    assert saved.shape[0] == 28 + MAX_SIDE // 2  # caption bar + image
