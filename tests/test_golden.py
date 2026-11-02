"""Golden set: fixed images whose decisions must not change.

Needs the exported model (scripts/export_model.py) and the leather images, which are never
committed, so the whole module is skipped when they are absent, as in CI.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vqi.config import load_serve_config
from vqi.imageio import load_rgb, resize_to
from vqi.inspection import Inspector, OpenVINOModel, build_inspector

ROOT = Path(__file__).parent.parent
GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "leather_openvino_fp32.json").read_text(encoding="utf-8")
)
IMAGES = GOLDEN["images"]
DATA = ROOT / "data" / "MVTec_AD"
_config = load_serve_config(ROOT / "configs" / "serve" / "leather.yaml")
CONFIG = replace(
    _config,
    model_file=ROOT / _config.model_file,
    thresholds_file=ROOT / _config.thresholds_file,
    validation_scores=ROOT / _config.validation_scores,
)

_missing = [p for p in [CONFIG.model_file, *(DATA / i["image"] for i in IMAGES)] if not p.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=f"golden set needs the exported model and the leather data ({len(_missing)} missing)",
)


@pytest.fixture(scope="module")
def inspector() -> Inspector:
    return build_inspector(CONFIG, commit="test")


@pytest.mark.parametrize("item", IMAGES, ids=lambda item: item["image"])
def test_golden_decisions_do_not_change(inspector: Inspector, item: dict) -> None:
    result = inspector.inspect(load_rgb(DATA / item["image"]))
    assert result.score == pytest.approx(item["score"], abs=GOLDEN["score_tolerance"])
    assert result.decision == item["decision"]
    assert result.needs_review == item["needs_review"]


def test_plain_runtime_matches_openvino_inferencer() -> None:
    deploy = pytest.importorskip("anomalib.deploy", reason="needs the [train] extra")
    model = OpenVINOModel(CONFIG.model_file)
    reference = deploy.OpenVINOInferencer(path=CONFIG.model_file, device="CPU")
    for item in IMAGES:
        image = resize_to(load_rgb(DATA / item["image"]), model.input_size)
        score, anomaly_map = model(image)
        expected = reference.predict(image)
        assert score == pytest.approx(float(np.ravel(expected.pred_score)[0]), abs=1e-4)
        np.testing.assert_allclose(anomaly_map, expected.anomaly_map.squeeze(), atol=1e-4)


def test_anomaly_map_marks_the_defects(inspector: Inspector) -> None:
    # The heatmap is coloured up to the threshold, so a defect's map must exceed it.
    # The first INT8 export failed this: its map was clipped at about 27.5 (D50).
    for item in IMAGES:
        if "/good/" in item["image"]:
            continue
        image = resize_to(load_rgb(DATA / item["image"]), inspector.input_size)
        _, anomaly_map = inspector.model(image)
        assert anomaly_map.max() > inspector.threshold, item["image"]
