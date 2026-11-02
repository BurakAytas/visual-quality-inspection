from pathlib import Path

import cv2
import numpy as np
import pytest

from vqi import inspection
from vqi.config import ServeConfig, load_serve_config
from vqi.defect_type import DefectType
from vqi.inspection import (
    Classifier,
    Inspector,
    Model,
    build_inspector,
    decide,
    model_version,
    read_validation_scores,
)
from vqi.threshold import save_threshold

SIZE = (32, 48)


def fake_model(score: float) -> Model:
    """A model that checks its input and returns a fixed score with one hot pixel."""

    def model(image: np.ndarray) -> tuple[float, np.ndarray]:
        assert image.shape == (*SIZE, 3) and image.dtype == np.float32
        anomaly_map = np.zeros((8, 8), np.float32)
        anomaly_map[2, 3] = score
        return score, anomaly_map

    return model


def make_inspector(score: float) -> Inspector:
    return Inspector(fake_model(score), SIZE, 10.0, (9.0, 11.0), 5.0, "fmt@abc+thr=10")


def test_decide_rejects_only_above_the_threshold() -> None:
    assert decide(10.0, 10.0) == "OK"
    assert decide(10.001, 10.0) == "NOK"


@pytest.mark.parametrize(
    ("score", "decision", "needs_review"),
    [
        (1.0, "OK", False),
        (9.0, "OK", False),
        (9.5, "OK", True),
        (10.5, "NOK", True),
        (11.0, "NOK", True),
        (11.5, "NOK", False),
    ],
)
def test_decision_and_review_band(score: float, decision: str, needs_review: bool) -> None:
    result = make_inspector(score).inspect(np.zeros((64, 64, 3), np.float32))
    assert (result.decision, result.needs_review) == (decision, needs_review)
    assert result.score == score and result.latency_ms > 0


def recording_classifier(crops: list[np.ndarray]) -> Classifier:
    """Classifier that keeps every crop it gets and always says "cut"."""

    def classify(crop: np.ndarray) -> DefectType:
        crops.append(crop)
        return DefectType("cut", 0.9)

    return classify


def test_only_nok_parts_are_classified() -> None:
    crops: list[np.ndarray] = []
    image = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    nok = Inspector(
        fake_model(20.0), SIZE, 10.0, (9.0, 11.0), 5.0, "v", recording_classifier(crops)
    )
    assert nok.inspect(image).defect_type == DefectType("cut", 0.9)
    # Smaller than the crop side: the whole original image is classified.
    assert len(crops) == 1
    np.testing.assert_array_equal(crops[0], image)
    ok = Inspector(fake_model(1.0), SIZE, 10.0, (9.0, 11.0), 5.0, "v", recording_classifier(crops))
    assert ok.inspect(image).defect_type is None and len(crops) == 1


def test_crop_comes_from_the_original_image_at_the_map_peak() -> None:
    crops: list[np.ndarray] = []
    inspector = Inspector(
        fake_model(20.0), SIZE, 10.0, (9.0, 11.0), 5.0, "v", recording_classifier(crops)
    )
    image = np.random.default_rng(0).random((1024, 1024, 3)).astype(np.float32)
    inspector.inspect(image)
    # Cell (2, 3) of the 8 x 8 map is centred on pixel (320, 448): crop rows 192..447,
    # columns 320..575.
    np.testing.assert_array_equal(crops[0], image[192:448, 320:576])


def test_heatmap_is_a_png_at_model_size_coloured_where_the_map_is_high() -> None:
    result = make_inspector(20.0).inspect(np.full((64, 64, 3), 0.5, np.float32))
    heatmap = cv2.imdecode(np.frombuffer(result.heatmap_png, np.uint8), cv2.IMREAD_COLOR)
    assert heatmap.shape == (*SIZE, 3)
    grey = round(0.5 * 255)
    assert (heatmap[0, 0] == grey).all()  # map below vmin: image unchanged
    assert not (heatmap[10, 20] == grey).all()  # the hot pixel, scaled up 6 x 4


def test_model_version_names_format_commit_and_threshold() -> None:
    version = model_version("openvino_int8", "52df362", 30.676389)
    assert version == "openvino_int8@52df362+thr=30.6764"


def test_read_validation_scores_rejects_defective_rows(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text("split,image,label,score\nvalidation,a.png,0,1.5\ntest,b.png,1,9\n")
    assert read_validation_scores(path).tolist() == [1.5]
    path.write_text("split,image,label,score\nvalidation,a.png,1,1.5\n")
    with pytest.raises(ValueError):
        read_validation_scores(path)


def test_serve_config_env_overrides_a_key() -> None:
    path = Path(__file__).parent.parent / "configs" / "serve" / "leather.yaml"
    assert load_serve_config(path).review_rate == 0.1
    config = load_serve_config(path, {"VQI_DB_PATH": "x.sqlite", "VQI_REVIEW_RATE": "0.2"})
    assert config.db_path == Path("x.sqlite") and config.review_rate == 0.2
    assert load_serve_config(path, {"VQI_CLASSIFIER_FILE": ""}).classifier_file is None
    with pytest.raises(ValueError):
        load_serve_config(path, {"VQI_REVIEW_RATE": "1.5"})


def test_build_inspector_reads_threshold_band_and_heatmap_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeOpenVINOModel:
        input_size = SIZE

        def __init__(self, model_file: Path) -> None:
            assert model_file == tmp_path / "model.xml"

    monkeypatch.setattr(inspection, "OpenVINOModel", FakeOpenVINOModel)
    scores = tmp_path / "scores.csv"
    rows = "".join(f"validation,{i}.png,0,{i}\n" for i in range(1, 11))
    scores.write_text("split,image,label,score\n" + rows + "test,x.png,1,99\n")
    save_threshold(tmp_path / "thresholds.json", "m/c/fmt", {"threshold": 10.0})
    config = ServeConfig(
        model_file=tmp_path / "model.xml",
        model_format="fmt",
        thresholds_file=tmp_path / "thresholds.json",
        threshold_key="m/c/fmt",
        validation_scores=scores,
        review_rate=0.2,
        db_path=tmp_path / "decisions.sqlite",
        classifier_file=None,
    )
    built = build_inspector(config, "abc")
    assert built.threshold == 10.0 and built.input_size == SIZE
    assert built.band == (8.0, 12.0)  # 2 of the 10 good scores lie above 8; mirrored at 10
    assert built.heatmap_vmin == 5.5  # median good validation score (the test row is ignored)
    assert built.model_version == "fmt@abc+thr=10"
