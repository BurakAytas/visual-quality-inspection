"""API tests with a fake model: no exported model, no OpenVINO inference, no GPU."""

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vqi.api import create_app
from vqi.decision_log import DecisionLog
from vqi.defect_type import DefectType
from vqi.inspection import Inspector


def png_bytes(height: int = 40, width: int = 60) -> bytes:
    return cv2.imencode(".png", np.full((height, width, 3), 100, np.uint8))[1].tobytes()


@pytest.fixture
def log(tmp_path: Path) -> DecisionLog:
    return DecisionLog(tmp_path / "decisions.sqlite")


def client_with_score(
    score: float, log: DecisionLog, defect_type: DefectType | None = None
) -> TestClient:
    def model(image: np.ndarray) -> tuple[float, np.ndarray]:
        assert image.shape == (16, 16, 3)  # the upload was resized to the model input
        return score, np.full((4, 4), score, np.float32)

    def classifier(crop: np.ndarray) -> DefectType:
        return defect_type

    inspector = Inspector(
        model,
        (16, 16),
        10.0,
        (9.0, 11.0),
        5.0,
        "fmt@abc+thr=10",
        classifier if defect_type else None,
    )
    return TestClient(create_app(inspector, log))


def test_inspect_returns_decision_and_logs_it(log: DecisionLog) -> None:
    client = client_with_score(12.0, log)
    response = client.post("/inspect", files={"file": ("part_001.png", png_bytes())})
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "NOK" and body["score"] == 12.0 and body["threshold"] == 10.0
    assert body["needs_review"] is False and body["review_band"] == [9.0, 11.0]
    assert body["defect_type"] is None and body["defect_probability"] is None
    assert body["model_version"] == "fmt@abc+thr=10"
    assert body["file"] == "part_001.png" and body["latency_ms"] > 0
    heatmap = cv2.imdecode(
        np.frombuffer(base64.b64decode(body["heatmap_png_base64"]), np.uint8), cv2.IMREAD_COLOR
    )
    assert heatmap.shape == (16, 16, 3)
    assert log.daily_summary()[0]["nok"] == 1


def test_nok_part_gets_its_defect_type_in_the_answer_and_the_log(log: DecisionLog) -> None:
    client = client_with_score(12.0, log, DefectType("fold", 0.75))
    body = client.post("/inspect", files={"file": ("a.png", png_bytes())}).json()
    assert (body["defect_type"], body["defect_probability"]) == ("fold", 0.75)
    assert log.daily_defect_types()[0]["defect_type"] == "fold"
    daily = client.get("/daily").json()
    assert daily["days"][0]["nok"] == 1
    assert daily["defect_types"][0]["defect_type"] == "fold"


def test_scores_near_the_threshold_reach_the_review_queue(log: DecisionLog) -> None:
    client = client_with_score(9.5, log)
    client.post("/inspect", files={"file": ("a.png", png_bytes())})
    queue = client.get("/review").json()
    assert [(row["file"], row["decision"]) for row in queue] == [("a.png", "OK")]


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_unreadable_upload_is_rejected_and_not_logged(log: DecisionLog, data: bytes) -> None:
    client = client_with_score(1.0, log)
    response = client.post("/inspect", files={"file": ("bad.png", data)})
    assert response.status_code == 400
    assert log.daily_summary() == []


def test_health_names_the_model(log: DecisionLog) -> None:
    body = client_with_score(1.0, log).get("/health").json()
    assert body == {"status": "ok", "model_version": "fmt@abc+thr=10"}
