"""Operator panel against a faked API, run headless with Streamlit's AppTest."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from vqi.panel import decision_banner


def panel_with_fake_api() -> None:
    """Panel over an API that has one part in review and one NOK fold on one day."""
    import httpx

    from vqi.panel import render

    def handler(request: httpx.Request) -> httpx.Response:
        part = {"timestamp": "2026-11-02T10:00", "file": "a.png", "decision": "NOK"}
        part |= {"score": 31.0, "threshold": 30.4, "defect_type": "fold", "model_version": "v"}
        answers = {
            "/health": {"status": "ok", "model_version": "openvino_fp32@abc+thr=30.4"},
            "/review": [part],
            "/daily": {
                "days": [{"day": "2026-11-02", "inspected": 3, "nok": 1, "needs_review": 1}],
                "defect_types": [{"day": "2026-11-02", "defect_type": "fold", "nok": 1}],
            },
        }
        return httpx.Response(200, json=answers[request.url.path])

    render(httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler)))


def panel_without_api() -> None:
    """Panel whose API refuses the connection."""
    import httpx

    from vqi.panel import render

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    render(httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler)))


def result_near_threshold() -> None:
    """One NOK answer inside the review band, with a 1-pixel heatmap."""
    import base64

    import cv2
    import numpy as np

    from vqi.panel import show_result

    png = cv2.imencode(".png", np.zeros((1, 1, 3), np.uint8))[1].tobytes()
    body = {
        "decision": "NOK",
        "defect_type": "cut",
        "defect_probability": 0.93,
        "needs_review": True,
        "score": 30.9,
        "threshold": 30.4,
        "latency_ms": 181.4,
        "file": "part.png",
        "model_version": "v",
        "heatmap_png_base64": base64.b64encode(png).decode("ascii"),
    }
    show_result(body, png)


def test_panel_shows_model_review_queue_and_report() -> None:
    app = AppTest.from_function(panel_with_fake_api).run()
    assert not app.exception
    assert "openvino_fp32@abc+thr=30.4" in app.caption[0].value
    assert [tab.label for tab in app.tabs] == ["Inspect", "Review queue", "Daily report"]
    review, report = app.dataframe[0].value, app.dataframe[1].value
    assert review["file"].tolist() == ["a.png"] and "model_version" not in review
    assert report["inspected"].tolist() == [3]


def test_panel_says_when_the_api_is_down() -> None:
    app = AppTest.from_function(panel_without_api).run()
    assert not app.exception
    assert "not reachable" in app.error[0].value
    assert len(app.tabs) == 0


def test_result_flags_review_and_shows_numbers() -> None:
    app = AppTest.from_function(result_near_threshold).run()
    assert not app.exception
    assert "operator" in app.warning[0].value
    assert [m.value for m in app.metric] == ["30.90", "30.40", "181 ms"]


def test_banner_names_the_defect_type_of_nok_parts_only() -> None:
    nok = decision_banner({"decision": "NOK", "defect_type": "cut", "defect_probability": 0.934})
    assert "NOK &middot; cut (93%)" in nok and "#b3261e" in nok
    ok = decision_banner({"decision": "OK", "defect_type": None})
    assert ">OK</div>" in ok and "#1b7f3b" in ok
