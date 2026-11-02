"""Line simulator and daily report, with a mocked API and a temporary decision log."""

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import httpx

from vqi.decision_log import Decision, DecisionLog

SCRIPTS = Path(__file__).parent.parent / "scripts"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_simulator_posts_each_new_image_once_oldest_first(tmp_path: Path) -> None:
    simulator = load_script("line_simulator")
    posted = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(request.content.split(b'filename="')[1].split(b'"')[0].decode())
        if b"broken" in request.content:
            return httpx.Response(400, json={"detail": "not a readable image"})
        body = {"decision": "OK", "score": 1.0, "threshold": 2.0, "latency_ms": 5.0}
        return httpx.Response(200, json={**body, "needs_review": False})

    for i, name in enumerate(["b.png", "a.jpg", "notes.txt", "broken.png"]):
        (tmp_path / name).write_bytes(b"broken" if name == "broken.png" else b"image")
        os.utime(tmp_path / name, (1000 + i, 1000 + i))
    client = httpx.Client(transport=httpx.MockTransport(handler))
    seen: set[Path] = set()

    assert simulator.post_new_images(client, "http://api", tmp_path, seen) == 3
    assert posted == ["b.png", "a.jpg", "broken.png"]  # oldest first, no text file
    assert simulator.post_new_images(client, "http://api", tmp_path, seen) == 0
    (tmp_path / "c.png").write_bytes(b"image")
    assert simulator.post_new_images(client, "http://api", tmp_path, seen) == 1
    assert posted[-1] == "c.png"


def test_daily_report_writes_one_row_per_day(tmp_path: Path) -> None:
    db = tmp_path / "decisions.sqlite"
    log = DecisionLog(db)
    for timestamp, verdict in [("2026-11-01T10:00", "NOK"), ("2026-11-02T10:00", "OK")]:
        log.add(Decision(timestamp, "a.png", 1.0, verdict, False, 2.0, "v1", 100.0))
    out = tmp_path / "reports" / "daily.csv"
    subprocess.run(
        [sys.executable, str(SCRIPTS / "daily_report.py"), "--db", str(db), "--out", str(out)],
        check=True,
    )
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [(r["day"], r["inspected"], r["nok"], r["defect_rate"]) for r in rows] == [
        ("2026-11-01", "1", "1", "1.0"),
        ("2026-11-02", "1", "0", "0.0"),
    ]


def test_daily_report_refuses_a_missing_log(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "daily_report.py"), "--db", str(tmp_path / "none.db")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "No decision log" in result.stderr
    assert not (tmp_path / "none.db").exists()
