import sqlite3
from pathlib import Path

import pytest

from vqi.decision_log import Decision, DecisionLog


def decision(timestamp: str, verdict: str, needs_review: bool = False) -> Decision:
    return Decision(timestamp, "a.png", 1.0, verdict, needs_review, 2.0, "v1", 150.0)


def test_daily_summary_counts_per_day(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path / "logs" / "decisions.sqlite")
    log.add(decision("2026-11-01T23:59:59.000+00:00", "NOK", needs_review=True))
    log.add(decision("2026-11-02T00:00:00.000+00:00", "OK"))
    log.add(decision("2026-11-02T08:00:00.000+00:00", "OK", needs_review=True))
    log.add(decision("2026-11-02T09:00:00.000+00:00", "OK"))
    log.add(decision("2026-11-02T10:00:00.000+00:00", "NOK"))
    assert log.daily_summary() == [
        {"day": "2026-11-01", "inspected": 1, "nok": 1, "needs_review": 1, "defect_rate": 1.0},
        {"day": "2026-11-02", "inspected": 4, "nok": 1, "needs_review": 1, "defect_rate": 0.25},
    ]


def test_daily_defect_types_count_nok_parts_only(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path / "decisions.sqlite")
    for hour, kind in [(8, "fold"), (9, "cut"), (10, "fold")]:
        stamp = f"2026-11-02T{hour:02d}:00"
        log.add(Decision(stamp, "a.png", 1.0, "NOK", False, 2.0, "v1", 1.0, kind))
    log.add(decision("2026-11-02T11:00:00.000+00:00", "OK"))
    log.add(decision("2026-11-03T08:00:00.000+00:00", "NOK"))
    assert log.daily_defect_types() == [
        {"day": "2026-11-02", "defect_type": "cut", "nok": 1},
        {"day": "2026-11-02", "defect_type": "fold", "nok": 2},
        {"day": "2026-11-03", "defect_type": "unclassified", "nok": 1},
    ]


def test_review_queue_holds_only_flagged_decisions_newest_first(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path / "decisions.sqlite")
    log.add(decision("2026-11-02T08:00:00.000+00:00", "OK", needs_review=True))
    log.add(decision("2026-11-02T09:00:00.000+00:00", "OK"))
    log.add(decision("2026-11-02T10:00:00.000+00:00", "NOK", needs_review=True))
    queue = log.review_queue()
    assert [row["timestamp"][11:13] for row in queue] == ["10", "08"]
    assert queue[0]["defect_type"] is None and queue[0]["model_version"] == "v1"


def test_log_survives_reopening_and_rejects_unknown_decisions(tmp_path: Path) -> None:
    path = tmp_path / "decisions.sqlite"
    DecisionLog(path).add(decision("2026-11-02T08:00:00.000+00:00", "OK"))
    log = DecisionLog(path)
    assert log.daily_summary()[0]["inspected"] == 1
    with pytest.raises(sqlite3.IntegrityError):
        log.add(decision("2026-11-02T08:00:00.000+00:00", "MAYBE"))
