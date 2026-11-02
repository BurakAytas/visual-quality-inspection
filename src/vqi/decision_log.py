"""SQLite log of every inspection decision: audit trail, review queue and daily report."""

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    file TEXT NOT NULL,
    score REAL NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('OK', 'NOK')),
    needs_review INTEGER NOT NULL,
    threshold REAL NOT NULL,
    model_version TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    defect_type TEXT
)
"""

# Days are the UTC date of the timestamp (ISO 8601 text, so its first 10 characters).
DAILY_SUMMARY = """
SELECT substr(timestamp, 1, 10) AS day,
       COUNT(*) AS inspected,
       SUM(decision = 'NOK') AS nok,
       SUM(needs_review) AS needs_review,
       ROUND(1.0 * SUM(decision = 'NOK') / COUNT(*), 4) AS defect_rate
FROM decisions
GROUP BY day
ORDER BY day
"""


@dataclass(frozen=True)
class Decision:
    """One logged decision."""

    timestamp: str  # UTC, ISO 8601
    file: str
    score: float
    decision: str
    needs_review: bool
    threshold: float
    model_version: str
    latency_ms: float
    defect_type: str | None = None  # set by the defect-type classifier (Phase 4)


class DecisionLog:
    """Append-only decision table; one short connection per call, so threads can share it."""

    def __init__(self, path: Path) -> None:
        """Create the database file and table if they do not exist."""
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._run(SCHEMA)

    def _run(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        """Execute one statement, commit, and return its rows as dicts."""
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            with db:  # commits on success, rolls back on error
                return [dict(row) for row in db.execute(sql, params)]
        finally:
            db.close()

    def add(self, decision: Decision) -> None:
        """Append one decision."""
        values = asdict(decision)
        columns = ", ".join(values)
        placeholders = ", ".join(f":{name}" for name in values)
        self._run(f"INSERT INTO decisions ({columns}) VALUES ({placeholders})", values)

    def review_queue(self, limit: int = 50) -> list[dict[str, Any]]:
        """Newest decisions whose score lies in the review band."""
        return self._run(
            "SELECT * FROM decisions WHERE needs_review = 1 ORDER BY id DESC LIMIT ?", (limit,)
        )

    def daily_summary(self) -> list[dict[str, Any]]:
        """Per UTC day: images inspected, NOK, needs review, and defect rate (NOK share)."""
        return self._run(DAILY_SUMMARY)
