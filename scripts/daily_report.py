"""Daily inspection report from the decision log: counts and defect rate per day, as CSV.

Days are UTC dates. The defect rate is the share of inspected parts decided NOK; a
per-defect-type breakdown follows once the classifier exists (Phase 4). The CSV opens in
Excel as it is.

Example:
    python scripts/daily_report.py --db logs/decisions.sqlite --out reports/daily.csv
"""

import argparse
import csv
import sys
from pathlib import Path

from vqi.decision_log import DecisionLog

COLUMNS = ["day", "inspected", "nok", "needs_review", "defect_rate"]


def write_report(db: Path, out: Path) -> list[dict]:
    """Write one CSV row per day and return the rows."""
    rows = DecisionLog(db).daily_summary()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    """Parse arguments, write the report and print it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=Path("logs/decisions.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("reports/daily_report.csv"))
    args = parser.parse_args()
    if not args.db.exists():  # DecisionLog would create an empty database
        sys.exit(f"No decision log at {args.db}")
    for row in write_report(args.db, args.out):
        print(", ".join(f"{key} {row[key]}" for key in COLUMNS))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
