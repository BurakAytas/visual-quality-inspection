"""Daily inspection report from the decision log: counts and defect rate per day, as CSV.

Days are UTC dates. The defect rate is the share of inspected parts decided NOK. A second
CSV, ``<out>_defect_types.csv``, splits each day's NOK parts by defect type. Both open in
Excel as they are.

Example:
    python scripts/daily_report.py --db logs/decisions.sqlite --out reports/daily.csv
"""

import argparse
import csv
import sys
from pathlib import Path

from vqi.decision_log import DecisionLog

COLUMNS = ["day", "inspected", "nok", "needs_review", "defect_rate"]
TYPE_COLUMNS = ["day", "defect_type", "nok"]


def defect_types_path(out: Path) -> Path:
    """Path of the per-defect-type CSV next to the daily report."""
    return out.with_name(f"{out.stem}_defect_types.csv")


def write_csv(rows: list[dict], columns: list[str], out: Path) -> None:
    """Write dict rows with a header line."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(db: Path, out: Path) -> list[dict]:
    """Write the daily CSV and the per-defect-type CSV; return the daily rows."""
    log = DecisionLog(db)
    rows = log.daily_summary()
    write_csv(rows, COLUMNS, out)
    write_csv(log.daily_defect_types(), TYPE_COLUMNS, defect_types_path(out))
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
    print(f"Wrote {args.out} and {defect_types_path(args.out)}")


if __name__ == "__main__":
    main()
