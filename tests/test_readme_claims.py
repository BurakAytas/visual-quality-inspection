import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "check_readme_claims.py"
REPO_ROOT = SCRIPT.parent.parent


def run_check(tmp_path: Path, readme: str, metrics: dict) -> subprocess.CompletedProcess:
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "m.json").write_text(json.dumps(metrics))
    (tmp_path / "README.md").write_text(readme)
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--readme", "README.md", "--root", "."],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "readme",
    [
        "AUROC 0.873<!-- claim: results/m.json#fabric.auroc -->",
        "Miss rate 4.2%<!-- claim: results/m.json#fabric.miss -->",
        "<!-- results:start -->| 0.87 <!-- claim: results/m.json#fabric.auroc --> |"
        "<!-- results:end -->",
        "AU-PRO 0.30<!-- claim: results/m.json#fabric.aupro_0.3 -->",
        "CI high 0.95<!-- claim: results/m.json#fabric.ci.1 -->",
    ],
    ids=["plain", "percent", "results-block", "dotted-key", "list-index"],
)
def test_matching_claims_pass(tmp_path: Path, readme: str) -> None:
    metrics = {"fabric": {"auroc": 0.87312, "miss": 0.0421, "aupro_0.3": 0.3, "ci": [0.8, 0.95]}}
    result = run_check(tmp_path, readme, metrics)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "readme",
    [
        "AUROC 0.990<!-- claim: results/m.json#fabric.auroc -->",
        "AUROC 0.873<!-- claim: results/m.json#fabric.missing_key -->",
        "AUROC 0.873<!-- claim: results/nope.json#fabric.auroc -->",
        "<!-- results:start -->| 0.95 |<!-- results:end -->",
    ],
    ids=["wrong-value", "missing-key", "missing-file", "hand-typed-in-results-block"],
)
def test_bad_claims_fail(tmp_path: Path, readme: str) -> None:
    result = run_check(tmp_path, readme, {"fabric": {"auroc": 0.87312}})
    assert result.returncode == 1


def test_repository_readme_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
