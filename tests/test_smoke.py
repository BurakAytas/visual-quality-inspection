"""Empty-model smoke test: the package installs and imports before any model exists."""

import tomllib
from pathlib import Path

import vqi


def test_package_version_matches_pyproject() -> None:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    assert vqi.__version__ == tomllib.loads(pyproject.read_text())["project"]["version"]
