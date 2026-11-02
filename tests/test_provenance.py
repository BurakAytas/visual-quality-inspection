"""Dirty-tree warning for the training and export scripts."""

import subprocess
from pathlib import Path

import pytest

from vqi.provenance import DIRTY_WARNING, uncommitted_changes, warn_if_dirty


def test_warns_only_when_the_tree_has_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert not uncommitted_changes(tmp_path)
    warn_if_dirty(tmp_path)
    assert capsys.readouterr().out == ""
    (tmp_path / "new.py").write_text("x = 1\n")
    assert uncommitted_changes(tmp_path)
    warn_if_dirty(tmp_path)
    assert capsys.readouterr().out == DIRTY_WARNING + "\n"
