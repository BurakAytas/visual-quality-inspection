"""Where a result came from: the git state of the code that produced it."""

import subprocess
from pathlib import Path

DIRTY_WARNING = (
    "WARNING: uncommitted changes; the recorded git commit is not exactly the code that runs"
)


def uncommitted_changes(cwd: Path | None = None) -> bool:
    """True if ``git status`` lists modified, staged or untracked files."""
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=cwd, check=False
    )
    return bool(result.stdout.strip())


def warn_if_dirty(cwd: Path | None = None) -> None:
    """Print a one-line warning when results would be recorded against a dirty tree."""
    if uncommitted_changes(cwd):
        print(DIRTY_WARNING, flush=True)
