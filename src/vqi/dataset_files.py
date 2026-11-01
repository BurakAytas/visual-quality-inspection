"""Put dataset files in place and check their folder layout.

Both MVTec datasets are downloaded by the user through the web forms on the MVTec
website, so they are always installed from a local path.
"""

import shutil
import tarfile
import tempfile
from pathlib import Path

from vqi.config import DatasetConfig

# Folders each dataset must have inside one category folder.
REQUIRED_DIRS = {
    "mvtec_ad2": (
        "train/good",
        "validation/good",
        "test_public/good",
        "test_public/bad",
        "test_public/ground_truth/bad",
    ),
    "mvtec_ad": ("train/good", "test/good", "ground_truth"),
}


def missing_dirs(category_dir: Path, dataset_name: str) -> list[str]:
    """Return the required sub-folders that are missing or empty."""
    missing = []
    for rel in REQUIRED_DIRS[dataset_name]:
        folder = category_dir / rel
        if not folder.is_dir() or not any(folder.iterdir()):
            missing.append(rel)
    return missing


def find_category_dir(search_root: Path, category: str) -> Path:
    """Find the single folder named ``category`` that contains ``train/good``.

    Downloads differ in how deep the category folder sits, so we search for it.
    """
    candidates = [search_root] if search_root.name == category else []
    candidates += [p for p in search_root.rglob(category) if p.is_dir()]
    matches = sorted({p for p in candidates if (p / "train" / "good").is_dir()})
    if len(matches) != 1:
        top = sorted(p.name for p in search_root.iterdir())[:20]
        raise FileNotFoundError(
            f"Expected one '{category}/train/good' folder under {search_root}, "
            f"found {len(matches)}. Top-level entries: {top}"
        )
    return matches[0]


def _category_archive(folder: Path, category: str) -> Path | None:
    """Return ``<folder>/<category>.tar*`` if exactly one such archive exists."""
    archives = sorted(folder.glob(f"{category}.tar*"))
    return archives[0] if len(archives) == 1 else None


def install_from_local(source: Path, dataset: DatasetConfig) -> Path:
    """Link a downloaded folder, or extract a downloaded archive, into ``dataset.category_dir``.

    ``source`` can be an unpacked folder (symlinked, no copy of several GB), a
    ``.tar``/``.tar.gz``/``.tar.xz`` archive, or a folder holding ``<category>.tar*``.
    Archives are extracted with the safe ``data`` filter.
    """
    target = dataset.category_dir
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists; remove it first")
    target.parent.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        archive = _category_archive(source, dataset.category)
        if archive is None:
            target.symlink_to(find_category_dir(source, dataset.category).resolve())
            return target
        source = archive
    if tarfile.is_tarfile(source):
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            with tarfile.open(source) as archive:
                archive.extractall(tmp, filter="data")
            shutil.move(find_category_dir(Path(tmp), dataset.category), target)
    else:
        raise ValueError(f"{source} is neither a folder nor a tar archive")
    return target
