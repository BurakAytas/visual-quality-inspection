import tarfile
from pathlib import Path

import pytest

from vqi.config import DatasetConfig
from vqi.dataset_files import REQUIRED_DIRS, find_category_dir, install_from_local, missing_dirs


def make_fake_category(root: Path, dataset_name: str) -> Path:
    """Create a minimal category folder with one dummy file in each required folder."""
    for rel in REQUIRED_DIRS[dataset_name]:
        (root / rel).mkdir(parents=True)
        (root / rel / "000.png").write_bytes(b"")
    return root


@pytest.mark.parametrize("dataset_name", sorted(REQUIRED_DIRS))
def test_complete_layout_has_nothing_missing(tmp_path: Path, dataset_name: str) -> None:
    category = make_fake_category(tmp_path / "fabric", dataset_name)
    assert missing_dirs(category, dataset_name) == []


def test_empty_validation_folder_is_reported(tmp_path: Path) -> None:
    category = make_fake_category(tmp_path / "fabric", "mvtec_ad2")
    (category / "validation/good/000.png").unlink()
    assert missing_dirs(category, "mvtec_ad2") == ["validation/good"]


def test_find_category_dir_searches_nested_folders(tmp_path: Path) -> None:
    category = make_fake_category(tmp_path / "download" / "mvtec_ad_2" / "fabric", "mvtec_ad2")
    assert find_category_dir(tmp_path / "download", "fabric") == category


def test_find_category_dir_fails_when_absent(tmp_path: Path) -> None:
    make_fake_category(tmp_path / "carpet", "mvtec_ad")
    with pytest.raises(FileNotFoundError):
        find_category_dir(tmp_path, "fabric")


def test_install_from_folder_creates_symlink(tmp_path: Path) -> None:
    source = make_fake_category(tmp_path / "download" / "fabric", "mvtec_ad2")
    dataset = DatasetConfig(name="mvtec_ad2", category="fabric", data_root=tmp_path / "data")
    target = install_from_local(tmp_path / "download", dataset)
    assert target == tmp_path / "data" / "MVTec_AD_2" / "fabric"
    assert target.is_symlink() and target.resolve() == source.resolve()


def test_install_from_archive_extracts_category(tmp_path: Path) -> None:
    make_fake_category(tmp_path / "src" / "mvtec_ad_2" / "fabric", "mvtec_ad2")
    archive = tmp_path / "fabric.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        tar.add(tmp_path / "src" / "mvtec_ad_2", arcname="mvtec_ad_2")
    dataset = DatasetConfig(name="mvtec_ad2", category="fabric", data_root=tmp_path / "data")

    target = install_from_local(archive, dataset)

    assert not target.is_symlink()
    assert missing_dirs(target, "mvtec_ad2") == []
    assert sorted(p.name for p in target.parent.iterdir()) == ["fabric"]  # temp dir cleaned up


def test_install_refuses_to_overwrite(tmp_path: Path) -> None:
    source = make_fake_category(tmp_path / "fabric", "mvtec_ad2")
    dataset = DatasetConfig(name="mvtec_ad2", category="fabric", data_root=tmp_path / "data")
    dataset.category_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        install_from_local(source, dataset)
