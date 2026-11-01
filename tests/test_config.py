from pathlib import Path

import pytest

from vqi.config import load_config

CONFIG_DIR = Path(__file__).parent.parent / "configs"
CATEGORY_CONFIGS = sorted(p for p in CONFIG_DIR.glob("*.yaml") if p.name != "base.yaml")


def test_category_configs_exist() -> None:
    assert {p.stem for p in CATEGORY_CONFIGS} >= {"fabric", "leather", "carpet"}


@pytest.mark.parametrize("path", CATEGORY_CONFIGS, ids=lambda p: p.stem)
def test_every_category_config_loads(path: Path) -> None:
    config = load_config(path)
    assert config.dataset.category == path.stem
    assert config.dataset.data_root == Path("data")  # inherited from base.yaml


def test_fabric_points_at_mvtec_ad2() -> None:
    dataset = load_config(CONFIG_DIR / "fabric.yaml").dataset
    assert dataset.category_dir == Path("data/MVTec_AD_2/fabric")


def test_child_overrides_base(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "data_root: a\nresults_dir: r\nseed: 1\ndataset: {name: mvtec_ad, category: x}\n"
    )
    (tmp_path / "child.yaml").write_text("extends: base.yaml\ndataset: {category: y}\n")
    dataset = load_config(tmp_path / "child.yaml").dataset
    assert (dataset.data_root, dataset.name, dataset.category) == (Path("a"), "mvtec_ad", "y")


VALID = "data_root: d\nresults_dir: r\nseed: 0\ndataset: {name: mvtec_ad, category: x}\n"


def test_minimal_config_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "ok.yaml"
    path.write_text(VALID)
    config = load_config(path)
    assert config.model.key == "patchcore"
    assert config.threshold.target_false_alarm_rate == 0.01


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("name: mvtec_ad", "name: imagenet"),
        (", category: x", ""),
        ("seed: 0\n", ""),
        ("seed: 0\n", "seed: 0\ntypo_key: 1\n"),
        ("seed: 0\n", "seed: 0\nmodel: {image_sise: 128}\n"),
        ("seed: 0\n", "seed: 0\nthreshold: {target_false_alarm_rate: 1.5}\n"),
    ],
    ids=["unknown-dataset", "no-category", "no-seed", "unknown-key", "model-typo", "bad-rate"],
)
def test_invalid_config_is_rejected(tmp_path: Path, old: str, new: str) -> None:
    assert old in VALID
    path = tmp_path / "bad.yaml"
    path.write_text(VALID.replace(old, new))
    with pytest.raises(ValueError):
        load_config(path)
