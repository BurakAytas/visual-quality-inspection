"""Load experiment configs from YAML.

A category config names one dataset and one category and may extend a base file.
Adding a new product means adding a new YAML file, not changing code.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Folder name of each supported dataset under ``data_root``.
DATASET_DIRS = {
    "mvtec_ad": "MVTec_AD",
    "mvtec_ad2": "MVTec_AD_2",
}


@dataclass(frozen=True)
class DatasetConfig:
    """Which dataset and category to use, and where the data lives."""

    name: str
    category: str
    data_root: Path

    @property
    def root(self) -> Path:
        """Dataset root that holds one folder per category."""
        return self.data_root / DATASET_DIRS[self.name]

    @property
    def category_dir(self) -> Path:
        """Folder of this category, e.g. ``data/MVTec_AD_2/fabric``."""
        return self.root / self.category


@dataclass(frozen=True)
class Config:
    """Top-level experiment config."""

    dataset: DatasetConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated with ``override``; nested dicts are merged key by key."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and resolve its optional ``extends`` key (relative path)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(_read_yaml(path.parent / parent), raw)


def load_config(path: str | Path) -> Config:
    """Load and validate a config file.

    Raises:
        ValueError: if a key is missing, unknown, or has an invalid value.
    """
    raw = _read_yaml(Path(path))
    unknown = set(raw) - {"data_root", "dataset"}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    dataset = raw.get("dataset") or {}
    name = dataset.get("name")
    category = dataset.get("category")
    if name not in DATASET_DIRS:
        raise ValueError(f"dataset.name must be one of {sorted(DATASET_DIRS)}, got {name!r}")
    if not category:
        raise ValueError("dataset.category is required")
    if "data_root" not in raw:
        raise ValueError("data_root is required")

    return Config(
        dataset=DatasetConfig(name=name, category=category, data_root=Path(raw["data_root"]))
    )
