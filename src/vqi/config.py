"""Load experiment configs from YAML.

A category config names one dataset and one category and extends a base file.
Adding a new product means adding a new YAML file, not changing code.
"""

from dataclasses import dataclass, field, fields
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
    # MVTec AD only: share of train/good moved into a good-only validation set.
    val_split_ratio: float = 0.2

    @property
    def root(self) -> Path:
        """Dataset root that holds one folder per category."""
        return self.data_root / DATASET_DIRS[self.name]

    @property
    def category_dir(self) -> Path:
        """Folder of this category, e.g. ``data/MVTec_AD_2/fabric``."""
        return self.root / self.category


@dataclass(frozen=True)
class ModelConfig:
    """Anomaly model: anomalib class name, input size, constructor and trainer arguments."""

    name: str = "Patchcore"
    image_size: int = 256
    batch_size: int = 8  # training batch size
    eval_batch_size: int = 8
    num_workers: int = 2
    params: dict[str, Any] = field(default_factory=dict)  # passed to the anomalib model
    trainer: dict[str, Any] = field(default_factory=dict)  # passed to the anomalib Engine

    @property
    def key(self) -> str:
        """Short lower-case name used in result paths, e.g. ``patchcore``."""
        return self.name.lower()


@dataclass(frozen=True)
class ThresholdConfig:
    """Threshold policy: accept this share of false alarms on good validation images."""

    target_false_alarm_rate: float = 0.01


@dataclass(frozen=True)
class EvaluationConfig:
    """Settings for test-set metrics and the failure gallery."""

    bootstrap_samples: int = 1000
    fixed_false_alarm_rate: float = 0.01
    overlays_per_group: int = 8


@dataclass(frozen=True)
class RobustnessConfig:
    """Fixed perturbations applied to test images (threshold stays the clean one)."""

    brightness_factors: tuple[float, ...] = (0.7, 1.3)
    blur_sigma_fraction: float = 0.004  # Gaussian sigma as a share of image width
    noise_std: float = 0.05  # on pixel values scaled to [0, 1]

    def __post_init__(self) -> None:
        """Store factors as a tuple (YAML gives a list) so the config stays immutable."""
        object.__setattr__(self, "brightness_factors", tuple(map(float, self.brightness_factors)))


@dataclass(frozen=True)
class Config:
    """Top-level experiment config."""

    dataset: DatasetConfig
    model: ModelConfig
    threshold: ThresholdConfig
    evaluation: EvaluationConfig
    robustness: RobustnessConfig
    seed: int
    results_dir: Path


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


def _check_keys(section: str, raw: dict[str, Any], cls: type) -> None:
    """Reject keys that the dataclass ``cls`` does not have (catches typos)."""
    unknown = set(raw) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown keys in '{section}': {sorted(unknown)}")


def _rate(name: str, value: float) -> float:
    """Check that a rate lies strictly between 0 and 1."""
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


def load_config(path: str | Path, model_path: str | Path | None = None) -> Config:
    """Load and validate a category config, optionally with a model config.

    ``model_path`` points to a file with a ``model`` section (``configs/models/*.yaml``);
    it replaces the whole ``model`` section, so one model's parameters never leak into
    another's.

    Raises:
        ValueError: if a key is missing, unknown, or has an invalid value.
    """
    raw = _read_yaml(Path(path))
    if model_path is not None:
        model_raw = _read_yaml(Path(model_path))
        if set(model_raw) != {"model"}:
            raise ValueError(f"{model_path} must contain only a 'model' section")
        raw["model"] = model_raw["model"]
    for key in ("dataset", "data_root", "seed", "results_dir"):
        if key not in raw:
            raise ValueError(f"'{key}' is required")
    data_root = Path(raw.pop("data_root"))
    _check_keys("top level", raw, Config)

    dataset_raw = raw["dataset"]
    _check_keys("dataset", dataset_raw, DatasetConfig)
    if dataset_raw.get("name") not in DATASET_DIRS:
        raise ValueError(f"dataset.name must be one of {sorted(DATASET_DIRS)}")
    if not dataset_raw.get("category"):
        raise ValueError("dataset.category is required")
    dataset = DatasetConfig(**dataset_raw, data_root=data_root)
    _rate("dataset.val_split_ratio", dataset.val_split_ratio)

    sections = {
        "model": ModelConfig,
        "threshold": ThresholdConfig,
        "evaluation": EvaluationConfig,
        "robustness": RobustnessConfig,
    }
    built = {}
    for name, cls in sections.items():
        section = raw.get(name) or {}
        _check_keys(name, section, cls)
        built[name] = cls(**section)
    _rate("threshold.target_false_alarm_rate", built["threshold"].target_false_alarm_rate)
    _rate("evaluation.fixed_false_alarm_rate", built["evaluation"].fixed_false_alarm_rate)

    return Config(
        dataset=dataset,
        seed=int(raw["seed"]),
        results_dir=Path(raw["results_dir"]),
        **built,
    )
