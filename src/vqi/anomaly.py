"""Train an anomalib model on good images and collect raw scores and anomaly maps.

Needs the ``[train]`` extra (anomalib, torch). Kept apart from the rest of the
package so the API and CI do not import torch.
"""

from dataclasses import dataclass
from pathlib import Path

import anomalib.models
import numpy as np
import torch
from anomalib.data import MVTecAD, MVTecAD2
from anomalib.data.datamodules.base import AnomalibDataModule
from anomalib.engine import Engine
from anomalib.metrics import AUPRO
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from torch.utils.data import DataLoader

from vqi.config import Config


@dataclass
class Predictions:
    """Model outputs for one split, in dataset order."""

    image_paths: list[str]
    mask_paths: list[str | None]
    labels: np.ndarray  # 0 = good, 1 = defective
    scores: np.ndarray  # raw image scores
    anomaly_maps: torch.Tensor  # (N, H, W) raw pixel scores at model resolution
    gt_masks: torch.Tensor | None  # (N, H, W) bool at model resolution, None if no defects


def build_datamodule(config: Config) -> AnomalibDataModule:
    """Data module whose validation split holds good images only."""
    dataset = config.dataset
    common = {
        "root": dataset.root,
        "category": dataset.category,
        "train_batch_size": config.model.batch_size,
        "eval_batch_size": config.model.batch_size,
        "num_workers": config.model.num_workers,
        "seed": config.seed,
    }
    if dataset.name == "mvtec_ad2":
        # Dedicated validation/good folder; "public" is the only labelled test set.
        return MVTecAD2(**common, test_type="public")
    # anomalib's default for MVTec AD copies the test set into validation (see decisions D6).
    return MVTecAD(**common, val_split_mode="from_train", val_split_ratio=dataset.val_split_ratio)


def build_model(config: Config) -> AnomalibModule:
    """Model that returns raw scores: anomalib normalization and thresholding are off."""
    model_class = getattr(anomalib.models, config.model.name)
    size = config.model.image_size
    return model_class(
        **config.model.params,
        pre_processor=model_class.configure_pre_processor(image_size=(size, size)),
        post_processor=PostProcessor(enable_normalization=False, enable_thresholding=False),
        evaluator=False,
        visualizer=False,
    )


def build_engine(run_dir: Path) -> Engine:
    """Lightning engine on one device; checkpoints go to ``run_dir`` (git-ignored)."""
    return Engine(default_root_dir=run_dir, accelerator="auto", devices=1, logger=False)


def predict(engine: Engine, model: AnomalibModule, loader: DataLoader) -> Predictions:
    """Run the model on a loader and stack the outputs of all batches."""
    batches = engine.predict(model=model, dataloaders=loader, return_predictions=True)
    paths, mask_paths, labels, scores, maps, masks = [], [], [], [], [], []
    for batch in batches:
        n = len(batch.image_path)
        height, width = batch.anomaly_map.shape[-2:]
        paths += [str(p) for p in batch.image_path]
        # Good images have no mask; anomalib may report that as None, "" or the text "None".
        mask_paths += [
            str(p) if p and Path(str(p)).is_file() else None
            for p in (batch.mask_path or [None] * n)
        ]
        labels.append(batch.gt_label.cpu().reshape(n))
        scores.append(batch.pred_score.detach().cpu().reshape(n))
        maps.append(batch.anomaly_map.detach().cpu().reshape(n, height, width))
        if batch.gt_mask is not None:
            masks.append(batch.gt_mask.cpu().reshape(n, height, width).bool())
    return Predictions(
        image_paths=paths,
        mask_paths=mask_paths,
        labels=torch.cat(labels).numpy().astype(int),
        scores=torch.cat(scores).numpy().astype(float),
        anomaly_maps=torch.cat(maps),
        gt_masks=torch.cat(masks) if len(masks) == len(batches) else None,
    )


def pixel_aupro(predictions: Predictions, fpr_limit: float) -> float:
    """Area under the per-region-overlap curve up to ``fpr_limit``, normalised to [0, 1]."""
    if predictions.gt_masks is None:
        raise ValueError("AU-PRO needs ground-truth masks")
    metric = AUPRO(fields=["anomaly_map", "gt_mask"], fpr_limit=fpr_limit)
    metric.update(_MapBatch(predictions.anomaly_maps, predictions.gt_masks))
    return float(metric.compute())


@dataclass
class _MapBatch:
    """Minimal object with the two fields anomalib's AUPRO reads."""

    anomaly_map: torch.Tensor
    gt_mask: torch.Tensor
