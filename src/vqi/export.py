"""Export a trained anomalib model and score images with every exported format on CPU.

Needs the ``[train]`` and ``[export]`` extras. Each scorer maps one image (H, W, C float
in [0, 1]) to the model's raw anomaly score. The exported graphs contain the model's
pre-processing, so the scorers take the image at its original resolution.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from anomalib.data.datamodules.base import AnomalibDataModule
from anomalib.deploy import CompressionType, ExportType, OpenVINOInferencer
from anomalib.engine import Engine
from anomalib.models.components import AnomalibModule

Scorer = Callable[[np.ndarray], float]

# Format name -> (export type, OpenVINO compression or None).
FORMATS: dict[str, tuple[ExportType, CompressionType | None]] = {
    "onnx_fp32": (ExportType.ONNX, None),
    "openvino_fp32": (ExportType.OPENVINO, None),
    "openvino_fp16": (ExportType.OPENVINO, CompressionType.FP16),
    # Post-training quantization, calibrated on the (good-only) validation loader.
    "openvino_int8": (ExportType.OPENVINO, CompressionType.INT8_PTQ),
}


def export_all(
    engine: Engine, model: AnomalibModule, datamodule: AnomalibDataModule, out_dir: Path
) -> dict[str, Path]:
    """Export ``model`` to every format in ``FORMATS``; return the model file per format."""
    paths = {}
    for name, (export_type, compression) in FORMATS.items():
        paths[name] = Path(
            engine.export(
                model,
                export_type,
                export_root=out_dir / name,
                compression_type=compression,
                datamodule=datamodule if compression == CompressionType.INT8_PTQ else None,
            )
        )
    return paths


def model_size_mb(model_file: Path) -> float:
    """Size on disk of everything exported next to the model file.

    ONNX keeps large weights in a separate ``model.onnx.data`` file and OpenVINO in
    ``model.bin``, so counting only the model file would hide most of the size.
    """
    return sum(f.stat().st_size for f in model_file.parent.iterdir() if f.is_file()) / 2**20


def torch_scorer(model: AnomalibModule) -> Scorer:
    """The trained PyTorch model on CPU, as the baseline the exports are compared with."""
    model = model.to("cpu").eval()

    def score(image: np.ndarray) -> float:
        with torch.no_grad():
            batch = torch.from_numpy(image).permute(2, 0, 1)[None]
            return float(model(batch).pred_score.reshape(-1)[0])

    return score


def onnx_scorer(model_file: Path) -> Scorer:
    """ONNX Runtime on CPU."""
    session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def score(image: np.ndarray) -> float:
        (pred_score,) = session.run(["pred_score"], {input_name: image.transpose(2, 0, 1)[None]})
        return float(np.ravel(pred_score)[0])

    return score


def openvino_scorer(model_file: Path) -> Scorer:
    """OpenVINOInferencer (from anomalib) on CPU."""
    inferencer = OpenVINOInferencer(path=model_file, device="CPU")

    def score(image: np.ndarray) -> float:
        return float(np.ravel(inferencer.predict(image).pred_score)[0])

    return score


def scorer_for(name: str, model_file: Path) -> Scorer:
    """Pick the runtime that matches an exported format name."""
    return onnx_scorer(model_file) if name.startswith("onnx") else openvino_scorer(model_file)
