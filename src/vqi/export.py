"""Export a trained anomalib model and score images with every exported format on CPU.

Needs the ``[train]`` and ``[export]`` extras. Each scorer maps one image (H, W, C float
in [0, 1]) to the model's raw anomaly score. The exported graphs contain the model's
pre-processing, so the scorers take the image at its original resolution.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import nncf
import numpy as np
import onnxruntime as ort
import openvino as ov
import torch
from anomalib.deploy import CompressionType, ExportType, OpenVINOInferencer
from anomalib.engine import Engine
from anomalib.models.components import AnomalibModule

Scorer = Callable[[np.ndarray], float]

# Formats exported by anomalib: name -> (export type, OpenVINO compression or None).
# INT8 is done separately by ``quantize_openvino_int8``, see there.
ANOMALIB_FORMATS: dict[str, tuple[ExportType, CompressionType | None]] = {
    "onnx_fp32": (ExportType.ONNX, None),
    "openvino_fp32": (ExportType.OPENVINO, None),
    "openvino_fp16": (ExportType.OPENVINO, CompressionType.FP16),
}
INT8_FORMAT = "openvino_int8"


def export_format(engine: Engine, model: AnomalibModule, name: str, out_dir: Path) -> Path:
    """Export ``model`` in one of ``ANOMALIB_FORMATS``; return the model file."""
    export_type, compression = ANOMALIB_FORMATS[name]
    return Path(
        engine.export(model, export_type, export_root=out_dir / name, compression_type=compression)
    )


def quantize_openvino_int8(
    fp32_model_file: Path,
    calibration_images: Sequence[str],
    load: Callable[[str], np.ndarray],
    out_dir: Path,
) -> Path:
    """Post-training INT8 quantization of the OpenVINO FP32 model with NNCF.

    anomalib 2.7.0's own INT8_PTQ reads ``model.input(0).shape`` whenever the batch axis is
    static, which fails while height and width are still dynamic (seen on Windows). Here the
    input is set to batch 1 with dynamic height and width, as the API uses it, and NNCF is
    calibrated on the given good validation images, loaded one at a time.
    """
    core = ov.Core()
    model = core.read_model(fp32_model_file)
    model.reshape({model.input(0): ov.PartialShape([1, 3, -1, -1])})
    calibration = nncf.Dataset(
        list(calibration_images), lambda path: load(path).transpose(2, 0, 1)[None]
    )
    quantized = nncf.quantize(model, calibration, subset_size=len(calibration_images))
    model_file = out_dir / INT8_FORMAT / "model.xml"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, model_file)
    return model_file


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
