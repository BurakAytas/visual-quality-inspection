"""Export the anomaly model to ONNX and OpenVINO, then check accuracy and CPU speed.

For the PyTorch model (baseline) and each exported format:
  1. score the good validation images and set that format's own threshold (same policy);
  2. score and time the test images on CPU, one image at a time;
  3. report image AUROC, recall and false-alarm rate with 95% bootstrap intervals.

The exported graphs resize without antialiasing (ONNX cannot export it), so their scores
differ slightly from training; that is why every format gets its own threshold.

Writes ``results/export/<model>/<category>/``: export.json, export.md, scores_<format>.csv,
and adds ``<model>/<category>/<format>`` thresholds to ``results/thresholds.json``.
Exported model files go to ``exports/`` (git-ignored).

Example:
    python scripts/export_model.py --config configs/leather.yaml --model <model.yaml>
    (with <model.yaml> = configs/models/patchcore.yaml)
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path

import anomalib
import numpy as np
import onnxruntime
import openvino
import pandas as pd
import torch
from lightning.pytorch import seed_everything

from vqi.anomaly import build_datamodule, build_engine, build_model
from vqi.config import Config, load_config
from vqi.dataset_files import relative_to_category
from vqi.export import (
    ANOMALIB_FORMATS,
    INT8_FORMAT,
    Scorer,
    export_format,
    model_size_mb,
    quantize_openvino_int8,
    scorer_for,
    torch_scorer,
)
from vqi.imageio import load_rgb, resize_to
from vqi.latency import timed_map
from vqi.metrics import physical_part, summarize
from vqi.provenance import warn_if_dirty
from vqi.threshold import save_threshold, threshold_record

METRICS = ("image_auroc", "recall", "false_alarm_rate")


def _load_at_size(path: str, size: tuple[int, int]) -> np.ndarray:
    """Decode an image and bring it to the export input size (not timed)."""
    return resize_to(load_rgb(path), size)


def cpu_name() -> str:
    """Human-readable CPU model, so latency numbers can be compared fairly."""
    if platform.system() == "Windows":  # platform.processor() gives only the family there
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if platform.system() == "Linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def environment() -> dict:
    """Where and with what the numbers were measured."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "anomalib": anomalib.__version__,
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "openvino": openvino.__version__,
        "git_commit": commit.stdout.strip() or "unknown",
    }


def write_scores(path: Path, category: str, rows: list[tuple[str, str, int, float]]) -> None:
    """One CSV row per image: split, image path from the category folder, label, raw score."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image", "label", "score"])
        writer.writerows(
            (s, relative_to_category(p, category), y, f"{v:.6f}") for s, p, y, v in rows
        )


def markdown_table(formats: dict) -> str:
    """Format x size x latency x accuracy; failed formats keep a row with the error."""
    lines = [
        "| Format | Size (MB) | Median ms/image | p95 ms/image | Image AUROC | Recall "
        "| False-alarm rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, f in formats.items():
        if "error" in f:
            lines.append(f"| {name} | failed: {f['error'].splitlines()[0][:80]} ||||||")
            continue
        size = "-" if f["size_mb"] is None else f"{f['size_mb']:.0f}"
        auroc, recall, far = (f["image"][m] for m in METRICS)
        lines.append(
            f"| {name} | {size} | {f['latency']['median_ms']:.1f} | {f['latency']['p95_ms']:.1f} "
            f"| {auroc['value']:.3f} | {recall['value']:.3f} | {far['value']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def guarded(name: str, step: Callable[[], dict | Path], failures: dict[str, str]) -> object:
    """Run one format's step; on error record it and return None, so other formats go on."""
    try:
        return step()
    except Exception as exc:  # one broken format must not stop the others
        traceback.print_exc()
        failures[name] = f"{type(exc).__name__}: {exc}"
        print(f"{name}: FAILED, continuing with the other formats", flush=True)
        return None


def benchmark(
    config: Config,
    key: str,
    name: str,
    scorer: Scorer,
    model_file: Path | None,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
    load_image: Callable[[str], np.ndarray],
    warmup: int,
) -> dict:
    """Threshold from validation, then score and time the test images for one format."""
    val_scores, _ = timed_map(scorer, val.image_path, load=load_image, warmup=0)
    record = threshold_record(np.array(val_scores), config.threshold.target_false_alarm_rate)
    save_threshold(config.results_dir / "thresholds.json", f"{key}/{name}", record)

    labels = test.label_index.to_numpy().astype(int)
    test_scores, latency = timed_map(scorer, test.image_path, load=load_image, warmup=warmup)
    image = summarize(
        np.array(test_scores),
        labels,
        record["threshold"],
        config.evaluation.fixed_false_alarm_rate,
        config.evaluation.bootstrap_samples,
        config.seed,
        groups=np.array([physical_part(p) for p in test.image_path]),
        names=METRICS,
    )
    rows = [("validation", p, 0, s) for p, s in zip(val.image_path, val_scores, strict=True)]
    rows += [
        ("test", p, int(y), s) for p, y, s in zip(test.image_path, labels, test_scores, strict=True)
    ]
    write_scores(out_dir / f"scores_{name}.csv", config.dataset.category, rows)
    auroc = image["image_auroc"]["value"]
    print(f"{key} {name}: {latency['median_ms']:.1f} ms/image, AUROC {auroc:.3f}", flush=True)
    return {
        "size_mb": None if model_file is None else model_size_mb(model_file),
        "threshold": record["threshold"],
        "n_validation_images": record["n_validation_images"],
        "image": {m: image[m] for m in METRICS},
        "latency": latency,
    }


def run(config: Config, warmup: int) -> list[str]:
    """Fit, export, then score and time every format on CPU; return the failed formats."""
    seed_everything(config.seed, workers=True)
    key = f"{config.model.key}/{config.dataset.category}"
    out_dir = config.results_dir / "export" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    exports_dir = Path("exports") / key

    datamodule = build_datamodule(config)
    model = build_model(config)
    engine = build_engine(config, Path("runs") / "export" / key)
    engine.fit(model=model, datamodule=datamodule)
    val = datamodule.val_data.samples
    test = datamodule.test_data.samples
    if val.label_index.any():
        raise RuntimeError("Validation split contains defective images")

    # Export from the CPU, the deployment target: tracing on a GPU fixed the batch axis
    # to the example size 2 on the user's machine (D47), and CPU export is what was tested.
    model = model.to("cpu")

    # One fixed input size for every format: the camera image size of this category.
    size = load_rgb(val.image_path.iloc[0]).shape[:2]
    load_image = partial(_load_at_size, size=size)
    print(f"{key}: exporting with fixed input [1, 3, {size[0]}, {size[1]}]", flush=True)

    failures: dict[str, str] = {}
    model_files: dict[str, Path] = {}
    for name in ANOMALIB_FORMATS:
        step = partial(export_format, engine, model, name, exports_dir, size)
        path = guarded(name, step, failures)
        if path is not None:
            model_files[name] = path
    if "openvino_fp32" in model_files:
        step = partial(
            quantize_openvino_int8,
            model_files["openvino_fp32"],
            list(val.image_path),
            load_image,
            exports_dir,
        )
        path = guarded(INT8_FORMAT, step, failures)
        if path is not None:
            model_files[INT8_FORMAT] = path
    else:
        failures[INT8_FORMAT] = "skipped: needs the openvino_fp32 export"

    # The PyTorch baseline goes first; torch_scorer moves the model to the CPU, so it is
    # built only after all exports are done.
    formats: dict[str, dict] = {}
    for name, model_file in [("torch_fp32", None), *model_files.items()]:

        def step(name: str = name, model_file: Path | None = model_file) -> dict:
            scorer = torch_scorer(model) if model_file is None else scorer_for(name, model_file)
            return benchmark(
                config, key, name, scorer, model_file, val, test, out_dir, load_image, warmup
            )

        result = guarded(name, step, failures)
        if result is not None:
            formats[name] = result
    for name, error in failures.items():
        formats.setdefault(name, {"error": error})

    result = {"environment": environment(), "input_size": list(size), "formats": formats}
    (out_dir / "export.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out_dir / "export.md").write_text(markdown_table(formats), encoding="utf-8")
    return sorted(failures)


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="category config")
    parser.add_argument("--model", required=True, type=Path, help="configs/models/<model>.yaml")
    parser.add_argument("--warmup", type=int, default=10, help="untimed runs before timing")
    args = parser.parse_args()
    # torch's ONNX exporter prints emoji; a Windows console (e.g. cp1254) cannot encode
    # them and the export would crash. Write UTF-8 regardless of the console code page.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    warn_if_dirty()
    failed = run(load_config(args.config, args.model), args.warmup)
    if failed:
        print(f"Finished; results written, but these formats FAILED: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
