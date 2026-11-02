# Phase 5: export and CPU latency

## What was done

- `scripts/export_model.py`: trains PatchCore on `leather`, exports it to ONNX and to
  OpenVINO FP32 and FP16 with anomalib, and to OpenVINO INT8 with NNCF directly. Every
  format is exported from the CPU with one fixed input shape, [1, 3, 1024, 1024].
- For the PyTorch model and each format: a threshold from that format's own validation
  scores, then AUROC, recall and false-alarm rate on the test images, and CPU latency per
  image (warm-up, median, 95th percentile). One failing format no longer stops the others.
- `vqi.latency` (timing), `vqi.imageio` (image loading and resizing without torch, shared
  with the API), `vqi.export` (export, shape check, INT8, scorers).

## Why

- **The line runs on a CPU.** A GPU in every inspection station is expensive; the question
  is whether the model is fast enough on an ordinary industrial PC.
- **Accuracy must be re-checked after export.** The exported graph resizes without
  antialiasing, and INT8 rounds weights, so the scores shift. Each format therefore gets
  its own threshold, and its AUROC and false-alarm rate are measured again.
- **One fixed input shape.** Dynamic axes came out differently on different machines; a
  camera delivers one image of one size anyway, so a static shape is both safer and true
  to the use case.

## Bugs found on the way (all in `docs/decisions.md`)

A missing `onnxscript` dependency (D45); a Windows console that could not print torch's
emoji (D45); anomalib's INT8 step failing on a partly static shape (D46); a GPU-traced
export that fixed the batch size at 2 (D47); file names without folders in the score CSVs
(D48). Several of these only appeared on Windows, which is why the smoke tests now cover
the real architecture and a CPU export path.

## Summary

The PatchCore model kept its accuracy in every format: AUROC 1.000 and the same 12.5%
false-alarm rate on the test good parts, from PyTorch to OpenVINO INT8. On a 12th-gen
Intel desktop CPU the median time per 1024 x 1024 image fell from 316 ms in PyTorch to
195 ms with OpenVINO FP32 and 167 ms with INT8. INT8 gave less than I expected, and I
think the reason is that PatchCore's memory-bank search stays in floating point; I would
profile that part before promising a faster line. Getting there took five fixes, most of
them Windows- or GPU-specific, and each one is written down with how it was found.
