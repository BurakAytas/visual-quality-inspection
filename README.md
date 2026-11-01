# Visual Quality Inspection

> Work in progress. Phase 1 of 7 (project skeleton) is done. No model has been trained yet,
> so this README contains no results. Every number added later is checked against
> `results/*.json` in CI by `scripts/check_readme_claims.py`.

## Problem

Products leave the line with visual defects (scratches, holes, cuts, stains, colour errors,
contamination, missing parts) and reach the customer. Manual inspection is slow and
inconsistent. Defect samples are rare on a real line, so this system learns from good parts
only. It flags anomalies, shows where they are, guesses the defect type, and logs every
decision so quality engineers can audit it.

The pipeline is product-agnostic: a new product needs a new config file and new training
images, not new code.

## Demo data

Public datasets are used as examples only (non-commercial licence, see [DATA.md](DATA.md)):
MVTec AD 2 `fabric` (main), MVTec AD `leather` and `carpet` (secondary).

## Quick start

```bash
pip install -e ".[dev]"
pytest
```

Put a dataset in place (details in [DATA.md](DATA.md)):

```bash
python scripts/download_data.py --config configs/leather.yaml --source /path/to/leather.tar.xz
```

## Project status

| Phase | Content | Status |
|---|---|---|
| 1 | Skeleton, configs, data script, CI | done |
| 2 | PatchCore baseline, validation-only threshold, metrics, failure overlays | planned |
| 3 | EfficientAD comparison, robustness to perturbations | planned |
| 4 | Defect-type classifier (ResNet18 vs HOG + SVM), hybrid pipeline | planned |
| 5 | ONNX / OpenVINO export, CPU latency | planned |
| 6 | FastAPI line integration, SQLite decision log, Docker | planned |
| 7 | Operator panel, final README | planned |

Design decisions and negative results are logged in [docs/decisions.md](docs/decisions.md).

## License

Code: MIT. Datasets and any weights trained on them: CC BY-NC-SA 4.0 (see [DATA.md](DATA.md)).
