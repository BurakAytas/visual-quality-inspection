# Visual Quality Inspection

> Work in progress: Phase 2 of 7 is done. Every result number in this README is checked
> against `results/*.json` in CI by `scripts/check_readme_claims.py`.

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

## Results so far (Phase 2: PatchCore baseline)

PatchCore, trained on good images only. The decision threshold is set on good
validation images only, for a 1%<!-- claim: results/thresholds.json#patchcore/fabric.target_false_alarm_rate --> false-alarm target,
and the test split is scored once. Brackets are 95% bootstrap intervals over physical
parts (MVTec AD 2 photographs each test part under six lights; those six images are
resampled together). Pixel AU-PRO is integrated up to a 30% false-positive rate.

<!-- results:start -->
| Category | Test parts | Image AUROC | Recall at threshold | False-alarm rate at threshold | Pixel AU-PRO |
|---|---|---|---|---|---|
| fabric (main set) | 26<!-- claim: results/metrics.json#patchcore/fabric.image.n_test_parts --> | 0.657<!-- claim: results/metrics.json#patchcore/fabric.image.image_auroc.value --> [0.473<!-- claim: results/metrics.json#patchcore/fabric.image.image_auroc.ci95.0 -->, 0.825<!-- claim: results/metrics.json#patchcore/fabric.image.image_auroc.ci95.1 -->] | 0.144<!-- claim: results/metrics.json#patchcore/fabric.image.recall.value --> [0.022<!-- claim: results/metrics.json#patchcore/fabric.image.recall.ci95.0 -->, 0.311<!-- claim: results/metrics.json#patchcore/fabric.image.recall.ci95.1 -->] | 0.030<!-- claim: results/metrics.json#patchcore/fabric.image.false_alarm_rate.value --> [0.000<!-- claim: results/metrics.json#patchcore/fabric.image.false_alarm_rate.ci95.0 -->, 0.091<!-- claim: results/metrics.json#patchcore/fabric.image.false_alarm_rate.ci95.1 -->] | 0.296<!-- claim: results/metrics.json#patchcore/fabric.pixel.pixel_aupro_0.3 --> |
| leather | 124<!-- claim: results/metrics.json#patchcore/leather.image.n_test_parts --> | 1.000<!-- claim: results/metrics.json#patchcore/leather.image.image_auroc.value --> [1.000<!-- claim: results/metrics.json#patchcore/leather.image.image_auroc.ci95.0 -->, 1.000<!-- claim: results/metrics.json#patchcore/leather.image.image_auroc.ci95.1 -->] | 1.000<!-- claim: results/metrics.json#patchcore/leather.image.recall.value --> [1.000<!-- claim: results/metrics.json#patchcore/leather.image.recall.ci95.0 -->, 1.000<!-- claim: results/metrics.json#patchcore/leather.image.recall.ci95.1 -->] | 0.094<!-- claim: results/metrics.json#patchcore/leather.image.false_alarm_rate.value --> [0.000<!-- claim: results/metrics.json#patchcore/leather.image.false_alarm_rate.ci95.0 -->, 0.188<!-- claim: results/metrics.json#patchcore/leather.image.false_alarm_rate.ci95.1 -->] | 0.976<!-- claim: results/metrics.json#patchcore/leather.pixel.pixel_aupro_0.3 --> |
| carpet | 117<!-- claim: results/metrics.json#patchcore/carpet.image.n_test_parts --> | 0.988<!-- claim: results/metrics.json#patchcore/carpet.image.image_auroc.value --> [0.967<!-- claim: results/metrics.json#patchcore/carpet.image.image_auroc.ci95.0 -->, 1.000<!-- claim: results/metrics.json#patchcore/carpet.image.image_auroc.ci95.1 -->] | 0.978<!-- claim: results/metrics.json#patchcore/carpet.image.recall.value --> [0.944<!-- claim: results/metrics.json#patchcore/carpet.image.recall.ci95.0 -->, 1.000<!-- claim: results/metrics.json#patchcore/carpet.image.recall.ci95.1 -->] | 0.071<!-- claim: results/metrics.json#patchcore/carpet.image.false_alarm_rate.value --> [0.000<!-- claim: results/metrics.json#patchcore/carpet.image.false_alarm_rate.ci95.0 -->, 0.179<!-- claim: results/metrics.json#patchcore/carpet.image.false_alarm_rate.ci95.1 -->] | 0.949<!-- claim: results/metrics.json#patchcore/carpet.pixel.pixel_aupro_0.3 --> |
<!-- results:end -->

What this shows:

- **The harder dataset is genuinely hard for this baseline.** On MVTec AD 2 `fabric` the
  validation threshold misses 85.6%<!-- claim: results/metrics.json#patchcore/fabric.image.miss_rate.value --> of defective images, and the AUROC
  interval reaches down to 0.473<!-- claim: results/metrics.json#patchcore/fabric.image.image_auroc.ci95.0 -->: with 26<!-- claim: results/metrics.json#patchcore/fabric.image.n_test_parts --> physical test parts it is not
  clearly better than chance. Defective and good fabric parts get almost the same scores.
  The missed defects in the failure gallery are thin lines and small marks that look
  like the printed pattern, and the input is downscaled to 256 px.
- **A 1% target on about 50 validation images is not a 1% guarantee.** The thresholds come
  from 43<!-- claim: results/thresholds.json#patchcore/fabric.n_validation_images -->, 49<!-- claim: results/thresholds.json#patchcore/leather.n_validation_images --> and 56<!-- claim: results/thresholds.json#patchcore/carpet.n_validation_images --> good images, so each one is simply the highest
  validation score. On the test good parts the false-alarm rate was
  3.0%<!-- claim: results/metrics.json#patchcore/fabric.image.false_alarm_rate.value --> (fabric), 9.4%<!-- claim: results/metrics.json#patchcore/leather.image.false_alarm_rate.value --> (leather) and 7.1%<!-- claim: results/metrics.json#patchcore/carpet.image.false_alarm_rate.value --> (carpet). On a line the threshold
  needs more good images, taken under the line's own lighting.
- `leather` and `carpet` (MVTec AD) are close to saturated, as expected for that dataset.

Full table with every metric: [`results/metrics.md`](results/metrics.md). Failure gallery:
`results/patchcore/<category>/overlays/` (true positives, misses, false alarms).
Decisions and negative results: [`docs/decisions.md`](docs/decisions.md).

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
| 2 | PatchCore baseline, validation-only threshold, metrics, failure overlays | done |
| 3 | EfficientAD comparison, robustness to perturbations | planned |
| 4 | Defect-type classifier (ResNet18 vs HOG + SVM), hybrid pipeline | planned |
| 5 | ONNX / OpenVINO export, CPU latency | planned |
| 6 | FastAPI line integration, SQLite decision log, Docker | planned |
| 7 | Operator panel, final README | planned |

Design decisions and negative results are logged in [docs/decisions.md](docs/decisions.md).

## License

Code: MIT. Datasets and any weights trained on them: CC BY-NC-SA 4.0 (see [DATA.md](DATA.md)).
