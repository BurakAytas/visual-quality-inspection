# Phase 1: project skeleton

## What was done

- A Python package `vqi` under `src/`, installed with `pip install -e .`.
- `pyproject.toml` with pinned versions in groups: `train` (GPU notebooks), `export`,
  `serve` (API and panel) and `dev` (pytest, ruff). The core needs only PyYAML.
- `configs/`: a base file plus one file per category (`fabric`, `leather`, `carpet`).
- `scripts/download_data.py`: installs one category from a local download (MVTec AD 2) or
  from the official MVTec host (MVTec AD), then checks the folder layout.
- `scripts/check_readme_claims.py`: makes sure every result number in the README matches a
  JSON file in `results/`.
- GitHub Actions CI: ruff lint, ruff format check, pytest, README claim check. No GPU.
- `DATA.md` (sources, licences, where to put files) and `docs/decisions.md`.

## Why

- **Config, not code, picks the product.** A factory changes products; the pipeline should
  not need a code change for that.
- **Small core install.** CI and the API should not download 2 GB of CUDA libraries.
- **Claims checked by CI.** A README number that does not come from a results file is the
  quickest way to lose a reviewer's trust, so the guard exists before any number does.
- **Reading the library source before using it.** This found that anomalib's default
  validation set for MVTec AD is a copy of the test set. Tuning a threshold on it would
  have been test-set leakage.

## Summary

Phase 1 is the skeleton, but I made three choices on purpose. First, everything is driven
by config: a new product is a new YAML file. Second, dependencies are split so CI runs in
seconds without a GPU, and the training stack is pinned to anomalib 2.7.0; I checked that
all pins resolve together. Third, I read anomalib's data code before trusting it. For MVTec
AD 2 the validation folder has only good images, so an F1-based threshold cannot be computed
there. For MVTec AD the default validation set is literally a copy of the test set, so I
will take validation images from the training set instead. That gives one honest rule for
both datasets: the threshold comes only from good validation images. I also added a CI check
that every number in the README must come from a results JSON file. The tests are real: each
one fails if the code it covers breaks, for example a config typo, a missing data folder, or
a README number that does not match its JSON.
