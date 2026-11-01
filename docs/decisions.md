# Decision log

Each entry: what was decided, why, and what was checked. Negative results and bugs found
are logged here too.

## Phase 1: skeleton

### D1. anomalib 2.7.0 is pinned; the rest of the stack is pinned to match it
anomalib 2.7.0 (released 2026-10-06) needs Python >= 3.10, `torch >= 2.10`,
`lightning >= 2.6` (excluding 2.6.2 and 2.6.3) and `opencv-python-headless`. Pins in
`pyproject.toml`: torch 2.14.1 + torchvision 0.29.1 (torch 2.15.0 was four days old; see
D22 for why not torchvision 0.29.2), scikit-learn 1.9.1, onnx 1.23.2,
onnxruntime 1.31.1, openvino 2026.4.1, nncf 3.4.0, fastapi 0.162.0, uvicorn 0.54.0,
streamlit 1.66.1. Checked: `pip install --dry-run ".[train,export,serve]"` resolves all pins
together on Python 3.13 / Linux. It still has to be confirmed on the notebook's Python in
Phase 2; if it fails there, the change is logged here.

### D2. `opencv-python-headless` instead of `opencv-python`
anomalib already depends on the headless build. Installing both puts two copies of `cv2`
in one environment, and the one that wins is unpredictable. The headless build has the
same image-processing API without GUI libraries, which is also what a server or container
needs.

### D3. Optional dependency groups (`train`, `export`, `serve`, `dev`)
The core package depends only on PyYAML. CI installs `.[dev]` and runs in seconds without
a GPU stack. Notebooks install `.[train]`; the API container will install `.[serve]` plus
the export runtime it needs.

### D4. One YAML file per category, with a shared base
`configs/<category>.yaml` extends `configs/base.yaml` and names the dataset and category.
Adding a product (e.g. `sheet_metal`) means copying a file. Unknown keys are rejected, so a
typo cannot be silently ignored.

### D5. MVTec AD 2 API in anomalib 2.7.0, verified in the installed source
Read in `anomalib/data/datamodules/image/mvtecad2.py` and
`anomalib/data/datasets/image/mvtecad2.py`:
- `from anomalib.data import MVTecAD2` exists. Arguments include `root`, `category`,
  `test_type` (default `"public"`), `seed`.
- `prepare_data()` downloads only if `root / category` is not a folder. We point `root` at
  our manual download, so it never downloads.
- Folders read: `train/good`, `validation/good` (normal only),
  `test_public/{good,bad}`, masks at `test_public/ground_truth/bad/<name>_mask.png`. A bad
  image without a mask raises an error. `test_private*` images get label `-1` and are never
  used for metrics.
- The validation set is a dedicated folder of good images only. This is why anomalib's
  default F1-based threshold cannot work on it, and why Phase 2 sets the threshold from a
  quantile of normal validation scores.

### D6. MVTec AD (leather, carpet): anomalib's default validation set IS the test set
Read in `anomalib/data/datamodules/image/mvtecad.py` and `base/image.py`: `MVTecAD` defaults
to `val_split_mode=SAME_AS_TEST`, which sets `val_data = deepcopy(test_data)`. Choosing a
threshold on that "validation" set is tuning on test, the exact leak this project must
avoid. Decision for Phase 2: use `val_split_mode="from_train"` with a fixed `seed`, which
moves a random part of `train/good` into validation. Validation then has good images only,
the same situation as MVTec AD 2, so one threshold policy covers both datasets. Cost: fewer
training images; the split ratio and seed will be recorded with the results.

### D7. MVTec AD download uses anomalib's official URL and checksum
`scripts/download_data.py` reuses `DOWNLOAD_INFO` from anomalib 2.7.0: the official MVTec
share on mydrive.ch with a SHA-256 hash. It downloads all 15 categories (about 5 GB) in one
archive; no per-category URL was used because none could be verified. MVTec AD 2 is never
downloaded by code: it needs the web form, so the script only accepts a local path.

### D8. Archive extraction uses `tarfile` with `filter="data"`
This blocks absolute paths, `..` paths and device files inside an archive. A folder source
is symlinked instead of copied, to save disk space in Colab.

### D9. README numbers are checked in CI
`scripts/check_readme_claims.py` compares each number marked
`<!-- claim: results/<file>.json#<key> -->` with the JSON value, and fails if a number in a
`results:start`/`results:end` block has no marker. Added in Phase 1 so it guards every
later phase.

### D10. Code under MIT, data and weights under CC BY-NC-SA 4.0
The code is the author's own work. The datasets are non-commercial and share-alike, so any
weights trained on them inherit that licence.

### D11. MVTec AD 2 licence checked against the official page
Read on 2026-11-01 from the official MVTec AD 2 page: CC BY-NC-SA 4.0, commercial use not
allowed. This matches the project plan, so no change was needed. The text is quoted word for
word in `DATA.md`. The page now cites the IJCV 2026 journal version of the paper, so
`DATA.md` cites that instead of the earlier arXiv preprint. The MVTec AD page also routes
downloads through a web form; `DATA.md` asks the user to fill it in before running the
download script.

## Phase 2: PatchCore baseline

### D12. Our own threshold policy: a false-alarm quantile of good validation scores
anomalib's default `F1AdaptiveThreshold` needs defective images in validation. Our
validation sets hold good images only (D5, D6), so it would fall back to the maximum
normal score with a warning. Instead `vqi.threshold.threshold_for_false_alarm_rate` takes
good-image scores and a target false-alarm rate (1%) and nothing else: with n images it
allows k = floor(rate * n) of them above the threshold. A unit test checks that the
function cannot receive labels. The threshold, quantile, n and the observed validation
false-alarm rate are stored in `results/thresholds.json` under `<model>/<category>`.
Limit: with n validation images the false-alarm rate cannot be resolved below 1/n; with
fewer than 100 images a 1% target becomes "maximum normal score".

### D13. First version used `np.quantile(method="higher")`; replaced (bug found by a test)
On 1000 scores the 99% "higher" quantile is the 991st value, which allows only 9 false
alarms (0.9%) when 10 (1.0%) are allowed. The unit test that checks "one step lower would
exceed the target" failed. Fixed by sorting and taking the (k+1)-th highest score directly.

### D14. anomalib normalization and thresholding are switched off
`PostProcessor(enable_normalization=False, enable_thresholding=False)`. The model then
returns raw scores and raw anomaly maps. Reasons: the threshold must be ours (D12), and
normalised scores depend on min/max statistics that anomalib collects on the validation
set, which adds a hidden second calibration step. Phase 5 (export) and the API must apply
our threshold to the same raw score. anomalib's evaluator and visualizer are off too; the
metrics are computed in `vqi.metrics` from saved raw scores.

### D15. Raw scores are saved, metrics are computed from them on CPU
The GPU notebook writes `results/<model>/<category>/scores.csv` (every validation and test
image, relative paths only). `scripts/evaluate.py` turns these into `results/metrics.json`
without a GPU or the dataset, so every README number can be recomputed from the committed
files.

### D16. Metrics reported
- Image AUROC (threshold-free ranking quality).
- F1, precision, recall, false-alarm rate and miss rate at **our validation threshold**:
  what would happen on the line.
- Miss rate at a fixed 1% false-alarm rate **on the test good images**. This threshold is
  read from test data, so it is a comparison metric between models (like TPR at fixed FPR),
  not a deployable setting. It is labelled that way everywhere.
- Pixel AU-PRO with FPR limits 0.3 (MVTec AD convention) and 0.05 (MVTec AD 2 paper
  convention), from anomalib's `AUPRO` on raw anomaly maps at model resolution.
- 95% CIs: stratified bootstrap (good and defective images resampled separately,
  1000 resamples, fixed seed). The threshold is held fixed, so the CI covers test-sample
  variation only, not threshold uncertainty from the validation set.

### D17. Input: 256 x 256 resize, no centre crop
PatchCore's default pre-processor in anomalib 2.7.0 is a resize to 256 x 256 (a centre crop
is optional). A crop would cut away defects near the image border, so none is used.
Downscaling high-resolution MVTec AD 2 images to 256 px can hide very small defects; this
is a known limit of the baseline and is recorded in the results.

### D18. Model built from config by class name
`model.name: Patchcore` is looked up as `anomalib.models.Patchcore`; `model.params` go to
its constructor unchanged. A second model in Phase 3 needs a config change, not new code.

### D19. Failure gallery
For each category: the 8 highest-scoring caught defects (TP), the 8 lowest-scoring missed
defects (FN) and the 8 highest-scoring false alarms (FP). The heatmap colour scale is fixed
per model and category: transparent at the median pixel score of good validation images,
strongest at the image threshold. Ground-truth defect regions are outlined in green.
Images are saved as 512 px JPEGs to keep the repository small.

### D20. pandas pinned to 2.3.3 (bug found in a CPU smoke test)
A CPU run of `scripts/train_anomaly.py` on a small synthetic dataset crashed inside
anomalib: `TypeError: Path must be None, a string, or Path object, got <class 'float'>`.
Cause: anomalib 2.7.0 has no upper bound on pandas, and pandas 3 stores a missing value in
a string column as `NaN` instead of `None`, so good images without a mask get
`mask_path = NaN`. Pinning `pandas==2.3.3` in the `train` extra fixed it; the same smoke
run then finished for both dataset layouts (MVTec AD 2, and MVTec AD with a train-based
validation split) and `scripts/evaluate.py` produced the table.

### D21. torch installed from the CUDA 12.6 build in notebooks (reverted, see D23)
The default PyPI wheel of torch 2.14.1 is built for CUDA 13, which needs a recent GPU
driver. Notebooks install `torch==2.14.1` from the official PyTorch CUDA 12.6 index (same
version, older driver requirement) and then stop with an assertion if no GPU is visible,
so a run cannot fall back to CPU without notice.

### D22. torchvision 0.29.2 replaced by 0.29.1 (failed on the first Colab run)
The first Colab run stopped at install: no torchvision 0.29.2 wheel exists for Python 3.12
on Linux, neither on PyPI nor on the PyTorch CUDA 12.6 index. The earlier dependency check
ran on Python 3.13, where that wheel exists, so it did not catch this. torchvision 0.29.1
was released together with torch 2.14.1 and has Python 3.12 Linux wheels on both indexes.
Lesson: check pins against the notebook's Python version, not the local one. All other
pinned binary packages were checked for Python 3.12 Linux wheels.

### D23. Notebook installs torch and torchvision from PyPI only (second Colab failure)
The second Colab run failed with a torch/torchvision CUDA mismatch. Colab now ships Python
3.13 with torch 2.14.1 built for CUDA 13 preinstalled. The notebook's
`pip install torch==2.14.1 --index-url .../cu126` saw "2.14.1 already installed", skipped
torch, and installed the CUDA 12.6 build of torchvision next to the CUDA 13 torch. D21 is
reverted: both now come from PyPI (same CUDA 13 build), which Colab's driver supports. The
install step now imports torchvision as well as torch, so a mismatch stops the run there.
All pinned binary packages were checked for Python 3.13 Linux wheels.
