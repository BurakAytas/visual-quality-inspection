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

### D7. MVTec AD download uses anomalib's official URL and checksum (removed, see D24)
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

### D24. Automatic MVTec AD download removed (404 on Colab)
The MVTec AD URL shipped with anomalib 2.7.0 returned `HTTP Error 404` on Colab; MVTec now
hands out downloads through a web form, like MVTec AD 2. `download_mvtec_ad` was deleted
and both datasets are installed from local archives. `--source` also accepts a folder that
holds `<category>.tar*`, so one Drive folder can serve `leather` and `carpet`. When the
category folder is not found, the error lists the top-level entries it did see.

### D25. Kaggle detected by `KAGGLE_KERNEL_RUN_TYPE`, not by a `/kaggle` folder
On the Colab run `os.path.exists("/kaggle")` was `True`, so the notebook used the Kaggle
paths and could not find the data. Kaggle sets the `KAGGLE_KERNEL_RUN_TYPE` environment
variable; Colab does not.

### D26. Notebook steps stop on the first error, and the notebook is simulated before release
With `!python ...` a failed step only printed an error and the next cells kept running,
so a data failure showed up later as a missing `scores.csv`. Every step now goes through a
small `run()` helper that raises on a non-zero exit code. Before this version was handed
over, all code cells were executed on CPU against synthetic archives laid out like the
Drive upload (`fabric.tar.gz`, `leather.tar.xz`, `carpet.tar.xz`), with a tiny untrained
backbone in place of the GPU model: install, all three training runs, evaluation and the
zip finished. Only the GPU check, Drive mount and browser download were stubbed.

### D27. Mask path "None" as text (bug found on the Colab run)
`leather` trained, then crashed while saving the failure gallery:
`cv2.imread("None")` returned nothing. When anomalib collates a batch that mixes defective
and good MVTec AD images, the good image's mask path becomes the string `'None'`
(reproduced on CPU by collating one defective and one good sample). The CPU smoke test had
missed it because its batches were never mixed. Fix: a mask path counts only if it is an
existing file; `save_overlay` now raises a clear error if a given mask cannot be read.
`fabric` (MVTec AD 2) was not affected and its results stand.

### D28. Bootstrap over physical parts, not images (MVTec AD 2 test design)
MVTec AD 2 `test_public` photographs every part under six lights (`_regular`,
`_overexposed`, `_underexposed`, `_shift_1..3`): 156 `fabric` test images are 26 parts
(15 defective, 11 good). The first metrics resampled images as if they were independent,
which made the intervals too narrow: image AUROC 0.657 had a 95% CI of [0.566, 0.739].
`scripts/evaluate.py` now maps each image to its part (`physical_part`), and
`vqi.metrics.bootstrap_ci` resamples whole parts. The `fabric` AUROC CI became
[0.473, 0.825], which includes 0.5. MVTec AD has one image per part, so `leather` and
`carpet` did not change. A unit test checks that grouping widens the interval when
images repeat one part.

### D29. README claim checker reads dotted keys and list items
Pixel metrics are stored as `pixel_aupro_0.3`, and CIs as `[low, high]` lists. The checker
now matches the longest key at each level and accepts a number as a list index
(`...ci95.0`). Both cases have tests.

### D30. Phase 2 results and what they say (computed from the committed `results/`)
Run on a Colab T4 with anomalib 2.7.0 and torch 2.14.1 (CUDA 13 build). `fabric` was
trained at commit `d7ea3ac`, `leather` and `carpet` at `0e019aa` (after the D27 fix, which
only touches the failure gallery). All metrics were recomputed on CPU from `scores.csv`
and matched the Colab output exactly before D28 changed the intervals.

Negative results, kept on purpose:
- `fabric` (MVTec AD 2): image AUROC 0.657 [0.473, 0.825]; at the validation threshold
  recall is 0.144, so 85.6% of defective images are missed. Pixel AU-PRO is 0.296 (FPR up
  to 0.3) and 0.029 (FPR up to 0.05).
- Scores by lighting, from `results/patchcore/fabric/scores.csv`: all 43 validation images
  are `_regular`, median score 36.9. Test good parts score higher even under regular light
  (median 38.9). In every lighting, defective and good medians are within about 1 point.
  Under regular light 1 of 15 defects is caught; under `shift_2`/`shift_3` 4 of 15, with
  1 of 11 good parts flagged. So the main problem is weak separation, and a second one is
  that validation covers only one lighting condition.
- Failure gallery: missed `fabric` defects are thin lines and small marks that resemble
  the printed pattern; the two false alarms come from shifted-light images, where the
  heatmap spreads over the lit areas. The missed `carpet` defects are a thin metal strand
  with the weave's colour and a thread defect. The `leather` false alarms have hot spots
  near the image border.
- Threshold vs. target: with 43 to 56 validation images the 1% target allows zero false
  alarms, so each threshold is the highest validation score. Test false-alarm rates were
  3.0% (`fabric`), 9.4% (`leather`) and 7.1% (`carpet`), with intervals up to 18.8%.
  Production consequence: calibrate on more good parts, under every lighting the line
  uses, and recheck the false-alarm rate on fresh good parts before trusting the 1%.
- `leather` (AUROC 1.000) and `carpet` (0.988) confirm that MVTec AD is close to saturated.

Ideas for later phases, not done here: higher input resolution or tiling for `fabric`
(defects are small relative to the image), and a validation set that includes every
lighting condition. Changing either would be a scope decision for the user.

## Phase 3: EfficientAD comparison and robustness

### D31. Model settings live in `configs/models/*.yaml`, passed with `--model`
A category config describes data; a model config describes the model. The model file
replaces the whole `model` section, so PatchCore's `backbone` cannot leak into
EfficientAD's constructor (a deep merge would have done that). `model.trainer` is passed
to the anomalib `Engine` (Lightning trainer), `model.eval_batch_size` is separate because
EfficientAD must train with batch size 1 but can score in batches.

### D32. EfficientAD settings and their limits
`EfficientAd(model_size="small")`, train batch size 1 (anomalib raises an error
otherwise), 20 epochs. The paper trains for 70,000 steps; 20 epochs is about 4,000 to
8,000 steps here, a budget for a free Colab GPU, so the result is "EfficientAD at this
budget", not the best EfficientAD can do. Epochs are used instead of steps because
EfficientAD sets its anomaly-map normalisation (90% and 99.5% quantiles) in
`on_validation_start` from the validation loader, and Lightning validates at the end of
each epoch. That loader holds good images only, so no labels are used, but the same good
images also set our threshold; this makes validation scores slightly optimistic for
EfficientAD. On first use anomalib downloads Imagenette (penalty images for the student)
and the pretrained teacher weights.

### D33. EfficientAD patent note is worded carefully
The brief asks for a README note that EfficientAD is patented with limited commercial use.
I could not confirm a specific patent: web searches found the paper (by MVTec authors) and
one US patent by an overlapping group of MVTec authors whose claims I could not read. The
README therefore says the method comes from MVTec researchers and that patent and licence
status must be checked before commercial use, without naming a patent number.

### D34. Robustness test design
Four fixed perturbations, applied to test images at full resolution before the model's
resize (`vqi.perturb`, wrapped as a torchvision transform in `vqi.anomaly.PerturbImage`;
masks are left unchanged): brightness x0.7 and x1.3, Gaussian blur with sigma = 0.4% of
the image width, Gaussian noise with std 0.05 on [0, 1] pixels. Noise is seeded from the
global seed and the image's own pixels, so it does not depend on loading order or worker
count. The threshold stays the clean one from validation: a camera that drifts does not
re-tune the threshold, so recall and false-alarm rate under perturbation show what the line
would see. AUROC is reported too, because it separates "the ranking broke" from "only the
threshold no longer fits".

### D35. Faster bootstrap: numpy counts, and only the metrics that are needed
The robustness table bootstraps 3 metrics for 5 conditions per model and category. With
sklearn's `confusion_matrix` inside the bootstrap loop and all 7 metrics per condition, the
CPU evaluation of two small synthetic categories took over two minutes. `operating_point`
now counts TP/FP/FN/TN with numpy, and `summarize(names=...)` computes only the requested
metrics: 40 seconds for two models and two categories.

### D36. Phase 3 notebook simulated before release
As in D26, all code cells of `notebooks/03_efficientad_robustness.ipynb` were run on CPU
against synthetic archives: install, data, PatchCore and EfficientAD on three categories
(with tiny settings and a randomly initialised teacher saved where anomalib looks for the
downloaded weights), robustness predictions, evaluation and the zip. Only the GPU check,
Drive mount and browser download were stubbed. PatchCore is retrained in this notebook
because no weights are kept; its clean numbers may differ slightly from Phase 2 because of
GPU non-determinism, and the README is regenerated from the new files.
