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
otherwise), 20 epochs, later cut to 10 (D38). The paper trains for 70,000 steps; 20 epochs is about 4,000 to
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

### D37. Notebooks keep results on Drive and resume after a disconnect
The first Phase 3 Colab run stopped during EfficientAD `fabric`, after all PatchCore runs
had finished; a Colab disconnect deletes the runtime's disk, so those runs were lost. On
Colab the notebooks now link `results/` to `MyDrive/vqi_results/phase<N>/`, so every
finished model and category is kept. On a rerun, a model and category is skipped when its
`run_info.json` (written last by `train_anomaly.py`) is at least as new as its
`scores_noise.csv`; a run cut off half-way is redone. Checked by running the simulated
notebook twice: the second run skipped all six model and category pairs.

### D38. Live output in notebooks, and EfficientAD cut to 10 epochs
On the second Phase 3 Colab run the notebook showed only the `$ command` lines: output of
a child process started with `subprocess.run` does not reach the notebook, so nobody could
tell whether EfficientAD `fabric` was training or stuck after 80 minutes. `run()` now reads
the child's output as it arrives and prints it, so epochs and progress bars are visible
(tested with a child that prints over time and one that fails). EfficientAD was cut from 20
to 10 epochs: on a free Colab GPU with two CPU cores, decoding the large MVTec AD 2 images
for every step is slow, and the PatchCore results already sat on Drive (D37), so only
EfficientAD had to be rerun. This is a compute budget, about 2,000 to 4,000 steps against
the paper's 70,000, and the results are reported as such.

### D39. Perturbations use `functools.partial`, not lambdas (bug found on Windows)
On the user's Windows machine the robustness step crashed with
`Can't get local object 'perturbations.<locals>.<lambda>'`: Windows starts data-loader
workers by spawning new processes and pickles what they need, and lambdas cannot be
pickled. Linux forks workers, so neither CI nor the CPU simulation saw it. `perturbations`
now returns `functools.partial` objects of module-level functions; a new test pickles every
perturbation and checks the restored one gives the same image (it fails on the old code).
The local run used `num_workers: 0` as a workaround (not committed); that only changes
loading speed, not results.

### D40. Phase 3 results: trained on a local RTX 3070 Ti
Colab's free GPU quota ran out, so Phase 3 ran on the user's Windows 11 PC (RTX 3070 Ti,
torch 2.14.1+cu130, commit `2fde1ef`, all six runs). The zip was made with
`Compress-Archive`, which stores Windows path separators; it was unpacked with them
converted, and six gallery images left over from the Phase 2 run in the same folder were
dropped (only images listed in each run's `overlays.json` are kept). All metrics were
recomputed on CPU here and matched the Windows output exactly.

PatchCore was retrained in this run, with the same code and seed as Phase 2, and its
numbers moved: `fabric` AUROC 0.657 (Colab T4) became 0.751, `leather` test false-alarm
rate 9.4% became 18.8%, `carpet` 7.1% became 10.7%. Each new value lies inside the old
95% interval and the other way round, so this is run-to-run variation (GPU and library
non-determinism), not a code change. The Phase 2 metrics are kept in
`results/history/phase2_colab_t4/` so the README can cite both.

### D41. EfficientAD lost to PatchCore at this budget (negative result)
Image AUROC, EfficientAD vs PatchCore: `fabric` 0.551 vs 0.751, `leather` 0.921 vs 1.000,
`carpet` 0.990 vs 0.987. At the validation threshold EfficientAD's test false-alarm rate was
56.1% (`fabric`), 18.8% (`leather`) and 25.0% (`carpet`). Two likely reasons, not tested
separately: 10 epochs is a small fraction of the paper's 70,000 steps (D38), and
EfficientAD normalises its maps with quantiles from the same good validation images that
set the threshold (D32), so validation scores are optimistic. PatchCore stays the model
for export and the API.

### D42. Robustness: ranking survives, threshold does not
With the clean threshold kept: PatchCore `leather` AUROC stays 1.000 under every
perturbation, but its false-alarm rate goes from 18.8% (clean) to 65.6% (brighter), 43.8%
(noise) and 100% (blur). `carpet` behaves the same way (blur: 92.9% false alarms).
EfficientAD loses ranking quality as well (`leather` brighter: AUROC 0.463). For a line this
means: monitor the score distribution of good parts, and recalibrate the threshold after any
change to optics or lighting. On `fabric` PatchCore blur drops AUROC to 0.549; the
likely reason (not tested) is that blur removes the fine detail its few caught
defects depend on.

## Phase 5: export and CPU latency

### D43. Each exported format gets its own threshold, from its own validation scores
anomalib's exported graph includes the pre-processing, but with antialiasing switched off
in the resize (`get_exportable_transform`; ONNX cannot export antialiased resizing). The
training pipeline resizes with antialiasing, so an exported model gives slightly different
raw scores for the same image, and a threshold from training would not fit it.
INT8 quantisation shifts scores further. `scripts/export_model.py` therefore scores the
good validation images with every format and applies the same threshold policy (D12),
storing `<model>/<category>/<format>` in `results/thresholds.json`. The API (Phase 6)
reads the threshold of the format it serves. anomalib's own 0.5 threshold is never used:
normalisation and thresholding are off in the model (D14).

### D44. How latency is measured
CPU only, batch size 1, one image at a time, as on a line. The image is decoded before the
timer starts (decoding depends on the file format and disk, not on the model); the timed
call includes the model's own resize and normalisation. 10 untimed warm-up calls first,
then every test image once; median and 95th percentile are reported, with the CPU model,
core count, thread count and library versions in `export.json`. Formats: PyTorch FP32
(baseline), ONNX Runtime FP32, OpenVINO FP32, FP16 and INT8 (post-training quantisation
calibrated on the good-only validation loader). Category: `leather`, where PatchCore works
well and which Phase 4's defect-type labels come from.

### D45. Export details found in a CPU smoke run before any real run
- The ONNX exporter in torch 2.14 needs `onnxscript`, which was missing from the `export`
  extra; added.
- The exporter stores weights in `model.onnx.data` next to `model.onnx`, so model size is
  measured over the whole export folder.
- During ONNX export, torch prints a `version_converter ... Assertion ... failed` traceback
  while trying to convert the graph to another opset; the export continues with the
  original opset and the exported model's scores match PyTorch's. It is noise, not a failure.
- OpenVINO writes a compiled-model cache to `openvino_cache/`; it is git-ignored.
- On Windows the export crashed with `UnicodeEncodeError: 'charmap' codec can't encode
  character '✅'`: torch's ONNX exporter prints a check-mark emoji, and the console
  code page (cp1254) cannot encode it. `scripts/export_model.py` now switches stdout and
  stderr to UTF-8 at start-up (`errors="replace"`), so no `PYTHONUTF8=1` is needed.
  Found by the user on the first local run; Linux consoles are UTF-8, so the CPU smoke run
  could not show it.

### D46. Own INT8 quantization, and one failing format no longer stops the benchmark
On Windows anomalib's `INT8_PTQ` crashed in `_post_training_quantization_ov` with
`to_shape was called on a dynamic shape`. Reproduced here: anomalib 2.7.0 reads
`model.input(0).shape` whenever the batch axis is static, but `.shape` needs every axis
static, and the exported model keeps height and width dynamic. The Windows export had a
static batch axis; the Linux smoke export had a dynamic one, so the smoke run never reached
that line. Fix: INT8 is no longer exported by anomalib. `vqi.export.quantize_openvino_int8`
reads the OpenVINO FP32 model, sets its input to batch 1 with dynamic height and width (the
way the API calls it), and runs `nncf.quantize` on the good validation images, loaded one at
a time. Whatever shape the export produces, the quantizer gets the same input.
`scripts/export_model.py` now wraps every export and every benchmark in its own guard: a
failure is printed with its traceback, stored as `"error"` in `export.json`, shown as a
"failed" row in `export.md`, and the other formats continue; the script exits with code 1
at the end if anything failed. Tested by forcing the ONNX export to fail: the other four
formats were benchmarked and the exit code was 1.

### D47. Export with one fixed input shape [1, 3, H, W], from the CPU
The second Windows run (commit `4109a0d`) benchmarked only PyTorch; all four exports had
input shape [2, 3, ?, ?]. anomalib traces dynamic axes with an example batch of 2 (torch
specialises sizes 0 and 1), and on that machine, where the model was traced on the GPU,
the batch axis came out fixed at that example size instead of dynamic. ONNX Runtime and
OpenVINO then rejected a single image. The INT8 step failed for a related reason: it
reshaped that batch-2 graph to batch 1, and OpenVINO's resize node could not follow.
The Linux smoke runs (CPU tracing) had kept the axis dynamic, so they could not show this.
Fix, chosen to remove the dependence on dynamic axes altogether:
- the model is moved to the CPU before export, the deployment target and the tested path;
- every format is exported with a fixed input [1, 3, H, W] (`input_size` = the category's
  image size, `dynamic_shapes=None`), as a line camera always delivers one image of one
  size; images of another size are resized first (`vqi.imageio.resize_to`);
- `check_input_shape` reads each exported model and raises at once if the input is not
  exactly [1, 3, H, W], so a wrong export fails before any benchmark;
- the INT8 quantizer no longer reshapes; it quantizes the static FP32 model as exported.
Checked on CPU with the real architecture (wide_resnet50_2, random weights, 1024 x 1024
synthetic leather images): all five formats exported with [1, 3, 1024, 1024] and were
benchmarked. The failed run's zip was not committed.

### D48. Phase 5 results, and file names in the export score files
Run on the user's Windows PC at commit `47a17eb` (training on the RTX 3070 Ti, export and
all timing on its Intel 12th-gen CPU, 12 logical cores): all five formats exported with a
fixed [1, 3, 1024, 1024] input and were benchmarked. Image AUROC is 1.000 and the test
false-alarm rate 12.5% for every format, so exporting and quantizing did not change which
parts are accepted. Median CPU latency: PyTorch 315.7 ms, ONNX Runtime 229.8 ms, OpenVINO
FP32 195.5 ms, FP16 192.1 ms, INT8 167.4 ms. INT8 helped less than expected (171 MB vs
213 MB, about 15% faster than OpenVINO FP32); NNCF quantizes the CNN, while PatchCore's
memory bank and nearest-neighbour search stay in floating point. That explanation is
likely but not measured; profiling the search separately would confirm it.
(Correction, Phase 6: the explanation is wrong. NNCF also quantized the memory-bank search
and clipped the anomaly map; see D50.)
The export run retrained PatchCore, so its leather threshold and false-alarm rate (12.5%)
differ from Phase 3's run (18.8%), the same run-to-run variation as in D40.

Bug found while checking the zip: `export_model.py` wrote only the file name (`000.png`)
into `scores_<format>.csv`, so images from different defect folders shared a name. The
benchmark itself used full paths and is correct; recomputing every threshold, metric and
interval from the CSVs (each `leather` image is its own physical part) matched
`export.json` exactly. The script now writes the path from the category folder
(`leather/test/cut/000.png`), as `train_anomaly.py` does; the helper moved to
`vqi.dataset_files.relative_to_category`, with a test. The committed CSVs from this run
keep bare file names.

## Phase 6: inspection API

### D49. The API runs the OpenVINO model with plain `openvino`, and flags a review band
The API compiles `model.xml` with `openvino.Core` and reads the `pred_score` and
`anomaly_map` outputs itself, instead of using anomalib's `OpenVINOInferencer`. The
inferencer imports torch and anomalib at module level (`openvino_inferencer.py`, line 57 in
the installed version), which would pull several gigabytes into a CPU image that only needs
the OpenVINO runtime. The `serve` extra therefore holds FastAPI, uvicorn, python-multipart
and openvino, and no torch. The risk of reading the outputs by hand is a silent mismatch, so
`tests/test_golden.py` compares the plain runtime with `OpenVINOInferencer` on a fixed set
of 11 leather images (score and full anomaly map, absolute tolerance 1e-4) whenever the
`train` extra and the data are present. On the Phase 5 INT8 model the largest difference
over every validation and test-good image was 4.9e-7, against the export CSV as well. The
same golden file pins each image's score (tolerance 0.001), decision and review flag, so a
changed model, threshold or band shows up as a failing test, not as a quiet shift on the
line.

The decision itself stays the per-format threshold from D43 (NOK when score > threshold).
Around it the API sets `needs_review` for scores in a band, derived from the good validation
scores and not chosen by hand: the lower edge is the threshold that would give a false-alarm
rate of `review_rate` on good validation parts (`review_rate: 0.1`, so roughly the top 10%
of good validation scores), and the band is mirrored the same distance above the threshold.
The reason is D42 and D43: the ranking transfers between runs and formats, the exact
threshold does not, so the parts closest to it on either side are the ones an operator
should look at. For OpenVINO FP32 the threshold is 30.399 and the band (29.015, 31.784];
it holds 4 of 49 good validation parts, 14 of 32 good test parts and none of the 92 defect
parts (the lowest defect scores 37.64). That 14 of 32 is itself a finding: the test good
parts score higher than the validation good parts, the same weak threshold transfer as in
D42, and the review queue is where it becomes visible in operation.

Smaller choices: the heatmap is coloured from the median good validation score to the
threshold, so anything red is above the decision line; the minimum was rejected because one
validation score is an outlier. The model version written with every decision is
`<format>@<git commit>+thr=<threshold>` (the commit from `VQI_GIT_COMMIT`, else
`git rev-parse`, else `unknown`), enough to tell which model and threshold made a decision.
Decisions go to SQLite (one file, no server, a CHECK constraint on OK/NOK), days in the
daily report are UTC dates, and the report is a CSV, which opens in Excel without an extra
dependency. Every setting lives in `configs/serve/leather.yaml` and can be overridden by an
environment variable `VQI_<KEY>`, so the container needs no edited files.

### D50. The INT8 export clips its anomaly map, so the API serves FP32; D48 corrected
The first API run served INT8, and its heatmaps were blank even for a part scoring 69.
Reading the quantized graph showed why: NNCF did not stop at the CNN. It put a FakeQuantize
on the patch-distance map, with a range calibrated on good images only ([0, 27.554]), so the
anomaly map is clipped at about 27.5, below the threshold. It also quantized the
memory-bank matmul and its inputs, which moves some scores far (test good 007: 16.75 in
INT8, 28.8 in FP32). The image score keeps AUROC 1.000 only because `pred_score` is taken
before the clipping node. The FP32 and FP16 maps are correct (glue 002 peaks at 68.55).

So D48's explanation, that the memory bank and nearest-neighbour search stay in floating
point, is wrong: they were quantized, and the modest INT8 speed-up has to be explained
again once the export is fixed. The API now serves OpenVINO FP32 (threshold and band in
D49), and `test_anomaly_map_marks_the_defects` requires the map of every defect in the
golden set to exceed the threshold; it failed on that INT8 model. The fix, keeping the
PatchCore head out of quantization, is D52.

### D51. Empty uploads, and a stale `.pyc` in mutation checks
An empty upload reached `cv2.imdecode`, which raised `cv2.error (-215:Assertion failed)
!buf.empty()` and became a 500. `decode_bytes` now raises `ValueError("empty image")`, which
the API answers with 400 like any unreadable image.

While breaking code on purpose to check that each test fails, one replacement had the same
length as the original and was restored within the same second. Python reused the cached
`.pyc` (it compares source size and modification time in seconds), so the test ran the
wrong code. Mutation checks now run with `PYTHONDONTWRITEBYTECODE=1` after deleting
`__pycache__`. The code under test was correct; only the check was misleading.

### D52. INT8 with the PatchCore head in floating point: correct maps, twice the false alarms
`quantize_openvino_int8` now passes an NNCF `IgnoredScope` with every op from the first
MatMul against the memory bank to the outputs (`vqi.export.patchcore_head`, 51 ops in the
leather model), so only the CNN in front of the search is quantized. The re-export has no
FakeQuantize in the head. Its anomaly maps are right again: on six golden defects their
correlation with the FP32 maps is at least 0.994 and every peak is far above the INT8
threshold. The largest score difference to FP32 over all validation and test images fell
from 24.36 to 8.35.

It is still not good enough to serve. The quantized features lower the scores on average,
but not uniformly, the threshold drops to 29.628 (FP32 30.399), and on the test good parts
INT8 raises false alarms on 8 of 32 (25.0%) where FP32 raises 4 (12.5%); AUROC stays
1.000. For about 13% less latency that is a bad trade on a line, so the API stays on FP32.
Recalibrating or quantization-aware training could close the gap; neither was tried (one
attempt was agreed).

The fix meant rerunning `export_model.py`. Training is deterministic on this machine: every
non-INT8 score is identical to Phase 5, so thresholds, the golden set and the served model
did not change. Only the score CSVs changed further, to the category paths of D48. The
first rerun ran while Docker Desktop's virtual machine was up and timed every format 17 to
57% slower than Phase 5, unevenly; it was discarded and the export run again with Docker
closed. Clean medians: PyTorch 286.7 ms, ONNX Runtime 218.0, OpenVINO FP32 177.3, FP16
184.9, INT8 154.4. These are lower than Phase 5's on the same PC, the usual run-to-run
spread of a desktop, which is why formats are only compared within one run. INT8 is now
142 MB instead of 171, as the memory bank stays in FP32 and the earlier weight compression
of the search is gone.

The Docker image is 1.65 GB, over the few hundred megabytes hoped for. It holds no torch;
the bulk is the OpenVINO runtime, OpenCV and the SciPy, pandas and pyarrow stack that
scikit-learn, matplotlib and Streamlit pull in (in the Windows venv the largest of these are
openvino 248 MB, scipy 115, OpenCV 113, pyarrow 90 and pandas 70). Shrinking it would
mean a separate API extra without Streamlit and matplotlib; not done.

### D53. The CPU name in `export.json`, and text claims in the README
On Windows `platform.processor()` gives only the family string (`Intel64 Family 6 Model
151 ...`), so `export.json` did not name the CPU the latency was measured on. `cpu_name()`
now reads `ProcessorNameString` from the registry on Windows. The current `export.json`
was not re-run for this; its `environment.cpu` was filled in with the same function on the
same machine (`12th Gen Intel(R) Core(TM) i5-12400F`), the only field changed. So the README
can state the model with a source, `check_readme_claims.py` now also accepts text claims:
backticked text followed by a claim marker must appear in the JSON string it names.

## Phase 4: defect-type classifier (built after Phase 6)

### D54. Crops come from the anomaly-map peak, and both models are judged by 5-fold CV
A rejected part is classified from a 256 px crop of the original image centred on the
maximum of the served anomaly map (OpenVINO FP32), resized to 224 px. Ground-truth masks
are never used, in training or in the API, because a line has none. On leather the map is
256 px for a 1024 px image; its peak lies inside the defect mask in 80 of 92 test defects
(all fold and glue, 13 of 19 color), and the 256 px crop around it touches the mask in all
92, which is why the crop is that large. Mask bounding boxes have median sides of 63 px
(poke) to 252 px (fold).

Leather has 92 labelled defects in five classes, about 18 per class, so there is no room
for a separate test set. Both models are compared by stratified 5-fold cross-validation
(seed 42): every image is predicted once by a model that never saw it, and the pooled
predictions give accuracy and macro F1 with per-image bootstrap intervals, recall per class
with its count, and a confusion matrix. ResNet18 (ImageNet weights, all layers fine-tuned)
uses a recipe fixed before any result: Adam at 1e-4, 20 epochs, batch 16, random flips and
quarter turns. Colour jitter is left out because one class is "color". HOG + RBF SVM is the
classical baseline; its C is picked by an inner 3-fold search inside each training fold.
HOG sees a grey crop, so it cannot use colour at all. The final ResNet18 is then trained on
all 92 images, with the ImageNet normalisation inside the graph, and exported to OpenVINO
with `classes.json` next to it.

### D55. HOG comes from scikit-image, because OpenCV 5 has no `HOGDescriptor`
The pinned `opencv-python-headless` 5.0.0.93 has no `cv2.HOGDescriptor`: OpenCV 5 moved it
out of the main modules, and neither the module nor its `__init__.pyi` stub has it.
`skimage.feature.hog` (scikit-image 0.26.0, already installed with anomalib) computes the
same kind of descriptor; it is pinned in the `train` extra and imported only inside
`hog_features`, so the API image does not need it. Its test is skipped where scikit-image
is missing, as in CI.

### D56. ResNet18 is served for NOK parts; what it does not do
Pooled out-of-fold results on leather: ResNet18 accuracy 0.967 (95% CI 0.924 to 1.0), macro
F1 0.967; HOG + SVM accuracy 0.641 (0.543 to 0.728), macro F1 0.637, with C = 10 chosen in
every fold. ResNet18's three errors are two cuts (taken for fold and poke) and one poke
(taken for fold). With about 18 parts per class one error moves a class recall by about
5 points, so the gap between the models is clear but the ResNet18 figure is not a precise
estimate. GPU training is not bit-for-bit deterministic, so a rerun can move an image or
two.

`classifier.json` records commit 957afa5 because training ran before the code was
committed. The code that ran is in 71bb53e, the next commit, unchanged; it was not re-run,
and the results stand. The training and export scripts now print a one-line warning when
the working tree has uncommitted changes (`vqi.provenance`).

The API classifies only NOK parts. The crop is taken from the uploaded image at the map
peak, capped at the image size for small uploads. The type goes into the decision log's
existing `defect_type` column; its probability is returned in the answer but not logged,
so the log schema did not change. The daily report gets a second CSV with NOK parts per
day and type. Limits: the classes are leather's five, so another product needs its own
labelled defects and its own model, and there is no "unknown" class or rejection
threshold, because there is no data to calibrate one. A new kind of defect will be named as
one of the five; the probability in the answer is the only hint, and it is not calibrated.

## Phase 7: operator panel and final README

### D57. The panel is a thin client of the API
`vqi.panel` (Streamlit) talks to the inspection API over HTTP and holds no model, no
database connection and no state beyond the answers of the current session. The panel and
the line therefore see the same model, the same threshold and the same decision log, and
the panel can run on another machine or container (`panel` in `compose.yaml`). The report
tab needed the daily counts, which only `scripts/daily_report.py` produced from the SQLite
file, so the API got `GET /daily`, returning `DecisionLog.daily_summary()` and
`daily_defect_types()`. Streamlit reruns the whole script on every interaction; each
upload is posted once and its answer kept in `st.session_state` by file id, or every click
would log the part again. Streamlit 1.66.1 deprecates `use_container_width`, so images and
tables use `width="stretch"`. Tests run the panel headless with `streamlit.testing`
`AppTest` against an `httpx.MockTransport` fake of the API.

### D58. The demo GIF is made by a script, not a screen recording
`scripts/make_demo_gif.py` starts the API on a temporary decision log, replays the leather
test images through `POST /inspect` as the line simulator would, starts the panel and
drives it with headless Chromium (Playwright 1.64.0, in the `demo` extra). Each step is a
screenshot; Pillow joins them into `docs/demo.gif`. The three parts inspected on camera
(`good/005`, `cut/003`, `color/000`) were picked by eye for marks visible at GIF size.
`good/000` was not used as the good part because it is a false alarm; the review queue
shows such parts instead. The first version took the screenshot as soon as the page was
idle and recorded the previous part's result: Streamlit shows the new upload before the
new answer arrives. The script now waits until the file name appears twice, in the upload
chip and in the image caption, which happens only once the new answer is drawn.
