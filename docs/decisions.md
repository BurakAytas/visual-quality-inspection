# Decision log

Each entry: what was decided, why, and what was checked. Negative results and bugs found
are logged here too.

## Phase 1: skeleton

### D1. anomalib 2.7.0 is pinned; the rest of the stack is pinned to match it
anomalib 2.7.0 (released 2026-10-06) needs Python >= 3.10, `torch >= 2.10`,
`lightning >= 2.6` (excluding 2.6.2 and 2.6.3) and `opencv-python-headless`. Pins in
`pyproject.toml`: torch 2.14.1 + torchvision 0.29.2 (torch 2.15.0 was four days old and
torchvision 0.29.2 targets the 2.14 line), scikit-learn 1.9.1, onnx 1.23.2,
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
