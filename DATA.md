# Data

The datasets are **demo data**. They show that the pipeline works; they are not the product
focus. No dataset file is committed to this repository (`data/` is in `.gitignore`).

| Dataset | Categories used | Role | Source |
|---|---|---|---|
| MVTec AD 2 | `fabric` | main anomaly-detection set | official MVTec website, manual download |
| MVTec AD | `leather`, `carpet` | secondary set; `leather` defect folders give defect-type labels | official MVTec host |

Only official MVTec sources are used, no mirrors.

## Licences

All three categories are published under **CC BY-NC-SA 4.0** (attribution, non-commercial,
share-alike). Consequences for this repository:

- The data is used for non-commercial research and portfolio purposes only.
- Model weights trained on this data are derived material. If they are published (as a
  GitHub release, never in git), they carry the same CC BY-NC-SA 4.0 licence.
- The MIT licence of this repository covers the code only, not the data or weights.

### MVTec AD 2 licence text

**Pending.** The official licence text must be read on the MVTec AD 2 page of the MVTec
website and quoted here word for word before any MVTec AD 2 image is used (Phase 2).
It has not been quoted yet because the build environment could not reach the MVTec website.

### Citations

- MVTec AD 2: Heckler-Kram, Neudeck, Scheler, König, Steger. *The MVTec AD 2 Dataset:
  Advanced Scenarios for Unsupervised Anomaly Detection.* arXiv:2503.21622, 2025.
- MVTec AD: Bergmann, Batzner, Fauser, Sattlegger, Steger. *The MVTec Anomaly Detection
  Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*
  IJCV 129, 2021.

## Folder layout

`scripts/download_data.py` puts each category under `data/` and checks the layout:

```
data/
├── MVTec_AD_2/fabric/     train/good, validation/good, test_public/{good,bad,ground_truth/bad}
└── MVTec_AD/
    ├── leather/           train/good, test/{good,color,cut,fold,glue,poke}, ground_truth/
    └── carpet/            train/good, test/{good,...}, ground_truth/
```

MVTec AD 2 also ships `test_private` and `test_private_mixed`. They have no labels, so this
project never uses them for metrics.

## How to get the data

### MVTec AD 2 (manual, web form)

1. Open the MVTec AD 2 page on the official MVTec website, fill in the form (name, email,
   non-commercial acknowledgment) and download the `fabric` archive.
2. Make it available to the notebook:
   - **Colab:** upload the archive to Google Drive, e.g. `MyDrive/datasets/mvtec_ad_2/`.
     The notebook mounts Drive at `/content/drive`.
   - **Kaggle:** create a private Kaggle Dataset from the archive. It appears read-only under
     `/kaggle/input/<your-dataset-name>/` (Kaggle may unpack it; both forms work).
3. Install it (the script accepts the archive or an already-unpacked folder and finds the
   `fabric` folder inside):

```bash
python scripts/download_data.py --config configs/fabric.yaml --source /path/to/archive-or-folder
```

A folder source is symlinked, not copied. An archive is extracted into `data/`.

### MVTec AD (`leather`, `carpet`)

```bash
pip install -e ".[train]"
python scripts/download_data.py --config configs/leather.yaml
python scripts/download_data.py --config configs/carpet.yaml
```

This downloads the official MVTec AD archive (all 15 categories, about 5 GB) with the URL
and SHA-256 checksum that anomalib 2.7.0 ships. A manually downloaded archive also works
with `--source`.

## Adding another category

Copy a config file (for example `configs/fabric.yaml` to `configs/sheet_metal.yaml`) and
change `dataset.category`. No code change is needed.
