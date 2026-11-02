# Data

The datasets are **demo data**. They show that the pipeline works; they are not the product
focus. No dataset file is committed to this repository (`data/` is in `.gitignore`).

| Dataset | Categories used | Role | Source |
|---|---|---|---|
| MVTec AD 2 | `fabric` | main anomaly-detection set | official MVTec website, manual download |
| MVTec AD | `leather`, `carpet` | secondary set; `leather` defect folders give defect-type labels | official MVTec website, manual download |

Only official MVTec sources are used, no mirrors.

## Licences

All three categories are published under **CC BY-NC-SA 4.0** (attribution, non-commercial,
share-alike). Consequences for this repository:

- The data is used for non-commercial research and portfolio purposes only.
- Model weights trained on this data are derived material. If they are published (as a
  GitHub release, never in git), they carry the same CC BY-NC-SA 4.0 licence.
- The MIT licence of this repository covers the code only, not the data or weights.
- `docs/demo.gif` shows MVTec AD leather test images, so it is under CC BY-NC-SA 4.0 too.

### Official licence text

Quoted word for word from the "License Terms" section of the official MVTec AD 2 page
(<https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2>, read on 2026-11-01).
The MVTec AD page (<https://www.mvtec.com/research-teaching/datasets/mvtec-ad>) has the
same text.

> The data is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
> International License (CC BY-NC-SA 4.0).
>
> In particular, it is not allowed to use the dataset for commercial purposes. If you are
> unsure whether or not your application violates the non-commercial use clause of the
> license, please contact us.

Licence: <https://creativecommons.org/licenses/by-nc-sa/4.0/>

### Citations

- MVTec AD 2: Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca König,
  Carsten Steger. *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly
  Detection.* International Journal of Computer Vision 134(4), 2026.
  DOI: 10.1007/s11263-026-02743-0
- MVTec AD: Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger.
  *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised
  Anomaly Detection.* International Journal of Computer Vision 129, 2021.

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

Both datasets are downloaded by you through the web forms on the official MVTec website
(name, email, non-commercial acknowledgment). The code never downloads them.

1. MVTec AD 2: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2>, archive
   for `fabric`.
2. MVTec AD: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>, archives for
   `leather` and `carpet`.
3. Upload them where the notebook expects them (or edit the two paths in its step 3):

| Archive | Colab (Google Drive) | Kaggle (private dataset) |
|---|---|---|
| MVTec AD 2 `fabric` | `MyDrive/datasets/mvtec_ad_2/fabric.tar.gz` | `/kaggle/input/mvtec-ad-2/` |
| MVTec AD `leather`, `carpet` | `MyDrive/datasets/mvtec_ad/` (`leather.tar.xz`, `carpet.tar.xz`) | `/kaggle/input/mvtec-ad/` |

4. Install each category (the notebook does this):

```bash
python scripts/download_data.py --config configs/fabric.yaml --source /path/to/fabric.tar.gz
python scripts/download_data.py --config configs/leather.yaml --source /path/to/mvtec_ad/
python scripts/download_data.py --config configs/carpet.yaml --source /path/to/mvtec_ad/
```

`--source` can be an archive, an unpacked folder, or a folder that holds
`<category>.tar*`. A folder is symlinked, not copied; an archive is extracted into `data/`.

## Adding another category

Copy a config file (for example `configs/fabric.yaml` to `configs/sheet_metal.yaml`) and
change `dataset.category`. No code change is needed.
