"""Defect-type classifier for rejected parts: ResNet18 against HOG + SVM, then export.

Labels come from the leather defect folders (color, cut, fold, glue, poke): 92 images, so
both models are compared by 5-fold stratified cross-validation and every image is
predicted once by a model that never saw it. Crops are taken at the peak of the served
anomaly model's map, as in the API (D54). The ResNet18 settings are fixed in advance; the
SVM's C is chosen inside each training fold. The final ResNet18 is then trained on all
images and exported to OpenVINO for the API.

Writes ``results/classifier/<category>/``: classifier.json, predictions.csv,
confusion_<model>.png; and the model to ``exports/classifier/<category>/``.

Example:
    python scripts/train_classifier.py --config configs/leather.yaml
"""

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import torch
import torchvision
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from vqi.config import Config, load_config, load_serve_config
from vqi.defect_type import (
    CROP_SIDE,
    INPUT_SIDE,
    classification_summary,
    crop_at_peak,
    hog_features,
)
from vqi.imageio import load_rgb, resize_to
from vqi.inspection import OpenVINOModel

FOLDS = 5
# ResNet18 recipe, fixed before any result was seen (there is no data left to tune it on).
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
SVM_C_GRID = [0.1, 1.0, 10.0, 100.0]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CropClassifier(nn.Module):
    """ResNet18 with ImageNet normalisation inside, so the export takes crops in [0, 1]."""

    def __init__(self, n_classes: int) -> None:
        """ImageNet-pretrained ResNet18 with a new last layer."""
        super().__init__()
        self.net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits for a [N, 3, H, W] batch of RGB crops in [0, 1]."""
        return self.net((x - self.mean) / self.std)


def defect_images(category_dir: Path) -> tuple[list[Path], np.ndarray, list[str]]:
    """Every defective test image, its class index and the sorted class names."""
    classes = sorted(p.name for p in (category_dir / "test").iterdir() if p.name != "good")
    paths, labels = [], []
    for index, name in enumerate(classes):
        found = sorted((category_dir / "test" / name).glob("*.png"))
        paths += found
        labels += [index] * len(found)
    return paths, np.array(labels), classes


def peak_crops(paths: list[Path], anomaly_model: OpenVINOModel) -> np.ndarray:
    """Crop the full-size image at the anomaly peak, resized to the classifier input.

    The model sees the image resized to its input; the crop comes from the original pixels,
    as in the API.
    """
    crops = []
    for path in paths:
        image = load_rgb(path)
        _, anomaly_map = anomaly_model(resize_to(image, anomaly_model.input_size))
        crop = crop_at_peak(image, anomaly_map, CROP_SIDE)
        crops.append(cv2.resize(crop, (INPUT_SIDE, INPUT_SIDE), interpolation=cv2.INTER_AREA))
    return np.stack(crops)


def augment(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random flips and quarter turns: leather texture has no up, down, left or right."""
    out = []
    for x in batch:
        if torch.rand(1, generator=generator) < 0.5:
            x = x.flip(-1)
        k = int(torch.randint(0, 4, (1,), generator=generator))
        out.append(torch.rot90(x, k, dims=(-2, -1)))
    return torch.stack(out)


def train_resnet(crops: np.ndarray, labels: np.ndarray, n_classes: int, seed: int) -> nn.Module:
    """Fine-tune all layers with the fixed recipe; returns the model in eval mode."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CropClassifier(n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x = torch.from_numpy(crops.transpose(0, 3, 1, 2)).float()
    y = torch.from_numpy(labels).long()
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            batch = augment(x[idx], generator).to(device)
            loss = nn.functional.cross_entropy(model(batch), y[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.no_grad()
def predict_resnet(model: nn.Module, crops: np.ndarray) -> np.ndarray:
    """Class index for each crop."""
    device = next(model.parameters()).device
    x = torch.from_numpy(crops.transpose(0, 3, 1, 2)).float().to(device)
    return model(x).argmax(1).cpu().numpy()


def fit_hog_svm(features: np.ndarray, labels: np.ndarray, seed: int) -> GridSearchCV:
    """Scaled HOG features into an RBF SVM; C picked by 3-fold CV on the training fold."""
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    search = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        {"svc__C": SVM_C_GRID},
        cv=inner,
        scoring="f1_macro",
    )
    return search.fit(features, labels)


def save_confusion(summary: dict, title: str, out: Path) -> None:
    """Confusion matrix with counts, rows true and columns predicted."""
    classes = summary["confusion"]["classes"]
    matrix = np.array(summary["confusion"]["rows_true_cols_predicted"])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), count in np.ndenumerate(matrix):
        color = "white" if count > matrix.max() / 2 else "black"
        ax.text(j, i, str(count), ha="center", va="center", color=color)
    ax.set_xticks(range(len(classes)), classes, rotation=45)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run(config: Config) -> dict:
    """Cross-validate both models, write the results, export the final ResNet18."""
    category = config.dataset.category
    serve = load_serve_config(Path("configs/serve") / f"{category}.yaml")
    anomaly_model = OpenVINOModel(serve.model_file)
    paths, labels, classes = defect_images(config.dataset.category_dir)
    print(f"{category}: {len(paths)} defect images, classes {classes}", flush=True)
    crops = peak_crops(paths, anomaly_model)
    features = np.stack([hog_features(c) for c in crops])

    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=config.seed)
    predicted = {"resnet18": np.zeros_like(labels), "hog_svm": np.zeros_like(labels)}
    chosen_c = []
    for fold, (train, test) in enumerate(folds.split(crops, labels)):
        resnet = train_resnet(crops[train], labels[train], len(classes), config.seed + fold)
        predicted["resnet18"][test] = predict_resnet(resnet, crops[test])
        svm = fit_hog_svm(features[train], labels[train], config.seed)
        predicted["hog_svm"][test] = svm.predict(features[test])
        chosen_c.append(svm.best_params_["svc__C"])
        print(f"fold {fold}: done", flush=True)

    out_dir = config.results_dir / "classifier" / category
    out_dir.mkdir(parents=True, exist_ok=True)
    n_bootstrap = config.evaluation.bootstrap_samples
    models = {}
    for name, title in [("resnet18", "ResNet18"), ("hog_svm", "HOG + SVM")]:
        summary = classification_summary(labels, predicted[name], classes, n_bootstrap, config.seed)
        save_confusion(summary, title, out_dir / f"confusion_{name}.png")
        models[name] = summary
        print(f"{name}: macro F1 {summary['macro_f1']['value']:.3f}", flush=True)
    models["hog_svm"]["chosen_c_per_fold"] = chosen_c

    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "true", "resnet18", "hog_svm"])
        for path, true, res, hog in zip(
            paths, labels, predicted["resnet18"], predicted["hog_svm"], strict=True
        ):
            rel = path.relative_to(path.parents[2]).as_posix()
            writer.writerow([f"{category}/{rel}", classes[true], classes[res], classes[hog]])

    final = train_resnet(crops, labels, len(classes), config.seed).cpu()
    export_dir = Path("exports") / "classifier" / category
    export_dir.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, INPUT_SIDE, INPUT_SIDE)
    ov.save_model(ov.convert_model(final, example_input=example), export_dir / "model.xml")
    (export_dir / "classes.json").write_text(json.dumps(classes) + "\n", encoding="utf-8")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    result = {
        "n_images": len(paths),
        "classes": classes,
        "crop": {
            "side_px": CROP_SIDE,
            "input_side_px": INPUT_SIDE,
            "peak_from": serve.model_file.as_posix(),
        },
        "settings": {
            "folds": FOLDS,
            "resnet18": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
            },
            "hog_svm": {"c_grid": SVM_C_GRID, "inner_folds": 3},
            "seed": config.seed,
            "bootstrap_samples": n_bootstrap,
        },
        "models": models,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "git_commit": commit.stdout.strip() or "unknown",
        },
    }
    (out_dir / "classifier.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    return result


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="category config")
    run(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
