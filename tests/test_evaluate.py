import csv
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "evaluate.py"


def test_evaluate_writes_metrics_table_and_plot(tmp_path: Path) -> None:
    results = tmp_path / "results"
    out = results / "patchcore" / "toy"
    out.mkdir(parents=True)
    with (out / "scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image", "label", "score"])
        writer.writerows(["validation", f"v{i}.png", 0, i / 10] for i in range(10))
        writer.writerows(["test", f"g{i}.png", 0, i / 10] for i in range(10))
        writer.writerows(["test", f"b{i}.png", 1, 0.5 + i / 10] for i in range(10))
    (results / "thresholds.json").write_text(json.dumps({"patchcore/toy": {"threshold": 0.85}}))
    (out / "pixel_metrics.json").write_text(
        json.dumps({"pixel_aupro_0.3": 0.6, "pixel_aupro_0.05": 0.2})
    )
    config = tmp_path / "toy.yaml"
    config.write_text(
        f"data_root: data\nresults_dir: {results}\nseed: 0\n"
        "dataset: {name: mvtec_ad, category: toy}\nevaluation: {bootstrap_samples: 50}\n"
    )

    subprocess.run([sys.executable, str(SCRIPT), str(config)], check=True, capture_output=True)

    metrics = json.loads((results / "metrics.json").read_text())["patchcore/toy"]
    image = metrics["image"]
    # Defects 0.5..1.4 vs good 0.0..0.9 at threshold 0.85: 6 caught, 4 missed, 1 false alarm.
    assert image["counts"] == {"tp": 6, "fp": 1, "fn": 4, "tn": 9}
    assert image["image_auroc"]["value"] == 0.875
    assert metrics["n_validation_images"] == 10
    assert "patchcore/toy" in (results / "metrics.md").read_text()
    assert (out / "score_distribution.png").stat().st_size > 0
