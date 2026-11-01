import csv
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "evaluate.py"
CONDITIONS = ("brightness_0.7", "brightness_1.3", "blur", "noise")


def write_scores(path: Path, shift: float, with_validation: bool) -> None:
    """5 good and 5 defective parts, each under two lights (MVTec AD 2 naming)."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image", "label", "score"])
        if with_validation:
            writer.writerows(["validation", f"v{i}.png", 0, i / 10] for i in range(10))
        for light in ("regular", "shift_1"):
            writer.writerows(
                ["test", f"good/{i}_{light}.png", 0, shift + i / 10] for i in range(0, 10, 2)
            )
            writer.writerows(
                ["test", f"bad/{i}_{light}.png", 1, shift + 0.5 + i / 10] for i in range(0, 10, 2)
            )


def test_evaluate_writes_metrics_robustness_and_plot(tmp_path: Path) -> None:
    results = tmp_path / "results"
    out = results / "patchcore" / "toy"
    out.mkdir(parents=True)
    write_scores(out / "scores.csv", shift=0.0, with_validation=True)
    for condition in CONDITIONS:
        # "noise" pushes every score up by 0.5, the others leave scores unchanged.
        write_scores(out / f"scores_{condition}.csv", 0.5 if condition == "noise" else 0.0, False)
    (results / "thresholds.json").write_text(json.dumps({"patchcore/toy": {"threshold": 0.85}}))
    (out / "pixel_metrics.json").write_text(
        json.dumps({"pixel_aupro_0.3": 0.6, "pixel_aupro_0.05": 0.2})
    )
    config = tmp_path / "toy.yaml"
    config.write_text(
        f"data_root: data\nresults_dir: {results}\nseed: 0\n"
        "dataset: {name: mvtec_ad, category: toy}\nevaluation: {bootstrap_samples: 50}\n"
    )
    model = Path(__file__).parent.parent / "configs" / "models" / "patchcore.yaml"

    subprocess.run(
        [sys.executable, str(SCRIPT), "--model", str(model), str(config)],
        check=True,
        capture_output=True,
    )

    metrics = json.loads((results / "metrics.json").read_text())["patchcore/toy"]
    image = metrics["image"]
    # Defects 0.5..1.3 vs good 0.0..0.8 (each twice) at threshold 0.85:
    # defects 0.9, 1.1, 1.3 caught, 0.5 and 0.7 missed, no false alarm.
    assert image["counts"] == {"tp": 6, "fp": 0, "fn": 4, "tn": 10}
    assert image["n_test_parts"] == 10
    assert metrics["n_validation_images"] == 10
    assert "patchcore/toy" in (results / "metrics.md").read_text()
    assert (out / "score_distribution.png").stat().st_size > 0

    robustness = json.loads((results / "robustness.json").read_text())["patchcore/toy"]
    assert set(robustness) == {"clean", *CONDITIONS}
    assert robustness["clean"]["recall"]["value"] == 0.6
    # Same ranking, so AUROC is unchanged, but with the clean threshold kept the
    # shifted good scores 0.9, 1.1 and 1.3 (3 of 5 parts) become false alarms.
    assert robustness["noise"]["image_auroc"]["value"] == image["image_auroc"]["value"]
    assert robustness["noise"]["false_alarm_rate"]["value"] == 0.6
    rows = [r for r in (results / "robustness.md").read_text().splitlines() if "toy" in r]
    assert rows[0].startswith("| patchcore/toy | clean |")  # clean first, then perturbations
