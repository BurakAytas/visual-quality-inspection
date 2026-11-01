# Phase 2: PatchCore baseline

## What was done

- `vqi.threshold`: the decision threshold comes from good validation images only. With n
  good images and a 1% target, at most floor(0.01 * n) of them may score above it. The
  function takes scores and the target rate, nothing else; a test checks it cannot take labels.
- `vqi.metrics`: precision, recall, F1, false-alarm rate and miss rate at that threshold,
  image AUROC, and miss rate at a fixed 1% false-alarm rate. Every metric gets a 95%
  bootstrap interval, resampled over physical parts.
- `vqi.overlay`: heatmap overlays with the true defect outlined in green.
- `scripts/train_anomaly.py` (GPU): trains PatchCore, saves raw scores of every validation
  and test image, the threshold, pixel AU-PRO, run info and the failure gallery.
- `scripts/evaluate.py` (CPU): turns the saved scores into `results/metrics.json`,
  `results/metrics.md` and score plots.
- `notebooks/02_patchcore_baseline.ipynb`: clone, install, data, train, evaluate, zip.
  Run on a Colab T4; results are committed under `results/`.

## Why

- **The threshold is the product decision.** AUROC says how well the model ranks parts;
  the line needs a yes/no. Picking it on test data or on defect labels would give a number
  that cannot be repeated in production, where defects are rare and unknown.
- **False-alarm rate as the knob.** A quality engineer can say "I accept 1 good part in
  100 being sent to manual check". That maps directly to a quantile of good scores.
- **Raw scores are saved.** Metrics can be recomputed on any laptop from committed files,
  and the README claim check reads them.
- **Intervals over parts.** MVTec AD 2 shows each part under six lights; counting those as
  six samples made the intervals look twice as precise as they are.
- **The failure gallery** shows what the model misses and why, which is what a quality
  engineer asks first.

## Bugs found on the way (all in `docs/decisions.md`)

pandas 3 broke anomalib (D20); a torchvision pin had no wheel for Colab's Python (D22);
torch and torchvision came from different CUDA builds (D23); Colab has a `/kaggle` folder
(D25); the MVTec AD download link was dead (D24); a mask path arrived as the text "None"
(D27); and the first intervals ignored the six-lights design (D28).

## Summary

Phase 2 is the baseline, and the point is the measurement, not the score. PatchCore
learns from good parts only. I set the threshold on good validation images for a 1%
false-alarm target; the function cannot even receive labels, and a test checks that.
Then I score the test set once.

On MVTec AD leather and carpet the AUROC is 1.000 and 0.988, which says little, because
that dataset is saturated. On MVTec AD 2 fabric, the harder set, AUROC is 0.657, and at
the threshold the model misses about 86% of defective images. The failure gallery shows
why: the missed defects are thin lines and small marks that look like the print, and the
image is shrunk to 256 pixels.

Two things I would point out to a quality engineer. First, the 1% target did not hold:
with about 50 validation images the threshold is just the highest validation score, and on
the test good parts the false-alarm rate was 3 to 9%. So on a real line I would calibrate
on more good parts, under every lighting the line uses, and re-measure. Second, MVTec AD 2
photographs each part under six lights. When I bootstrapped images, the AUROC interval
looked tight; when I resampled physical parts, it widened to 0.47 to 0.83, so with 26 parts
I cannot claim fabric is clearly better than chance. I report that instead of a nicer
number. Everything in the README is checked against the results JSON in CI.
