# Phase 2: PatchCore baseline

## What was done

- `vqi.threshold`: the decision threshold comes from good validation images only. With n
  good images and a 1% target, at most floor(0.01 * n) of them may score above it. The
  function takes scores and the target rate, nothing else; a test checks it cannot take labels.
- `vqi.metrics`: precision, recall, F1, false-alarm rate and miss rate at that threshold,
  image AUROC, and miss rate at a fixed 1% false-alarm rate. Every metric gets a 95%
  bootstrap confidence interval.
- `vqi.overlay`: heatmap overlays with the true defect outlined in green.
- `scripts/train_anomaly.py` (GPU): trains PatchCore, saves raw scores of every validation
  and test image, the threshold, pixel AU-PRO, run info and the failure gallery.
- `scripts/evaluate.py` (CPU): turns the saved scores into `results/metrics.json`,
  `results/metrics.md` and score plots.
- `notebooks/02_patchcore_baseline.ipynb`: clone, install, data, train, evaluate, zip.

## Why

- **The threshold is the product decision.** AUROC says how well the model ranks parts;
  the line needs a yes/no. Picking it on test data or on defect labels would give a number
  that cannot be repeated in production, where defects are rare and unknown.
- **False-alarm rate as the knob.** A quality engineer can say "I accept 1 good part in
  100 being sent to manual check". That maps directly to a quantile of good scores.
- **Raw scores are saved.** Metrics can be recomputed on any laptop from committed files,
  and the README claim check reads them.
- **Confidence intervals.** Test sets have tens of defects, so a recall of 0.80 could
  easily be 0.65 or 0.92. Showing the interval is more honest than three decimals.
- **The failure gallery** shows what the model misses and why, which is what a quality
  engineer asks first.

## What to say in 2 minutes

Written after the GPU run, from the numbers in `results/`.
