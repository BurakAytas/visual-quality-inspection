# Phase 3: EfficientAD comparison and robustness

## What was done

- `configs/models/efficient_ad.yaml`: EfficientAD (small), trained on the same splits with
  the same threshold policy as PatchCore. Model settings now live in `configs/models/` and
  are passed with `--model`, so a second model needed no code change.
- `vqi.perturb`: four fixed test perturbations (darker, brighter, blur, noise), seeded and
  applied at full resolution before the model's resize. The clean threshold is kept.
- `scripts/evaluate.py` writes one robustness table: model x category x condition, with
  95% bootstrap intervals over physical parts.
- Notebooks keep results on Drive and resume after a disconnect, and stream progress live.
  In the end Colab's free GPU quota ran out, and the run was done on a local RTX 3070 Ti.

## Why

- **A second model shows the measurement is fair.** If both models were judged with
  different rules, a comparison would mean nothing; here they share splits, threshold
  policy, metrics and intervals.
- **Perturbations with the clean threshold** mimic what happens on a line after a lamp
  ages or a lens gets dirty: nobody retunes the threshold on the spot.
- **AUROC and false-alarm rate together** separate "the model can no longer tell good from
  bad" from "the model still can, but the threshold no longer fits".

## Bugs found on the way (all in `docs/decisions.md`)

`onnxscript` missing for ONNX export (found in a local test before any GPU run);
notebook output from child processes was invisible (D38); Colab lost finished runs on a
disconnect (D37); lambdas broke data-loader workers on Windows (D39).

## Summary

PatchCore beat EfficientAD at the compute budget used here: AUROC 0.751 vs 0.551 on the
harder fabric set and 1.000 vs 0.921 on leather, with similar results on carpet.
EfficientAD's validation threshold also transferred badly, with false-alarm rates of 25 to
56% on the test good parts. I report that as a negative result, with the two likely causes:
only 10 epochs, and map normalisation computed on the same validation images that set the
threshold.

The robustness test gave the most useful finding for a production line. On leather,
PatchCore's ranking is unaffected by brightness, blur and noise (AUROC 1.000 every time),
but with the clean threshold its false-alarm rate jumps from 19% to 100% under blur. So
the model is not the weak point; the fixed threshold is. On a real line I would watch the
score distribution of good parts and recalibrate after any optics or lighting change.

Retraining PatchCore with the same code and seed on a different GPU moved the fabric AUROC
from 0.657 to 0.751. Both values sit inside each other's 95% interval, which is exactly
why every number in this project carries one.
