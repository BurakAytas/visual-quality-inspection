# Phase 6: inspection API and line integration

## What was done

`vqi.api` is a FastAPI service with three endpoints. `POST /inspect` takes one image and
returns the decision (OK/NOK), the score, the threshold, the latency, a heatmap PNG and a
`needs_review` flag. `GET /review` lists the parts flagged for an operator. `GET /health`
reports the model version. The model runs on the plain OpenVINO runtime (`vqi.inspection`),
and every decision is written to a SQLite log (`vqi.decision_log`) with its timestamp,
score, threshold, model version and latency. All settings come from
`configs/serve/leather.yaml` and can be overridden by `VQI_<KEY>` environment variables.

Two scripts stand in for the line around the API. `scripts/line_simulator.py` watches a
folder like a camera output directory and posts each new image once, oldest first.
`scripts/daily_report.py` turns the log into a CSV with one row per day: parts inspected,
NOK, flagged for review, and the defect rate. A Dockerfile and `compose.yaml` run the API
and the simulator together on a CPU, with the exported model mounted as a volume.

A golden set of 11 leather images (`tests/golden/leather_openvino_fp32.json`) pins each
image's score, decision and review flag, checks the plain runtime against anomalib's
`OpenVINOInferencer`, and requires the anomaly map of every defect to exceed the threshold.

## Why

The API must be small and predictable on a line PC. anomalib's inferencer imports torch,
so the image would carry several gigabytes it never uses; reading the two model outputs
directly removes that, and the golden-set test guards against reading them wrong.

A single threshold hides how uncertain a decision is. Phase 3 showed that the ranking
transfers but the exact threshold does not, so parts close to the threshold on either side
are the ones a person should see. The review band is computed from the good validation
scores (about the top 10% of them sit above its lower edge) and mirrored above the
threshold, so it is reproducible and not a hand-picked number. On the leather test set it
flags no defects, but many good parts, which shows that the test good parts score higher
than the validation ones; on a real line that queue is the early warning to recalibrate.

The log records the model version (format, git commit, threshold) with each decision, so
any past decision can be traced to the model that made it. SQLite and a CSV report need no
server and no extra dependency; the CSV opens in Excel, which is what a quality engineer
would use.

## Bugs found on the way

The biggest one was in Phase 5's INT8 model. The first API run served INT8 and its heatmaps
were blank even for clear defects. NNCF had quantized not only the CNN but also PatchCore's
memory-bank search, and had put a quantization node on the distance map with a range
calibrated on good images only, which clipped the map below the threshold. The image score
is taken before that node, so the Phase 5 metrics did not show it, and the explanation
written in Phase 5 (that the search stays in floating point) was wrong. The API now serves
OpenVINO FP32, a test fails if a defect's map does not reach the threshold, and D48 carries
a correction (D50).

An empty upload crashed `cv2.imdecode` with an assertion and became a server error; it is
now a 400 like any unreadable image. While checking that tests fail when their target
breaks, one check was misleading: a same-length edit restored within the same second left
a stale `.pyc`, so the old code ran. Mutation checks now run without bytecode files (D51).

## Summary

The exported model now has a working line integration: an API that decides, explains with a
heatmap, flags uncertain parts and logs everything, plus a simulated camera and a daily
report. Building it exposed a real defect in the INT8 export that the Phase 5 metrics had
missed, which is the strongest argument for testing the whole output of a model, not only
its headline score. Fixing INT8 means excluding the PatchCore head from quantization and
rerunning part of Phase 5; that waits for approval. The defect-type field in the log stays
empty until the classifier of Phase 4 exists.
