# Phase 7: operator panel and final README

## What was done

`vqi.panel` is a Streamlit operator panel with three tabs. Inspect uploads a part to the
API and shows the decision with the defect type, the score, threshold and latency, and the
image next to its heatmap. Review queue lists the parts the API flagged for an operator.
Daily report shows the counts per day and a bar chart of NOK parts per defect type, served
by a new `GET /daily` endpoint. `compose.yaml` starts the panel next to the API and the
line simulator.

`scripts/make_demo_gif.py` records the README GIF without a screen recorder: it starts the
API and the panel, replays the leather test set as one shift, drives the panel with a
headless browser and joins the screenshots into `docs/demo.gif`. The README now opens with
the problem, the approach, the main results with their claim markers, and the GIF.

## Why

The panel holds no model and no data. Everything it shows comes from the API, so it cannot
disagree with the line, and it can run anywhere the API is reachable. The only API change
was the daily report, which until now existed only as a CSV script.

A screen recording cannot be repeated after the next model or panel change; a script can,
and it records the real panel against the real exported models, not a mock-up. The parts
shown were chosen for visible marks; the good part is not `good/000`, a known false alarm,
so the demo does not hide false alarms but leaves them to the review queue, where they
belong.

## Bugs found on the way

The first GIF showed the previous part's result under the new file name. Streamlit draws
the new upload before the API answer arrives, so waiting for an idle page was not enough.
The script now waits until the file name appears in both the upload chip and the image
caption.

Streamlit 1.66.1 warns that `use_container_width` is deprecated; the panel uses
`width="stretch"` instead.

## Summary

The project is complete end to end: a model trained on good parts only decides and
locates, a classifier names the defect, both run on a CPU behind an API that logs every
decision, and an operator sees the result, the review queue and the daily report in a
panel. The README's first screen now says what the problem is, how it is solved and how
well, with every number checked in CI. The main limit stays the one Phase 2 found: on the
harder fabric set the method misses most defects, and the README says so on its first
screen.
