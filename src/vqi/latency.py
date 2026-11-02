"""Measure inference time per image: warm-up first, then median and 95th percentile."""

import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


def timed_map(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    load: Callable[[Any], Any],
    warmup: int,
) -> tuple[list[Any], dict[str, float]]:
    """Apply ``fn`` to every loaded item and time only the ``fn`` call.

    Loading (decoding an image from disk) is done outside the timer. The first loaded
    item is run ``warmup`` extra times before timing starts, so one-off costs (memory
    allocation, kernel compilation, caches) do not count.
    """
    items = list(items)
    if not items:
        raise ValueError("no items to time")
    first = load(items[0])
    for _ in range(warmup):
        fn(first)

    outputs, times_ms = [], []
    for item in items:
        data = load(item)
        start = time.perf_counter()
        outputs.append(fn(data))
        times_ms.append((time.perf_counter() - start) * 1000)
    stats = {
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "mean_ms": float(np.mean(times_ms)),
        "n_timed": len(times_ms),
        "n_warmup": warmup,
    }
    return outputs, stats
