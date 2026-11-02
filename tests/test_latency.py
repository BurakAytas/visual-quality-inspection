import time

import pytest

from vqi.latency import timed_map


def test_outputs_keep_order_and_only_fn_is_timed() -> None:
    def slow_load(x: int) -> int:
        time.sleep(0.02)  # decoding must not count
        return x

    def fn(x: int) -> int:
        time.sleep(0.005)
        return x * 2

    outputs, stats = timed_map(fn, [1, 2, 3], load=slow_load, warmup=2)
    assert outputs == [2, 4, 6]
    assert stats["n_timed"] == 3 and stats["n_warmup"] == 2
    assert 4 < stats["median_ms"] < 15


def test_warmup_runs_are_not_timed() -> None:
    calls: list[int] = []

    def fn(x: int) -> int:
        calls.append(x)
        time.sleep(0.05 if len(calls) == 1 else 0.001)  # only the first call is slow
        return x

    _, stats = timed_map(fn, [7, 8, 9, 10], load=lambda x: x, warmup=1)
    assert len(calls) == 5
    assert stats["p95_ms"] < 20


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        timed_map(lambda x: x, [], load=lambda x: x, warmup=0)
