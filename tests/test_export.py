"""INT8 quantization scope: only the CNN, never the PatchCore head (D50)."""

import numpy as np
import pytest

ov = pytest.importorskip("openvino")
export = pytest.importorskip("vqi.export", reason="needs the [train] and [export] extras")
ops = ov.opset13


def tiny_patchcore() -> ov.Model:
    """Feature layer, search against a memory bank, then a score; the shape of PatchCore."""
    image = ops.parameter([1, 4], np.float32, name="image")
    features = ops.matmul(image, ops.constant(np.ones((4, 3), np.float32)), False, False)
    features.set_friendly_name("backbone")
    bank = ops.constant(np.ones((3, 5), np.float32))
    bank.set_friendly_name("model.memory_bank")
    distances = ops.matmul(features, bank, False, False)
    distances.set_friendly_name("search")
    nearest = ops.reduce_min(distances, ops.constant([1]), keep_dims=False)
    nearest.set_friendly_name("nearest")
    score = ops.multiply(nearest, ops.constant(np.float32(2)))
    score.set_friendly_name("pred_score")
    return ov.Model([ops.result(score)], [image])


def test_patchcore_head_holds_the_search_and_everything_after_it() -> None:
    assert export.patchcore_head(tiny_patchcore()) == ["nearest", "pred_score", "search"]


def test_patchcore_head_refuses_a_model_without_memory_bank() -> None:
    image = ops.parameter([1, 4], np.float32)
    model = ov.Model([ops.result(ops.relu(image))], [image])
    with pytest.raises(ValueError, match="memory bank"):
        export.patchcore_head(model)
