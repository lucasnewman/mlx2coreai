from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

from mlx2coreai import ConversionConfig, convert_mlx_to_coreai
from mlx2coreai.ir import Graph, Node
from mlx2coreai.passes import normalize_graph
from mlx2coreai.runtime import run_aimodel_sync


@pytest.mark.parametrize("dtype", [np.float16, ml_dtypes.bfloat16, np.float32, np.uint32])
@pytest.mark.parametrize("shape", [(), (2,)])
def test_small_constants_keep_dtype_and_rank(dtype, shape):
    value = np.ones(shape, dtype=dtype)
    graph = normalize_graph(Graph([], [Node("const", (), "x", {"value": value})], ["x"]))
    actual = np.asarray(graph.nodes[0].attrs["value"])
    assert actual.dtype == value.dtype
    assert actual.shape == value.shape


def test_contiguous_hint_is_omitted_only_during_export(tmp_path):
    import mlx.core as mx
    from mlx2coreai.from_mlx import capture_graph_from_mlx_function

    original = mx.contiguous
    x = np.ones((2, 3), dtype=np.float32)
    converted = convert_mlx_to_coreai(lambda x: mx.contiguous(x.T) + 1, {"x": x},
                                     output_path=tmp_path / "contiguous.aimodel")
    assert mx.contiguous is original
    assert not any(n.op == "contiguous" for n in converted.prepared.normalized_graph.nodes)
    def fail(x):
        mx.contiguous(x)
        raise ValueError("capture failed")
    with pytest.raises(ValueError, match="capture failed"):
        capture_graph_from_mlx_function(None, inputs={"x": x}, function=fail)
    assert mx.contiguous is original


@pytest.mark.parametrize("kind", ["conv_split", "conv_tail", "partial_rope"])
def test_dynamic_mlx_model_ops(tmp_path, kind):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions, ComputeUnitKind

    if not SpecializationOptions.is_supported():
        pytest.skip("requires macOS 27 OS runtime")
    rng = np.random.default_rng(4)
    if kind == "conv_split":
        weight = mx.array(rng.normal(size=(4, 3, 1)).astype(np.float32))
        def fn(x):
            return mx.split(mx.conv1d(x, weight, padding=1, groups=4), 2, axis=-1)
        shape, axis = (1, 5, 4), 1
    elif kind == "conv_tail":
        history = mx.zeros((1, 2, 4), dtype=mx.float32)
        mx.eval(history)
        def fn(x):
            return mx.concatenate([history, x], axis=1)[:, -2:, :]
        shape, axis = (1, 5, 4), 1
    else:
        def fn(x):
            return mx.fast.rope(x, dims=8, traditional=False, base=10000.0, scale=1.0, offset=3)
        shape, axis = (1, 2, 5, 16), 2
    base = rng.normal(size=shape).astype(np.float32)
    probe_shape = list(shape)
    probe_shape[axis] += 1
    converted = convert_mlx_to_coreai(fn, {"x": base}, output_path=tmp_path / "ops.aimodel",
        config=ConversionConfig(capture_shapeless=True, dynamic_axes={"x": [axis]},
                                dynamic_probe_inputs={"x": np.ones(probe_shape, np.float32)}))
    opts = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    for length in (3, 7):
        runtime_shape = list(shape)
        runtime_shape[axis] = length
        x = rng.normal(size=runtime_shape).astype(np.float32)
        expected = fn(mx.array(x))
        expected = expected if isinstance(expected, list) else [expected]
        result = run_aimodel_sync(converted.asset, {"x": x}, specialization_options=opts)
        for name, value in zip(converted.prepared.normalized_graph.outputs, expected, strict=True):
            np.testing.assert_allclose(result.outputs[name], np.asarray(value), rtol=2e-4, atol=2e-5)
