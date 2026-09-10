import numpy as np
import pytest

from mlx2coreai import ConversionConfig, convert_mlx_to_coreai
from mlx2coreai.runtime import run_aimodel_sync


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4)])
def test_layernorm_capture_axis_and_epsilon(tmp_path, shape):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions

    rng = np.random.default_rng(18)
    inputs = {"x": rng.normal(size=shape).astype(np.float32),
              "weight": rng.normal(size=4).astype(np.float32),
              "bias": rng.normal(size=4).astype(np.float32)}
    eps = 0.03
    forward = lambda x, weight, bias: mx.fast.layer_norm(x, weight, bias, eps)
    converted = convert_mlx_to_coreai(
        forward, inputs,
        config=ConversionConfig(capture_shapeless=True, dynamic_axes={"x": [0]},
            dynamic_probe_inputs={**inputs, "x": np.ones((5, *shape[1:]), np.float32)}),
        output_path=tmp_path / "layernorm.aimodel",
    )
    node = next(n for n in converted.prepared.normalized_graph.nodes if n.op == "layernorm")
    assert node.attrs["axes"] == [-1]
    assert node.attrs["eps"] == pytest.approx(eps)
    for batch in (2, 7):
        values = {**inputs, "x": rng.normal(size=(batch, *shape[1:])).astype(np.float32)}
        x = values["x"]
        expected = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)
        expected = expected * values["weight"] + values["bias"]
        result = run_aimodel_sync(converted.asset, values,
                                 specialization_options=SpecializationOptions.cpu_only())
        np.testing.assert_allclose(next(iter(result.outputs.values())), expected, atol=2e-6, rtol=2e-6)
