import numpy as np
import pytest

from mlx2coreai import ConversionConfig, convert_mlx_to_coreai
from mlx2coreai.runtime import run_aimodel_sync


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.75, -0.5), (0.0, 1.0), (1.0, 0.0)])
def test_live_addmm_order_and_scaling(tmp_path, alpha, beta):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions
    from mlx2coreai._legacy_capture import _eval_node_with_mlx

    rng = np.random.default_rng(12)
    inputs = {"x": rng.normal(size=(2, 3)).astype(np.float32),
              "w": rng.normal(size=(3, 4)).astype(np.float32),
              "b": rng.normal(size=(4,)).astype(np.float32)}
    converted = convert_mlx_to_coreai(
        lambda x, w, b: mx.addmm(b, x, w, alpha=alpha, beta=beta), inputs,
        config=ConversionConfig(dynamic_axes={"x": [0]}, capture_shapeless=True,
            dynamic_probe_inputs={**inputs, "x": np.ones((5, 3), np.float32)}),
        output_path=tmp_path / "addmm.aimodel",
    )
    graph = converted.prepared.normalized_graph
    node = next(n for n in graph.nodes if n.op == "addmm")
    assert node.attrs == {"input_order": "abc", "alpha": alpha, "beta": beta}
    for length in (2, 7):
        values = {**inputs, "x": rng.normal(size=(length, 3)).astype(np.float32)}
        expected = beta * values["b"] + alpha * (values["x"] @ values["w"])
        result = run_aimodel_sync(converted.asset, values,
                                 specialization_options=SpecializationOptions.cpu_only())
        np.testing.assert_allclose(next(iter(result.outputs.values())), expected, atol=2e-6, rtol=2e-6)
    replay_values = dict(zip(node.inputs, (mx.array(inputs["x"]), mx.array(inputs["w"]),
                                          mx.array(np.broadcast_to(inputs["b"], (2, 4))))))
    np.testing.assert_allclose(np.asarray(_eval_node_with_mlx(node, replay_values, mx)),
                               beta * inputs["b"] + alpha * (inputs["x"] @ inputs["w"]),
                               atol=2e-6, rtol=2e-6)


def test_batched_linear_derived_dynamic_extent(tmp_path):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions

    rng = np.random.default_rng(19)
    w = rng.normal(size=(3, 4)).astype(np.float32)
    b = rng.normal(size=4).astype(np.float32)
    inputs = {"x": rng.normal(size=(1, 6, 3)).astype(np.float32)}
    # The b2 authoring optimizer mis-folds this graph's reshape/addmm pattern.
    converted = convert_mlx_to_coreai(
        lambda x: mx.addmm(mx.array(b), x, mx.array(w)), inputs,
        config=ConversionConfig(optimize=False, capture_shapeless=True, dynamic_axes={"x": [0]},
            dynamic_probe_inputs={"x": np.ones((2, 6, 3), np.float32)}),
        output_path=tmp_path / "batched_linear.aimodel",
    )
    for batch in (1, 2, 3):
        x = rng.normal(size=(batch, 6, 3)).astype(np.float32)
        result = run_aimodel_sync(converted.asset, {"x": x},
                                 specialization_options=SpecializationOptions.cpu_only())
        np.testing.assert_allclose(next(iter(result.outputs.values())), x @ w + b, atol=2e-6, rtol=2e-6)
