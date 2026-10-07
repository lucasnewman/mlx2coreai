"""Generic projection compression, native execution, and MLX checkpoint preservation."""
import asyncio
from dataclasses import replace

import numpy as np
import pytest

from mlx2coreai import ConversionConfig, Graph, Node, TensorSpec, WeightQuantization
from mlx2coreai import convert_mlx_to_coreai, prepare_quantized_linears, run_aimodel_sync
from mlx2coreai.quantization import quantize_linear_weights
from mlx2coreai.recipe import Build, Component, export


def projection(weight, *, transpose=True, bias=False):
    nodes = [Node("constant", (), "weight", {"value": weight}, source="linear.weight")]
    rhs = "weight"
    if transpose:
        nodes.append(Node("transpose", (rhs,), "wt", {"perm": [1, 0]}))
        rhs = "wt"
    if bias:
        nodes += [Node("constant", (), "bias", {"value": np.full((1, 3), 0.25, np.float32)}),
                  Node("addmm", ("x", rhs, "bias"), "out", {"input_order": "abc"})]
    else:
        nodes.append(Node("matmul", ("x", rhs), "out"))
    return Graph([TensorSpec("x", (2, 128))], nodes, ["out"])


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("transpose,bias", [(True, False), (True, True), (False, False)])
def test_native_grouped_projection(tmp_path, bits, transpose, bias):
    from coreai.runtime import SpecializationOptions
    from mlx2coreai.conversion import lower_graph_to_coreai

    rng = np.random.default_rng(67)
    weight = rng.uniform(-1, 1, (3, 128)).astype(np.float32)
    # Each 64-value group has a known [-1, 1] quantization range.
    weight[:, ::64], weight[:, 63::64] = -1, 1
    graph = projection(weight if transpose else weight.T, transpose=transpose, bias=bias)
    converted, report = quantize_linear_weights(graph, WeightQuantization(bits=bits, group_size=64))
    assert report["selected_weights"] == report["quantized_weights"] == 1
    assert report["weights"][0]["axis"] == (1 if transpose else 0)
    assert report["weights"][0]["compressed_bytes"] < weight.nbytes
    assert report["weights"][0]["max_abs_error"] <= 1 / (2**bits - 1) + 1e-6
    x = rng.normal(size=(2, 128)).astype(np.float32)
    scale = np.float32(2 / (2**bits - 1))
    expected_weight = np.clip(np.rint((weight.astype(np.float64) + 1) / scale), 0, 2**bits - 1).astype(np.float32) * scale - 1
    expected = x @ expected_weight.T + (0.25 if bias else 0)
    lowered = lower_graph_to_coreai(converted, config=ConversionConfig(optimize=False))
    asset = lowered.program.save_asset(tmp_path / "linear.aimodel")
    actual = next(iter(run_aimodel_sync(asset, {"x": x},
        specialization_options=SpecializationOptions.cpu_only()).outputs.values()))
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
    assert graph.nodes[0].op == "constant"  # Original graph remains usable.


def test_weight_selection_and_skips():
    graph = projection(np.zeros((3, 128), np.float32))
    graph.nodes.insert(1, Node("constant", (), "unrelated", {"value": np.eye(128, dtype=np.float32)}))
    graph.nodes.insert(2, Node("constant", (), "mask", {"value": np.ones((3, 128), bool)}))
    result, report = quantize_linear_weights(graph, WeightQuantization(exclude=("linear.*",)))
    assert report["selected_weights"] == 1 and report["quantized_weights"] == 0
    assert report["weights"][0]["reason"] == "excluded"
    assert result.nodes == graph.nodes
    for weight, reason in ((np.zeros((3, 130), np.float32), "divisible"),
                           (np.zeros((3, 128), np.float16), "FP32")):
        _, report = quantize_linear_weights(projection(weight), WeightQuantization())
        assert report["quantized_weights"] == 0 and reason in report["weights"][0]["reason"]
    graph = Graph([TensorSpec("x", (2, 128)), TensorSpec("weight", (128, 3))],
                  [Node("matmul", ("x", "weight"), "out")], ["out"])
    _, report = quantize_linear_weights(graph, WeightQuantization())
    assert report["selected_weights"] == 0  # Runtime inputs are never quantized.


def test_constant_groups_shared_weights_and_names():
    graph = projection(np.full((3, 128), 2.5, np.float32))
    graph.nodes.insert(1, Node("constant", (), "weight__packed_codes", {"value": np.array(1, np.float32)}))
    graph.nodes.append(Node("matmul", ("x", "wt"), "again"))
    graph.outputs.append("again")
    result, report = quantize_linear_weights(graph, WeightQuantization())
    assert report["quantized_weights"] == 1 and report["weights"][0]["max_abs_error"] == 0
    assert sum(n.op == "blockwise_shift_scale" for n in result.nodes) == 1
    result.validate()
    bad = projection(np.full((3, 128), np.nan, np.float32))
    with pytest.raises(ValueError, match="nonfinite"):
        quantize_linear_weights(bad, WeightQuantization())


@pytest.mark.parametrize("kwargs", [{"bits": 2}, {"bits": True}, {"group_size": 0}, {"exclude": "weight"}])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        WeightQuantization(**kwargs)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_preserve_standard_mlx_checkpoint(tmp_path, bits):
    import mlx.core as mx
    import mlx.nn as nn
    from coreai.runtime import SpecializationOptions

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(256, 16)

        def __call__(self, x):
            return self.layer(x)

    mx.random.seed(68)
    model = Model()
    nn.quantize(model, group_size=64, bits=bits)
    mx.eval(model.parameters())
    x = np.random.default_rng(69).normal(size=(2, 256)).astype(np.float32)
    original = np.asarray(model(mx.array(x)))
    packed = prepare_quantized_linears(model, require_all=True)
    assert packed.module_count == 1 and len(packed.weights) == 1
    converted = convert_mlx_to_coreai(model, {"x": x},
        config=ConversionConfig(optimize=False, graph_transform=packed), output_path=tmp_path / "preserved.aimodel")
    actual = next(iter(run_aimodel_sync(converted.asset, {"x": x},
        specialization_options=SpecializationOptions.cpu_only()).outputs.values()))
    np.testing.assert_allclose(actual, original, atol=2e-6, rtol=2e-6)
    weight = next(iter(packed.weights.values()))
    graph = Graph([], [Node("constant", (), "transposed", {"value": np.asarray(model.layer.weight).T})], ["transposed"])
    assert packed(graph).nodes[-1].op == "blockwise_shift_scale"
    assert weight.bits == bits


@pytest.mark.parametrize("bits", [4, 8])
def test_existing_recipe_export_option(tmp_path, bits):
    import mlx.core as mx
    import mlx.nn as nn
    from coreai.runtime import SpecializationOptions

    mx.random.seed(70)
    model = nn.Linear(128, 16)
    mx.eval(model.parameters())
    x = np.ones((1, 128), np.float32)
    component = Component(lambda x: model(x), {"x": x}, ("result",), ConversionConfig(optimize=False))
    plan = Build("linear", {"main": component}, runtime_metadata={})
    baseline = export(plan, tmp_path / "fp32")
    report = {}
    quantized = export(plan, tmp_path / "quantized", quantization=WeightQuantization(bits=bits),
                       quantization_report=report)
    assert report["main"]["quantized_weights"] == 1
    assert component.config.weight_quantization is None  # Export does not modify the recipe plan.
    def size(path):
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    assert size(quantized.path / "main.aimodel") < size(baseline.path / "main.aimodel")
    policy_component = replace(component, config=replace(component.config, weight_quantization=WeightQuantization(bits=bits)))
    policy_bundle = export(Build("linear", {"main": policy_component}), tmp_path / "component_policy")

    async def check():
        options = SpecializationOptions.cpu_only()
        async with baseline.session(specialization_options=options) as session:
            original = (await session.run("main", {"x": x}, readback=True))["result"]
        async with quantized.session(specialization_options=options) as session:
            actual = (await session.run("main", {"x": x}, readback=True))["result"]
        async with policy_bundle.session(specialization_options=options) as session:
            per_component = (await session.run("main", {"x": x}, readback=True))["result"]
        np.testing.assert_array_equal(actual, per_component)
        error_bound = 128 * report["main"]["weights"][0]["max_abs_error"] + 1e-5
        assert np.max(np.abs(actual - original)) <= error_bound

    asyncio.run(check())


def test_batched_dynamic_linear_and_attention_selection(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    from coreai.runtime import SpecializationOptions

    mx.random.seed(71)
    model = nn.Linear(128, 128, bias=False)
    mx.eval(model.parameters())

    def forward(x):
        projected = model(x)
        # The second matmul is attention; its RHS is an activation, not a weight.
        return projected @ projected.swapaxes(-1, -2)

    x = np.random.default_rng(72).normal(size=(1, 3, 128)).astype(np.float32)
    config = ConversionConfig(optimize=False, capture_shapeless=True, dynamic_axes={"x": [1]},
                              dynamic_probe_inputs={"x": np.ones((1, 5, 128), np.float32)},
                              weight_quantization=WeightQuantization())
    converted = convert_mlx_to_coreai(forward, {"x": x}, config=config, output_path=tmp_path / "batched.aimodel")
    assert converted.prepared.quantization_report["selected_weights"] == 1
    assert converted.prepared.quantization_report["quantized_weights"] == 1
    x = np.ones((1, 7, 128), np.float32)
    actual = next(iter(run_aimodel_sync(converted.asset, {"x": x},
        specialization_options=SpecializationOptions.cpu_only()).outputs.values()))
    assert actual.shape == (1, 7, 7) and np.isfinite(actual).all()


def test_quantization_with_mutable_state_and_optimization(tmp_path):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions
    from mlx2coreai import CaptureSignature, StateBinding, StateSpec

    weight = mx.array(np.random.default_rng(73).normal(size=(16, 128)).astype(np.float32))

    def forward(x, cache):
        updated = cache + x
        return updated, updated @ weight.T

    component = Component(forward, {"x": np.ones((1, 128), np.float32), "cache": np.zeros((1, 128), np.float32)},
        ("projected",), ConversionConfig(signature=CaptureSignature(output_count=2,
            states=(StateBinding(StateSpec("cache", (1, 128), "fp32"), 0),))))
    report = {}
    bundle = export(Build("stateful_linear", {"main": component}), tmp_path / "stateful",
                    quantization=WeightQuantization(), quantization_report=report)
    assert report["main"]["quantized_weights"] == 1

    async def check():
        async with bundle.session(specialization_options=SpecializationOptions.cpu_only()) as session:
            session.reset_state({"main": 1})
            first = (await session.run("main", {"x": np.ones((1, 128), np.float32)}, readback=True))["projected"]
            second = (await session.run("main", {"x": np.ones((1, 128), np.float32)}, readback=True))["projected"]
            np.testing.assert_allclose(second, first * 2, atol=1e-5, rtol=1e-5)
            np.testing.assert_array_equal(session.snapshot_state("main")["cache"], np.full((1, 128), 2, np.float32))

    asyncio.run(check())
