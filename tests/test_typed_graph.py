from dataclasses import replace

import numpy as np
import pytest

from mlx2coreai import ConversionConfig, convert_mlx_to_coreai, prepare_mlx_conversion
from mlx2coreai.ir import Graph, Node, TensorSpec, TensorType, dynamic_dim_ref
from mlx2coreai.passes import infer_graph_specs, normalize_graph
from mlx2coreai.lower_to_coreai import build_coreai_program
from mlx2coreai.runtime import run_aimodel_sync


def test_capture_retains_types_without_reinferring(monkeypatch):
    import mlx.core as mx
    import mlx2coreai.passes as passes

    def unexpected_inference(*args):
        pytest.fail("Captured tensor types should not be reconstructed.")

    monkeypatch.setattr(passes, "_infer_node_spec", unexpected_inference)
    converted = convert_mlx_to_coreai(
        lambda x: (mx.transpose(x), x.astype(mx.float16)),
        {"x": np.arange(6, dtype=np.float32).reshape(2, 3)},
    )
    graph = converted.prepared.normalized_graph
    assert all(name in graph.value_types for node in graph.nodes for name in node.outputs)
    assert graph.value_types[graph.outputs[0]] == TensorType((3, 2), "fp32")
    assert graph.value_types[graph.outputs[1]] == TensorType((2, 3), "fp16")


@pytest.mark.parametrize("probe", [False, True])
def test_dynamic_types_do_not_freeze_trace_shape(probe):
    prepared = prepare_mlx_conversion(
        lambda x: x * 2,
        {"x": np.ones((2, 3), np.float32)},
        config=ConversionConfig(
            dynamic_axes={"x": [0]},
            dynamic_probe_inputs={"x": np.ones((5, 3), np.float32)} if probe else None,
        ),
    )
    name = prepared.normalized_graph.outputs[0]
    assert prepared.normalized_graph.value_types[name].shape == ((-1, 3) if probe else None)
    assert prepared.analysis.specs[name].shape == (-1, 3)


def test_unprobed_axes_retain_conservative_inference():
    inputs = {"x": np.ones((2, 3), np.float32), "y": np.ones((2, 3), np.float32)}
    prepared = prepare_mlx_conversion(
        lambda x, y: (x + x, y + y), inputs,
        config=ConversionConfig(dynamic_axes={"x": [0], "y": [0]},
            dynamic_probe_inputs={**inputs, "x": np.ones((5, 3), np.float32)}),
    )
    graph = prepared.normalized_graph
    assert graph.value_types[graph.outputs[0]].shape == (-1, 3)
    assert graph.value_types[graph.outputs[1]].shape is None
    assert all(prepared.analysis.specs[name].shape == (-1, 3) for name in graph.outputs)


def test_renaming_and_identity_removal_preserve_types_and_dimension_refs():
    graph = Graph(
        [TensorSpec("x input", (-1, 3))],
        [Node("copy", ("x input",), "old copy"),
         Node("reshape", ("old copy",), "my result", {"shape": [dynamic_dim_ref("old copy", 0), 3]})],
        ["my result"],
        {"old copy": TensorType((-1, 3), "fp32"), "my result": TensorType((-1, 3), "fp32")},
    )
    normalized = normalize_graph(graph)
    assert normalized.value_types == {"my_result": TensorType((-1, 3), "fp32")}
    assert normalized.nodes[0].inputs == ("x_input",)
    assert normalized.nodes[0].attrs["shape"][0] == dynamic_dim_ref("x_input", 0)
    assert graph.nodes[1].attrs["shape"][0] == dynamic_dim_ref("old copy", 0)
    build_coreai_program(normalized)


def test_node_constructor_and_replace_preserve_multi_output_contract():
    single = Node("add", ("x", "y"), "z", {"flag": True})
    assert single.outputs == ("z",)
    assert single.to_dict()["output"] == "z"
    multi = Node("divmod", ("x", "y"), outputs=("quotient", "remainder"))
    assert replace(multi, attrs={"flag": True}).outputs == multi.outputs
    assert multi.to_dict()["outputs"] == ["quotient", "remainder"]
    assert multi.result_node(1).output == "remainder"
    with pytest.raises(ValueError, match="Duplicate tensor"):
        Graph([TensorSpec("x", (2,)), TensorSpec("y", (2,))],
              [replace(multi, outputs=("z", "z"))], ["z"]).validate()


@pytest.mark.parametrize("kind", ["split", "broadcast_arrays", "divmod", "meshgrid"])
def test_multi_output_operation_runtime(tmp_path, kind):
    from coreai.runtime import SpecializationOptions

    if kind == "split":
        inputs = {"x": np.arange(8, dtype=np.float32).reshape(2, 4)}
        node = Node("split", tuple(inputs), outputs=("a", "b"), attrs={"axis": 1, "num_splits": 2})
        expected = np.split(inputs["x"], 2, axis=1)
    elif kind == "broadcast_arrays":
        inputs = {"x": np.ones((2, 1), np.float32), "y": np.arange(3, dtype=np.float32)[None]}
        node = Node(kind, tuple(inputs), outputs=("a", "b"))
        expected = np.broadcast_arrays(*inputs.values())
    elif kind == "meshgrid":
        inputs = {"x": np.arange(2, dtype=np.float32), "y": np.arange(3, dtype=np.float32)}
        node = Node(kind, tuple(inputs), outputs=("a", "b"))
        expected = np.meshgrid(*inputs.values())
    else:
        inputs = {"x": np.array([3, 7], np.float32), "y": np.array([2, 4], np.float32)}
        node = Node(kind, tuple(inputs), outputs=("a", "b"))
        expected = np.divmod(inputs["x"], inputs["y"])
    graph = Graph([TensorSpec(name, value.shape) for name, value in inputs.items()], [node], ["a", "b"])
    assert set(infer_graph_specs(graph)) == {*inputs, "a", "b"}
    asset = build_coreai_program(graph).program.save_asset(tmp_path / "multi.aimodel")
    actual = run_aimodel_sync(asset, inputs, specialization_options=SpecializationOptions.cpu_only()).outputs
    for name, reference in zip(graph.outputs, expected, strict=True):
        np.testing.assert_array_equal(actual[name], reference)


@pytest.mark.parametrize("kind", ["broadcast_arrays", "meshgrid"])
def test_multi_output_inference_preserves_each_input_dtype(kind):
    graph = Graph(
        [TensorSpec("x", (2, 1) if kind == "broadcast_arrays" else (2,), "fp32"),
         TensorSpec("y", (3,), "int32")],
        [Node(kind, ("x", "y"), outputs=("a", "b"))], ["a", "b"],
    )
    specs = infer_graph_specs(graph)
    assert specs["a"].dtype == "fp32"
    assert specs["b"].dtype == "int32"
    expected_shape = (2, 3) if kind == "broadcast_arrays" else (3, 2)
    assert specs["a"].shape == specs["b"].shape == expected_shape


def test_multi_output_gated_delta_runtime(tmp_path):
    from coreai.runtime import SpecializationOptions
    from tests.test_gated_delta import delta_graph, reference

    legacy = delta_graph(sequence=3)
    graph = replace(legacy, nodes=[Node("gated_delta_update", tuple(spec.name for spec in legacy.inputs),
                                       outputs=("y", "s"))])
    rng = np.random.default_rng(17)
    inputs = {spec.name: rng.normal(0, 0.1, spec.shape).astype(np.float32) for spec in graph.inputs}
    inputs["decay"] = np.exp(-np.abs(inputs["decay"]))
    asset = build_coreai_program(graph).program.save_asset(tmp_path / "delta.aimodel")
    actual = run_aimodel_sync(asset, inputs, specialization_options=SpecializationOptions.cpu_only()).outputs
    for name, expected in zip(graph.outputs, reference(inputs), strict=True):
        np.testing.assert_allclose(actual[name], expected, rtol=3e-4, atol=2e-6)
