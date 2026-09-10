import numpy as np
import pytest

from mlx2coreai import ConversionConfig, convert_mlx_to_coreai
from mlx2coreai.ir import Graph, Node, StateSpec, TensorSpec
from mlx2coreai.signature import CaptureSignature, StateBinding


def test_state_binding_and_input_order():
    graph = Graph(
        [TensorSpec("cache", (2,)), TensorSpec("x", (2,))],
        [Node("add", ("x", "cache"), "sum")], ["sum", "sum"],
    )
    expected = {"sum": np.array([3, 4], dtype=np.float32)}
    signature = CaptureSignature(("x", "cache"), (StateBinding(StateSpec("cache", (2,), "fp32"), 1),))
    bound, outputs = signature.bind(graph, expected)
    assert [s.name for s in bound.inputs] == ["x", "cache"]
    assert bound.outputs == ["sum", "cache__updated"]
    assert outputs["sum"] is outputs["cache__updated"]
    assert graph.outputs == ["sum", "sum"]


def test_signature_rejects_invalid_bindings():
    graph = Graph([TensorSpec("x", (2,))], [], ["x"])
    for signature in (
        CaptureSignature(input_order=("missing",)),
        CaptureSignature(states=(StateBinding(StateSpec("x", (2,)), 2),)),
        CaptureSignature(output_count=2),
    ):
        with pytest.raises(ValueError):
            signature.bind(graph, {"x": np.zeros(2)})


def test_conversion_analyzes_once_and_derives_state_specs(monkeypatch):
    import mlx2coreai.passes as passes

    calls = []
    original = passes.normalize_graph

    def normalize(graph):
        calls.append(graph)
        return original(graph)

    monkeypatch.setattr(passes, "normalize_graph", normalize)
    spec = StateSpec("cache", (2,), "fp32", capacity_axis=0)
    converted = convert_mlx_to_coreai(
        lambda x, cache: (x + cache, x + cache),
        {"x": np.ones(2, np.float32), "cache": np.zeros(2, np.float32)},
        config=ConversionConfig(signature=CaptureSignature(states=(StateBinding(spec, 1),))),
    )
    assert len(calls) == 1
    assert converted.lowered.graph is converted.prepared.normalized_graph
    assert 'MutableBuffers.buffer_mutation = "cache"' in str(converted.program)
    assert spec.resolved_shape(4) == (4,)
    assert spec.to_dict()["capacity_axis"] == 0
