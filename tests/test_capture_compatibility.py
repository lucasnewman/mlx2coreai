import subprocess
import sys

import numpy as np

from mlx2coreai.ir import Graph, Node, TensorSpec


def test_normal_import_does_not_load_legacy_capture():
    subprocess.run([sys.executable, "-c", """
import sys
import mlx2coreai
import mlx2coreai.cli
import mlx2coreai.from_mlx
import mlx2coreai._convert_mlx_lm_stateful
assert 'mlx2coreai._legacy_capture' not in sys.modules
"""], check=True, capture_output=True, text=True)


def test_old_capture_imports_remain_available():
    from mlx2coreai import from_mlx, _legacy_capture

    for name in from_mlx._LEGACY_EXPORTS:
        assert name in dir(from_mlx)
        assert getattr(from_mlx, name) is getattr(_legacy_capture, name)


def test_dot_function_capture_keeps_legacy_behavior(tmp_path):
    from mlx2coreai.from_mlx import capture_graph_from_mlx_function

    inputs = {"x": np.array([1, 2], np.float32), "y": np.array([3, 4], np.float32)}
    graph, captured, expected = capture_graph_from_mlx_function(
        tmp_path / "capture.dot", inputs, lambda x, y: x + y, capture_mode="dot",
    )
    graph.validate()
    assert {node.op for node in graph.nodes} == {"add"}
    assert set(captured) == set(inputs)
    np.testing.assert_array_equal(next(iter(expected.values())), [4, 6])


def test_ir_to_mlx_debug_export_is_still_available(tmp_path):
    from mlx2coreai.from_mlx import capture_graph_from_ir

    graph = Graph([TensorSpec("x", (2,)), TensorSpec("y", (2,))],
                  [Node("add", ("x", "y"), "z")], ["z"])
    captured = capture_graph_from_ir(tmp_path / "legacy.dot", graph,
                                     {"x": np.ones(2, np.float32), "y": np.ones(2, np.float32)})
    assert [node.op for node in captured.nodes] == ["add"]


def test_ir_visualization_does_not_use_reverse_interpreter(tmp_path, monkeypatch):
    from mlx2coreai import _legacy_capture
    from mlx2coreai.from_mlx import parse_mlx_dot_to_graph
    from tests import model_zoo

    def unexpected(*args, **kwargs):
        raise AssertionError("IR visualization must not execute MLX")

    monkeypatch.setattr(_legacy_capture, "_eval_node_with_mlx", unexpected)
    name = model_zoo.available_model_names()[0]
    spec = model_zoo.capture_model_spec(name, 0, tmp_path)
    graph = parse_mlx_dot_to_graph((tmp_path / "capture_graph.dot").read_text(), spec.graph.inputs)
    assert len(graph.nodes) == len(spec.graph.nodes)


def test_multi_output_meshgrid_legacy_replay():
    import mlx.core as mx
    from mlx2coreai._legacy_capture import _eval_node_with_mlx

    node = Node("meshgrid", ("x", "y"), outputs=("a", "b"))
    inputs = {"x": mx.arange(2), "y": mx.arange(3)}
    expected = np.meshgrid(np.arange(2), np.arange(3))
    for index, reference in enumerate(expected):
        actual = _eval_node_with_mlx(node.result_node(index), inputs, mx)
        np.testing.assert_array_equal(np.asarray(actual), reference)
