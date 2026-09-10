import subprocess
import sys

import numpy as np

from mlx2coreai.ir import Graph, Node, TensorSpec
from mlx2coreai.reporting import write_graph_dot


def test_callback_capture_with_dot_visualization(tmp_path):
    from mlx2coreai.from_mlx import capture_graph_from_mlx_function

    path = tmp_path / "capture.dot"
    inputs = {"x": np.array([1, 2], np.float32), "y": np.array([3, 4], np.float32)}
    graph, captured, expected = capture_graph_from_mlx_function(path, inputs, lambda x, y: x + y)
    graph.validate()
    assert {node.op for node in graph.nodes} == {"add"}
    assert set(captured) == set(inputs)
    np.testing.assert_array_equal(next(iter(expected.values())), [4, 6])
    assert "digraph" in path.read_text()


def test_ir_visualization_without_mlx(tmp_path):
    subprocess.run([sys.executable, "-c", """
import sys
from pathlib import Path
sys.modules['mlx'] = None
sys.modules['mlx.core'] = None
from mlx2coreai.ir import Graph, Node, TensorSpec
from mlx2coreai.reporting import write_graph_dot
graph = Graph([TensorSpec('x', (2,)), TensorSpec('y', (2,))],
              [Node('add', ('x', 'y'), 'z')], ['z'])
write_graph_dot(Path(sys.argv[1]), graph)
""", str(tmp_path / "graph.dot")], check=True, capture_output=True, text=True)
    assert (tmp_path / "graph.dot").read_text() == (
        'digraph {\n'
        '  {rank=source; "x";}\n'
        '  {rank=source; "y";}\n'
        '  {rank=sink; "z";}\n'
        '  { 0 [label="add", shape=rectangle]; }\n'
        '  "x" -> 0;\n'
        '  "y" -> 0;\n'
        '  0 -> "z";\n'
        '}\n'
    )


def test_ir_visualization_preserves_multiple_outputs(tmp_path):
    graph = Graph([TensorSpec("0", (2,)), TensorSpec("y", (3,))],
                  [Node("meshgrid", ("0", "y"), outputs=("a", "b"))], ["a", "b"])
    path = tmp_path / "nested" / "graph.dot"
    write_graph_dot(path, graph)
    dot = path.read_text()
    assert dot.count('label="meshgrid"') == 1
    assert '"0" -> 1;' in dot
    assert '1 -> "a";' in dot
    assert '1 -> "b";' in dot


def test_model_zoo_ir_visualization(tmp_path):
    from tests import model_zoo

    spec = model_zoo.capture_model_spec(model_zoo.available_model_names()[0], 0, tmp_path)
    dot = (tmp_path / "capture_graph.dot").read_text()
    assert dot.count("shape=rectangle") == len(spec.graph.nodes)
