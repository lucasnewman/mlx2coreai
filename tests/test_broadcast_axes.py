from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import pytest

from mlx2coreai.conversion import ConversionConfig, convert_mlx_to_coreai, lower_graph_to_coreai
from mlx2coreai.from_mlx import parse_mlx_export_events_to_graph
from mlx2coreai.ir import Graph, Node, TensorSpec
from mlx2coreai.passes import infer_graph_specs


def broadcast_graph(shapes, axes):
    inputs = [TensorSpec(f"x{i}", shape, "fp32") for i, shape in enumerate(shapes)]
    return Graph(inputs, [Node("broadcast_axes", tuple(x.name for x in inputs), "out",
                               attrs={"ignore_axes": axes})], ["out"])


@pytest.mark.parametrize("shapes,axes,expected", [
    ([(2, 1, 3), (1, 4, 1)], [], (2, 4, 3)),
    ([(2, 3), (5, 3, 4)], [-2, -1], (5, 2, 3)),
    ([(1, 2, 3), (5, 7, 3)], [-2], (5, 2, 3)),
    ([(1, 2, 1), (5, 7, 3), (1, 9, 1)], [-2], (5, 2, 3)),
    ([(3, 1, 2), (7, 4, 2)], [-3], (3, 4, 2)),
    ([(), (2, 0)], [], (2, 0)),
    ([(1, 0), (3, 1)], [], (3, 0)),
    ([(), ()], [], ()),
    ([(1, -1, 3), (-1, 3, 4)], [-2, -1], (-1, -1, 3)),
    ([(-1, 3), (7, 4)], [-2, -1], (-1, 3)),
])
def test_broadcast_axes_shape_and_asset(tmp_path: Path, shapes, axes, expected):
    graph = broadcast_graph(shapes, axes)
    assert infer_graph_specs(graph)["out"].shape == expected
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=False))
    lowered.program.save_asset(tmp_path / "broadcast.aimodel")


@pytest.mark.parametrize("shapes,axes", [
    ([(2, 3), (4, 3)], []),
    ([(2, 3), (3,)], [-2]),
    ([(2, 3)], [0]),
    ([(2, 3)], [-1, -2]),
    ([(2, 3)], [-1, -1]),
    ([], []),
])
def test_broadcast_axes_rejects_invalid_shapes_and_axes(shapes, axes):
    with pytest.raises(ValueError, match="broadcast_axes"):
        infer_graph_specs(broadcast_graph(shapes, axes))


@pytest.mark.parametrize("axes", [[], [-2, -1]])
def test_broadcast_axes_export_preserves_axes_and_first_input_dtype(axes):
    graph = parse_mlx_export_events_to_graph([
        {"type": "inputs", "inputs": [("a", (1, 2, 3), "float16"), ("b", (5, 2, 3), "int32")]},
        {"type": "primitive", "name": "BroadcastAxes", "arguments": [axes],
         "inputs": [("a", (1, 2, 3), "float16"), ("b", (5, 2, 3), "int32")],
         "outputs": [("out", (5, 2, 3), "float16")]},
        {"type": "outputs", "outputs": [("out", (5, 2, 3), "float16")]},
    ], input_specs=[])
    assert graph.nodes[0].op == "broadcast_axes"
    assert graph.nodes[0].attrs == {"ignore_axes": axes}
    assert infer_graph_specs(graph)["out"].dtype == "fp16"


def test_live_mlx_matmul_broadcast_axes_capture(tmp_path: Path):
    pytest.importorskip("mlx.core")
    from coreai.runtime import SpecializationOptions
    from mlx2coreai.runtime import run_aimodel_sync
    converted = convert_mlx_to_coreai(
        lambda x, y: x @ y,
        {"x": np.ones((2, 3), dtype=np.float32), "y": np.ones((5, 3, 4), dtype=np.float32)},
        config=ConversionConfig(
            capture_shapeless=True, dynamic_axes={"x": [0], "y": [0]}, optimize=True,
        ),
        output_path=tmp_path / "matmul.aimodel",
    )
    nodes = [node for node in converted.prepared.normalized_graph.nodes if node.op == "broadcast_axes"]
    if not nodes:
        pytest.skip("Installed MLX predates BroadcastAxes export")
    assert len(nodes) == 2
    assert all(node.attrs["ignore_axes"] == [-2, -1] for node in nodes)
    rng = np.random.default_rng(7)
    for rows, batch in ((2, 5), (7, 3)):
        values = {"x": rng.normal(size=(rows, 3)).astype(np.float32),
                  "y": rng.normal(size=(batch, 3, 4)).astype(np.float32)}
        actual = run_aimodel_sync(converted.asset, values,
                                 specialization_options=SpecializationOptions.cpu_only()).outputs
        np.testing.assert_allclose(next(iter(actual.values())), values["x"] @ values["y"],
                                   rtol=2e-6, atol=2e-6)


def test_broadcast_axes_uses_lowered_rank_when_graph_inference_is_incomplete(tmp_path: Path):
    graph = Graph(
        [TensorSpec("x", (2, -1), "fp32"), TensorSpec("y", (5, 3, 4), "fp32")],
        [Node("transpose", ("x",), "transposed"),
         Node("broadcast_axes", ("transposed", "y"), "out", attrs={"ignore_axes": [-2, -1]})],
        ["out"],
    )
    assert infer_graph_specs(graph)["out"].shape is None
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=True))
    lowered.program.save_asset(tmp_path / "inferred_rank.aimodel")


@pytest.mark.parametrize("shapes,axes,cases", [
    ([(-1, -1, 3), (-1, 3, -1)], [-2, -1], [
        ((1, 2, 3), (5, 3, 4), (5, 2, 3)),
        ((4, 7, 3), (1, 3, 9), (4, 7, 3)),
    ]),
    ([(-1, 1, 3), (1, -1, 1)], [], [
        ((2, 1, 3), (1, 4, 1), (2, 4, 3)),
        ((5, 1, 3), (1, 7, 1), (5, 7, 3)),
    ]),
    ([(-1, 2, -1), (-1, -1, 3)], [-2], [
        ((1, 2, 3), (5, 7, 3), (5, 2, 3)),
        ((4, 2, 1), (1, 9, 3), (4, 2, 3)),
    ]),
])
@pytest.mark.skipif(platform.system() != "Darwin" or int(platform.mac_ver()[0].split(".")[0] or 0) < 27,
                    reason="requires the macOS 27 CoreAI runtime")
def test_dynamic_broadcast_axes_runtime(tmp_path: Path, shapes, axes, cases):
    from coreai.runtime import SpecializationOptions
    from mlx2coreai.runtime import run_aimodel_sync

    graph = broadcast_graph(shapes, axes)
    # A scalar output lets Python run this dynamic graph without outputViews.
    graph.nodes.append(Node("sum", ("out",), "total", attrs={"axes": [0, 1, 2], "keep_dims": False}))
    graph.outputs = ["total"]
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=True))
    asset = lowered.program.save_asset(tmp_path / "dynamic.aimodel")
    for x_shape, y_shape, expected_shape in cases:
        x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape) + 1
        result = run_aimodel_sync(asset, {"x0": x, "x1": np.ones(y_shape, dtype=np.float32)},
                                  specialization_options=SpecializationOptions.cpu_only())
        np.testing.assert_allclose(next(iter(result.outputs.values())), np.broadcast_to(x, expected_shape).sum())
