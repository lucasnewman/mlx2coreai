from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlx2coreai.conversion import ConversionConfig, convert_mlx_to_coreai, lower_graph_to_coreai
from mlx2coreai.ir import Graph, Node, StateSpec, TensorSpec


def _save(tmp_path: Path, name: str, graph: Graph, config: ConversionConfig | None = None) -> None:
    lowered = lower_graph_to_coreai(graph, config=config or ConversionConfig(optimize=False))
    asset_path = tmp_path / f"{name}.aimodel"
    lowered.program.save_asset(asset_path)
    assert (asset_path / "main.mlirb").exists()


@pytest.mark.parametrize(
    "op",
    [
        "add",
        "sub",
        "mul",
        "divide",
        "pow",
        "mod",
        "maximum",
        "minimum",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "equal",
        "not_equal",
    ],
)
def test_binary_op_assets(tmp_path: Path, op: str) -> None:
    graph = Graph(
        inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (2, 3), "fp32")],
        nodes=[Node(op, ("x", "y"), "out")],
        outputs=["out"],
    )
    _save(tmp_path, op, graph)


@pytest.mark.parametrize(
    "op",
    [
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "sigmoid",
        "silu",
        "gelu",
        "tanh",
        "sin",
        "cos",
        "erf",
        "negative",
        "abs",
    ],
)
def test_unary_op_assets(tmp_path: Path, op: str) -> None:
    graph = Graph(
        inputs=[TensorSpec("x", (2, 3), "fp32")],
        nodes=[Node(op, ("x",), "out")],
        outputs=["out"],
    )
    _save(tmp_path, op, graph)


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "prod", "logsumexp", "all", "any", "var", "std"])
def test_reduction_op_assets(tmp_path: Path, op: str) -> None:
    dtype = "bool" if op in {"all", "any"} else "fp32"
    graph = Graph(
        inputs=[TensorSpec("x", (2, 3), dtype)],
        nodes=[Node(op, ("x",), "out", attrs={"axes": [1], "keep_dims": False})],
        outputs=["out"],
    )
    _save(tmp_path, op, graph)


@pytest.mark.parametrize("op", ["argmax", "argmin"])
def test_arg_reduction_op_assets(tmp_path: Path, op: str) -> None:
    graph = Graph(
        inputs=[TensorSpec("x", (2, 3, 4), "fp32")],
        nodes=[Node(op, ("x",), "out", attrs={"axis": 2, "keep_dims": False})],
        outputs=["out"],
    )
    _save(tmp_path, op, graph)


@pytest.mark.parametrize(
    ("name", "graph"),
    [
        (
            "reshape",
            Graph(
                [TensorSpec("x", (2, 3), "fp32")],
                [Node("reshape", ("x",), "out", attrs={"shape": [3, 2]})],
                ["out"],
            ),
        ),
        (
            "transpose",
            Graph(
                [TensorSpec("x", (2, 3, 4), "fp32")],
                [Node("transpose", ("x",), "out", attrs={"perm": [0, 2, 1]})],
                ["out"],
            ),
        ),
        (
            "slice",
            Graph(
                [TensorSpec("x", (2, 4), "fp32")],
                [Node("slice", ("x",), "out", attrs={"begin": [0, 1], "end": [2, 3], "stride": [1, 1]})],
                ["out"],
            ),
        ),
        (
            "concat",
            Graph(
                [TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (2, 3), "fp32")],
                [Node("concatenate", ("x", "y"), "out", attrs={"axis": 1})],
                ["out"],
            ),
        ),
        (
            "broadcast_to",
            Graph(
                [TensorSpec("x", (1, 3), "fp32")],
                [Node("broadcast_to", ("x",), "out", attrs={"shape": [2, 3]})],
                ["out"],
            ),
        ),
        (
            "where",
            Graph(
                [
                    TensorSpec("c", (2, 3), "bool"),
                    TensorSpec("x", (2, 3), "fp32"),
                    TensorSpec("y", (2, 3), "fp32"),
                ],
                [Node("where", ("c", "x", "y"), "out")],
                ["out"],
            ),
        ),
        (
            "outer",
            Graph(
                [TensorSpec("x", (2,), "fp32"), TensorSpec("y", (3,), "fp32")],
                [Node("outer", ("x", "y"), "out")],
                ["out"],
            ),
        ),
        (
            "inner",
            Graph(
                [TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (2, 3), "fp32")],
                [Node("inner", ("x", "y"), "out")],
                ["out"],
            ),
        ),
        (
            "tensordot",
            Graph(
                [TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (3, 4), "fp32")],
                [Node("tensordot", ("x", "y"), "out", attrs={"axes": 1})],
                ["out"],
            ),
        ),
        (
            "logaddexp",
            Graph(
                [TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (2, 3), "fp32")],
                [Node("logaddexp", ("x", "y"), "out")],
                ["out"],
            ),
        ),
        (
            "isclose",
            Graph(
                [TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (2, 3), "fp32")],
                [Node("isclose", ("x", "y"), "out")],
                ["out"],
            ),
        ),
        (
            "allclose",
            Graph(
                [TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (2, 3), "fp32")],
                [Node("allclose", ("x", "y"), "out")],
                ["out"],
            ),
        ),
        (
            "nan_to_num",
            Graph(
                [TensorSpec("x", (2, 3), "fp32")],
                [Node("nan_to_num", ("x",), "out")],
                ["out"],
            ),
        ),
        (
            "moveaxis_take",
            Graph(
                [
                    TensorSpec("x", (2, 3, 4), "fp32"),
                    TensorSpec("idx", (2,), "int32"),
                ],
                [
                    Node("moveaxis", ("x",), "mx", attrs={"source": 2, "destination": 0}),
                    Node("take", ("mx", "idx"), "out", attrs={"axis": 1}),
                ],
                ["out"],
            ),
        ),
        (
            "diag_vector",
            Graph(
                [TensorSpec("x", (4,), "fp32")],
                [Node("diag", ("x",), "out", attrs={"k": 1})],
                ["out"],
            ),
        ),
        (
            "diagonal_trace",
            Graph(
                [TensorSpec("x", (2, 3, 4), "fp32")],
                [
                    Node("diagonal", ("x",), "d", attrs={"axis1": 1, "axis2": 2, "offset": 1}),
                    Node("trace", ("x",), "t", attrs={"axis1": 1, "axis2": 2}),
                    Node("sum", ("d",), "ds", attrs={"axes": [0, 1], "keep_dims": False}),
                    Node("sum", ("t",), "ts", attrs={"axes": [0], "keep_dims": False}),
                    Node("add", ("ds", "ts"), "out"),
                ],
                ["out"],
            ),
        ),
        (
            "tri_eye",
            Graph(
                [TensorSpec("x", (2, 4, 5), "fp32")],
                [
                    Node("tril", ("x",), "l"),
                    Node("triu", ("x",), "u", attrs={"k": 1}),
                    Node("tri", tuple(), "t", attrs={"n": 4, "m": 5, "k": -1}),
                    Node("eye", tuple(), "e", attrs={"n": 4, "m": 5, "k": 1}),
                    Node("sum", ("l",), "ls", attrs={"axes": [0, 1, 2], "keep_dims": False}),
                    Node("sum", ("u",), "us", attrs={"axes": [0, 1, 2], "keep_dims": False}),
                    Node("sum", ("t",), "ts", attrs={"axes": [0, 1], "keep_dims": False}),
                    Node("sum", ("e",), "es", attrs={"axes": [0, 1], "keep_dims": False}),
                    Node("add", ("ls", "us"), "a"),
                    Node("add", ("ts", "es"), "b"),
                    Node("add", ("a", "b"), "out"),
                ],
                ["out"],
            ),
        ),
        (
            "meshgrid_kron",
            Graph(
                [
                    TensorSpec("x", (3,), "fp32"),
                    TensorSpec("y", (2,), "fp32"),
                    TensorSpec("a", (2, 2), "fp32"),
                    TensorSpec("b", (2, 1), "fp32"),
                ],
                [
                    Node("meshgrid", ("x", "y"), "gx", attrs={"input_index": 0, "indexing": "xy"}),
                    Node("meshgrid", ("x", "y"), "gy", attrs={"input_index": 1, "indexing": "xy"}),
                    Node("kron", ("a", "b"), "k"),
                    Node("sum", ("gx",), "gxs", attrs={"axes": [0, 1], "keep_dims": False}),
                    Node("sum", ("gy",), "gys", attrs={"axes": [0, 1], "keep_dims": False}),
                    Node("sum", ("k",), "ks", attrs={"axes": [0, 1], "keep_dims": False}),
                    Node("add", ("gxs", "gys"), "s"),
                    Node("add", ("s", "ks"), "out"),
                ],
                ["out"],
            ),
        ),
        (
            "conv_block",
            Graph(
                [
                    TensorSpec("x", (1, 2, 4, 4), "fp32"),
                    TensorSpec("w", (3, 2, 1, 1), "fp32"),
                    TensorSpec("b", (3,), "fp32"),
                    TensorSpec("wt", (3, 2, 1, 1), "fp32"),
                    TensorSpec("bt", (2,), "fp32"),
                ],
                [
                    Node("conv2d", ("x", "w", "b"), "c", attrs={"pad_type": "valid"}),
                    Node("conv_transpose2d", ("c", "wt", "bt"), "out", attrs={"pad_type": "valid"}),
                ],
                ["out"],
            ),
        ),
    ],
)
def test_shape_and_select_assets(tmp_path: Path, name: str, graph: Graph) -> None:
    _save(tmp_path, name, graph)


def test_mutable_buffer_metadata_is_emitted() -> None:
    graph = Graph(
        inputs=[TensorSpec("cache", (2, 3), "fp32"), TensorSpec("v", (2, 3), "fp32")],
        nodes=[Node("write_state", ("cache", "v"), "cache_out")],
        outputs=["cache_out"],
    )
    lowered = lower_graph_to_coreai(
        graph,
        config=ConversionConfig(
            optimize=False,
            state_specs=[StateSpec("cache", (2, 3), "fp32")],
        ),
    )
    assert 'MutableBuffers.buffer_mutation = "cache_out"' in str(lowered.program)


def test_take_supports_rank2_embedding_indices(tmp_path: Path) -> None:
    graph = Graph(
        inputs=[
            TensorSpec("embedding", (32, 8), "fp32"),
            TensorSpec("input_ids", (1, 16), "int32"),
        ],
        nodes=[Node("take", ("embedding", "input_ids"), "out", attrs={"axis": 0})],
        outputs=["out"],
    )
    _save(tmp_path, "rank2_embedding_take", graph)


def test_callback_gather_preserves_slice_shape_for_embedding(tmp_path: Path) -> None:
    graph = Graph(
        inputs=[
            TensorSpec("embedding", (32, 8), "fp32"),
            TensorSpec("input_ids", (1, 16), "int32"),
        ],
        nodes=[
            Node(
                "gather",
                ("embedding", "input_ids"),
                "gathered",
                attrs={"axis": 0, "slice_shape": [1, 8], "shape": [1, 16, 1, 8]},
            ),
            Node("squeeze", ("gathered",), "out", attrs={"axes": [2]}),
        ],
        outputs=["out"],
    )
    _save(tmp_path, "callback_embedding_gather", graph)


def test_toy_transformer_like_mlx_capture_saves(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(0)
    batch, seq, dim, heads = 1, 4, 8, 2
    wq = mx.array(rng.standard_normal((dim, dim)).astype(np.float32) * 0.01)
    wk = mx.array(rng.standard_normal((dim, dim)).astype(np.float32) * 0.01)
    wv = mx.array(rng.standard_normal((dim, dim)).astype(np.float32) * 0.01)
    wo = mx.array(rng.standard_normal((dim, dim)).astype(np.float32) * 0.01)
    w1 = mx.array(rng.standard_normal((dim, 16)).astype(np.float32) * 0.01)
    w2 = mx.array(rng.standard_normal((16, dim)).astype(np.float32) * 0.01)
    scale = mx.ones((dim,), dtype=mx.float32)

    def rms(x):
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5) * scale

    def block(x):
        h = rms(x)
        q = mx.reshape(mx.matmul(h, wq), (batch, seq, heads, dim // heads)).transpose(0, 2, 1, 3)
        k = mx.reshape(mx.matmul(h, wk), (batch, seq, heads, dim // heads)).transpose(0, 2, 1, 3)
        v = mx.reshape(mx.matmul(h, wv), (batch, seq, heads, dim // heads)).transpose(0, 2, 1, 3)
        scores = mx.matmul(q, mx.swapaxes(k, -1, -2)) / np.float32(np.sqrt(dim // heads))
        att = mx.softmax(scores, axis=-1)
        y = mx.matmul(att, v).transpose(0, 2, 1, 3).reshape(batch, seq, dim)
        y = mx.matmul(y, wo)
        m = mx.maximum(mx.matmul(h, w1), 0)
        m = mx.matmul(m, w2)
        return x + y + m

    converted = convert_mlx_to_coreai(
        block,
        {"x": rng.standard_normal((batch, seq, dim)).astype(np.float32)},
        config=ConversionConfig(optimize=False),
        output_path=tmp_path / "toy_block.aimodel",
    )
    assert (tmp_path / "toy_block.aimodel" / "main.mlirb").exists()
    assert len(converted.prepared.normalized_graph.nodes) > 20


def test_mlx_fast_rmsnorm_and_rope_emit_composites(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(0)

    def rms(x, w):
        return mx.fast.rms_norm(x, w, 1e-5)

    def rope(x):
        return mx.fast.rope(x, dims=4, traditional=False, base=1000000.0, scale=1.0, offset=0)

    rms_model = convert_mlx_to_coreai(
        rms,
        {"x": rng.standard_normal((2, 4)).astype(np.float32), "w": np.ones((4,), np.float32)},
        config=ConversionConfig(optimize=False),
        output_path=tmp_path / "rms.aimodel",
    )
    rope_model = convert_mlx_to_coreai(
        rope,
        {"x": rng.standard_normal((1, 2, 3, 4)).astype(np.float32)},
        config=ConversionConfig(optimize=False),
        output_path=tmp_path / "rope.aimodel",
    )
    assert 'composite_declaration<"rms_norm"' in str(rms_model.program)
    assert 'composite_declaration<"rope"' in str(rope_model.program)
    rope_node = next(node for node in rope_model.prepared.normalized_graph.nodes if node.op == "rope")
    assert rope_node.attrs["dims"] == 4
    assert rope_node.attrs["traditional"] is False
    assert rope_node.attrs["base"] == 1000000.0
