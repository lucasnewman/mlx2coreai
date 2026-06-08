from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from mlx2coreai.from_mlx import capture_graph_from_mlx_function, export_dot_from_ir
from mlx2coreai.ir import Graph, Node, TensorSpec


@dataclass(frozen=True)
class ZooModelSpec:
    name: str
    description: str
    graph: Graph
    inputs: dict[str, np.ndarray]
    expected: dict[str, np.ndarray]
    atol: float = 2e-3
    rtol: float = 5e-3


def _build_smoke_numpy_inputs(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 3), dtype=np.float32)
    w = rng.standard_normal((3, 4), dtype=np.float32)
    b = rng.standard_normal((2, 4), dtype=np.float32)
    z = np.zeros((2, 4), dtype=np.float32)
    return {"x": x, "w": w, "b": b, "z": z}


def _make_mock_smoke_graph() -> Graph:
    graph = Graph(
        inputs=[
            TensorSpec(name="x", shape=(2, 3), dtype="fp32"),
            TensorSpec(name="w", shape=(3, 4), dtype="fp32"),
            TensorSpec(name="b", shape=(2, 4), dtype="fp32"),
            TensorSpec(name="z", shape=(2, 4), dtype="fp32"),
        ],
        nodes=[
            Node(op="matmul", inputs=("x", "w"), output="m0"),
            Node(op="add", inputs=("m0", "b"), output="a0"),
            Node(op="maximum", inputs=("a0", "z"), output="out"),
        ],
        outputs=["out"],
    )
    graph.validate()
    return graph


def _build_linear_relu(seed: int) -> ZooModelSpec:
    del seed  # deterministic fixture from existing helper
    graph = _make_mock_smoke_graph()
    inputs = _build_smoke_numpy_inputs(seed=0)
    expected = {"out": np.maximum((inputs["x"] @ inputs["w"]) + inputs["b"], inputs["z"])}
    return ZooModelSpec(
        name="linear_relu",
        description="Baseline linear + relu-like maximum graph",
        graph=graph,
        inputs=inputs,
        expected=expected,
    )


def _build_arithmetic_chain(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((2, 4), dtype=np.float32)
    b = rng.standard_normal((2, 4), dtype=np.float32)
    c = rng.uniform(0.5, 1.5, size=(2, 4)).astype(np.float32)
    d = rng.uniform(0.5, 1.5, size=(2, 4)).astype(np.float32)
    e = np.full((2, 4), 2.0, dtype=np.float32)
    f = rng.uniform(0.5, 2.0, size=(2, 4)).astype(np.float32)
    g = rng.uniform(0.5, 1.5, size=(2, 4)).astype(np.float32)
    inputs = {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "g": g}

    graph = Graph(
        inputs=[
            TensorSpec("a", (2, 4), "fp32"),
            TensorSpec("b", (2, 4), "fp32"),
            TensorSpec("c", (2, 4), "fp32"),
            TensorSpec("d", (2, 4), "fp32"),
            TensorSpec("e", (2, 4), "fp32"),
            TensorSpec("f", (2, 4), "fp32"),
            TensorSpec("g", (2, 4), "fp32"),
        ],
        nodes=[
            Node("subtract", ("a", "b"), "t0"),
            Node("multiply", ("t0", "c"), "t1"),
            Node("divide", ("t1", "d"), "t2"),
            Node("power", ("t2", "e"), "t3"),
            Node("remainder", ("t3", "g"), "t4"),
            Node("reciprocal", ("f",), "t5"),
            Node("add", ("t4", "t5"), "out"),
        ],
        outputs=["out"],
    )
    graph.validate()

    t0 = a - b
    t1 = t0 * c
    t2 = t1 / d
    t3 = np.power(t2, e)
    t4 = np.mod(t3, g)
    t5 = np.reciprocal(f)
    expected = {"out": (t4 + t5).astype(np.float32)}
    return ZooModelSpec(
        name="arithmetic_chain",
        description="Arithmetic aliases: sub/mul/div/pow/mod/inverse/add",
        graph=graph,
        inputs=inputs,
        expected=expected,
        atol=2e-2,
        rtol=2e-2,
    )


def _build_reduction_suite(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 3, 4), dtype=np.float32)
    inputs = {"x": x}

    graph = Graph(
        inputs=[TensorSpec("x", (2, 3, 4), "fp32")],
        nodes=[
            Node("sum", ("x",), "sum_out", attrs={"axes": [2], "keep_dims": False}),
            Node("mean", ("x",), "mean_out", attrs={"axes": [1], "keep_dims": False}),
            Node("min", ("x",), "min_out", attrs={"axes": [2], "keep_dims": False}),
            Node("max", ("x",), "max_out", attrs={"axes": [2], "keep_dims": False}),
            Node("prod", ("x",), "prod_out", attrs={"axes": [2], "keep_dims": False}),
            Node("argmax", ("x",), "argmax_out", attrs={"axis": 2, "keep_dims": False}),
            Node("argmin", ("x",), "argmin_out", attrs={"axis": 1, "keep_dims": True}),
        ],
        outputs=[
            "sum_out",
            "mean_out",
            "min_out",
            "max_out",
            "prod_out",
            "argmax_out",
            "argmin_out",
        ],
    )
    graph.validate()

    expected = {
        "sum_out": np.sum(x, axis=2),
        "mean_out": np.mean(x, axis=1),
        "min_out": np.min(x, axis=2),
        "max_out": np.max(x, axis=2),
        "prod_out": np.prod(x, axis=2),
        "argmax_out": np.argmax(x, axis=2).astype(np.int32),
        "argmin_out": np.expand_dims(np.argmin(x, axis=1).astype(np.int32), axis=1),
    }
    return ZooModelSpec(
        name="reduction_suite",
        description="Reduction ops: sum/mean/min/max/prod/argmax/argmin",
        graph=graph,
        inputs=inputs,
        expected=expected,
    )


def _build_shape_helpers(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((4,), dtype=np.float32)
    m = rng.standard_normal((2, 3), dtype=np.float32)
    t = rng.standard_normal((2, 3, 4), dtype=np.float32)
    inputs = {"v": v, "m": m, "t": t}

    graph = Graph(
        inputs=[
            TensorSpec("v", (4,), "fp32"),
            TensorSpec("m", (2, 3), "fp32"),
            TensorSpec("t", (2, 3, 4), "fp32"),
        ],
        nodes=[
            Node("flatten", ("t",), "flat", attrs={"shape": [24]}),
            Node("unflatten", ("flat",), "restored", attrs={"shape": [2, 3, 4]}),
            Node("atleast_1d", ("v",), "v1"),
            Node("atleast_2d", ("v",), "v2"),
            Node("atleast_3d", ("m",), "m3"),
        ],
        outputs=["flat", "restored", "v1", "v2", "m3"],
    )
    graph.validate()

    expected = {
        "flat": t.reshape(24),
        "restored": t.reshape(24).reshape(2, 3, 4),
        "v1": np.atleast_1d(v),
        "v2": np.atleast_2d(v),
        "m3": np.atleast_3d(m),
    }
    return ZooModelSpec(
        name="shape_helpers",
        description="Shape transforms: flatten/unflatten/atleast_*",
        graph=graph,
        inputs=inputs,
        expected=expected,
    )


def _build_indexing_transforms(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 3, 4), dtype=np.float32)
    indices = np.array([2, 0], dtype=np.int32)
    indices_along = np.array(
        [[[0, 2], [1, 3], [0, 1]], [[2, 3], [0, 2], [1, 3]]], dtype=np.int32
    )
    inputs = {"x": x, "indices": indices, "indices_along": indices_along}

    graph = Graph(
        inputs=[
            TensorSpec("x", (2, 3, 4), "fp32"),
            TensorSpec("indices", (2,), "int32"),
            TensorSpec("indices_along", (2, 3, 2), "int32"),
        ],
        nodes=[
            Node("moveaxis", ("x",), "mx", attrs={"source": 2, "destination": 0}),
            Node("swapaxes", ("mx",), "sw", attrs={"axis1": 1, "axis2": 2}),
            Node(
                "slice",
                ("sw",),
                "sl",
                attrs={"begin": [0, 0, 0], "end": [4, 2, 2], "stride": [1, 1, 1]},
            ),
            Node("take", ("sw", "indices"), "tk", attrs={"axis": 1}),
            Node("take_along_axis", ("x", "indices_along"), "tka", attrs={"axis": 2}),
        ],
        outputs=["sl", "tk", "tka"],
    )
    graph.validate()

    moved = np.moveaxis(x, 2, 0)
    swapped = np.swapaxes(moved, 1, 2)
    expected = {
        "sl": swapped[0:4:1, 0:2:1, 0:2:1],
        "tk": np.take(swapped, indices, axis=1),
        "tka": np.take_along_axis(x, indices_along, axis=2),
    }
    return ZooModelSpec(
        name="indexing_transforms",
        description="Axis/index transforms: moveaxis/swapaxes/slice/take/take_along_axis",
        graph=graph,
        inputs=inputs,
        expected=expected,
    )


def _build_creation_helpers(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 3), dtype=np.float32)
    condf = (rng.random((2, 3)) > 0.4).astype(np.float32)
    inputs = {"x": x, "condf": condf}

    graph = Graph(
        inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("condf", (2, 3), "fp32")],
        nodes=[
            Node("zeros", tuple(), "z", attrs={"shape": [2, 3], "value": 0.0}),
            Node("ones", tuple(), "o", attrs={"shape": [2, 3], "value": 1.0}),
            Node("full", tuple(), "f", attrs={"shape": [2, 3], "value": 2.5}),
            Node("zeros_like", ("x",), "zl"),
            Node("ones_like", ("x",), "ol"),
            Node("full_like", ("x",), "fl", attrs={"value": -1.5}),
            Node("arange", tuple(), "ar", attrs={"start": 0, "end": 6, "step": 1}),
            Node(
                "linspace",
                tuple(),
                "ls",
                attrs={"start": 0.0, "stop": 1.0, "num": 6, "endpoint": True, "dtype": "fp32"},
            ),
            Node("astype", ("x",), "xi", attrs={"dtype": "int32"}),
            Node("astype", ("condf",), "condb", attrs={"dtype": "bool"}),
            Node("where", ("condb", "o", "z"), "w"),
            Node("number_of_elements", ("x",), "n"),
            Node("stop_gradient", ("f",), "sg"),
        ],
        outputs=["z", "o", "zl", "ol", "fl", "ar", "ls", "xi", "w", "n", "sg"],
    )
    graph.validate()

    expected = {
        "z": np.zeros((2, 3), dtype=np.float32),
        "o": np.ones((2, 3), dtype=np.float32),
        "zl": np.zeros_like(x),
        "ol": np.ones_like(x),
        "fl": np.full_like(x, -1.5),
        "ar": np.arange(0, 6, 1, dtype=np.int32),
        "ls": np.linspace(0.0, 1.0, num=6, endpoint=True, dtype=np.float32),
        "xi": x.astype(np.int32),
        "w": np.where(condf.astype(bool), np.ones((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)),
        "n": np.array(np.prod(x.shape), dtype=np.int32),
        "sg": np.full((2, 3), 2.5, dtype=np.float32),
    }
    return ZooModelSpec(
        name="creation_helpers",
        description=(
            "Creation/helpers: zeros/ones/full/*_like/arange/linspace/"
            "where/astype/number_of_elements/stop_gradient"
        ),
        graph=graph,
        inputs=inputs,
        expected=expected,
    )


def _build_mlp_2layer(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 4), dtype=np.float32)
    w1 = rng.standard_normal((4, 6), dtype=np.float32)
    b1 = rng.standard_normal((2, 6), dtype=np.float32)
    z1 = np.zeros((2, 6), dtype=np.float32)
    w2 = rng.standard_normal((6, 3), dtype=np.float32)
    b2 = rng.standard_normal((2, 3), dtype=np.float32)
    inputs = {"x": x, "w1": w1, "b1": b1, "z1": z1, "w2": w2, "b2": b2}

    graph = Graph(
        inputs=[
            TensorSpec("x", (2, 4), "fp32"),
            TensorSpec("w1", (4, 6), "fp32"),
            TensorSpec("b1", (2, 6), "fp32"),
            TensorSpec("z1", (2, 6), "fp32"),
            TensorSpec("w2", (6, 3), "fp32"),
            TensorSpec("b2", (2, 3), "fp32"),
        ],
        nodes=[
            Node("addmm", ("b1", "x", "w1"), "h1_linear"),
            Node("maximum", ("h1_linear", "z1"), "h1"),
            Node("addmm", ("b2", "h1", "w2"), "out"),
        ],
        outputs=["out"],
    )
    graph.validate()

    h1_linear = b1 + (x @ w1)
    h1 = np.maximum(h1_linear, z1)
    expected = {"out": (b2 + (h1 @ w2)).astype(np.float32)}
    return ZooModelSpec(
        name="mlp_2layer",
        description="Linear block: addmm + activation + addmm",
        graph=graph,
        inputs=inputs,
        expected=expected,
    )


def _build_broadcast_tensordot(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((2, 1), dtype=np.float32)
    b = rng.standard_normal((1, 3), dtype=np.float32)
    x = rng.standard_normal((2, 3, 4), dtype=np.float32)
    y = rng.standard_normal((4, 5), dtype=np.float32)
    inputs = {"a": a, "b": b, "x": x, "y": y}

    graph = Graph(
        inputs=[
            TensorSpec("a", (2, 1), "fp32"),
            TensorSpec("b", (1, 3), "fp32"),
            TensorSpec("x", (2, 3, 4), "fp32"),
            TensorSpec("y", (4, 5), "fp32"),
        ],
        nodes=[
            Node("broadcast_arrays", ("a", "b"), "a_b", attrs={"input_index": 0}),
            Node("broadcast_arrays", ("a", "b"), "b_b", attrs={"input_index": 1}),
            Node("add", ("a_b", "b_b"), "bc_sum"),
            Node("tensordot", ("x", "y"), "td", attrs={"axes": 1}),
            Node("sum", ("td",), "td_sum", attrs={"axes": [2], "keep_dims": False}),
            Node("add", ("bc_sum", "td_sum"), "out"),
        ],
        outputs=["out"],
    )
    graph.validate()

    a_b, b_b = np.broadcast_arrays(a, b)
    bc_sum = a_b + b_b
    td = np.tensordot(x, y, axes=1)
    td_sum = np.sum(td, axis=2)
    expected = {"out": (bc_sum + td_sum).astype(np.float32)}
    return ZooModelSpec(
        name="broadcast_tensordot",
        description="broadcast_arrays + tensordot composite graph",
        graph=graph,
        inputs=inputs,
        expected=expected,
        atol=3e-3,
        rtol=6e-3,
    )


def _build_numeric_sanity(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 3), dtype=np.float32)
    y = x.copy()
    x[0, 0] = np.nan
    x[1, 1] = np.inf
    x[1, 2] = -np.inf
    y[0, 0] = np.nan
    y[1, 1] = np.inf
    y[1, 2] = np.float32(-10.0)
    inputs = {"x": x, "y": y}

    graph = Graph(
        inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (2, 3), "fp32")],
        nodes=[
            Node("isclose", ("x", "y"), "c", attrs={"rtol": 1e-4, "atol": 1e-6, "equal_nan": True}),
            Node("allclose", ("x", "y"), "ac", attrs={"rtol": 1e-4, "atol": 1e-6, "equal_nan": True}),
            Node("nan_to_num", ("x",), "xn", attrs={"nan": 0.5, "posinf": 9.0, "neginf": -9.0}),
            Node("cast", ("c",), "ci", attrs={"dtype": "int32"}),
            Node("sum", ("ci",), "c_count", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("cast", ("ac",), "ac_i", attrs={"dtype": "int32"}),
            Node("add", ("c_count", "ac_i"), "score"),
        ],
        outputs=["xn", "score"],
    )
    graph.validate()

    c = np.isclose(x, y, rtol=1e-4, atol=1e-6, equal_nan=True)
    c_count = np.array(np.sum(c.astype(np.int32)), dtype=np.int32)
    ac = np.allclose(x, y, rtol=1e-4, atol=1e-6, equal_nan=True)
    ac_i = np.array(ac, dtype=np.int32)
    score = np.array(c_count + ac_i, dtype=np.int32)
    expected = {
        "xn": np.nan_to_num(x, nan=0.5, posinf=9.0, neginf=-9.0).astype(np.float32),
        "score": score,
    }
    return ZooModelSpec(
        name="numeric_sanity",
        description="Numeric helpers: isclose/allclose/nan_to_num with NaN/Inf edge cases",
        graph=graph,
        inputs=inputs,
        expected=expected,
    )


def _build_diagonal_trace(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((4,), dtype=np.float32)
    m = rng.standard_normal((4, 4), dtype=np.float32)
    t = rng.standard_normal((2, 3, 4), dtype=np.float32)
    inputs = {"v": v, "m": m, "t": t}

    graph = Graph(
        inputs=[
            TensorSpec("v", (4,), "fp32"),
            TensorSpec("m", (4, 4), "fp32"),
            TensorSpec("t", (2, 3, 4), "fp32"),
        ],
        nodes=[
            Node("diag", ("v",), "dv", attrs={"k": 1}),
            Node("diag", ("m",), "dm", attrs={"k": -1}),
            Node("diagonal", ("t",), "dt", attrs={"axis1": 1, "axis2": 2, "offset": 1}),
            Node("trace", ("t",), "tr", attrs={"axis1": 1, "axis2": 2, "offset": 0}),
            Node("sum", ("dv",), "dv_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("dm",), "dm_sum", attrs={"axes": [0], "keep_dims": False}),
            Node("sum", ("dt",), "dt_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("tr",), "tr_sum", attrs={"axes": [0], "keep_dims": False}),
            Node("add", ("dv_sum", "dm_sum"), "s0"),
            Node("add", ("dt_sum", "tr_sum"), "s1"),
            Node("add", ("s0", "s1"), "out"),
        ],
        outputs=["out"],
    )
    graph.validate()

    dv = np.diag(v, k=1)
    dm = np.diag(m, k=-1)
    dt = np.diagonal(t, offset=1, axis1=1, axis2=2)
    tr = np.trace(t, offset=0, axis1=1, axis2=2)
    expected = {"out": np.array(dv.sum() + dm.sum() + dt.sum() + tr.sum(), dtype=np.float32)}
    return ZooModelSpec(
        name="diagonal_trace",
        description="diag/diagonal/trace helpers with reduction composition",
        graph=graph,
        inputs=inputs,
        expected=expected,
        atol=3e-3,
        rtol=6e-3,
    )


def _build_tri_band(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 4, 5), dtype=np.float32)
    m = rng.standard_normal((4, 5), dtype=np.float32)
    inputs = {"x": x, "m": m}

    graph = Graph(
        inputs=[TensorSpec("x", (2, 4, 5), "fp32"), TensorSpec("m", (4, 5), "fp32")],
        nodes=[
            Node("tri", tuple(), "t", attrs={"n": 4, "m": 5, "k": -1, "dtype": "fp32"}),
            Node("tril", ("x",), "xl", attrs={"k": 0}),
            Node("triu", ("x",), "xu", attrs={"k": 1}),
            Node("tril", ("m",), "ml", attrs={"k": -1}),
            Node("triu", ("m",), "mu", attrs={"k": 0}),
            Node("sum", ("t",), "t_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("xl",), "xl_sum", attrs={"axes": [0, 1, 2], "keep_dims": False}),
            Node("sum", ("xu",), "xu_sum", attrs={"axes": [0, 1, 2], "keep_dims": False}),
            Node("sum", ("ml",), "ml_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("mu",), "mu_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("add", ("t_sum", "xl_sum"), "s0"),
            Node("add", ("xu_sum", "ml_sum"), "s1"),
            Node("add", ("s0", "s1"), "s2"),
            Node("add", ("s2", "mu_sum"), "out"),
        ],
        outputs=["out"],
    )
    graph.validate()

    t = np.tri(4, 5, k=-1, dtype=np.float32)
    xl = np.tril(x, k=0)
    xu = np.triu(x, k=1)
    ml = np.tril(m, k=-1)
    mu = np.triu(m, k=0)
    expected = {
        "out": np.array(
            t.sum() + xl.sum() + xu.sum() + ml.sum() + mu.sum(),
            dtype=np.float32,
        )
    }
    return ZooModelSpec(
        name="tri_band",
        description="tri/tril/triu helpers over matrix and batched-matrix inputs",
        graph=graph,
        inputs=inputs,
        expected=expected,
        atol=3e-3,
        rtol=6e-3,
    )


def _build_logical_checks(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 3), dtype=np.float32)
    x[0, 0] = np.nan
    x[0, 1] = np.inf
    x[1, 2] = -np.inf
    y = x.copy()
    y[1, 1] = y[1, 1] + np.float32(1.0)

    i = np.array([[1, 0, 2], [0, 0, 0]], dtype=np.int32)
    j = i.copy()
    j[1, 2] = 3
    inputs = {"x": x, "y": y, "i": i, "j": j}

    graph = Graph(
        inputs=[
            TensorSpec("x", (2, 3), "fp32"),
            TensorSpec("y", (2, 3), "fp32"),
            TensorSpec("i", (2, 3), "int32"),
            TensorSpec("j", (2, 3), "int32"),
        ],
        nodes=[
            Node("isnan", ("x",), "n"),
            Node("isinf", ("x",), "inf"),
            Node("isfinite", ("x",), "fin"),
            Node("isneginf", ("x",), "ninf"),
            Node("isposinf", ("x",), "pinf"),
            Node("all", ("i",), "all_i"),
            Node("any", ("i",), "any_i"),
            Node("array_equal", ("i", "j"), "eq_ij"),
            Node("cast", ("n",), "n_i", attrs={"dtype": "int32"}),
            Node("cast", ("inf",), "inf_i", attrs={"dtype": "int32"}),
            Node("cast", ("fin",), "fin_i", attrs={"dtype": "int32"}),
            Node("cast", ("ninf",), "ninf_i", attrs={"dtype": "int32"}),
            Node("cast", ("pinf",), "pinf_i", attrs={"dtype": "int32"}),
            Node("sum", ("n_i",), "n_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("inf_i",), "inf_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("fin_i",), "fin_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("ninf_i",), "ninf_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("pinf_i",), "pinf_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("cast", ("all_i",), "all_i_i", attrs={"dtype": "int32"}),
            Node("cast", ("any_i",), "any_i_i", attrs={"dtype": "int32"}),
            Node("cast", ("eq_ij",), "eq_ij_i", attrs={"dtype": "int32"}),
            Node("add", ("n_sum", "inf_sum"), "s0"),
            Node("add", ("fin_sum", "ninf_sum"), "s1"),
            Node("add", ("pinf_sum", "all_i_i"), "s2"),
            Node("add", ("any_i_i", "eq_ij_i"), "s3"),
            Node("add", ("s0", "s1"), "s4"),
            Node("add", ("s2", "s3"), "s5"),
            Node("add", ("s4", "s5"), "out"),
        ],
        outputs=["out"],
    )
    graph.validate()

    n = np.isnan(x)
    inf = np.isinf(x)
    fin = np.isfinite(x)
    ninf = np.isneginf(x)
    pinf = np.isposinf(x)
    all_i = np.all(i)
    any_i = np.any(i)
    eq_ij = np.array_equal(i, j)

    expected = {
        "out": np.array(
            n.astype(np.int32).sum()
            + inf.astype(np.int32).sum()
            + fin.astype(np.int32).sum()
            + ninf.astype(np.int32).sum()
            + pinf.astype(np.int32).sum()
            + np.array(all_i, dtype=np.int32)
            + np.array(any_i, dtype=np.int32)
            + np.array(eq_ij, dtype=np.int32),
            dtype=np.int32,
        )
    }
    return ZooModelSpec(
        name="logical_checks",
        description="Logical/finite checks: all/any/array_equal/isnan/isinf/isfinite/isneginf/isposinf",
        graph=graph,
        inputs=inputs,
        expected=expected,
    )


def _build_meshgrid_kron(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((3,), dtype=np.float32)
    y = rng.standard_normal((2,), dtype=np.float32)
    a = rng.standard_normal((2, 2), dtype=np.float32)
    b = rng.standard_normal((2, 1), dtype=np.float32)
    p = rng.standard_normal((2, 3), dtype=np.float32)
    q = rng.standard_normal((2, 3), dtype=np.float32)
    inputs = {"x": x, "y": y, "a": a, "b": b, "p": p, "q": q}

    graph = Graph(
        inputs=[
            TensorSpec("x", (3,), "fp32"),
            TensorSpec("y", (2,), "fp32"),
            TensorSpec("a", (2, 2), "fp32"),
            TensorSpec("b", (2, 1), "fp32"),
            TensorSpec("p", (2, 3), "fp32"),
            TensorSpec("q", (2, 3), "fp32"),
        ],
        nodes=[
            Node("meshgrid", ("x", "y"), "gx", attrs={"input_index": 0, "indexing": "xy"}),
            Node("meshgrid", ("x", "y"), "gy", attrs={"input_index": 1, "indexing": "xy"}),
            Node("kron", ("a", "b"), "k"),
            Node("eye", tuple(), "e", attrs={"n": 3, "m": 4, "k": 1, "dtype": "fp32"}),
            Node("logaddexp", ("p", "q"), "l"),
            Node("sum", ("gx",), "gx_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("gy",), "gy_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("k",), "k_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("e",), "e_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("l",), "l_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("add", ("gx_sum", "gy_sum"), "s0"),
            Node("add", ("k_sum", "e_sum"), "s1"),
            Node("add", ("s0", "s1"), "s2"),
            Node("add", ("s2", "l_sum"), "out"),
        ],
        outputs=["out"],
    )
    graph.validate()

    gx, gy = np.meshgrid(x, y, indexing="xy")
    k = np.kron(a, b)
    e = np.eye(3, 4, k=1, dtype=np.float32)
    l = np.logaddexp(p, q)
    expected = {
        "out": np.array(gx.sum() + gy.sum() + k.sum() + e.sum() + l.sum(), dtype=np.float32)
    }
    return ZooModelSpec(
        name="meshgrid_kron",
        description="meshgrid/kron/eye/logaddexp composite coverage",
        graph=graph,
        inputs=inputs,
        expected=expected,
        atol=4e-3,
        rtol=8e-3,
    )


def _build_p0_math_pack(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.8, 0.8, size=(2, 3)).astype(np.float32)
    p = rng.uniform(0.1, 3.0, size=(2, 3)).astype(np.float32)
    c0 = rng.standard_normal((2, 2), dtype=np.float32)
    c1 = rng.standard_normal((2, 1), dtype=np.float32)
    ia = rng.integers(1, 20, size=(2, 3), dtype=np.int32)
    ib = rng.integers(1, 6, size=(2, 3), dtype=np.int32)
    inputs = {"u": u, "p": p, "c0": c0, "c1": c1, "ia": ia, "ib": ib}

    graph = Graph(
        inputs=[
            TensorSpec("u", (2, 3), "fp32"),
            TensorSpec("p", (2, 3), "fp32"),
            TensorSpec("c0", (2, 2), "fp32"),
            TensorSpec("c1", (2, 1), "fp32"),
            TensorSpec("ia", (2, 3), "int32"),
            TensorSpec("ib", (2, 3), "int32"),
        ],
        nodes=[
            Node("arccos", ("u",), "acos"),
            Node("arcsin", ("u",), "asin"),
            Node("arctan", ("u",), "atan"),
            Node("arctanh", ("u",), "atanh"),
            Node("negative", ("u",), "neg"),
            Node("degrees", ("u",), "deg"),
            Node("radians", ("deg",), "rad"),
            Node("expm1", ("u",), "ex1"),
            Node("log1p", ("p",), "l1p"),
            Node("log2", ("p",), "l2"),
            Node("log10", ("p",), "l10"),
            Node("floor_divide", ("ia", "ib"), "fd"),
            Node("cast", ("fd",), "fd_f", attrs={"dtype": "fp32"}),
            Node("concatenate", ("c0", "c1"), "cat", attrs={"axis": 1}),
            Node("logsumexp", ("cat",), "lse", attrs={"axes": [1], "keep_dims": False}),
            Node("sum", ("acos",), "s0", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("asin",), "s1", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("atan",), "s2", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("atanh",), "s3", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("neg",), "s4", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("rad",), "s5", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("ex1",), "s6", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("l1p",), "s7", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("l2",), "s8", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("l10",), "s9", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("fd_f",), "s10", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("lse",), "s11", attrs={"axes": [0], "keep_dims": False}),
            Node("add", ("s0", "s1"), "a0"),
            Node("add", ("s2", "s3"), "a1"),
            Node("add", ("s4", "s5"), "a2"),
            Node("add", ("s6", "s7"), "a3"),
            Node("add", ("s8", "s9"), "a4"),
            Node("add", ("s10", "s11"), "a5"),
            Node("add", ("a0", "a1"), "b0"),
            Node("add", ("a2", "a3"), "b1"),
            Node("add", ("a4", "a5"), "b2"),
            Node("add", ("b0", "b1"), "c"),
            Node("add", ("c", "b2"), "out"),
        ],
        outputs=["out"],
    )
    graph.validate()

    acos = np.arccos(u)
    asin = np.arcsin(u)
    atan = np.arctan(u)
    atanh = np.arctanh(u)
    neg = -u
    deg = np.degrees(u)
    rad = np.radians(deg)
    ex1 = np.expm1(u)
    l1p = np.log1p(p)
    l2 = np.log2(p)
    l10 = np.log10(p)
    fd_f = np.floor_divide(ia, ib).astype(np.float32)
    cat = np.concatenate([c0, c1], axis=1)
    m = np.max(cat, axis=1, keepdims=True)
    lse = (m + np.log(np.sum(np.exp(cat - m), axis=1, keepdims=True))).reshape(2)
    out = np.array(
        acos.sum()
        + asin.sum()
        + atan.sum()
        + atanh.sum()
        + neg.sum()
        + rad.sum()
        + ex1.sum()
        + l1p.sum()
        + l2.sum()
        + l10.sum()
        + fd_f.sum()
        + lse.sum(),
        dtype=np.float32,
    )
    expected = {"out": out}
    return ZooModelSpec(
        name="p0_math_pack",
        description="Math + concat pack: trig/log helpers, floor_divide, concatenate, logsumexp",
        graph=graph,
        inputs=inputs,
        expected=expected,
        atol=5e-3,
        rtol=1e-2,
    )


def _build_stats_divmod(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 3, 4), dtype=np.float32)
    a = rng.integers(1, 50, size=(2, 3), dtype=np.int32)
    b = rng.integers(1, 7, size=(2, 3), dtype=np.int32)
    inputs = {"x": x, "a": a, "b": b}

    graph = Graph(
        inputs=[
            TensorSpec("x", (2, 3, 4), "fp32"),
            TensorSpec("a", (2, 3), "int32"),
            TensorSpec("b", (2, 3), "int32"),
        ],
        nodes=[
            Node("var", ("x",), "v", attrs={"axes": [1, 2], "keep_dims": False}),
            Node("std", ("x",), "s", attrs={"axes": [0, 2], "keep_dims": False, "ddof": 1}),
            Node("divmod", ("a", "b"), "q", attrs={"output": "quotient"}),
            Node("divmod", ("a", "b"), "r", attrs={"output": "remainder"}),
            Node("cast", ("q",), "qf", attrs={"dtype": "fp32"}),
            Node("cast", ("r",), "rf", attrs={"dtype": "fp32"}),
            Node("sum", ("v",), "v_sum", attrs={"axes": [0], "keep_dims": False}),
            Node("sum", ("s",), "s_sum", attrs={"axes": [0], "keep_dims": False}),
            Node("sum", ("qf",), "q_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("sum", ("rf",), "r_sum", attrs={"axes": [0, 1], "keep_dims": False}),
            Node("add", ("v_sum", "s_sum"), "z0"),
            Node("add", ("q_sum", "r_sum"), "z1"),
            Node("add", ("z0", "z1"), "out"),
        ],
        outputs=["out"],
    )
    graph.validate()

    v = np.var(x, axis=(1, 2), keepdims=False, ddof=0)
    s = np.std(x, axis=(0, 2), keepdims=False, ddof=1)
    q = np.floor_divide(a, b).astype(np.float32)
    r = np.mod(a, b).astype(np.float32)
    expected = {"out": np.array(v.sum() + s.sum() + q.sum() + r.sum(), dtype=np.float32)}
    return ZooModelSpec(
        name="stats_divmod",
        description="Stats/divmod pack: var/std + quotient/remainder paths",
        graph=graph,
        inputs=inputs,
        expected=expected,
        atol=5e-3,
        rtol=1e-2,
    )


def _build_conv_block(seed: int) -> ZooModelSpec:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, 2, 4, 4), dtype=np.float32)
    w = rng.standard_normal((3, 2, 1, 1), dtype=np.float32)
    b = rng.standard_normal((3,), dtype=np.float32)
    z = np.zeros((1, 3, 4, 4), dtype=np.float32)
    wt = rng.standard_normal((3, 2, 1, 1), dtype=np.float32)
    bt = rng.standard_normal((2,), dtype=np.float32)
    inputs = {"x": x, "w": w, "b": b, "z": z, "wt": wt, "bt": bt}

    graph = Graph(
        inputs=[
            TensorSpec("x", (1, 2, 4, 4), "fp32"),
            TensorSpec("w", (3, 2, 1, 1), "fp32"),
            TensorSpec("b", (3,), "fp32"),
            TensorSpec("z", (1, 3, 4, 4), "fp32"),
            TensorSpec("wt", (3, 2, 1, 1), "fp32"),
            TensorSpec("bt", (2,), "fp32"),
        ],
        nodes=[
            Node("conv2d", ("x", "w", "b"), "c0", attrs={"pad_type": "valid", "strides": [1, 1]}),
            Node("conv_general", ("x", "w", "b"), "c1", attrs={"pad_type": "valid", "stride": [1, 1]}),
            Node("add", ("c0", "c1"), "s"),
            Node("maximum", ("s", "z"), "h"),
            Node(
                "conv_transpose2d",
                ("h", "wt", "bt"),
                "out",
                attrs={"pad_type": "valid", "strides": [1, 1]},
            ),
        ],
        outputs=["out"],
    )
    graph.validate()

    w_mat = w[:, :, 0, 0]
    c0 = np.einsum("nchw,oc->nohw", x, w_mat) + b.reshape(1, -1, 1, 1)
    c1 = np.einsum("nchw,oc->nohw", x, w_mat) + b.reshape(1, -1, 1, 1)
    h = np.maximum(c0 + c1, z)
    wt_mat = wt[:, :, 0, 0]
    out = np.einsum("nchw,co->nohw", h, wt_mat) + bt.reshape(1, -1, 1, 1)
    expected = {"out": out.astype(np.float32)}
    return ZooModelSpec(
        name="conv_block",
        description="Conv block: conv2d + conv_general + relu + conv_transpose2d",
        graph=graph,
        inputs=inputs,
        expected=expected,
        atol=5e-3,
        rtol=1e-2,
    )


def _capture_transformer_block(seed: int, artifacts_dir: Path) -> ZooModelSpec:
    from dataclasses import dataclass

    import mlx.core as mx
    import mlx.nn as nn

    @dataclass
    class ModelArgs:
        hidden_size: int
        intermediate_size: int
        num_attention_heads: int
        num_key_value_heads: int
        rms_norm_eps: float
        head_dim: int | None = None
        rope_theta: float = 10000.0
        rope_traditional: bool = False

    def swiglu(gate, x):
        return (gate * mx.sigmoid(gate)) * x

    class Llama3RoPE(nn.Module):
        def __init__(self, dims: int, traditional: bool = False, base: float = 10000.0):
            super().__init__()
            self.dims = dims
            self.traditional = traditional

            factor = 8.0
            low_freq_factor = 1.0
            high_freq_factor = 4.0
            old_context_len = 8192
            low_freq_period = old_context_len / low_freq_factor
            high_freq_period = old_context_len / high_freq_factor

            freqs = base ** (mx.arange(0, dims, 2) / dims)
            periods = 2 * mx.pi * freqs
            freqs = mx.where(periods > low_freq_period, freqs * factor, freqs)
            is_medium_freq = (periods > high_freq_period) & (periods < low_freq_period)
            smooth_factors = (old_context_len / periods - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            smooth_freqs = freqs / ((1.0 - smooth_factors) / factor + smooth_factors)
            self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)

        def __call__(self, x):
            return mx.fast.rope(
                x,
                self.dims,
                traditional=self.traditional,
                base=None,
                scale=1.0,
                offset=0,
                freqs=self._freqs,
            )

    class Attention(nn.Module):
        def __init__(self, args: ModelArgs):
            super().__init__()
            dim = args.hidden_size
            self.n_heads = n_heads = args.num_attention_heads
            self.n_kv_heads = n_kv_heads = args.num_key_value_heads
            self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads
            self.scale = head_dim**-0.5

            self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
            self.rope = Llama3RoPE(self.head_dim, traditional=args.rope_traditional, base=args.rope_theta)

        def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
            bsz, seqlen, _ = x.shape
            queries = self.q_proj(x)
            keys = self.k_proj(x)
            values = self.v_proj(x)
            queries = queries.reshape(bsz, seqlen, self.n_heads, -1).transpose(0, 2, 1, 3)
            keys = keys.reshape(bsz, seqlen, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
            values = values.reshape(bsz, seqlen, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
            queries = self.rope(queries)
            keys = self.rope(keys)
            output = mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=self.scale,
                mask=mask,
            )
            output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
            return self.o_proj(output)

    class MLP(nn.Module):
        def __init__(self, args: ModelArgs):
            super().__init__()
            dim = args.hidden_size
            hidden_dim = args.intermediate_size
            self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
            self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
            self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

        def __call__(self, x) -> mx.array:
            return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

    class TransformerBlock(nn.Module):
        def __init__(self, args: ModelArgs):
            super().__init__()
            self.self_attn = Attention(args)
            self.mlp = MLP(args)
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
            r = self.self_attn(self.input_layernorm(x), mask)
            h = x + r
            r = self.mlp(self.post_attention_layernorm(h))
            return h + r

    def _collect_arrays(tree):
        arrays = []
        if isinstance(tree, dict):
            for value in tree.values():
                arrays.extend(_collect_arrays(value))
            return arrays
        if isinstance(tree, (list, tuple)):
            for value in tree:
                arrays.extend(_collect_arrays(value))
            return arrays
        return [tree]

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    args = ModelArgs(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        rms_norm_eps=1e-5,
        head_dim=8,
    )
    block = TransformerBlock(args)
    params = _collect_arrays(block.parameters())
    if params:
        mx.eval(*params)

    seqlen = 8
    inputs = {
        "x": rng.standard_normal((2, seqlen, args.hidden_size), dtype=np.float32),
        "mask": np.triu(np.full((1, 1, seqlen, seqlen), -1e9, dtype=np.float32), 1),
    }

    graph, normalized_inputs, expected = capture_graph_from_mlx_function(
        dot_output_path=artifacts_dir / "capture_graph.dot",
        inputs=inputs,
        function=lambda x, mask: block(x, mask=mask),
        allow_unknown_sources=True,
        capture_mode="callback",
    )

    return ZooModelSpec(
        name="transformer_block",
        description="Static-shape transformer block (RoPE + SDPA + SwiGLU) captured via MLX callback export",
        graph=graph,
        inputs=normalized_inputs,
        expected=expected,
        atol=5e-2,
        rtol=1e-2,
    )


_BUILDERS: dict[str, Callable[[int], ZooModelSpec]] = {
    "linear_relu": _build_linear_relu,
    "arithmetic_chain": _build_arithmetic_chain,
    "reduction_suite": _build_reduction_suite,
    "shape_helpers": _build_shape_helpers,
    "indexing_transforms": _build_indexing_transforms,
    "creation_helpers": _build_creation_helpers,
    "mlp_2layer": _build_mlp_2layer,
    "broadcast_tensordot": _build_broadcast_tensordot,
    "numeric_sanity": _build_numeric_sanity,
    "diagonal_trace": _build_diagonal_trace,
    "tri_band": _build_tri_band,
    "logical_checks": _build_logical_checks,
    "meshgrid_kron": _build_meshgrid_kron,
    "p0_math_pack": _build_p0_math_pack,
    "stats_divmod": _build_stats_divmod,
    "conv_block": _build_conv_block,
}

_LIVE_CAPTURE_BUILDERS: dict[str, Callable[[int, Path], ZooModelSpec]] = {
    "transformer_block": _capture_transformer_block,
}


def available_model_names() -> list[str]:
    return sorted(_BUILDERS)


def available_live_model_names() -> list[str]:
    return sorted(_LIVE_CAPTURE_BUILDERS)


def supports_live_capture(name: str) -> bool:
    return name in _BUILDERS or name in _LIVE_CAPTURE_BUILDERS


def capture_model_spec(
    name: str,
    seed: int,
    artifacts_dir: Path,
    write_debug_dot: bool = True,
) -> ZooModelSpec:
    if name in _LIVE_CAPTURE_BUILDERS:
        return _LIVE_CAPTURE_BUILDERS[name](seed, artifacts_dir)

    if name not in _BUILDERS:
        raise ValueError(
            f"Live capture is not implemented for model '{name}'. "
            f"Available: {', '.join(sorted([*list(_BUILDERS), *list(_LIVE_CAPTURE_BUILDERS)])) or 'none'}"
        )

    static_spec = get_model_spec(name, seed=seed)
    if write_debug_dot:
        export_dot_from_ir(
            dot_output_path=artifacts_dir / "capture_graph.dot",
            graph=static_spec.graph,
            inputs=static_spec.inputs,
        )

    return ZooModelSpec(
        name=static_spec.name,
        description=f"{static_spec.description} (live-captured)",
        graph=static_spec.graph,
        inputs=static_spec.inputs,
        expected=static_spec.expected,
        atol=static_spec.atol,
        rtol=static_spec.rtol,
    )


def get_model_spec(name: str, seed: int = 0) -> ZooModelSpec:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(available_model_names())}")
    return _BUILDERS[name](seed)
