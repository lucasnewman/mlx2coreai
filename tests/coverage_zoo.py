from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mlx2coreai.ir import Graph, Node, TensorSpec


@dataclass(frozen=True)
class CoverageModelSpec:
    name: str
    description: str
    graph: Graph


def _binary_canonical(seed: int) -> CoverageModelSpec:
    del seed
    graph = Graph(
        inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (2, 3), "fp32")],
        nodes=[
            Node("sub", ("x", "y"), "sub_out"),
            Node("mul", ("x", "y"), "mul_out"),
            Node("real_div", ("x", "y"), "div_out"),
            Node("pow", ("x", "y"), "pow_out"),
            Node("mod", ("x", "y"), "mod_out"),
            Node("minimum", ("x", "y"), "minimum_out"),
            Node("greater", ("x", "y"), "greater_out"),
            Node("greater_equal", ("x", "y"), "greater_equal_out"),
            Node("less", ("x", "y"), "less_out"),
            Node("less_equal", ("x", "y"), "less_equal_out"),
            Node("equal", ("x", "y"), "equal_out"),
            Node("not_equal", ("x", "y"), "not_equal_out"),
        ],
        outputs=[
            "sub_out",
            "mul_out",
            "div_out",
            "pow_out",
            "mod_out",
            "minimum_out",
            "greater_out",
            "greater_equal_out",
            "less_out",
            "less_equal_out",
            "equal_out",
            "not_equal_out",
        ],
    )
    return CoverageModelSpec("supplemental_binary_canonical", "Canonical binary op aliases", graph)


def _unary_canonical(seed: int) -> CoverageModelSpec:
    del seed
    ops = ["exp", "log", "sqrt", "rsqrt", "sigmoid", "silu", "gelu", "tanh", "sin", "cos", "erf", "abs"]
    graph = Graph(
        inputs=[TensorSpec("x", (2, 3), "fp32")],
        nodes=[Node(op, ("x",), f"{op}_out") for op in ops],
        outputs=[f"{op}_out" for op in ops],
    )
    return CoverageModelSpec("supplemental_unary_canonical", "Unary math ops", graph)


def _shape_index(seed: int) -> CoverageModelSpec:
    del seed
    graph = Graph(
        inputs=[
            TensorSpec("x", (2, 3, 4), "fp32"),
            TensorSpec("update", (2, 2, 4), "fp32"),
            TensorSpec("s", (1, 2, 1), "fp32"),
            TensorSpec("b", (1, 3), "fp32"),
            TensorSpec("idx", (2,), "int32"),
            TensorSpec("start", (3,), "int32"),
        ],
        nodes=[
            Node("reshape", ("x",), "reshape_out", attrs={"shape": [6, 4]}),
            Node("transpose", ("x",), "transpose_out", attrs={"perm": [0, 2, 1]}),
            Node("expand_dims", ("b",), "expand_out", attrs={"axes": [0]}),
            Node("squeeze", ("s",), "squeeze_out", attrs={"axes": [0, 2]}),
            Node("broadcast_to", ("b",), "broadcast_out", attrs={"shape": [2, 3]}),
            Node("slice_by_index", ("x",), "slice_out", attrs={"begin": [0, 0, 0], "end": [2, 2, 4], "stride": [1, 1, 1]}),
            Node("slice_update", ("x", "update"), "slice_update_out", attrs={"begin": [0, 0, 0], "end": [2, 2, 4], "stride": [1, 1, 1]}),
            Node("dynamic_slice_update", ("x", "update", "start"), "dynamic_slice_update_out", attrs={"axes": [0, 1, 2]}),
            Node("dynamicsliceupdate", ("x", "update", "start"), "dynamicsliceupdate_out", attrs={"axes": [0, 1, 2]}),
            Node("split", ("x",), "split_out", attrs={"axis": 2, "num_splits": 2, "output_index": 1}),
            Node("gather", ("x", "idx"), "gather_out", attrs={"axis": 1}),
        ],
        outputs=[
            "reshape_out",
            "transpose_out",
            "expand_out",
            "squeeze_out",
            "broadcast_out",
            "slice_out",
            "slice_update_out",
            "dynamic_slice_update_out",
            "dynamicsliceupdate_out",
            "split_out",
            "gather_out",
        ],
    )
    return CoverageModelSpec("supplemental_shape_index", "Shape and index lowerings", graph)


def _constants_and_identity(seed: int) -> CoverageModelSpec:
    del seed
    graph = Graph(
        inputs=[TensorSpec("x", (2, 3), "fp32")],
        nodes=[
            Node("const", tuple(), "const_out", attrs={"value": np.ones((2, 3), dtype=np.float32)}),
            Node("constant", tuple(), "constant_out", attrs={"value": np.zeros((2, 3), dtype=np.float32)}),
            Node("copy", ("x",), "copy_out"),
            Node("contiguous", ("x",), "contiguous_out"),
            Node("broadcast", ("x",), "broadcast_alias_out", attrs={"shape": [2, 3]}),
        ],
        outputs=["const_out", "constant_out", "copy_out", "contiguous_out", "broadcast_alias_out"],
    )
    return CoverageModelSpec("supplemental_constants_identity", "Constants and identity aliases", graph)


def _reductions_canonical(seed: int) -> CoverageModelSpec:
    del seed
    graph = Graph(
        inputs=[TensorSpec("x", (2, 3, 4), "fp32")],
        nodes=[
            Node("reduce", ("x",), "reduce_out", attrs={"axes": [2], "mode": 2, "keep_dims": False}),
            Node("reduce_sum", ("x",), "reduce_sum_out", attrs={"axes": [2], "keep_dims": False}),
            Node("reduce_mean", ("x",), "reduce_mean_out", attrs={"axes": [1], "keep_dims": False}),
            Node("reduce_min", ("x",), "reduce_min_out", attrs={"axes": [2], "keep_dims": False}),
            Node("reduce_max", ("x",), "reduce_max_out", attrs={"axes": [2], "keep_dims": False}),
            Node("reduce_prod", ("x",), "reduce_prod_out", attrs={"axes": [2], "keep_dims": False}),
            Node("reduce_argmax", ("x",), "reduce_argmax_out", attrs={"axis": 2, "keep_dims": False}),
            Node("reduce_argmin", ("x",), "reduce_argmin_out", attrs={"axis": 1, "keep_dims": True}),
        ],
        outputs=[
            "reduce_out",
            "reduce_sum_out",
            "reduce_mean_out",
            "reduce_min_out",
            "reduce_max_out",
            "reduce_prod_out",
            "reduce_argmax_out",
            "reduce_argmin_out",
        ],
    )
    return CoverageModelSpec("supplemental_reductions_canonical", "Canonical reduction lowering keys", graph)


def _nn_composites(seed: int) -> CoverageModelSpec:
    del seed
    graph = Graph(
        inputs=[
            TensorSpec("x", (2, 4), "fp32"),
            TensorSpec("scale", (4,), "fp32"),
            TensorSpec("bias", (4,), "fp32"),
            TensorSpec("rope_x", (1, 2, 3, 4), "fp32"),
            TensorSpec("q", (1, 2, 3, 4), "fp32"),
            TensorSpec("k", (1, 2, 3, 4), "fp32"),
            TensorSpec("v", (1, 2, 3, 4), "fp32"),
        ],
        nodes=[
            Node("layernorm", ("x", "scale", "bias"), "layernorm_out", attrs={"axes": [-1], "eps": 1e-5}),
            Node("rmsnorm", ("x", "scale"), "rmsnorm_out", attrs={"axes": [-1], "eps": 1e-5}),
            Node("rope", ("rope_x",), "rope_out", attrs={"dims": 4, "base": 10000.0}),
            Node("scaled_dot_product_attention", ("q", "k", "v"), "sdpa_out", attrs={"scale": 0.5}),
            Node("softmax", ("x",), "softmax_out", attrs={"axis": -1}),
        ],
        outputs=["layernorm_out", "rmsnorm_out", "rope_out", "sdpa_out", "softmax_out"],
    )
    return CoverageModelSpec("supplemental_nn_composites", "NN and composite lowerings", graph)


def _linear_misc(seed: int) -> CoverageModelSpec:
    del seed
    graph = Graph(
        inputs=[
            TensorSpec("x", (2,), "fp32"),
            TensorSpec("y", (3,), "fp32"),
            TensorSpec("a", (2, 3), "fp32"),
            TensorSpec("b", (2, 3), "fp32"),
            TensorSpec("cond", (2, 3), "bool"),
        ],
        nodes=[
            Node("outer", ("x", "y"), "outer_out"),
            Node("inner", ("a", "b"), "inner_out"),
            Node("select", ("cond", "a", "b"), "select_out"),
        ],
        outputs=["outer_out", "inner_out", "select_out"],
    )
    return CoverageModelSpec("supplemental_linear_misc", "Outer/inner/select lowerings", graph)


def _convolutions(seed: int) -> CoverageModelSpec:
    del seed
    graph = Graph(
        inputs=[
            TensorSpec("x1", (1, 2, 4), "fp32"),
            TensorSpec("w1", (3, 2, 1), "fp32"),
            TensorSpec("t1", (1, 3, 4), "fp32"),
            TensorSpec("wt1", (3, 2, 1), "fp32"),
            TensorSpec("x3", (1, 2, 3, 3, 3), "fp32"),
            TensorSpec("w3", (3, 2, 1, 1, 1), "fp32"),
            TensorSpec("t3", (1, 3, 3, 3, 3), "fp32"),
            TensorSpec("wt3", (3, 2, 1, 1, 1), "fp32"),
            TensorSpec("x2", (1, 2, 4, 4), "fp32"),
            TensorSpec("w2", (3, 2, 1, 1), "fp32"),
        ],
        nodes=[
            Node("conv1d", ("x1", "w1"), "conv1d_out", attrs={"pad_type": "valid", "strides": [1]}),
            Node("conv_transpose1d", ("t1", "wt1"), "conv_transpose1d_out", attrs={"pad_type": "valid", "strides": [1]}),
            Node("conv3d", ("x3", "w3"), "conv3d_out", attrs={"pad_type": "valid", "strides": [1, 1, 1]}),
            Node("conv_transpose3d", ("t3", "wt3"), "conv_transpose3d_out", attrs={"pad_type": "valid", "strides": [1, 1, 1]}),
            Node("convolution", ("x2", "w2"), "convolution_out", attrs={"pad_type": "valid", "strides": [1, 1]}),
        ],
        outputs=["conv1d_out", "conv_transpose1d_out", "conv3d_out", "conv_transpose3d_out", "convolution_out"],
    )
    return CoverageModelSpec("supplemental_convolutions", "Additional convolution variants", graph)


def _state_ops(seed: int) -> CoverageModelSpec:
    del seed
    graph = Graph(
        inputs=[
            TensorSpec("state", (2, 3), "fp32"),
            TensorSpec("value", (2, 3), "fp32"),
            TensorSpec("mask", (2, 3), "bool"),
        ],
        nodes=[
            Node("read_state", ("state",), "read_out"),
            Node("write_state", ("state", "value"), "write_out"),
            Node("state_update_masked", ("state", "value", "mask"), "masked_out"),
        ],
        outputs=["read_out", "write_out", "masked_out"],
    )
    return CoverageModelSpec("supplemental_state_ops", "State read/write/update lowerings", graph)


def _aliases_and_bitwise(seed: int) -> CoverageModelSpec:
    del seed
    graph = Graph(
        inputs=[
            TensorSpec("x", (2, 3), "fp32"),
            TensorSpec("y", (2, 3), "fp32"),
            TensorSpec("i", (2, 3), "int32"),
            TensorSpec("j", (2, 3), "int32"),
            TensorSpec("b0", (2, 3), "bool"),
            TensorSpec("b1", (2, 3), "bool"),
            TensorSpec("update", (2, 2), "fp32"),
            TensorSpec("q", (1, 2, 3, 4), "fp32"),
            TensorSpec("k", (1, 2, 3, 4), "fp32"),
            TensorSpec("v", (1, 2, 3, 4), "fp32"),
        ],
        nodes=[
            Node("bitwisebinary", ("b0", "b1"), "bitwise_out", attrs={"mode": "and"}),
            Node("expanddims", ("x",), "expanddims_out", attrs={"axes": [0]}),
            Node("floor_div", ("i", "j"), "floor_div_out"),
            Node("inverse", ("x",), "inverse_out"),
            Node("greaterequal", ("x", "y"), "greaterequal_out"),
            Node("lessequal", ("x", "y"), "lessequal_out"),
            Node("notequal", ("x", "y"), "notequal_out"),
            Node("scaleddotproductattention", ("q", "k", "v"), "sdpa_alias_out", attrs={"scale": 0.5}),
            Node("sliceupdate", ("x", "update"), "sliceupdate_out", attrs={"begin": [0, 0], "end": [2, 2], "stride": [1, 1]}),
        ],
        outputs=[
            "bitwise_out",
            "expanddims_out",
            "floor_div_out",
            "inverse_out",
            "greaterequal_out",
            "lessequal_out",
            "notequal_out",
            "sdpa_alias_out",
            "sliceupdate_out",
        ],
    )
    return CoverageModelSpec("supplemental_aliases_and_bitwise", "Alias spellings and bitwise binary", graph)


_BUILDERS: dict[str, Callable[[int], CoverageModelSpec]] = {
    "supplemental_aliases_and_bitwise": _aliases_and_bitwise,
    "supplemental_binary_canonical": _binary_canonical,
    "supplemental_unary_canonical": _unary_canonical,
    "supplemental_shape_index": _shape_index,
    "supplemental_constants_identity": _constants_and_identity,
    "supplemental_reductions_canonical": _reductions_canonical,
    "supplemental_nn_composites": _nn_composites,
    "supplemental_linear_misc": _linear_misc,
    "supplemental_convolutions": _convolutions,
    "supplemental_state_ops": _state_ops,
}


def available_model_names() -> list[str]:
    return sorted(_BUILDERS)


def get_model_spec(name: str, seed: int = 0) -> CoverageModelSpec:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(available_model_names())}")
    return _BUILDERS[name](seed)
