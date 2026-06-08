from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .ir import Graph

# Canonical MLX -> CoreAI lowering-key mapping for this translator.
SUPPORTED_MLX_TO_COREAI_OPS: dict[str, str] = {
    "matmul": "matmul",
    "add": "add",
    "maximum": "maximum",
    "minimum": "minimum",
    # Arithmetic aliases
    "subtract": "sub",
    "sub": "sub",
    "multiply": "mul",
    "mul": "mul",
    "divide": "real_div",
    "real_div": "real_div",
    "power": "pow",
    "pow": "pow",
    "reciprocal": "inverse",
    "inverse": "inverse",
    "remainder": "mod",
    "mod": "mod",
    # Reductions
    "reduce": "reduce",
    "sum": "reduce_sum",
    "reduce_sum": "reduce_sum",
    "mean": "reduce_mean",
    "reduce_mean": "reduce_mean",
    "min": "reduce_min",
    "reduce_min": "reduce_min",
    "max": "reduce_max",
    "reduce_max": "reduce_max",
    "prod": "reduce_prod",
    "reduce_prod": "reduce_prod",
    "argmax": "reduce_argmax",
    "reduce_argmax": "reduce_argmax",
    "argmin": "reduce_argmin",
    "reduce_argmin": "reduce_argmin",
    # Shape/index transforms
    "flatten": "flatten",
    "unflatten": "unflatten",
    "reshape": "reshape",
    "transpose": "transpose",
    "atleast_1d": "atleast_1d",
    "atleast_2d": "atleast_2d",
    "atleast_3d": "atleast_3d",
    "moveaxis": "moveaxis",
    "swapaxes": "swapaxes",
    "slice": "slice_by_index",
    "slice_by_index": "slice_by_index",
    "slice_update": "slice_update",
    "sliceupdate": "slice_update",
    "take": "gather",
    "take_along_axis": "gather_along_axis",
    "gather": "gather",
    "squeeze": "squeeze",
    # Tensor creation/helpers
    "zeros": "zeros",
    "ones": "ones",
    "full": "full",
    "zeros_like": "zeros_like",
    "ones_like": "ones_like",
    "full_like": "full_like",
    "arange": "arange",
    "linspace": "linspace",
    "where": "select",
    "select": "select",
    "greater": "greater",
    "greaterequal": "greater_equal",
    "greater_equal": "greater_equal",
    "less": "less",
    "lessequal": "less_equal",
    "less_equal": "less_equal",
    "equal": "equal",
    "not_equal": "not_equal",
    "notequal": "not_equal",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "rsqrt": "rsqrt",
    "abs": "abs",
    "split": "split",
    "expanddims": "expand_dims",
    "expand_dims": "expand_dims",
    "bitwisebinary": "bitwisebinary",
    "scaled_dot_product_attention": "scaled_dot_product_attention",
    "scaleddotproductattention": "scaled_dot_product_attention",
    "rope": "rope",
    "softmax": "softmax",
    "sigmoid": "sigmoid",
    "silu": "silu",
    "gelu": "gelu",
    "tanh": "tanh",
    "sin": "sin",
    "cos": "cos",
    "erf": "erf",
    "layernorm": "layernorm",
    "rmsnorm": "rmsnorm",
    "astype": "cast",
    "cast": "cast",
    "number_of_elements": "number_of_elements",
    "stop_gradient": "identity",
    # Linear/broadcast/composite helpers
    "addmm": "addmm",
    "broadcast": "broadcast_to",
    "broadcast_to": "broadcast_to",
    "broadcast_arrays": "broadcast_arrays",
    "outer": "outer",
    "inner": "inner",
    "tensordot": "tensordot",
    "isclose": "isclose",
    "allclose": "allclose",
    "nan_to_num": "nan_to_num",
    "diag": "diag",
    "diagonal": "diagonal",
    "trace": "trace",
    "tri": "tri",
    "tril": "tril",
    "triu": "triu",
    "all": "all",
    "any": "any",
    "array_equal": "array_equal",
    "isnan": "isnan",
    "isinf": "isinf",
    "isfinite": "isfinite",
    "isneginf": "isneginf",
    "isposinf": "isposinf",
    "eye": "eye",
    "meshgrid": "meshgrid",
    "kron": "kron",
    "logaddexp": "logaddexp",
    "concatenate": "concat",
    "copy": "identity",
    "contiguous": "identity",
    "arccos": "acos",
    "arcsin": "asin",
    "arctan": "atan",
    "arctanh": "atanh",
    "negative": "negative",
    "degrees": "degrees",
    "radians": "radians",
    "expm1": "expm1",
    "log1p": "log1p",
    "log2": "log2",
    "log10": "log10",
    "logsumexp": "reduce_log_sum_exp",
    "floor_divide": "floor_div",
    "floor_div": "floor_div",
    "var": "var",
    "std": "std",
    "divmod": "divmod",
    "conv1d": "conv",
    "conv2d": "conv",
    "conv3d": "conv",
    "conv_general": "conv",
    "conv_transpose1d": "conv_transpose",
    "conv_transpose2d": "conv_transpose",
    "conv_transpose3d": "conv_transpose",
    "convolution": "conv",
    "const": "const",
    "constant": "const",
    # Stateful lowering primitives
    "read_state": "read_state",
    "write_state": "write_state",
    "state_update_masked": "state_update_masked",
}

# Backward-compatible alias for the shared normalization code ported from
# mlx2coreml. The values are lowering keys, not necessarily literal CoreAI op
# names.
SUPPORTED_MLX_TO_MIL_OPS = SUPPORTED_MLX_TO_COREAI_OPS

# Human-friendly label normalization from MLX dot export labels.
_DOT_LABEL_ALIASES: dict[str, str] = {
    "Matmul": "matmul",
    "Add": "add",
    "Maximum": "maximum",
    "Minimum": "minimum",
    "Sub": "sub",
    "Subtract": "subtract",
    "Mul": "mul",
    "Multiply": "multiply",
    "RealDiv": "real_div",
    "Divide": "divide",
    "Pow": "pow",
    "Power": "power",
    "Inverse": "inverse",
    "Reciprocal": "reciprocal",
    "Mod": "mod",
    "Remainder": "remainder",
    "ReduceSum": "reduce_sum",
    "Reduce": "reduce",
    "ReduceMean": "reduce_mean",
    "ReduceMin": "reduce_min",
    "ReduceMax": "reduce_max",
    "ReduceProd": "reduce_prod",
    "ReduceArgmax": "reduce_argmax",
    "ReduceArgmin": "reduce_argmin",
    "Flatten": "flatten",
    "Unflatten": "unflatten",
    "Reshape": "reshape",
    "Transpose": "transpose",
    "Atleast1d": "atleast_1d",
    "Atleast2d": "atleast_2d",
    "Atleast3d": "atleast_3d",
    "Moveaxis": "moveaxis",
    "Swapaxes": "swapaxes",
    "SliceByIndex": "slice_by_index",
    "Slice": "slice",
    "SliceUpdate": "slice_update",
    "Take": "take",
    "TakeAlongAxis": "take_along_axis",
    "Gather": "gather",
    "Squeeze": "squeeze",
    "Zeros": "zeros",
    "Ones": "ones",
    "Full": "full",
    "ZerosLike": "zeros_like",
    "OnesLike": "ones_like",
    "FullLike": "full_like",
    "Arange": "arange",
    "Linspace": "linspace",
    "Where": "where",
    "Select": "select",
    "Greater": "greater",
    "GreaterEqual": "greaterequal",
    "Less": "less",
    "LessEqual": "lessequal",
    "Equal": "equal",
    "NotEqual": "not_equal",
    "Exp": "exp",
    "Log": "log",
    "Sqrt": "sqrt",
    "Rsqrt": "rsqrt",
    "Abs": "abs",
    "Split": "split",
    "ExpandDims": "expanddims",
    "BitwiseBinary": "bitwisebinary",
    "ScaledDotProductAttention": "scaled_dot_product_attention",
    "RoPE": "rope",
    "Rope": "rope",
    "Softmax": "softmax",
    "Sigmoid": "sigmoid",
    "Silu": "silu",
    "SiLU": "silu",
    "Gelu": "gelu",
    "GELU": "gelu",
    "Tanh": "tanh",
    "Sin": "sin",
    "Cos": "cos",
    "Erf": "erf",
    "LayerNorm": "layernorm",
    "RMSNorm": "rmsnorm",
    "Astype": "astype",
    "Cast": "cast",
    "NumberOfElements": "number_of_elements",
    "StopGradient": "stop_gradient",
    "Addmm": "addmm",
    "AddMM": "addmm",
    "Broadcast": "broadcast",
    "BroadcastTo": "broadcast_to",
    "BroadcastArrays": "broadcast_arrays",
    "Outer": "outer",
    "Inner": "inner",
    "Tensordot": "tensordot",
    "TensorDot": "tensordot",
    "Isclose": "isclose",
    "Allclose": "allclose",
    "NanToNum": "nan_to_num",
    "NanToNumV2": "nan_to_num",
    "Diag": "diag",
    "Diagonal": "diagonal",
    "Trace": "trace",
    "Tri": "tri",
    "Tril": "tril",
    "Triu": "triu",
    "All": "all",
    "Any": "any",
    "ArrayEqual": "array_equal",
    "Isnan": "isnan",
    "IsInf": "isinf",
    "Isinf": "isinf",
    "IsFinite": "isfinite",
    "Isfinite": "isfinite",
    "IsNegInf": "isneginf",
    "Isneginf": "isneginf",
    "IsPosInf": "isposinf",
    "Isposinf": "isposinf",
    "Eye": "eye",
    "Meshgrid": "meshgrid",
    "MeshGrid": "meshgrid",
    "Kron": "kron",
    "LogAddExp": "logaddexp",
    "Logaddexp": "logaddexp",
    "Concatenate": "concatenate",
    "Concat": "concatenate",
    "Copy": "copy",
    "Contiguous": "contiguous",
    "Arccos": "arccos",
    "Arcsin": "arcsin",
    "Arctan": "arctan",
    "Arctanh": "arctanh",
    "Negative": "negative",
    "Neg": "negative",
    "Degrees": "degrees",
    "Radians": "radians",
    "Expm1": "expm1",
    "Log1p": "log1p",
    "Log2": "log2",
    "Log10": "log10",
    "LogSumExp": "logsumexp",
    "ReduceLogSumExp": "logsumexp",
    "FloorDivide": "floor_divide",
    "FloorDiv": "floor_div",
    "Var": "var",
    "Std": "std",
    "Divmod": "divmod",
    "Conv1d": "conv1d",
    "Conv2d": "conv2d",
    "Conv3d": "conv3d",
    "ConvGeneral": "conv_general",
    "ConvTranspose1d": "conv_transpose1d",
    "ConvTranspose2d": "conv_transpose2d",
    "ConvTranspose3d": "conv_transpose3d",
    "ConvTranspose": "conv_transpose2d",
    "Convolution": "convolution",
    "Const": "const",
    "Constant": "constant",
    "ReadState": "read_state",
    "WriteState": "write_state",
    "StateUpdateMasked": "state_update_masked",
}

_OPS_STATUS_CACHE: dict[str, str] | None = None


class UnsupportedOpsError(ValueError):
    def __init__(self, first_op: str, all_ops: list[str], details: list[dict[str, Any]]):
        self.first_op = first_op
        self.all_ops = all_ops
        self.details = details

        lines = [
            f"Unsupported MLX op encountered first: {first_op}",
            "All unsupported ops: " + ", ".join(all_ops),
            "Recommendations:",
        ]
        for detail in details:
            source = detail.get("source", "unknown")
            status = detail.get("status", "unlisted")
            count = int(detail.get("count", 1))
            primitive = detail.get("primitive", "")
            primitive_text = f", primitive={primitive}" if primitive else ""
            lines.append(
                f"- {detail['op']} (count={count}, source={source}{primitive_text}) "
                f"-> backlog status {status}; {detail['recommendation']}"
            )
            sample_contexts = detail.get("sample_contexts") or []
            if sample_contexts:
                sample = sample_contexts[0]
                lines.append(
                    f"  sample: output={sample.get('output')}, "
                    f"inputs={sample.get('inputs')}, attrs={sample.get('attrs')}"
                )
        lines.append("Add mappings/lowerings in mlx2coreai/op_registry.py and mlx2coreai/lower_to_coreai.py.")
        super().__init__("\n".join(lines))


def _load_ops_statuses() -> dict[str, str]:
    global _OPS_STATUS_CACHE
    if _OPS_STATUS_CACHE is not None:
        return _OPS_STATUS_CACHE

    statuses: dict[str, str] = {}
    ops_path = Path(__file__).resolve().parents[1] / "docs" / "ops_status.md"
    if ops_path.exists():
        section = ""
        row_re = re.compile(r"^\|\s*`(?P<op>[^`]+)`\s*\|")
        for line in ops_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                if stripped == "## Supported":
                    section = "supported"
                elif stripped == "## Not Yet Implemented":
                    section = "not_yet_implemented"
                elif stripped == "## Not Supported":
                    section = "not_supported"
                else:
                    section = ""
                continue

            if not section:
                continue

            match = row_re.match(stripped)
            if match:
                statuses[normalize_mlx_op_name(match.group("op"))] = section

    _OPS_STATUS_CACHE = statuses
    return statuses


def normalize_mlx_op_name(name: str) -> str:
    cleaned = name.strip()
    if cleaned in _DOT_LABEL_ALIASES:
        return _DOT_LABEL_ALIASES[cleaned]
    return cleaned.lower()


def mil_op_for_mlx(name: str) -> str | None:
    normalized = normalize_mlx_op_name(name)
    return SUPPORTED_MLX_TO_COREAI_OPS.get(normalized)


def coreai_op_for_mlx(name: str) -> str | None:
    normalized = normalize_mlx_op_name(name)
    return SUPPORTED_MLX_TO_COREAI_OPS.get(normalized)


def _source_primitive_context(source: str) -> tuple[str, str | None, str | None]:
    if source.startswith("mlx_export:"):
        parts = source.split(":", 2)
        if len(parts) == 3:
            return "mlx_export", parts[2], parts[1]
        return "mlx_export", None, None
    if source.startswith("mlx_dot:"):
        parts = source.split(":", 2)
        if len(parts) == 3:
            return "mlx_dot", parts[2], parts[1]
        return "mlx_dot", None, None
    return "ir", None, None


def _preview_attr(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        if len(value) <= 6 and all(
            isinstance(v, (str, int, float, bool)) or v is None for v in value
        ):
            return list(value)
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, dict):
        keys = list(value.keys())
        return f"dict(keys={keys[:6]}{'...' if len(keys) > 6 else ''})"
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            shape_list = [int(v) for v in shape]
        except Exception:  # pragma: no cover - defensive
            shape_list = list(shape)
        return f"array(shape={shape_list},dtype={dtype})"
    return type(value).__name__


def unsupported_op_details(graph: Graph) -> list[dict[str, Any]]:
    statuses = _load_ops_statuses()
    details_by_op: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for node in graph.nodes:
        normalized = normalize_mlx_op_name(node.op)
        if mil_op_for_mlx(normalized) is not None:
            continue
        source = node.source or node.output
        source_kind, primitive, primitive_index = _source_primitive_context(source)

        if normalized not in details_by_op:
            status = statuses.get(normalized, "unlisted")
            if status == "not_yet_implemented":
                recommendation = (
                    "In scope but not implemented yet; add mapping/lowering in the translator."
                )
            elif status == "not_supported":
                recommendation = (
                    "Out of scope; keep explicit unsupported guard and document rationale."
                )
            elif status == "supported":
                recommendation = (
                    "Expected to be supported; verify normalization/mapping for this primitive."
                )
            else:
                recommendation = (
                    "Classify this op in docs/ops_status.md and then implement or defer explicitly."
                )
            details_by_op[normalized] = {
                "op": normalized,
                "source": source,
                "status": status,
                "recommendation": recommendation,
                "count": 0,
                "source_kind": source_kind,
                "primitive": primitive,
                "primitive_index": primitive_index,
                "sample_sources": [],
                "sample_contexts": [],
            }
            order.append(normalized)

        detail = details_by_op[normalized]
        detail["count"] = int(detail["count"]) + 1

        sample_sources = detail["sample_sources"]
        if source not in sample_sources and len(sample_sources) < 5:
            sample_sources.append(source)

        sample_contexts = detail["sample_contexts"]
        if len(sample_contexts) < 3:
            sample_contexts.append(
                {
                    "source": source,
                    "output": node.output,
                    "inputs": list(node.inputs),
                    "attrs": {
                        str(k): _preview_attr(v)
                        for k, v in node.attrs.items()
                    },
                }
            )

    return [details_by_op[op] for op in order]


def unsupported_ops(graph: Graph) -> list[str]:
    return [detail["op"] for detail in unsupported_op_details(graph)]


def ensure_supported(graph: Graph) -> None:
    details = unsupported_op_details(graph)
    if details:
        raise UnsupportedOpsError(
            first_op=details[0]["op"],
            all_ops=[detail["op"] for detail in details],
            details=details,
        )
