"""Legacy DOT capture and IR-to-MLX debugging, loaded only on explicit use.

Production model conversion uses the callback frontend in from_mlx.py.
These entrypoints remain available for older callers and diagnostic scripts.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import re
import numpy as np
from .dtypes import capture_numpy_dtype as _numpy_dtype_to_ir
from .ir import Graph, Node, TensorSpec
from .op_registry import normalize_mlx_op_name
from .passes import infer_broadcast_axes_shape
from ._capture_utils import (
    _shape_tuple, _constant_to_numpy, _normalize_numpy_inputs,
    _default_input_specs, _normalize_outputs,
)

_SOURCE_RE = re.compile(r'rank=source;\s*"([^"]+)"')
_SINK_RE = re.compile(r'rank=sink;\s*"([^"]+)"')
_OP_RE = re.compile(r'^\{\s*(\d+)\s+\[label\s*=\s*"(.*?)",\s*shape=rectangle\];\s*\}$')
_EDGE_RE = re.compile(r'^"?([^"]+?)"?\s*->\s*"?([^"]+?)"?$')


def build_smoke_numpy_inputs(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 3), dtype=np.float32)
    w = rng.standard_normal((3, 4), dtype=np.float32)
    b = rng.standard_normal((2, 4), dtype=np.float32)
    z = np.zeros((2, 4), dtype=np.float32)
    return {"x": x, "w": w, "b": b, "z": z}


def evaluate_smoke_numpy(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.maximum((inputs["x"] @ inputs["w"]) + inputs["b"], inputs["z"])


def make_mock_smoke_graph() -> Graph:
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


def parse_mlx_dot_to_graph(
    dot_text: str,
    input_specs: list[TensorSpec],
    allow_unknown_sources: bool = False,
) -> Graph:
    source_names: list[str] = []
    sink_names: list[str] = []
    op_labels: dict[str, str] = {}
    op_raw_labels: dict[str, str] = {}
    op_order: list[str] = []
    edges: list[tuple[str, str]] = []

    for raw_line in dot_text.splitlines():
        line = raw_line.strip()
        if not line or line in {"digraph {", "}"}:
            continue

        source_match = _SOURCE_RE.search(line)
        if source_match:
            source_names.append(source_match.group(1))
            continue

        sink_match = _SINK_RE.search(line)
        if sink_match:
            sink_names.append(sink_match.group(1))
            continue

        op_match = _OP_RE.match(line)
        if op_match:
            op_id, label = op_match.group(1), op_match.group(2)
            op_labels[op_id] = normalize_mlx_op_name(label)
            op_raw_labels[op_id] = label
            op_order.append(op_id)
            continue

        if "->" in line:
            edge_line = line.rstrip(";")
            edge_match = _EDGE_RE.match(edge_line)
            if edge_match:
                edges.append((edge_match.group(1), edge_match.group(2)))

    op_ids = set(op_labels)
    op_inputs: dict[str, list[str]] = {op_id: [] for op_id in op_ids}
    op_outputs: dict[str, list[str]] = {op_id: [] for op_id in op_ids}

    for src, dst in edges:
        if dst in op_ids:
            op_inputs[dst].append(src)
        if src in op_ids and dst not in op_ids:
            if dst not in op_outputs[src]:
                op_outputs[src].append(dst)

    missing_outputs = [op_id for op_id in op_order if not op_outputs.get(op_id)]
    if missing_outputs:
        raise ValueError(
            "DOT parse error: some op nodes are missing outputs: "
            + ", ".join(missing_outputs)
        )

    producer_of_tensor = {
        tensor_name: op_id
        for op_id, tensor_names in op_outputs.items()
        for tensor_name in tensor_names
    }
    dependencies: dict[str, set[str]] = {}
    for op_id in op_order:
        deps = {
            producer_of_tensor[input_name]
            for input_name in op_inputs[op_id]
            if input_name in producer_of_tensor
        }
        dependencies[op_id] = deps

    ready = [op_id for op_id in op_order if not dependencies[op_id]]
    sorted_op_ids: list[str] = []
    while ready:
        current = ready.pop(0)
        sorted_op_ids.append(current)
        for candidate in op_order:
            if current in dependencies[candidate]:
                dependencies[candidate].remove(current)
                if not dependencies[candidate] and candidate not in sorted_op_ids and candidate not in ready:
                    ready.append(candidate)

    if len(sorted_op_ids) != len(op_order):
        raise ValueError("DOT parse error: graph contains unresolved/cyclic op dependencies.")

    nodes: list[Node] = []
    for op_id in sorted_op_ids:
        outputs_for_op = op_outputs[op_id]
        for output_index, output_name in enumerate(outputs_for_op):
            attrs: dict[str, Any] = {}
            if len(outputs_for_op) > 1:
                attrs["output_index"] = output_index
                attrs["num_outputs"] = len(outputs_for_op)
            nodes.append(
                Node(
                    op=op_labels[op_id],
                    inputs=tuple(op_inputs[op_id]),
                    output=output_name,
                    attrs=attrs,
                    source=f"mlx_dot:{op_id}:{op_raw_labels[op_id]}",
                )
            )

    spec_by_name = {spec.name: spec for spec in input_specs}
    missing_specs = [name for name in source_names if name not in spec_by_name]
    if missing_specs and not allow_unknown_sources:
        raise ValueError(
            "DOT graph has source nodes without input specs: "
            + ", ".join(missing_specs)
            + ". For v0, constants must be explicit inputs."
        )
    if allow_unknown_sources:
        for name in missing_specs:
            spec_by_name[name] = TensorSpec(name=name, shape=tuple(), dtype="fp32")

    ordered_inputs = [spec_by_name[name] for name in source_names]
    if sink_names:
        outputs = sink_names
    elif nodes:
        outputs = [nodes[-1].output]
    else:
        raise ValueError("DOT parse error: no outputs discovered.")

    graph = Graph(inputs=ordered_inputs, nodes=nodes, outputs=outputs)
    graph.validate()
    return graph


@contextmanager
def _temporary_dot_output_path(dot_output_path: Path | None):
    if dot_output_path is not None:
        yield Path(dot_output_path)
        return

    with TemporaryDirectory(prefix="mlx2coreai_dot_") as temp_dir:
        yield Path(temp_dir) / "capture.dot"


def _capture_graph_from_precomputed_outputs(
    dot_output_path: Path | None,
    numpy_inputs: dict[str, np.ndarray],
    mx_inputs: dict[str, Any],
    outputs: Any,
    *,
    input_specs: list[TensorSpec] | None = None,
    allow_unknown_sources: bool = False,
) -> tuple[Graph, dict[str, np.ndarray], dict[str, np.ndarray]]:
    output_values = _normalize_outputs(outputs)

    if not output_values:
        raise ValueError("capture requires at least one output.")

    with _temporary_dot_output_path(dot_output_path) as resolved_dot_output_path:
        resolved_dot_output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(output_values) == 1:
            # MLX import is intentionally lazy to allow non-live operation in restricted envs.
            import mlx.core as mx  # noqa: PLC0415

            mx.export_to_dot(str(resolved_dot_output_path), output_values[0], **mx_inputs)
        else:
            import mlx.core as mx  # noqa: PLC0415

            mx.export_to_dot(str(resolved_dot_output_path), *output_values, **mx_inputs)

        dot_text = resolved_dot_output_path.read_text(encoding="utf-8")
    parser_specs = input_specs if input_specs is not None else _default_input_specs(numpy_inputs)
    graph = parse_mlx_dot_to_graph(
        dot_text,
        input_specs=list(parser_specs),
        allow_unknown_sources=allow_unknown_sources,
    )

    expected_arrays = [_constant_to_numpy(value) for value in output_values]
    if len(graph.outputs) > len(expected_arrays):
        # MLX DOT export can include additional sink tensors unrelated to requested outputs.
        selected_outputs = graph.outputs[-len(expected_arrays) :]
        graph = replace(graph, outputs=selected_outputs)
        graph.validate()
    elif len(graph.outputs) < len(expected_arrays):
        raise ValueError(
            f"Captured output mismatch: parser found {len(graph.outputs)} outputs, "
            f"but {len(expected_arrays)} outputs were provided."
        )
    expected = {
        graph.outputs[index]: expected_arrays[index]
        for index in range(len(graph.outputs))
    }
    return graph, numpy_inputs, expected


def capture_graph_from_mlx_outputs(
    dot_output_path: Path | None,
    inputs: dict[str, Any],
    outputs: Any,
    *,
    input_specs: list[TensorSpec] | None = None,
    allow_unknown_sources: bool = False,
) -> tuple[Graph, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Capture a DOT graph from precomputed MLX outputs and parse into translator IR.

    Args:
        dot_output_path: Optional DOT file path. When omitted, a temporary file is used.
        inputs: Named input tensors/arrays used to produce outputs.
        outputs: Single output tensor, sequence of output tensors, or dict of outputs.
        input_specs: Optional explicit input specs used by DOT parser.
        allow_unknown_sources: Allow parser to keep source nodes without input specs.
    """
    # MLX import is intentionally lazy to allow non-live operation in restricted envs.
    import mlx.core as mx  # noqa: PLC0415

    numpy_inputs = _normalize_numpy_inputs(inputs)
    mx_inputs = {name: mx.array(value) for name, value in numpy_inputs.items()}
    return _capture_graph_from_precomputed_outputs(
        dot_output_path=dot_output_path,
        numpy_inputs=numpy_inputs,
        mx_inputs=mx_inputs,
        outputs=outputs,
        input_specs=input_specs,
        allow_unknown_sources=allow_unknown_sources,
    )


def capture_smoke_graph(dot_output_path: Path, seed: int = 0) -> tuple[Graph, dict[str, np.ndarray], np.ndarray]:
    from .from_mlx import capture_graph_from_mlx_function

    # MLX import is intentionally lazy to allow mock-mode operation in restricted envs.
    import mlx.core as mx  # noqa: PLC0415

    inputs = build_smoke_numpy_inputs(seed=seed)
    graph, numpy_inputs, expected = capture_graph_from_mlx_function(
        dot_output_path=dot_output_path,
        inputs=inputs,
        function=lambda x, w, b, z: mx.maximum(mx.add(mx.matmul(x, w), b), z),
    )
    if len(expected) != 1:
        raise ValueError(f"Smoke capture expected 1 output but got {len(expected)}.")
    return graph, numpy_inputs, next(iter(expected.values()))


def _ir_dtype_to_mx(dtype: str, mx: Any) -> Any:
    mapping = {
        "fp16": mx.float16,
        "fp32": mx.float32,
        "int32": mx.int32,
        "int64": mx.int64,
        "bool": mx.bool_,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported IR dtype for MLX replay capture: {dtype}")
    return mapping[dtype]


def _as_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    return (int(value),)


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _conv_padding_from_attrs(attrs: dict[str, Any]) -> int | tuple[int, int]:
    pad = str(attrs.get("pad_type", "valid")).lower()
    if pad == "valid":
        return 0
    if pad == "same":
        return tuple(int(v) for v in attrs.get("padding", [0, 0]))
    raise ValueError(f"Unsupported pad_type for MLX replay capture: {pad}")


def _eval_node_with_mlx(node: Node, values: dict[str, Any], mx: Any) -> Any:
    if len(node.outputs) > 1:
        return tuple(_eval_node_with_mlx(node.result_node(index), values, mx) for index in range(len(node.outputs)))
    args = [values[name] for name in node.inputs]
    attrs = dict(node.attrs)
    op = node.op

    if op == "pad":
        padding = attrs["padding"]
        return mx.pad(args[0], list(zip(padding[::2], padding[1::2])), constant_values=args[1])
    if op == "argreduce":
        fn = mx.argmin if attrs["mode"] == 0 else mx.argmax
        return fn(args[0], axis=attrs["axis"], keepdims=attrs.get("keep_dims", True))

    if op == "gated_delta_update":
        from mlx_lm.models.gated_delta import gated_delta_ops

        return gated_delta_ops(*args)[int(attrs.get("output_index", 0))]

    if op == "add":
        return mx.add(args[0], args[1])
    if op == "subtract":
        return mx.subtract(args[0], args[1])
    if op == "multiply":
        return mx.multiply(args[0], args[1])
    if op == "divide":
        return mx.divide(args[0], args[1])
    if op == "power":
        return mx.power(args[0], args[1])
    if op == "reciprocal":
        return mx.reciprocal(args[0])
    if op == "remainder":
        return mx.remainder(args[0], args[1])
    if op == "matmul":
        return mx.matmul(args[0], args[1])
    if op == "maximum":
        return mx.maximum(args[0], args[1])

    if op in {"sum", "mean", "min", "max", "prod"}:
        axes = _as_tuple(attrs.get("axes"))
        keepdims = _as_bool(attrs.get("keep_dims"), False)
        fn = {
            "sum": mx.sum,
            "mean": mx.mean,
            "min": mx.min,
            "max": mx.max,
            "prod": mx.prod,
        }[op]
        return fn(args[0], axis=axes, keepdims=keepdims)

    if op in {"argmax", "argmin"}:
        axis = _as_int(attrs.get("axis"), 0)
        keepdims = _as_bool(attrs.get("keep_dims"), False)
        fn = mx.argmax if op == "argmax" else mx.argmin
        out = fn(args[0], axis=axis)
        if keepdims:
            out = mx.expand_dims(out, axis=axis)
        return out

    if op == "flatten":
        shape = tuple(int(v) for v in attrs["shape"])
        return mx.reshape(args[0], shape)
    if op == "unflatten":
        shape = tuple(int(v) for v in attrs["shape"])
        return mx.reshape(args[0], shape)
    if op == "atleast_1d":
        return mx.atleast_1d(args[0])
    if op == "atleast_2d":
        return mx.atleast_2d(args[0])
    if op == "atleast_3d":
        return mx.atleast_3d(args[0])
    if op == "moveaxis":
        return mx.moveaxis(args[0], int(attrs["source"]), int(attrs["destination"]))
    if op == "swapaxes":
        return mx.swapaxes(args[0], int(attrs["axis1"]), int(attrs["axis2"]))
    if op == "slice":
        begin = [int(v) for v in attrs.get("begin", [])]
        end = [int(v) for v in attrs.get("end", [])]
        stride = [int(v) for v in attrs.get("stride", [1] * len(begin))]
        index = tuple(slice(b, e, s) for b, e, s in zip(begin, end, stride))
        return args[0][index]
    if op == "take":
        axis = attrs.get("axis")
        return mx.take(args[0], args[1], axis=None if axis is None else int(axis))
    if op == "take_along_axis":
        axis = attrs.get("axis")
        return mx.take_along_axis(args[0], args[1], axis=None if axis is None else int(axis))

    if op == "zeros":
        shape = tuple(int(v) for v in attrs["shape"])
        dtype = _ir_dtype_to_mx(str(attrs.get("dtype", "fp32")), mx)
        return mx.zeros(shape, dtype=dtype)
    if op == "ones":
        shape = tuple(int(v) for v in attrs["shape"])
        dtype = _ir_dtype_to_mx(str(attrs.get("dtype", "fp32")), mx)
        return mx.ones(shape, dtype=dtype)
    if op == "full":
        shape = tuple(int(v) for v in attrs["shape"])
        value = attrs.get("value", 0.0)
        dtype = _ir_dtype_to_mx(str(attrs.get("dtype", "fp32")), mx)
        return mx.full(shape, value, dtype=dtype)
    if op == "zeros_like":
        return mx.zeros_like(args[0])
    if op == "ones_like":
        return mx.ones_like(args[0])
    if op == "full_like":
        if hasattr(mx, "full_like"):
            return mx.full_like(args[0], attrs.get("value", 0.0))
        return mx.full(tuple(int(v) for v in args[0].shape), attrs.get("value", 0.0), dtype=args[0].dtype)
    if op == "arange":
        return mx.arange(int(attrs["start"]), int(attrs["end"]), int(attrs.get("step", 1)))
    if op == "linspace":
        dtype = _ir_dtype_to_mx(str(attrs.get("dtype", "fp32")), mx)
        start = float(attrs["start"])
        stop = float(attrs["stop"])
        num = int(attrs["num"])
        endpoint = bool(attrs.get("endpoint", True))
        if not endpoint and num > 1:
            stop = stop - ((stop - start) / num)
        return mx.linspace(
            start,
            stop,
            num,
            dtype=dtype,
        )
    if op == "where":
        return mx.where(args[0], args[1], args[2])
    if op in {"astype", "cast"}:
        dtype = _ir_dtype_to_mx(str(attrs["dtype"]), mx)
        return args[0].astype(dtype)
    if op == "number_of_elements":
        return mx.array(int(np.prod(tuple(int(v) for v in args[0].shape))), dtype=mx.int32)
    if op == "stop_gradient":
        return mx.stop_gradient(args[0])

    if op == "addmm":
        alpha = float(attrs.get("alpha", 1.0))
        beta = float(attrs.get("beta", 1.0))
        x, y, bias = args if attrs.get("input_order", "cab") == "abc" else (args[1], args[2], args[0])
        return (beta * bias) + (alpha * mx.matmul(x, y))
    if op == "broadcast_arrays":
        outputs = mx.broadcast_arrays(*args)
        return outputs[int(attrs.get("input_index", 0))]
    if op == "broadcast_axes":
        shape = infer_broadcast_axes_shape([arg.shape for arg in args], attrs.get("ignore_axes", []))
        return mx.broadcast_to(args[0], shape)
    if op == "tensordot":
        axes = attrs.get("axes", 2)
        return mx.tensordot(args[0], args[1], axes=axes)
    if op == "isclose":
        return mx.isclose(
            args[0],
            args[1],
            rtol=float(attrs.get("rtol", 1e-5)),
            atol=float(attrs.get("atol", 1e-8)),
            equal_nan=bool(attrs.get("equal_nan", False)),
        )
    if op == "allclose":
        out = mx.allclose(
            args[0],
            args[1],
            rtol=float(attrs.get("rtol", 1e-5)),
            atol=float(attrs.get("atol", 1e-8)),
            equal_nan=bool(attrs.get("equal_nan", False)),
        )
        if isinstance(out, bool):
            return mx.array(out, dtype=mx.bool_)
        return out
    if op == "nan_to_num":
        return mx.nan_to_num(
            args[0],
            nan=float(attrs.get("nan", 0.0)),
            posinf=float(attrs.get("posinf", 0.0)),
            neginf=float(attrs.get("neginf", 0.0)),
        )
    if op == "diag":
        return mx.diag(args[0], k=int(attrs.get("k", 0)))
    if op == "diagonal":
        return mx.diagonal(
            args[0],
            offset=int(attrs.get("offset", 0)),
            axis1=int(attrs.get("axis1", 0)),
            axis2=int(attrs.get("axis2", 1)),
        )
    if op == "trace":
        return mx.trace(
            args[0],
            offset=int(attrs.get("offset", 0)),
            axis1=int(attrs.get("axis1", 0)),
            axis2=int(attrs.get("axis2", 1)),
        )
    if op == "tri":
        dtype = _ir_dtype_to_mx(str(attrs.get("dtype", "fp32")), mx)
        return mx.tri(int(attrs["n"]), int(attrs["m"]), k=int(attrs.get("k", 0)), dtype=dtype)
    if op == "tril":
        return mx.tril(args[0], k=int(attrs.get("k", 0)))
    if op == "triu":
        return mx.triu(args[0], k=int(attrs.get("k", 0)))

    if op == "all":
        return mx.all(args[0])
    if op == "any":
        return mx.any(args[0])
    if op == "array_equal":
        out = mx.array_equal(args[0], args[1])
        if isinstance(out, bool):
            return mx.array(out, dtype=mx.bool_)
        return out
    if op == "isnan":
        return mx.isnan(args[0])
    if op == "isinf":
        return mx.isinf(args[0])
    if op == "isfinite":
        return mx.isfinite(args[0])
    if op == "isneginf":
        if hasattr(mx, "isneginf"):
            return mx.isneginf(args[0])
        return mx.logical_and(mx.isinf(args[0]), args[0] < 0)
    if op == "isposinf":
        if hasattr(mx, "isposinf"):
            return mx.isposinf(args[0])
        return mx.logical_and(mx.isinf(args[0]), args[0] > 0)

    if op == "eye":
        dtype = _ir_dtype_to_mx(str(attrs.get("dtype", "fp32")), mx)
        return mx.eye(
            int(attrs["n"]),
            int(attrs.get("m", attrs["n"])),
            k=int(attrs.get("k", 0)),
            dtype=dtype,
        )
    if op == "meshgrid":
        outputs = mx.meshgrid(*args, indexing=str(attrs.get("indexing", "xy")))
        return outputs[int(attrs.get("input_index", attrs.get("output_index", 0)))]
    if op == "kron":
        return mx.kron(args[0], args[1])
    if op == "logaddexp":
        return mx.logaddexp(args[0], args[1])
    if op == "concatenate":
        return mx.concatenate(args, axis=int(attrs.get("axis", 0)))

    if op == "arccos":
        return mx.arccos(args[0])
    if op == "arcsin":
        return mx.arcsin(args[0])
    if op == "arctan":
        return mx.arctan(args[0])
    if op == "arctanh":
        return mx.arctanh(args[0])
    if op == "negative":
        return mx.negative(args[0])
    if op == "degrees":
        return mx.degrees(args[0])
    if op == "radians":
        return mx.radians(args[0])
    if op == "expm1":
        return mx.expm1(args[0])
    if op == "log1p":
        return mx.log1p(args[0])
    if op == "log2":
        return mx.log2(args[0])
    if op == "log10":
        return mx.log10(args[0])
    if op == "logsumexp":
        axes = _as_tuple(attrs.get("axes"))
        keepdims = _as_bool(attrs.get("keep_dims"), False)
        return mx.logsumexp(args[0], axis=axes, keepdims=keepdims)
    if op == "floor_divide":
        return mx.floor_divide(args[0], args[1])

    if op == "var":
        axes = _as_tuple(attrs.get("axes"))
        keepdims = _as_bool(attrs.get("keep_dims"), False)
        ddof = int(attrs.get("ddof", 0))
        return mx.var(args[0], axis=axes, keepdims=keepdims, ddof=ddof)
    if op == "std":
        axes = _as_tuple(attrs.get("axes"))
        keepdims = _as_bool(attrs.get("keep_dims"), False)
        ddof = int(attrs.get("ddof", 0))
        return mx.std(args[0], axis=axes, keepdims=keepdims, ddof=ddof)
    if op == "divmod":
        q, r = mx.divmod(args[0], args[1])
        output = str(attrs.get("output", "quotient"))
        return q if output == "quotient" else r

    if op in {"conv2d", "conv_general"}:
        x = args[0]
        w = args[1]
        b = args[2] if len(args) > 2 else None
        # MLX core conv kernels use NHWC; zoo fixtures are NCHW.
        x_nhwc = mx.transpose(x, (0, 2, 3, 1))
        w_hwio = mx.transpose(w, (0, 2, 3, 1))
        stride = tuple(int(v) for v in attrs.get("strides", attrs.get("stride", [1, 1])))
        padding = _conv_padding_from_attrs(attrs)
        if op == "conv2d":
            y_nhwc = mx.conv2d(x_nhwc, w_hwio, stride=stride, padding=padding)
        else:
            y_nhwc = mx.conv_general(x_nhwc, w_hwio, stride=stride, padding=padding)
        y = mx.transpose(y_nhwc, (0, 3, 1, 2))
        if b is not None:
            y = y + mx.reshape(b, (1, int(b.shape[0]), 1, 1))
        return y

    if op == "conv_transpose2d":
        x = args[0]
        w = args[1]
        b = args[2] if len(args) > 2 else None
        x_nhwc = mx.transpose(x, (0, 2, 3, 1))
        # Zoo fixtures use (C_in, C_out, KH, KW) for transposed conv weights.
        w_hwio = mx.transpose(w, (1, 2, 3, 0))
        stride = tuple(int(v) for v in attrs.get("strides", attrs.get("stride", [1, 1])))
        padding = _conv_padding_from_attrs(attrs)
        y_nhwc = mx.conv_transpose2d(x_nhwc, w_hwio, stride=stride, padding=padding)
        y = mx.transpose(y_nhwc, (0, 3, 1, 2))
        if b is not None:
            y = y + mx.reshape(b, (1, int(b.shape[0]), 1, 1))
        return y

    raise ValueError(f"MLX replay capture does not support op '{op}' yet.")


def export_dot_from_ir(
    dot_output_path: Path,
    graph: Graph,
    inputs: dict[str, np.ndarray],
) -> None:
    # MLX import is intentionally lazy to allow non-live operation in restricted envs.
    import mlx.core as mx  # noqa: PLC0415

    dot_output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.validate()

    missing = [spec.name for spec in graph.inputs if spec.name not in inputs]
    if missing:
        raise ValueError(f"Missing numpy inputs for live capture: {', '.join(missing)}")

    mx_inputs = {spec.name: mx.array(inputs[spec.name]) for spec in graph.inputs}
    values: dict[str, Any] = dict(mx_inputs)
    for node in graph.nodes:
        result = _eval_node_with_mlx(node, values, mx)
        if len(node.outputs) == 1:
            values[node.output] = result
        else:
            values.update(zip(node.outputs, result, strict=True))

    output_values = [values[name] for name in graph.outputs]
    if len(output_values) == 1:
        mx.export_to_dot(str(dot_output_path), output_values[0], **mx_inputs)
    else:
        mx.export_to_dot(str(dot_output_path), *output_values, **mx_inputs)


def capture_graph_from_ir(
    dot_output_path: Path,
    graph: Graph,
    inputs: dict[str, np.ndarray],
) -> Graph:
    export_dot_from_ir(dot_output_path=dot_output_path, graph=graph, inputs=inputs)

    dot_text = dot_output_path.read_text(encoding="utf-8")
    input_specs = [
        TensorSpec(
            name=spec.name,
            shape=tuple(int(v) for v in inputs[spec.name].shape),
            dtype=_numpy_dtype_to_ir(np.asarray(inputs[spec.name]).dtype),
        )
        for spec in graph.inputs
    ]
    return parse_mlx_dot_to_graph(dot_text, input_specs=input_specs)
