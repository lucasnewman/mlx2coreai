from __future__ import annotations

from contextlib import contextmanager
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import numpy as np
import ml_dtypes

from .ir import Graph, Node, TensorSpec
from .op_registry import normalize_mlx_op_name

_SOURCE_RE = re.compile(r'rank=source;\s*"([^"]+)"')
_SINK_RE = re.compile(r'rank=sink;\s*"([^"]+)"')
_OP_RE = re.compile(r'^\{\s*(\d+)\s+\[label\s*=\s*"(.*?)",\s*shape=rectangle\];\s*\}$')
_EDGE_RE = re.compile(r'^"?([^"]+?)"?\s*->\s*"?([^"]+?)"?$')


def _numpy_dtype_to_ir(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == ml_dtypes.bfloat16:
        return "bf16"
    if dtype == np.float16:
        return "fp16"
    if dtype == np.float32:
        return "fp32"
    if dtype == np.int32:
        return "int32"
    if dtype == np.int64:
        return "int64"
    if dtype == np.bool_:
        return "bool"
    if np.issubdtype(dtype, np.floating):
        if dtype.itemsize <= np.dtype(np.float16).itemsize:
            return "fp16"
        return "fp32"
    if np.issubdtype(dtype, np.signedinteger):
        if dtype.itemsize <= np.dtype(np.int32).itemsize:
            return "int32"
        return "int64"
    if np.issubdtype(dtype, np.unsignedinteger):
        if dtype.itemsize <= np.dtype(np.uint32).itemsize:
            return "int32"
        return "int64"
    raise ValueError(f"Unsupported dtype for v0 translator: {dtype}")


def _mlx_dtype_to_ir(dtype: Any) -> str:
    if isinstance(dtype, np.dtype):
        return _numpy_dtype_to_ir(dtype)
    text = str(dtype).strip().lower()
    if "bfloat16" in text or text.endswith("bf16"):
        return "bf16"
    if text.endswith("float16") or text.endswith("fp16"):
        return "fp16"
    if text.endswith("float32") or text.endswith("fp32"):
        return "fp32"
    if text.endswith("int32"):
        return "int32"
    if text.endswith("int64"):
        return "int64"
    if text.endswith("bool"):
        return "bool"
    raise ValueError(f"Unsupported MLX dtype for capture parser: {dtype}")


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if value is None:
        return tuple()
    return (int(value),)


def _constant_to_numpy(value: Any) -> np.ndarray:
    try:
        return np.asarray(value)
    except Exception:
        # MLX bf16 arrays can fail direct numpy conversion via buffer protocol.
        if hasattr(value, "astype"):
            try:
                import mlx.core as mx  # noqa: PLC0415

                if "bfloat16" in str(getattr(value, "dtype", "")).lower():
                    return np.asarray(value.astype(mx.float32)).astype(ml_dtypes.bfloat16)
                return np.asarray(value.astype(mx.float32))
            except Exception:
                pass
        raise


def _tensor_spec_from_event_entry(entry: Any, *, name_override: str | None = None) -> TensorSpec:
    if not isinstance(entry, (tuple, list)) or len(entry) != 3:
        raise ValueError(f"Invalid MLX callback tensor entry: {entry!r}")
    name = str(name_override if name_override is not None else entry[0])
    shape = _shape_tuple(entry[1])
    dtype = _mlx_dtype_to_ir(entry[2])
    return TensorSpec(name=name, shape=shape, dtype=dtype)


def _int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [int(v) for v in value]
    return [int(value)]


def _primitive_attrs_from_arguments(
    op: str,
    arguments: list[Any],
    output_shape: tuple[int, ...] | None,
    output_dtype: str | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}

    if op in {"reshape", "flatten", "unflatten", "broadcast", "broadcast_to"} and output_shape is not None:
        attrs["shape"] = list(output_shape)

    if op == "transpose" and arguments:
        perm = _int_list(arguments[0])
        if perm is not None:
            attrs["perm"] = perm

    if op == "moveaxis" and len(arguments) >= 2:
        attrs["source"] = int(arguments[0])
        attrs["destination"] = int(arguments[1])

    if op == "swapaxes" and len(arguments) >= 2:
        attrs["axis1"] = int(arguments[0])
        attrs["axis2"] = int(arguments[1])

    if op in {"slice", "slice_update", "sliceupdate"}:
        if len(arguments) >= 1:
            begin = _int_list(arguments[0])
            if begin is not None:
                attrs["begin"] = begin
        if len(arguments) >= 2:
            end = _int_list(arguments[1])
            if end is not None:
                attrs["end"] = end
        if len(arguments) >= 3:
            stride = _int_list(arguments[2])
            if stride is not None:
                attrs["stride"] = stride

    if op in {"dynamic_slice_update", "dynamicsliceupdate"} and arguments:
        axes = _int_list(arguments[0])
        if axes is not None:
            attrs["axes"] = axes

    if op in {"sum", "mean", "min", "max", "prod", "all", "any", "var", "std", "logsumexp"}:
        if arguments:
            axes = _int_list(arguments[0])
            if axes is not None:
                attrs["axes"] = axes
        if len(arguments) >= 2:
            attrs["keep_dims"] = bool(arguments[1])
        if op in {"var", "std"} and len(arguments) >= 3:
            attrs["ddof"] = int(arguments[2])

    if op == "reduce":
        if arguments:
            attrs["mode"] = int(arguments[0])
        if len(arguments) >= 2:
            axes = _int_list(arguments[1])
            if axes is not None:
                attrs["axes"] = axes
        # MLX callback Reduce currently emits rank-preserving reductions.
        attrs["keep_dims"] = bool(arguments[2]) if len(arguments) >= 3 else True

    if op in {"argmax", "argmin"}:
        if arguments:
            attrs["axis"] = int(arguments[0])
        if len(arguments) >= 2:
            attrs["keep_dims"] = bool(arguments[1])

    if op in {"take", "take_along_axis"} and arguments:
        attrs["axis"] = int(arguments[0])

    if op == "split" and arguments:
        split_arg = arguments[0]
        if isinstance(split_arg, (list, tuple)):
            attrs["split_indices"] = [int(v) for v in split_arg]
        else:
            attrs["num_splits"] = int(split_arg)
        if len(arguments) >= 2:
            axis = _int_list(arguments[1])
            if axis:
                attrs["axis"] = int(axis[0])
        else:
            attrs["axis"] = 0

    if op == "expanddims" and arguments:
        axes = _int_list(arguments[0])
        if axes is not None:
            attrs["axes"] = axes

    if op == "gather" and arguments:
        axes = _int_list(arguments[0])
        if axes:
            attrs["axis"] = int(axes[0])
        if len(arguments) >= 2:
            slice_shape = _int_list(arguments[1])
            if slice_shape is not None:
                attrs["slice_shape"] = slice_shape
        if output_shape is not None:
            attrs["shape"] = list(output_shape)

    if op == "squeeze" and arguments:
        axes = _int_list(arguments[0])
        if axes is not None:
            attrs["axes"] = axes

    if op in {"concatenate", "concat"} and arguments:
        attrs["axis"] = int(arguments[0])

    if op == "softmax" and arguments:
        # MLX export can emit [precise] when axis is defaulted to -1.
        first = arguments[0]
        if isinstance(first, bool):
            attrs["precise"] = bool(first)
        else:
            attrs["axis"] = int(first)
            if len(arguments) >= 2 and isinstance(arguments[1], bool):
                attrs["precise"] = bool(arguments[1])

    if op == "rmsnorm" and arguments:
        # Primitive state: [eps]
        attrs["eps"] = float(arguments[0])

    if op == "rope" and arguments:
        # Primitive state: [dims, traditional, base, scale, ...].
        state = [arg for arg in arguments if arg is not None]
        if len(state) >= 1:
            attrs["dims"] = int(state[0])
        if len(state) >= 2:
            attrs["traditional"] = bool(state[1])
        if len(state) >= 3:
            attrs["base"] = float(state[2])
        if len(state) >= 4:
            attrs["scale"] = float(state[3])

    if op in {"astype", "cast"} and output_dtype is not None:
        attrs["dtype"] = output_dtype

    if op == "bitwisebinary" and arguments:
        attrs["mode"] = int(arguments[0])

    if op == "scaled_dot_product_attention" and arguments:
        # MLX fast primitive state is [scale, do_causal, has_sinks, output_logsumexp].
        # Some serializers may preserve placeholder nulls; filter them first.
        state = [arg for arg in arguments if arg is not None]
        if len(state) >= 4:
            attrs["scale"] = float(state[0])
            attrs["do_causal"] = bool(state[1])
            attrs["has_sinks"] = bool(state[2])
            attrs["output_logsumexp"] = bool(state[3])

    if op == "convolution" and arguments:
        strides = _int_list(arguments[0])
        if strides is not None:
            attrs["strides"] = strides

        if len(arguments) >= 2:
            pad_lo = _int_list(arguments[1])
            if pad_lo is not None:
                attrs["padding"] = pad_lo
                attrs["pad_type"] = "custom"

        if len(arguments) >= 3:
            pad_hi = _int_list(arguments[2])
            if pad_hi is not None:
                pad_lo = attrs.get("padding")
                if isinstance(pad_lo, list) and len(pad_lo) == len(pad_hi):
                    attrs["padding"] = [int(lo) + int(hi) for lo, hi in zip(pad_lo, pad_hi)]

        if len(arguments) >= 4:
            dilations = _int_list(arguments[3])
            if dilations is not None:
                attrs["dilations"] = dilations

        if len(arguments) >= 6:
            try:
                attrs["groups"] = int(arguments[5])
            except Exception:
                pass

        if len(arguments) >= 7 and isinstance(arguments[6], (bool, np.bool_)):
            attrs["transpose"] = bool(arguments[6])

    if op == "arange" and len(arguments) >= 3:
        attrs["start"] = int(arguments[0])
        attrs["end"] = int(arguments[1])
        attrs["step"] = int(arguments[2])

    if op == "linspace" and len(arguments) >= 3:
        attrs["start"] = float(arguments[0])
        attrs["stop"] = float(arguments[1])
        attrs["num"] = int(arguments[2])
        if len(arguments) >= 4:
            attrs["endpoint"] = bool(arguments[3])

    return attrs


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


def parse_mlx_export_events_to_graph(
    events: list[dict[str, Any]],
    input_specs: list[TensorSpec],
    allow_unknown_sources: bool = False,
) -> Graph:
    input_entries: list[TensorSpec] = []
    keyword_pairs: list[tuple[str, str]] = []
    callback_outputs: list[TensorSpec] = []
    constant_entries: list[tuple[str, np.ndarray]] = []
    primitive_events: list[dict[str, Any]] = []
    tensor_specs_by_name: dict[str, TensorSpec] = {}

    for event in events:
        event_type = str(event.get("type", ""))
        if event_type == "inputs":
            entries = event.get("inputs", [])
            for entry in entries:
                spec = _tensor_spec_from_event_entry(entry)
                input_entries.append(spec)
                tensor_specs_by_name[spec.name] = spec
            continue

        if event_type == "keyword_inputs":
            for pair in event.get("keywords", []):
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError(f"Invalid MLX keyword input entry: {pair!r}")
                keyword_pairs.append((str(pair[0]), str(pair[1])))
            continue

        if event_type == "outputs":
            for entry in event.get("outputs", []):
                spec = _tensor_spec_from_event_entry(entry)
                callback_outputs.append(spec)
                tensor_specs_by_name[spec.name] = spec
            continue

        if event_type == "constants":
            for pair in event.get("constants", []):
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError(f"Invalid MLX constants entry: {pair!r}")
                const_name = str(pair[0])
                const_value = _constant_to_numpy(pair[1])
                constant_entries.append((const_name, const_value))
                tensor_specs_by_name[const_name] = TensorSpec(
                    name=const_name,
                    shape=tuple(int(v) for v in const_value.shape),
                    dtype=_numpy_dtype_to_ir(const_value.dtype),
                )
            continue

        if event_type == "primitive":
            primitive_events.append(event)
            for entry in event.get("inputs", []):
                spec = _tensor_spec_from_event_entry(entry)
                tensor_specs_by_name.setdefault(spec.name, spec)
            for entry in event.get("outputs", []):
                spec = _tensor_spec_from_event_entry(entry)
                tensor_specs_by_name[spec.name] = spec

    alias_by_tensor_name: dict[str, str] = {tensor_name: kw for kw, tensor_name in keyword_pairs}
    if not keyword_pairs and input_specs and len(input_entries) == len(input_specs):
        for entry, spec in zip(input_entries, input_specs):
            alias_by_tensor_name[entry.name] = spec.name

    def _rename(name: str) -> str:
        return alias_by_tensor_name.get(name, name)

    provided_specs = {spec.name: spec for spec in input_specs}
    ordered_inputs: list[TensorSpec] = []
    used_input_names: set[str] = set()

    if keyword_pairs:
        for kw_name, tensor_name in keyword_pairs:
            if kw_name in used_input_names:
                continue
            if kw_name in provided_specs:
                ordered_inputs.append(provided_specs[kw_name])
            else:
                base = tensor_specs_by_name.get(tensor_name)
                if base is None:
                    raise ValueError(
                        f"Missing tensor spec for keyword input '{kw_name}' ({tensor_name}) in callback payload."
                    )
                ordered_inputs.append(TensorSpec(name=kw_name, shape=base.shape, dtype=base.dtype))
            used_input_names.add(kw_name)
    else:
        for entry in input_entries:
            name = _rename(entry.name)
            if name in used_input_names:
                continue
            if name in provided_specs:
                ordered_inputs.append(provided_specs[name])
            else:
                ordered_inputs.append(TensorSpec(name=name, shape=entry.shape, dtype=entry.dtype))
            used_input_names.add(name)

    nodes: list[Node] = []
    produced: set[str] = set()
    for const_name, const_value in constant_entries:
        output_name = _rename(const_name)
        nodes.append(
            Node(
                op="const",
                inputs=tuple(),
                output=output_name,
                attrs={"value": const_value},
                source=f"mlx_export:const:{const_name}",
            )
        )
        produced.add(output_name)

    for primitive_index, primitive in enumerate(primitive_events):
        raw_name = str(primitive.get("name", ""))
        op = normalize_mlx_op_name(raw_name)
        input_entries_raw = primitive.get("inputs", [])
        output_entries_raw = primitive.get("outputs", [])
        arguments = list(primitive.get("arguments", []))

        inputs = []
        for entry in input_entries_raw:
            if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                raise ValueError(f"Invalid primitive input entry: {entry!r}")
            inputs.append(_rename(str(entry[0])))

        parsed_outputs: list[TensorSpec] = []
        for entry in output_entries_raw:
            spec = _tensor_spec_from_event_entry(entry)
            parsed_outputs.append(TensorSpec(name=_rename(spec.name), shape=spec.shape, dtype=spec.dtype))

        if not parsed_outputs:
            continue

        for output_index, output_spec in enumerate(parsed_outputs):
            attrs = _primitive_attrs_from_arguments(
                op,
                arguments,
                output_spec.shape,
                output_dtype=output_spec.dtype,
            )
            if len(parsed_outputs) > 1:
                attrs["output_index"] = output_index
                attrs["num_outputs"] = len(parsed_outputs)
            nodes.append(
                Node(
                    op=op,
                    inputs=tuple(inputs),
                    output=output_spec.name,
                    attrs=attrs,
                    source=f"mlx_export:{primitive_index}:{raw_name}",
                )
            )
            produced.add(output_spec.name)

    outputs = [_rename(spec.name) for spec in callback_outputs]
    if not outputs and nodes:
        outputs = [nodes[-1].output]
    if not outputs:
        raise ValueError("MLX callback parse error: no outputs discovered.")

    source_names: list[str] = []
    source_name_set: set[str] = set()
    available = {spec.name for spec in ordered_inputs}
    available.update(produced)
    for node in nodes:
        for input_name in node.inputs:
            if input_name in available:
                continue
            if input_name not in source_name_set:
                source_name_set.add(input_name)
                source_names.append(input_name)

    if source_names and not allow_unknown_sources:
        raise ValueError(
            "MLX callback graph has source nodes without input specs: "
            + ", ".join(source_names)
            + ". Pass allow_unknown_sources=True to keep them as explicit inputs."
        )

    if allow_unknown_sources:
        present = {spec.name for spec in ordered_inputs}
        for name in source_names:
            if name in present:
                continue
            base = tensor_specs_by_name.get(name)
            if base is None:
                spec = TensorSpec(name=name, shape=tuple(), dtype="fp32")
            else:
                spec = TensorSpec(name=name, shape=base.shape, dtype=base.dtype)
            ordered_inputs.append(spec)
            present.add(name)

    graph = Graph(inputs=ordered_inputs, nodes=nodes, outputs=outputs)
    graph.validate()
    return graph


def _normalize_numpy_inputs(inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {name: _constant_to_numpy(value) for name, value in inputs.items()}


def _default_input_specs(inputs: dict[str, np.ndarray]) -> list[TensorSpec]:
    return [
        TensorSpec(
            name=name,
            shape=tuple(int(v) for v in array.shape),
            dtype=_numpy_dtype_to_ir(np.asarray(array).dtype),
        )
        for name, array in inputs.items()
    ]


def _normalize_outputs(outputs: Any) -> list[Any]:
    if isinstance(outputs, dict):
        return list(outputs.values())
    if isinstance(outputs, (tuple, list)):
        return list(outputs)
    return [outputs]


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
        graph = Graph(inputs=list(graph.inputs), nodes=list(graph.nodes), outputs=selected_outputs)
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


def _capture_graph_from_mlx_function_callback(
    dot_output_path: Path | None,
    numpy_inputs: dict[str, np.ndarray],
    mx_inputs: dict[str, Any],
    function: Callable[..., Any],
    *,
    input_specs: list[TensorSpec] | None = None,
    allow_unknown_sources: bool = False,
    write_dot_debug: bool = True,
    shapeless: bool = False,
) -> tuple[Graph, dict[str, np.ndarray], dict[str, np.ndarray]]:
    # MLX import is intentionally lazy to allow non-live operation in restricted envs.
    import mlx.core as mx  # noqa: PLC0415

    events: list[dict[str, Any]] = []

    def _callback(payload: dict[str, Any]) -> None:
        events.append(payload)

    mx.export_function(_callback, function, shapeless=bool(shapeless), **mx_inputs)
    parser_specs = input_specs if input_specs is not None else _default_input_specs(numpy_inputs)
    graph = parse_mlx_export_events_to_graph(
        events,
        input_specs=list(parser_specs),
        allow_unknown_sources=allow_unknown_sources,
    )

    outputs = function(**mx_inputs)
    output_values = _normalize_outputs(outputs)
    if not output_values:
        raise ValueError("capture requires at least one output.")
    expected_arrays = [_constant_to_numpy(value) for value in output_values]

    if len(graph.outputs) > len(expected_arrays):
        selected_outputs = graph.outputs[-len(expected_arrays) :]
        graph = Graph(inputs=list(graph.inputs), nodes=list(graph.nodes), outputs=selected_outputs)
        graph.validate()
    elif len(graph.outputs) < len(expected_arrays):
        raise ValueError(
            f"Captured output mismatch: parser found {len(graph.outputs)} outputs, "
            f"but {len(expected_arrays)} outputs were provided."
        )

    if write_dot_debug and dot_output_path is not None:
        dot_output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(output_values) == 1:
            mx.export_to_dot(str(dot_output_path), output_values[0], **mx_inputs)
        else:
            mx.export_to_dot(str(dot_output_path), *output_values, **mx_inputs)

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


def capture_graph_from_mlx_function(
    dot_output_path: Path | None,
    inputs: dict[str, Any],
    function: Callable[..., Any],
    *,
    input_specs: list[TensorSpec] | None = None,
    allow_unknown_sources: bool = False,
    capture_mode: str = "callback",
    shapeless: bool = False,
) -> tuple[Graph, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Capture a graph by invoking a Python callable with named MLX inputs.

    The callable is invoked as `function(**mx_inputs)`.

    `capture_mode` controls the source graph format:
    - `callback` (default): uses `mx.export_function(..., callback=...)` and preserves
      primitive arguments needed for shape ops.
    - `dot`: preserves legacy `mx.export_to_dot` parsing behavior.
    """
    # MLX import is intentionally lazy to allow non-live operation in restricted envs.
    import mlx.core as mx  # noqa: PLC0415

    numpy_inputs = _normalize_numpy_inputs(inputs)
    mx_inputs = {name: mx.array(value) for name, value in numpy_inputs.items()}

    if capture_mode == "callback":
        return _capture_graph_from_mlx_function_callback(
            dot_output_path=dot_output_path,
            numpy_inputs=numpy_inputs,
            mx_inputs=mx_inputs,
            function=function,
            input_specs=input_specs,
            allow_unknown_sources=allow_unknown_sources,
            write_dot_debug=dot_output_path is not None,
            shapeless=shapeless,
        )

    if capture_mode == "dot":
        outputs = function(**mx_inputs)
        return _capture_graph_from_precomputed_outputs(
            dot_output_path=dot_output_path,
            numpy_inputs=numpy_inputs,
            mx_inputs=mx_inputs,
            outputs=outputs,
            input_specs=input_specs,
            allow_unknown_sources=allow_unknown_sources,
        )

    raise ValueError(
        f"Unsupported capture_mode={capture_mode!r}. Expected one of: 'callback', 'dot'."
    )


def capture_smoke_graph(dot_output_path: Path, seed: int = 0) -> tuple[Graph, dict[str, np.ndarray], np.ndarray]:
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
    args = [values[name] for name in node.inputs]
    attrs = dict(node.attrs)
    op = node.op

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
        return (beta * args[0]) + (alpha * mx.matmul(args[1], args[2]))
    if op == "broadcast_arrays":
        outputs = mx.broadcast_arrays(*args)
        return outputs[int(attrs.get("input_index", 0))]
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
        return outputs[int(attrs.get("input_index", 0))]
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
        values[node.output] = _eval_node_with_mlx(node, values, mx)

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
