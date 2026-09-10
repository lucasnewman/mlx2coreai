"""MLX callback capture: preserve primitive arguments, results, and tensor types."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
import numpy as np
from .dtypes import capture_mlx_dtype as _mlx_dtype_to_ir, capture_numpy_dtype as _numpy_dtype_to_ir
from .ir import Graph, Node, TensorSpec, TensorType
from .op_registry import decode_primitive_arguments, normalize_mlx_op_name
from ._capture_utils import (
    _shape_tuple, _constant_to_numpy, _normalize_numpy_inputs,
    _default_input_specs, _normalize_outputs,
)


def _tensor_spec_from_event_entry(entry: Any, *, name_override: str | None = None) -> TensorSpec:
    if not isinstance(entry, (tuple, list)) or len(entry) != 3:
        raise ValueError(f"Invalid MLX callback tensor entry: {entry!r}")
    name = str(name_override if name_override is not None else entry[0])
    shape = _shape_tuple(entry[1])
    dtype = _mlx_dtype_to_ir(entry[2])
    return TensorSpec(name=name, shape=shape, dtype=dtype)


def _primitive_attrs_from_arguments(op, arguments, output_shape, output_dtype=None):
    return decode_primitive_arguments(op, arguments, output_shape, output_dtype)


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

        if (
            raw_name == "CustomKernel" and arguments
            and str(arguments[0]).startswith("custom_kernel_gated_delta_step__")
        ):
            if len(input_entries_raw) != 7 or len(output_entries_raw) != 2:
                raise ValueError("Unexpected MLX gated-delta kernel signature.")
            op = "gated_delta_update"
            # The exported kernel's scalar T is a trace constant. The composite
            # derives sequence length from query at runtime instead.
            input_entries_raw = input_entries_raw[:6]
            arguments = []

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

        attrs = _primitive_attrs_from_arguments(
            op, arguments, parsed_outputs[0].shape, output_dtype=parsed_outputs[0].dtype,
        )
        names = tuple(spec.name for spec in parsed_outputs)
        nodes.append(Node(op, tuple(inputs), outputs=names, attrs=attrs,
                          source=f"mlx_export:{primitive_index}:{raw_name}"))
        produced.update(names)

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

    graph = Graph(inputs=ordered_inputs, nodes=nodes, outputs=outputs, value_types={
        _rename(name): TensorType(spec.shape, spec.dtype)
        for name, spec in tensor_specs_by_name.items() if _rename(name) in produced
    })
    graph.validate()
    return graph


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

    # MLX 0.32.2 cannot serialize Contiguous. It is only a layout hint; CoreAI
    # chooses its own layouts. Keep the original operation for reference execution.
    original_contiguous = mx.contiguous
    try:
        mx.contiguous = lambda value, *args, **kwargs: value
        mx.export_function(_callback, function, shapeless=bool(shapeless), **mx_inputs)
    finally:
        mx.contiguous = original_contiguous
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
        graph = replace(graph, outputs=selected_outputs)
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


def capture_graph_from_mlx_function(
    dot_output_path: Path | None,
    inputs: dict[str, Any],
    function: Callable[..., Any],
    *,
    input_specs: list[TensorSpec] | None = None,
    allow_unknown_sources: bool = False,
    shapeless: bool = False,
) -> tuple[Graph, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Capture a graph by invoking a Python callable with named MLX inputs.

    The callable is invoked as `function(**mx_inputs)`.

    Uses `mx.export_function(..., callback=...)` to preserve primitive arguments
    and tensor metadata. An optional DOT export is for visualization only.
    """
    # MLX import is intentionally lazy to allow non-live operation in restricted envs.
    import mlx.core as mx  # noqa: PLC0415

    numpy_inputs = _normalize_numpy_inputs(inputs)
    mx_inputs = {name: mx.array(value) for name, value in numpy_inputs.items()}

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


__all__ = [
    "Graph", "Node", "TensorSpec", "TensorType",
    "parse_mlx_export_events_to_graph", "capture_graph_from_mlx_function",
]
