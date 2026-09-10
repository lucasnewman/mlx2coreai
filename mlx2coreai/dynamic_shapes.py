from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from .ir import Graph, Node, TensorSpec, TensorType, dynamic_dim_ref, is_dynamic_dim_ref
from .op_registry import coreai_op_for_mlx


DynamicAxes = Mapping[str, Sequence[int] | Mapping[int, Any] | str]


def normalize_dynamic_axes(dynamic_axes: DynamicAxes | None, graph: Graph) -> dict[str, tuple[int, ...]]:
    if dynamic_axes is None:
        return {}
    specs = {spec.name: spec for spec in graph.inputs}
    out: dict[str, tuple[int, ...]] = {}
    for name, raw_axes in dynamic_axes.items():
        if name not in specs:
            raise ValueError(f"dynamic axis spec references unknown input '{name}'.")
        rank = len(specs[name].shape)
        if raw_axes == "all":
            axes = tuple(range(rank))
        elif isinstance(raw_axes, Mapping):
            axes = tuple(int(axis) for axis in raw_axes)
        else:
            axes = tuple(int(axis) for axis in raw_axes)
        normalized: list[int] = []
        for axis in axes:
            axis = axis + rank if axis < 0 else axis
            if axis < 0 or axis >= rank:
                raise ValueError(f"dynamic axis {axis} is out of range for input '{name}' rank {rank}.")
            if axis not in normalized:
                normalized.append(axis)
        out[str(name)] = tuple(normalized)
    return out


def apply_dynamic_axes(graph: Graph, dynamic_axes: DynamicAxes | None) -> Graph:
    axes_by_input = normalize_dynamic_axes(dynamic_axes, graph)
    if not axes_by_input:
        return graph
    inputs: list[TensorSpec] = []
    for spec in graph.inputs:
        axes = axes_by_input.get(spec.name, ())
        if not axes:
            inputs.append(spec)
            continue
        shape = list(spec.shape)
        for axis in axes:
            shape[axis] = -1
        inputs.append(replace(spec, shape=tuple(shape)))
    dynamic_values = set(axes_by_input)
    value_types = dict(graph.value_types)
    for node in graph.nodes:
        if dynamic_values.intersection(node.inputs):
            dynamic_values.update(node.outputs)
            for name in node.outputs:
                if name in value_types:
                    value_types[name] = replace(value_types[name], shape=None)
    out = replace(graph, inputs=inputs, value_types=value_types)
    out.validate()
    return out


def dynamicize_graph_from_probe(
    graph: Graph,
    probe_graph: Graph,
    *,
    dynamic_axes: DynamicAxes | None,
    base_inputs: Mapping[str, Any],
    probe_inputs: Mapping[str, Any],
) -> Graph:
    """Replace attrs that vary with requested input axes by dynamic-dim refs.

    MLX's callback export still reports concrete primitive shapes, even with
    shapeless export. Capturing one nearby probe shape lets us identify which
    reshape/broadcast/range/slice attributes are really input dimensions.
    """

    axes_by_input = normalize_dynamic_axes(dynamic_axes, graph)
    captured_graph = graph
    graph = apply_dynamic_axes(graph, axes_by_input)
    if not axes_by_input:
        return graph
    _validate_probe_compatibility(graph, probe_graph)

    candidates: list[tuple[Any, Any, dict[str, Any]]] = []
    unprobed_inputs = set()
    for input_name, axes in axes_by_input.items():
        if input_name not in base_inputs or input_name not in probe_inputs:
            continue
        base_shape = tuple(int(v) for v in np.asarray(base_inputs[input_name]).shape)
        probe_shape = tuple(int(v) for v in np.asarray(probe_inputs[input_name]).shape)
        for axis in axes:
            if axis >= len(base_shape) or axis >= len(probe_shape):
                continue
            base_dim = int(base_shape[axis])
            probe_dim = int(probe_shape[axis])
            if base_dim != probe_dim:
                candidates.append((base_dim, probe_dim, dynamic_dim_ref(input_name, axis)))
            else:
                unprobed_inputs.add(input_name)

    if not candidates:
        return graph

    nodes: list[Node] = []
    value_types = dict(graph.value_types)
    base_types = {spec.name: TensorType(spec.shape, spec.dtype) for spec in captured_graph.inputs}
    base_types.update(captured_graph.value_types)
    probe_types = {spec.name: TensorType(spec.shape, spec.dtype) for spec in probe_graph.inputs}
    probe_types.update(probe_graph.value_types)
    shape_candidates = list(candidates)
    for node, probe_node in zip(graph.nodes, probe_graph.nodes, strict=True):
        if unprobed_inputs.intersection(node.inputs):
            unprobed_inputs.update(node.outputs)
        for name, probe_name in zip(node.outputs, probe_node.outputs, strict=True):
            base, probe = base_types.get(name), probe_types.get(probe_name)
            if name in unprobed_inputs or base is None or probe is None or base.shape is None or probe.shape is None:
                continue
            if len(base.shape) != len(probe.shape) or base.dtype != probe.dtype:
                raise ValueError(f"Dynamic shape probe changed rank or dtype of {name}.")
            value_types[name] = TensorType(
                tuple(a if a == b else -1 for a, b in zip(base.shape, probe.shape, strict=True)), base.dtype,
            )
        attrs = _dynamicize_attr_value(node.attrs, probe_node.attrs, shape_candidates)
        if coreai_op_for_mlx(node.op) == "as_strided":
            # Strides and window sizes can be products or affine expressions of
            # input dimensions. Never freeze a varying value that probing could
            # not represent, since that would silently read the wrong elements.
            _require_dynamic_attr_resolution(node.attrs, probe_node.attrs, attrs)
        if coreai_op_for_mlx(node.op) == "arange":
            _require_dynamic_attr_resolution(node.attrs, probe_node.attrs, attrs, operation="Arange")
        shape, probe_shape = node.attrs.get("shape"), probe_node.attrs.get("shape")
        if isinstance(shape, (list, tuple)) and isinstance(probe_shape, (list, tuple)):
            attrs["shape"] = _dynamicize_attr_value(shape, probe_shape, shape_candidates)
            if coreai_op_for_mlx(node.op) in {"reshape", "flatten", "unflatten"}:
                unresolved = [i for i, (a, b) in enumerate(zip(shape, probe_shape, strict=True))
                              if a != b and not is_dynamic_dim_ref(attrs["shape"][i])]
                # A flattened product (e.g. batch * frames) can be inferred from
                # the input's element count, without fitting arithmetic to a probe.
                if len(unresolved) == 1 and -1 not in attrs["shape"]:
                    attrs["shape"] = list(attrs["shape"])
                    attrs["shape"][unresolved[0]] = -1
        # A slice to the end of an intermediate (e.g. concat(history, tokens))
        # need not match any input dimension. Preserve that runtime extent.
        if coreai_op_for_mlx(node.op) == "slice_by_index" and node.inputs:
            base = base_types.get(node.inputs[0])
            probe = probe_types.get(probe_node.inputs[0])
            # Legacy graphs may still carry the old slice-only observation.
            base_shape = base.shape if base is not None else node.attrs.get("slice_input_shape")
            probe_shape = probe.shape if probe is not None else probe_node.attrs.get("slice_input_shape")
            end, probe_end = node.attrs.get("end"), probe_node.attrs.get("end")
            if base_shape is not None and probe_shape is not None and isinstance(end, (list, tuple)):
                updated_end = list(attrs["end"])
                for axis, (base_dim, probe_dim) in enumerate(zip(base_shape, probe_shape, strict=True)):
                    if (axis < len(end) and isinstance(probe_end, (list, tuple)) and axis < len(probe_end)
                            and base_dim >= 0 and probe_dim >= 0 and base_dim != probe_dim
                            and end[axis] == base_dim and probe_end[axis] == probe_dim):
                        updated_end[axis] = dynamic_dim_ref(node.inputs[0], axis)
                    elif (axis < len(end) and isinstance(probe_end, (list, tuple)) and axis < len(probe_end)
                            and base_dim >= 0 and probe_dim >= 0 and base_dim != probe_dim
                            and isinstance(end[axis], int) and isinstance(probe_end[axis], int)
                            and 0 < base_dim - end[axis] == probe_dim - probe_end[axis]):
                        updated_end[axis] = end[axis] - base_dim
                attrs["end"] = updated_end
            attrs.pop("slice_input_shape", None)
        nodes.append(replace(node, attrs=attrs))
        # Later shape operands may use a derived extent rather than an input
        # dimension, such as the flattened batch used to broadcast a linear bias.
        for name, probe_name in zip(node.outputs, probe_node.outputs, strict=True):
            base, probe = base_types.get(name), probe_types.get(probe_name)
            if name in unprobed_inputs or base is None or probe is None or base.shape is None or probe.shape is None:
                continue
            for axis, (a, b) in enumerate(zip(base.shape, probe.shape, strict=True)):
                if a >= 0 and b >= 0 and a != b:
                    shape_candidates.append((a, b, dynamic_dim_ref(name, axis)))
    out = replace(graph, nodes=nodes, value_types=value_types)
    out.validate()
    return out


def _validate_probe_compatibility(graph: Graph, probe_graph: Graph) -> None:
    if len(graph.nodes) != len(probe_graph.nodes):
        raise ValueError(
            "dynamic shape probe produced a different graph structure: "
            f"{len(graph.nodes)} nodes vs {len(probe_graph.nodes)} nodes."
        )
    for index, (node, probe_node) in enumerate(zip(graph.nodes, probe_graph.nodes, strict=True)):
        if (node.op != probe_node.op or len(node.inputs) != len(probe_node.inputs)
                or len(node.outputs) != len(probe_node.outputs)):
            raise ValueError(
                "dynamic shape probe produced a different graph structure at "
                f"node {index}: {node.op}/{len(node.inputs)} vs {probe_node.op}/{len(probe_node.inputs)}."
            )


def _require_dynamic_attr_resolution(base, probe, resolved, *, operation="AsStrided"):
    if is_dynamic_dim_ref(resolved):
        return
    if isinstance(base, dict):
        for key in base:
            _require_dynamic_attr_resolution(base[key], probe[key], resolved[key], operation=operation)
    elif isinstance(base, (list, tuple)):
        for before, after, value in zip(base, probe, resolved, strict=True):
            _require_dynamic_attr_resolution(before, after, value, operation=operation)
    elif base != probe:
        raise ValueError(f'Dynamic {operation} expression could not be resolved from the probe; '
                         'expose the derived extent as a tensor dimension rather than fixing capture dimensions.')


def _dynamicize_attr_value(value: Any, probe_value: Any, candidates: list[tuple[Any, Any, dict[str, Any]]]) -> Any:
    if is_dynamic_dim_ref(value):
        return value
    replacement = _candidate_replacement(value, probe_value, candidates)
    if replacement is not None:
        return replacement
    if isinstance(value, dict) and isinstance(probe_value, dict):
        return {
            key: _dynamicize_attr_value(value[key], probe_value.get(key), candidates)
            for key in value
        }
    if isinstance(value, tuple) and isinstance(probe_value, tuple) and len(value) == len(probe_value):
        return tuple(_dynamicize_attr_value(v, p, candidates) for v, p in zip(value, probe_value, strict=True))
    if isinstance(value, list) and isinstance(probe_value, list) and len(value) == len(probe_value):
        return [_dynamicize_attr_value(v, p, candidates) for v, p in zip(value, probe_value, strict=True)]
    return value


def _candidate_replacement(value: Any, probe_value: Any, candidates: list[tuple[Any, Any, dict[str, Any]]]) -> dict[str, Any] | None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(probe_value, np.generic):
        probe_value = probe_value.item()
    if isinstance(value, bool) or isinstance(probe_value, bool):
        return None
    if not isinstance(value, (int, np.integer)) or not isinstance(probe_value, (int, np.integer)):
        return None
    for base_dim, probe_dim, ref in candidates:
        if int(value) == int(base_dim) and int(probe_value) == int(probe_dim):
            return dict(ref)
    return None
