from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from .ir import Graph, Node, TensorSpec, dynamic_dim_ref, is_dynamic_dim_ref


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
    out = Graph(inputs=inputs, nodes=list(graph.nodes), outputs=list(graph.outputs))
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
    graph = apply_dynamic_axes(graph, axes_by_input)
    if not axes_by_input:
        return graph
    _validate_probe_compatibility(graph, probe_graph)

    candidates: list[tuple[Any, Any, dict[str, Any]]] = []
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

    if not candidates:
        return graph

    nodes: list[Node] = []
    for node, probe_node in zip(graph.nodes, probe_graph.nodes, strict=True):
        attrs = _dynamicize_attr_value(node.attrs, probe_node.attrs, candidates)
        nodes.append(replace(node, attrs=attrs))
    out = Graph(inputs=list(graph.inputs), nodes=nodes, outputs=list(graph.outputs))
    out.validate()
    return out


def _validate_probe_compatibility(graph: Graph, probe_graph: Graph) -> None:
    if len(graph.nodes) != len(probe_graph.nodes):
        raise ValueError(
            "dynamic shape probe produced a different graph structure: "
            f"{len(graph.nodes)} nodes vs {len(probe_graph.nodes)} nodes."
        )
    for index, (node, probe_node) in enumerate(zip(graph.nodes, probe_graph.nodes, strict=True)):
        if node.op != probe_node.op or len(node.inputs) != len(probe_node.inputs):
            raise ValueError(
                "dynamic shape probe produced a different graph structure at "
                f"node {index}: {node.op}/{len(node.inputs)} vs {probe_node.op}/{len(probe_node.inputs)}."
            )


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
