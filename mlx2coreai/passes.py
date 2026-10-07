from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any

import numpy as np

from .dtypes import normalize_dtype as _normalize_input_dtype
from .ir import Graph, Node, TensorType, remap_dim_refs
from .op_registry import ensure_supported, coreai_op_for_mlx, normalize_mlx_op_name, rule_for_mlx
from ._type_inference import InferredTensorSpec, infer_broadcast_axes_shape, unknown_spec

_IDENTITY_OPS = {"identity", "stop_gradient", "copy", "contiguous"}
_CONSTANT_OPS = {"const", "constant", "literal"}
_SAFE_NAME_RE = re.compile(r"[^0-9a-zA-Z_]")
_MAX_INLINE_ARRAY_VALUES = 128


@dataclass(frozen=True)
class AnalyzedGraph:
    """A normalized, validated graph and its reusable tensor analysis."""

    graph: Graph
    specs: dict[str, InferredTensorSpec]


def analyze_graph(graph: Graph) -> AnalyzedGraph:
    graph = normalize_graph(graph)
    ensure_supported(graph)
    return AnalyzedGraph(graph, infer_graph_specs(graph))


def canonicalize_input_specs(graph: Graph) -> Graph:
    inputs = [
        replace(spec,
            name=str(spec.name).strip(),
            shape=tuple(int(v) for v in spec.shape),
            dtype=_normalize_input_dtype(spec.dtype),
        )
        for spec in graph.inputs
    ]
    normalized = replace(graph, inputs=inputs, value_types={
        name: replace(spec, dtype=_normalize_input_dtype(spec.dtype) if spec.dtype is not None else None)
        for name, spec in graph.value_types.items()
    })
    normalized.validate()
    return normalized


def _sanitize_tensor_name(raw: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", raw.strip())
    if cleaned == "":
        cleaned = "t"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


def canonicalize_tensor_names(graph: Graph) -> Graph:
    used: set[str] = set()
    name_map: dict[str, str] = {}

    def reserve(old: str) -> str:
        if old in name_map:
            return name_map[old]
        base = _sanitize_tensor_name(old)
        candidate = base
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        used.add(candidate)
        name_map[old] = candidate
        return candidate

    inputs = [
        replace(spec,
            name=reserve(spec.name),
            shape=spec.shape,
            dtype=spec.dtype,
        )
        for spec in graph.inputs
    ]
    for node in graph.nodes:
        for name in node.outputs:
            reserve(name)
    nodes = []
    for node in graph.nodes:
        mapped_inputs = tuple(name_map.get(name, name) for name in node.inputs)
        nodes.append(replace(node, inputs=mapped_inputs,
                             outputs=tuple(name_map[name] for name in node.outputs),
                             attrs=remap_dim_refs(node.attrs, name_map)))

    outputs = [name_map.get(name, name) for name in graph.outputs]
    normalized = replace(graph, inputs=inputs, nodes=nodes, outputs=outputs, value_types={
        name_map.get(name, name): spec for name, spec in graph.value_types.items()
    })
    normalized.validate()
    return normalized


def _normalize_attr_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        # Avoid exploding memory/time for large weight tensors during normalization.
        if value.size <= _MAX_INLINE_ARRAY_VALUES:
            return value.tolist()
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_normalize_attr_value(v) for v in value]
    if isinstance(value, list):
        return [_normalize_attr_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_attr_value(v) for k, v in value.items()}
    return value


def canonicalize_constant_attrs(graph: Graph) -> Graph:
    nodes: list[Node] = []
    for node in graph.nodes:
        op = node.op
        attrs = {str(k): _normalize_attr_value(v) for k, v in node.attrs.items()}
        if op in _CONSTANT_OPS:
            # Constants are tensor data, not shape attributes. Converting small
            # arrays to Python lists loses BF16/FP16 dtype and scalar rank.
            for key in ("value", "val", "data", "tensor"):
                if isinstance(node.attrs.get(key), (np.ndarray, np.generic)):
                    attrs[key] = np.asarray(node.attrs[key])
            op = "constant"
            if "value" not in attrs:
                for key in ("val", "data", "tensor"):
                    if key in attrs:
                        attrs["value"] = attrs.pop(key)
                        break
            dtype = attrs.get("dtype")
            if isinstance(dtype, str):
                attrs["dtype"] = _normalize_input_dtype(dtype)
        nodes.append(replace(node, op=op, attrs=attrs))

    normalized = replace(graph, nodes=nodes)
    normalized.validate()
    return normalized


def canonicalize_op_names(graph: Graph) -> Graph:
    nodes = [
        replace(node, op=normalize_mlx_op_name(node.op), attrs=dict(node.attrs))
        for node in graph.nodes
    ]
    normalized = replace(graph, nodes=nodes)
    normalized.validate()
    return normalized


def eliminate_identity_noops(graph: Graph) -> Graph:
    replacements: dict[str, str] = {}
    kept_nodes: list[Node] = []
    graph_outputs = set(graph.outputs)

    def resolve(name: str) -> str:
        while name in replacements:
            name = replacements[name]
        return name

    for node in graph.nodes:
        mapped_inputs = tuple(resolve(name) for name in node.inputs)
        canonical_node = replace(node, inputs=mapped_inputs, attrs=dict(node.attrs))
        if (
            canonical_node.op in _IDENTITY_OPS
            and len(canonical_node.inputs) == 1
            and len(canonical_node.outputs) == 1
            and canonical_node.output not in graph_outputs
        ):
            replacements[canonical_node.output] = canonical_node.inputs[0]
            continue
        kept_nodes.append(canonical_node)

    outputs = [resolve(name) for name in graph.outputs]
    names = {name: resolve(name) for name in replacements}
    kept_nodes = [replace(node, attrs=remap_dim_refs(node.attrs, names)) for node in kept_nodes]
    live_outputs = {name for node in kept_nodes for name in node.outputs}
    normalized = replace(graph, nodes=kept_nodes, outputs=outputs, value_types={
        name: spec for name, spec in graph.value_types.items() if name in live_outputs
    })
    normalized.validate()
    return normalized


def canonicalize_sdpa_masks(graph: Graph) -> Graph:
    nodes: list[Node] = []
    for node in graph.nodes:
        coreai_op = coreai_op_for_mlx(node.op)
        if coreai_op != "scaled_dot_product_attention":
            nodes.append(node)
            continue

        attrs = dict(node.attrs)
        attrs["do_causal"] = bool(attrs.get("do_causal", False))
        attrs["has_sinks"] = bool(attrs.get("has_sinks", False))
        attrs["output_logsumexp"] = bool(attrs.get("output_logsumexp", False))
        if "output_index" in attrs:
            try:
                attrs["output_index"] = int(attrs["output_index"])
            except Exception:
                attrs["output_index"] = attrs["output_index"]
        if attrs.get("scale") is not None:
            try:
                attrs["scale"] = float(attrs["scale"])
            except Exception:
                pass

        if len(node.inputs) < 4:
            attrs["mask_mode"] = "none"
        else:
            mask_mode_raw = str(attrs.get("mask_mode", "auto")).strip().lower()
            if mask_mode_raw not in {"auto", "bool", "additive"}:
                mask_mode_raw = "auto"
            if attrs["do_causal"] and mask_mode_raw == "auto":
                attrs["mask_mode"] = "causal_plus_explicit"
            else:
                attrs["mask_mode"] = mask_mode_raw

        nodes.append(replace(node, attrs=attrs))

    normalized = replace(graph, nodes=nodes)
    normalized.validate()
    return normalized


def _infer_node_spec(node: Node, input_specs: list[InferredTensorSpec]) -> InferredTensorSpec:
    rule = rule_for_mlx(node.op)
    if rule is None:
        return InferredTensorSpec(shape=None, dtype=None)
    return rule.infer(node, input_specs) if rule.infer is not None else unknown_spec(input_specs)


def infer_graph_specs(graph: Graph) -> dict[str, InferredTensorSpec]:
    graph.validate()
    inferred: dict[str, InferredTensorSpec] = {
        spec.name: InferredTensorSpec(
            shape=tuple(int(v) for v in spec.shape),
            dtype=_normalize_input_dtype(spec.dtype),
        )
        for spec in graph.inputs
    }
    for node in graph.nodes:
        input_specs = [inferred.get(name, InferredTensorSpec(shape=None, dtype=None)) for name in node.inputs]
        for index, name in enumerate(node.outputs):
            captured = graph.value_types.get(name)
            if captured is not None and captured.shape is not None and captured.dtype is not None:
                inferred[name] = captured
            else:
                fallback = _infer_node_spec(node.result_node(index), input_specs)
                inferred[name] = TensorType(
                    shape=fallback.shape,
                    dtype=captured.dtype if captured is not None and captured.dtype is not None else fallback.dtype,
                )
    return inferred


def summarize_inference(inferred: dict[str, InferredTensorSpec]) -> dict[str, int]:
    total = len(inferred)
    with_shape = sum(1 for spec in inferred.values() if spec.shape is not None)
    with_dtype = sum(1 for spec in inferred.values() if spec.dtype is not None)
    return {"total_tensors": total, "with_shape": with_shape, "with_dtype": with_dtype}


def normalize_graph(graph: Graph) -> Graph:
    graph.validate()
    canonical = canonicalize_op_names(graph)
    canonical = canonicalize_input_specs(canonical)
    canonical = canonicalize_tensor_names(canonical)
    canonical = canonicalize_constant_attrs(canonical)
    canonical = canonicalize_sdpa_masks(canonical)
    canonical = eliminate_identity_noops(canonical)
    return canonical
