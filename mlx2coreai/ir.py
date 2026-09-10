from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


_DYNAMIC_DIM_KEY = "__mlx2coreai_dynamic_dim__"


def dynamic_dim_ref(source: str, axis: int) -> dict[str, Any]:
    return {_DYNAMIC_DIM_KEY: True, "source": str(source), "axis": int(axis)}


def is_dynamic_dim_ref(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get(_DYNAMIC_DIM_KEY))


def remap_dim_refs(value: Any, names: dict[str, str]) -> Any:
    if is_dynamic_dim_ref(value):
        return dynamic_dim_ref(names.get(value["source"], value["source"]), value["axis"])
    if isinstance(value, dict):
        return {key: remap_dim_refs(item, names) for key, item in value.items()}
    if isinstance(value, list):
        return [remap_dim_refs(item, names) for item in value]
    if isinstance(value, tuple):
        return tuple(remap_dim_refs(item, names) for item in value)
    return value


def dimension_sources(value: Any):
    if is_dynamic_dim_ref(value):
        yield value["source"]
    elif isinstance(value, dict):
        for item in value.values():
            yield from dimension_sources(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from dimension_sources(item)


def _json_attr_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        # Keep small constants readable; summarize large arrays to avoid massive JSON artifacts.
        if value.size <= 128:
            return value.tolist()
        return {
            "__ndarray__": True,
            "shape": [int(v) for v in value.shape],
            "dtype": str(value.dtype),
            "numel": int(value.size),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_attr_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_attr_value(v) for v in value]
    return value


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...] | None
    dtype: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"shape": list(self.shape) if self.shape is not None else None, "dtype": self.dtype}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str = "fp32"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True)
class StateSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str = "fp16"
    capacity_axis: int | None = None

    def resolved_shape(self, capacity: int | None = None) -> tuple[int, ...]:
        shape = list(self.shape)
        if capacity is not None and self.capacity_axis is not None:
            if capacity <= 0:
                raise ValueError("State capacity must be positive.")
            shape[self.capacity_axis] = int(capacity)
        return tuple(shape)

    def to_dict(self) -> dict[str, Any]:
        result = {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}
        if self.capacity_axis is not None:
            result["capacity_axis"] = self.capacity_axis
        return result


@dataclass(frozen=True, init=False)
class Node:
    op: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    def __init__(
        self, op: str, inputs: tuple[str, ...], output: str | None = None,
        attrs: dict[str, Any] | None = None, source: str | None = None, *,
        outputs: tuple[str, ...] | None = None,
    ):
        if output is not None and outputs is not None:
            raise ValueError("Specify output or outputs, not both.")
        resolved = tuple(outputs) if outputs is not None else ((output,) if output is not None else ())
        if not resolved or any(not isinstance(name, str) or not name for name in resolved):
            raise ValueError("An operation requires named outputs.")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "inputs", tuple(inputs))
        object.__setattr__(self, "outputs", resolved)
        object.__setattr__(self, "attrs", {} if attrs is None else attrs)
        object.__setattr__(self, "source", source)

    @property
    def output(self) -> str:
        """First output; retained for single-output IR callers and diagnostics."""
        return self.outputs[0]

    def result_node(self, index: int) -> Node:
        """Adapt one result to a legacy single-result lowering or type rule."""
        if len(self.outputs) == 1:
            return self
        return replace(self, outputs=(self.outputs[index],), attrs={
            **self.attrs, "output_index": index, "num_outputs": len(self.outputs),
        })

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "op": self.op,
            "inputs": list(self.inputs),
            "attrs": {str(k): _json_attr_value(v) for k, v in self.attrs.items()},
        }
        if len(self.outputs) == 1:
            payload["output"] = self.output
        else:
            payload["outputs"] = list(self.outputs)
        if self.source is not None:
            payload["source"] = self.source
        return payload


@dataclass
class Graph:
    inputs: list[TensorSpec]
    nodes: list[Node]
    outputs: list[str]
    value_types: dict[str, TensorType] = field(default_factory=dict)

    def validate(self) -> None:
        input_names = [spec.name for spec in self.inputs]
        if len(input_names) != len(set(input_names)):
            raise ValueError("Graph input names must be unique.")

        available = set(input_names)
        for node in self.nodes:
            dependencies = (*node.inputs, *dimension_sources(node.attrs))
            missing_inputs = [name for name in dependencies if name not in available]
            if missing_inputs:
                raise ValueError(
                    f"Node '{node.op}' has missing inputs: {', '.join(missing_inputs)}"
                )
            for name in node.outputs:
                if name in available:
                    raise ValueError(f"Duplicate tensor name detected: {name}")
                available.add(name)

        if unknown := set(self.value_types) - available:
            raise ValueError(f"Tensor types reference unknown values: {sorted(unknown)}")

        missing_outputs = [name for name in self.outputs if name not in available]
        if missing_outputs:
            raise ValueError(
                f"Graph outputs reference unknown tensors: {', '.join(missing_outputs)}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "inputs": [spec.to_dict() for spec in self.inputs],
            "nodes": [node.to_dict() for node in self.nodes],
            "outputs": list(self.outputs),
        }
        if self.value_types:
            payload["value_types"] = {name: spec.to_dict() for name, spec in self.value_types.items()}
        return payload
