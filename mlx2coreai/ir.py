from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


_DYNAMIC_DIM_KEY = "__mlx2coreai_dynamic_dim__"


def dynamic_dim_ref(source: str, axis: int) -> dict[str, Any]:
    return {_DYNAMIC_DIM_KEY: True, "source": str(source), "axis": int(axis)}


def is_dynamic_dim_ref(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get(_DYNAMIC_DIM_KEY))


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

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True)
class Node:
    op: str
    inputs: tuple[str, ...]
    output: str
    attrs: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "op": self.op,
            "inputs": list(self.inputs),
            "output": self.output,
            "attrs": {str(k): _json_attr_value(v) for k, v in self.attrs.items()},
        }
        if self.source is not None:
            payload["source"] = self.source
        return payload


@dataclass
class Graph:
    inputs: list[TensorSpec]
    nodes: list[Node]
    outputs: list[str]

    def validate(self) -> None:
        input_names = [spec.name for spec in self.inputs]
        if len(input_names) != len(set(input_names)):
            raise ValueError("Graph input names must be unique.")

        available = set(input_names)
        for node in self.nodes:
            if node.output in available:
                raise ValueError(f"Duplicate tensor name detected: {node.output}")
            missing_inputs = [name for name in node.inputs if name not in available]
            if missing_inputs:
                raise ValueError(
                    f"Node '{node.op}' has missing inputs: {', '.join(missing_inputs)}"
                )
            available.add(node.output)

        missing_outputs = [name for name in self.outputs if name not in available]
        if missing_outputs:
            raise ValueError(
                f"Graph outputs reference unknown tensors: {', '.join(missing_outputs)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": [spec.to_dict() for spec in self.inputs],
            "nodes": [node.to_dict() for node in self.nodes],
            "outputs": list(self.outputs),
        }
