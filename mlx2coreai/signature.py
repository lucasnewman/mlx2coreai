"""Public argument order and functional-output bindings for mutable state."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .ir import Graph, Node, StateSpec, TensorType


@dataclass(frozen=True)
class StateBinding:
    spec: StateSpec
    output_index: int


@dataclass(frozen=True)
class CaptureSignature:
    input_order: tuple[str, ...] = ()
    states: tuple[StateBinding, ...] = ()
    output_count: int | None = None

    def bind(
        self, graph: Graph, expected: Mapping[str, np.ndarray],
    ) -> tuple[Graph, dict[str, np.ndarray]]:
        inputs = {spec.name: spec for spec in graph.inputs}
        if self.output_count is not None and len(graph.outputs) != self.output_count:
            raise ValueError(f"Capture returned {len(graph.outputs)} outputs; expected {self.output_count}.")
        if len(set(self.input_order)) != len(self.input_order):
            raise ValueError("Signature input names must be unique.")
        missing = set(self.input_order) - inputs.keys()
        if missing:
            raise ValueError(f"Signature references missing inputs: {sorted(missing)}")
        ordered = [inputs[name] for name in self.input_order]
        ordered.extend(spec for spec in graph.inputs if spec.name not in self.input_order)
        nodes, outputs = list(graph.nodes), list(graph.outputs)
        value_types = dict(graph.value_types)
        expected = dict(expected)
        indices, names = set(), set()
        for binding in self.states:
            spec, index = binding.spec, binding.output_index
            if index < 0 or index >= len(outputs) or index in indices:
                raise ValueError(f"Invalid or duplicate state output index: {index}")
            if spec.name not in inputs or spec.name in names:
                raise ValueError(f"Missing or duplicate state input: {spec.name}")
            indices.add(index)
            names.add(spec.name)
            value_name = outputs[index]
            output_name = f"{spec.name}__updated"
            nodes.append(Node(
                "write_state", (spec.name, value_name), output_name,
                attrs={"coreai_output_name": spec.name},
                source="mlx2coreai:state_binding",
            ))
            outputs[index] = output_name
            value_types[output_name] = TensorType(inputs[spec.name].shape, inputs[spec.name].dtype)
            expected[output_name] = expected[value_name]
        graph = replace(graph, inputs=ordered, nodes=nodes, outputs=outputs, value_types=value_types)
        graph.validate()
        return graph, {name: expected[name] for name in outputs}
