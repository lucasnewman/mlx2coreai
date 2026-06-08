from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .dynamic_shapes import DynamicAxes, apply_dynamic_axes, dynamicize_graph_from_probe
from .from_mlx import capture_graph_from_mlx_function
from .ir import Graph, StateSpec
from .lower_to_coreai import (
    CoreAILoweringConfig,
    LoweredCoreAIProgram,
    build_coreai_program,
    save_coreai_program,
)
from .op_registry import ensure_supported, unsupported_op_details
from .passes import infer_graph_specs, normalize_graph, summarize_inference


@dataclass(slots=True)
class ConversionConfig:
    capture_mode: str = "callback"
    allow_unknown_sources: bool = True
    capture_shapeless: bool = False
    dynamic_axes: DynamicAxes | None = None
    dynamic_probe_inputs: Mapping[str, Any] | None = None
    capture_is_training: bool = False
    optimize: bool = True
    entrypoint_name: str = "main"
    state_specs: list[StateSpec] | None = None
    externalize_weights: bool = True
    external_weight_threshold: int = 10
    min_runtime_target: str = "macOS27"
    constant_inputs: Mapping[str, Any] | None = None


@dataclass(slots=True)
class CapturedMLXGraph:
    graph: Graph
    normalized_inputs: dict[str, np.ndarray]
    expected_outputs: dict[str, np.ndarray]


@dataclass(slots=True)
class PreparedMLXGraph:
    captured: CapturedMLXGraph
    normalized_graph: Graph
    expected_outputs: dict[str, np.ndarray]
    inference_summary: dict[str, int]
    unsupported_details: list[dict[str, Any]]
    extra_input_names: list[str]

    @property
    def graph(self) -> Graph:
        return self.captured.graph

    @property
    def normalized_inputs(self) -> dict[str, np.ndarray]:
        return self.captured.normalized_inputs

    @property
    def weights_captured_as_constants(self) -> bool:
        return len(self.extra_input_names) == 0


@dataclass(slots=True)
class ConvertedCoreAIModel:
    prepared: PreparedMLXGraph
    lowered: LoweredCoreAIProgram
    asset: Any | None
    asset_path: Path | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def program(self) -> Any:
        return self.lowered.program

    @property
    def weight_manifest(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.lowered.weight_manifest]


@contextmanager
def temporary_capture_training_mode(obj: Any, enabled: bool):
    if not bool(enabled):
        yield
        return

    prior_training: bool | None = None
    if hasattr(obj, "training"):
        try:
            prior_training = bool(getattr(obj, "training"))
        except Exception:
            prior_training = None
    train_fn = getattr(obj, "train", None)
    eval_fn = getattr(obj, "eval", None)
    if callable(train_fn):
        train_fn()
    try:
        yield
    finally:
        if prior_training is not None:
            if prior_training and callable(train_fn):
                train_fn()
            elif (not prior_training) and callable(eval_fn):
                eval_fn()


def capture_mlx_graph(
    target: Any,
    inputs: Mapping[str, Any],
    *,
    dot_output_path: Path | None = None,
    capture_mode: str = "callback",
    capture_shapeless: bool = False,
    allow_unknown_sources: bool = True,
    capture_is_training: bool = False,
    capture_function: Callable[..., Any] | None = None,
) -> CapturedMLXGraph:
    normalized_inputs = {name: np.asarray(value) for name, value in inputs.items()}
    resolved_capture_function, capture_target = _resolve_capture_components(target, capture_function)
    with temporary_capture_training_mode(capture_target, enabled=capture_is_training):
        graph, captured_inputs, expected_outputs = capture_graph_from_mlx_function(
            dot_output_path=dot_output_path,
            inputs=normalized_inputs,
            function=resolved_capture_function,
            allow_unknown_sources=allow_unknown_sources,
            capture_mode=capture_mode,
            shapeless=capture_shapeless,
        )
    return CapturedMLXGraph(
        graph=graph,
        normalized_inputs=captured_inputs,
        expected_outputs=expected_outputs,
    )


def prepare_mlx_conversion(
    target: Any,
    inputs: Mapping[str, Any],
    *,
    config: ConversionConfig | None = None,
    dot_output_path: Path | None = None,
    capture_function: Callable[..., Any] | None = None,
) -> PreparedMLXGraph:
    resolved = config or ConversionConfig()
    resolved_capture_function, capture_target = _resolve_capture_components(target, capture_function)
    captured = capture_mlx_graph(
        capture_target,
        inputs,
        dot_output_path=dot_output_path,
        capture_mode=resolved.capture_mode,
        capture_shapeless=resolved.capture_shapeless,
        allow_unknown_sources=resolved.allow_unknown_sources,
        capture_is_training=resolved.capture_is_training,
        capture_function=resolved_capture_function,
    )
    graph = captured.graph
    if resolved.dynamic_axes:
        if resolved.dynamic_probe_inputs is not None:
            probe = capture_mlx_graph(
                capture_target,
                resolved.dynamic_probe_inputs,
                dot_output_path=None,
                capture_mode=resolved.capture_mode,
                capture_shapeless=resolved.capture_shapeless,
                allow_unknown_sources=resolved.allow_unknown_sources,
                capture_is_training=resolved.capture_is_training,
                capture_function=resolved_capture_function,
            )
            graph = dynamicize_graph_from_probe(
                graph,
                probe.graph,
                dynamic_axes=resolved.dynamic_axes,
                base_inputs=captured.normalized_inputs,
                probe_inputs=probe.normalized_inputs,
            )
        else:
            graph = apply_dynamic_axes(graph, resolved.dynamic_axes)
    captured = CapturedMLXGraph(
        graph=graph,
        normalized_inputs=captured.normalized_inputs,
        expected_outputs=captured.expected_outputs,
    )
    normalized_graph = normalize_graph(captured.graph)
    inference_summary = summarize_inference(infer_graph_specs(normalized_graph))
    unsupported_details = unsupported_op_details(normalized_graph)
    ensure_supported(normalized_graph)
    extra_input_names = find_extra_input_names(normalized_graph, captured.normalized_inputs)
    return PreparedMLXGraph(
        captured=captured,
        normalized_graph=normalized_graph,
        expected_outputs=captured.expected_outputs,
        inference_summary=inference_summary,
        unsupported_details=unsupported_details,
        extra_input_names=extra_input_names,
    )


def lower_graph_to_coreai(
    graph: Graph,
    *,
    config: ConversionConfig | None = None,
    public_input_names: set[str] | None = None,
) -> LoweredCoreAIProgram:
    resolved = config or ConversionConfig()
    return build_coreai_program(
        graph,
        config=CoreAILoweringConfig(
            entrypoint_name=resolved.entrypoint_name,
            optimize=resolved.optimize,
            state_specs=resolved.state_specs,
            constant_inputs=resolved.constant_inputs,
            public_input_names=public_input_names,
            externalize_weights=resolved.externalize_weights,
            external_weight_threshold=resolved.external_weight_threshold,
        ),
    )


def convert_mlx_to_coreai(
    target: Any,
    inputs: Mapping[str, Any],
    *,
    config: ConversionConfig | None = None,
    output_path: Path | None = None,
    dot_output_path: Path | None = None,
    capture_function: Callable[..., Any] | None = None,
) -> ConvertedCoreAIModel:
    resolved = config or ConversionConfig()
    prepared = prepare_mlx_conversion(
        target,
        inputs,
        config=resolved,
        dot_output_path=dot_output_path,
        capture_function=capture_function,
    )
    lowered = lower_graph_to_coreai(
        prepared.normalized_graph,
        config=resolved,
        public_input_names=set(prepared.normalized_inputs),
    )
    asset = None
    asset_path = None
    if output_path is not None:
        asset_path = Path(output_path)
        asset = save_coreai_program(lowered.program, asset_path)
    metadata = {
        "entrypoint_name": resolved.entrypoint_name,
        "min_runtime_target": resolved.min_runtime_target,
        "capture_shapeless": bool(resolved.capture_shapeless),
        "dynamic_axes": _metadata_dynamic_axes(resolved.dynamic_axes),
        "optimized": bool(lowered.optimized),
        "optimization_skip_reason": lowered.optimization_skip_reason,
        "externalize_weights": bool(resolved.externalize_weights),
        "external_weight_threshold": int(resolved.external_weight_threshold),
        "extra_input_names": prepared.extra_input_names,
        "unresolved_extra_inputs": lowered.unresolved_extra_inputs,
        "weight_manifest": [entry.to_dict() for entry in lowered.weight_manifest],
        "inference_summary": prepared.inference_summary,
    }
    return ConvertedCoreAIModel(
        prepared=prepared,
        lowered=lowered,
        asset=asset,
        asset_path=asset_path,
        metadata=metadata,
    )


def find_extra_input_names(
    graph: Graph,
    normalized_inputs: Mapping[str, Any],
) -> list[str]:
    input_names = set(normalized_inputs.keys())
    return [spec.name for spec in graph.inputs if spec.name not in input_names]


def _metadata_dynamic_axes(dynamic_axes: DynamicAxes | None) -> dict[str, Any]:
    if dynamic_axes is None:
        return {}
    out: dict[str, Any] = {}
    for name, axes in dynamic_axes.items():
        if axes == "all":
            out[str(name)] = "all"
        elif isinstance(axes, Mapping):
            out[str(name)] = [int(axis) for axis in axes]
        else:
            out[str(name)] = [int(axis) for axis in axes]
    return out


def _resolve_capture_components(
    target: Any,
    capture_function: Callable[..., Any] | None,
) -> tuple[Callable[..., Any], Any]:
    if capture_function is None:
        if not callable(target):
            raise TypeError("target must be callable when capture_function is not provided.")
        capture_function = target
    return capture_function, target
