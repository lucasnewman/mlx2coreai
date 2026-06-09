from __future__ import annotations

from importlib import import_module
from typing import Any

from .conversion import (
    CapturedMLXGraph,
    ConversionConfig,
    ConvertedCoreAIModel,
    PreparedMLXGraph,
    capture_mlx_graph,
    convert_mlx_to_coreai,
    lower_graph_to_coreai,
    prepare_mlx_conversion,
)
from .ir import Graph, Node, StateSpec, TensorSpec
from .runtime import (
    CoreAIOutputComparison,
    CoreAIRuntimeOutputs,
    CoreAIRuntimeUnavailableError,
    CoreAIValidationResult,
    compare_coreai_outputs,
    coreai_runtime_available,
    run_aimodel,
    run_aimodel_sync,
    run_converted_model,
    run_converted_model_sync,
    run_coreai_program,
    run_coreai_program_sync,
    validate_aimodel_outputs,
    validate_aimodel_outputs_sync,
    validate_converted_model,
    validate_converted_model_sync,
)

_LAZY_EXPORTS = {
    "MLXLMConversionInputs": "MLXLMConversionInputs",
    "MLXLMStatefulConversion": ("._convert_mlx_lm_stateful", "MLXLMStatefulConversion"),
    "build_mlx_lm_inputs": "build_mlx_lm_inputs",
    "convert_mlx_lm": "convert_mlx_lm",
    "convert_mlx_lm_stateful": ("._convert_mlx_lm_stateful", "convert_mlx_lm_stateful"),
}

__all__ = [
    "CapturedMLXGraph",
    "ConversionConfig",
    "ConvertedCoreAIModel",
    "CoreAIOutputComparison",
    "CoreAIRuntimeOutputs",
    "CoreAIRuntimeUnavailableError",
    "CoreAIValidationResult",
    "Graph",
    "MLXLMConversionInputs",
    "MLXLMStatefulConversion",
    "Node",
    "PreparedMLXGraph",
    "StateSpec",
    "TensorSpec",
    "build_mlx_lm_inputs",
    "capture_mlx_graph",
    "compare_coreai_outputs",
    "convert_mlx_lm",
    "convert_mlx_lm_stateful",
    "convert_mlx_to_coreai",
    "coreai_runtime_available",
    "lower_graph_to_coreai",
    "prepare_mlx_conversion",
    "run_aimodel",
    "run_aimodel_sync",
    "run_converted_model",
    "run_converted_model_sync",
    "run_coreai_program",
    "run_coreai_program_sync",
    "validate_aimodel_outputs",
    "validate_aimodel_outputs_sync",
    "validate_converted_model",
    "validate_converted_model_sync",
]


def __getattr__(name: str) -> Any:
    export = _LAZY_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if isinstance(export, tuple):
        module_name, export_name = export
    else:
        module_name, export_name = "._convert_mlx_lm", export
    module = import_module(module_name, __name__)
    value = getattr(module, export_name)
    globals()[name] = value
    return value
