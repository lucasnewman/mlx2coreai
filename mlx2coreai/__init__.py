from __future__ import annotations

# Initialize authoring before the compiler dialect to avoid a beta 3 import cycle.
import coreai.authoring  # noqa: F401

from .conversion import (
    CapturedMLXGraph,
    ConversionConfig,
    ConvertedCoreAIModel,
    PreparedMLXGraph,
    capture_mlx_graph,
    convert_mlx_to_coreai,
    convert_prepared_mlx_to_coreai,
    lower_graph_to_coreai,
    prepare_mlx_conversion,
)
from .ir import Graph, Node, StateSpec, TensorSpec, TensorType
from .signature import CaptureSignature, StateBinding
from .runtime import (
    CoreAIOutputComparison,
    CoreAIRuntimeOutputs,
    CoreAIRuntimeUnavailableError,
    CoreAISession,
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

__all__ = [
    "CapturedMLXGraph",
    "CaptureSignature",
    "ConversionConfig",
    "ConvertedCoreAIModel",
    "CoreAIOutputComparison",
    "CoreAIRuntimeOutputs",
    "CoreAIRuntimeUnavailableError",
    "CoreAISession",
    "CoreAIValidationResult",
    "Graph",
    "Node",
    "PreparedMLXGraph",
    "StateSpec",
    "StateBinding",
    "TensorSpec",
    "TensorType",
    "capture_mlx_graph",
    "compare_coreai_outputs",
    "convert_mlx_to_coreai",
    "convert_prepared_mlx_to_coreai",
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
