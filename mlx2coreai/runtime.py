from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import ml_dtypes
import numpy as np


class CoreAIRuntimeUnavailableError(RuntimeError):
    """Raised when coreai-core runtime imports are not available."""


@dataclass(frozen=True, slots=True)
class CoreAIRuntimeOutputs:
    outputs: dict[str, np.ndarray]
    function_name: str
    asset_path: Path | None = None
    raw_outputs: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class CoreAIOutputComparison:
    expected_name: str
    actual_name: str | None
    passed: bool
    expected_shape: tuple[int, ...]
    actual_shape: tuple[int, ...] | None
    expected_dtype: str
    actual_dtype: str | None
    max_abs_error: float | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class CoreAIValidationResult:
    runtime: CoreAIRuntimeOutputs
    comparisons: list[CoreAIOutputComparison]

    @property
    def passed(self) -> bool:
        return bool(self.comparisons) and all(
            comparison.passed for comparison in self.comparisons
        )


@dataclass(frozen=True, slots=True)
class _CoreAIRuntimeBindings:
    AIModelAsset: Any
    NDArray: Any
    StorageKind: Any
    SpecializationOptions: Any | None = None
    ComputeUnitKind: Any | None = None


def coreai_runtime_available() -> bool:
    try:
        _load_coreai_runtime()
    except CoreAIRuntimeUnavailableError:
        return False
    return True


async def run_aimodel(
    asset_or_path: Any,
    inputs: Mapping[str, Any],
    *,
    function_name: str = "main",
    specialization_options: Any | None = None,
    storage_kind: Any | str | None = None,
) -> CoreAIRuntimeOutputs:
    """Load and run a saved CoreAI .aimodel asset with coreai.runtime."""

    bindings = _load_coreai_runtime()
    asset, asset_path = _coerce_asset(asset_or_path, bindings.AIModelAsset)
    resolved_storage_kind = _resolve_storage_kind(storage_kind, bindings.StorageKind)
    nd_inputs = {
        name: _to_ndarray(value, bindings.NDArray, resolved_storage_kind)
        for name, value in inputs.items()
    }

    async with asset.executable(specialization_options=specialization_options) as ai_model:
        function = ai_model.load_function(function_name)
        raw_outputs = await function(inputs=nd_inputs)

    outputs = {
        str(name): _output_to_numpy(value)
        for name, value in raw_outputs.items()
    }
    return CoreAIRuntimeOutputs(
        outputs=outputs,
        function_name=function_name,
        asset_path=asset_path,
        raw_outputs=raw_outputs,
    )


def run_aimodel_sync(
    asset_or_path: Any,
    inputs: Mapping[str, Any],
    *,
    function_name: str = "main",
    specialization_options: Any | None = None,
    storage_kind: Any | str | None = None,
) -> CoreAIRuntimeOutputs:
    return _run_sync(
        run_aimodel(
            asset_or_path,
            inputs,
            function_name=function_name,
            specialization_options=specialization_options,
            storage_kind=storage_kind,
        )
    )


async def run_coreai_program(
    program: Any,
    inputs: Mapping[str, Any],
    *,
    asset_path: str | Path | None = None,
    function_name: str = "main",
    specialization_options: Any | None = None,
    storage_kind: Any | str | None = None,
) -> CoreAIRuntimeOutputs:
    """Save an AIProgram if needed, then run it through coreai.runtime."""

    if asset_path is not None:
        asset = program.save_asset(Path(asset_path))
        return await run_aimodel(
            asset,
            inputs,
            function_name=function_name,
            specialization_options=specialization_options,
            storage_kind=storage_kind,
        )

    with TemporaryDirectory(prefix="mlx2coreai_runtime_") as temp_dir:
        asset = program.save_asset(Path(temp_dir) / "model.aimodel")
        result = await run_aimodel(
            asset,
            inputs,
            function_name=function_name,
            specialization_options=specialization_options,
            storage_kind=storage_kind,
        )
        return replace(result, asset_path=None)


def run_coreai_program_sync(
    program: Any,
    inputs: Mapping[str, Any],
    *,
    asset_path: str | Path | None = None,
    function_name: str = "main",
    specialization_options: Any | None = None,
    storage_kind: Any | str | None = None,
) -> CoreAIRuntimeOutputs:
    return _run_sync(
        run_coreai_program(
            program,
            inputs,
            asset_path=asset_path,
            function_name=function_name,
            specialization_options=specialization_options,
            storage_kind=storage_kind,
        )
    )


async def run_converted_model(
    converted: Any,
    inputs: Mapping[str, Any] | None = None,
    *,
    function_name: str | None = None,
    specialization_options: Any | None = None,
    storage_kind: Any | str | None = None,
) -> CoreAIRuntimeOutputs:
    """Run a ConvertedCoreAIModel with its captured inputs by default."""

    resolved_inputs = inputs
    if resolved_inputs is None:
        resolved_inputs = converted.prepared.normalized_inputs
    resolved_function_name = function_name or converted.metadata.get("entrypoint_name") or "main"

    asset = getattr(converted, "asset", None)
    if asset is not None:
        return await run_aimodel(
            asset,
            resolved_inputs,
            function_name=resolved_function_name,
            specialization_options=specialization_options,
            storage_kind=storage_kind,
        )

    asset_path = getattr(converted, "asset_path", None)
    if asset_path is not None:
        return await run_aimodel(
            asset_path,
            resolved_inputs,
            function_name=resolved_function_name,
            specialization_options=specialization_options,
            storage_kind=storage_kind,
        )

    return await run_coreai_program(
        converted.program,
        resolved_inputs,
        function_name=resolved_function_name,
        specialization_options=specialization_options,
        storage_kind=storage_kind,
    )


def run_converted_model_sync(
    converted: Any,
    inputs: Mapping[str, Any] | None = None,
    *,
    function_name: str | None = None,
    specialization_options: Any | None = None,
    storage_kind: Any | str | None = None,
) -> CoreAIRuntimeOutputs:
    return _run_sync(
        run_converted_model(
            converted,
            inputs,
            function_name=function_name,
            specialization_options=specialization_options,
            storage_kind=storage_kind,
        )
    )


def compare_coreai_outputs(
    actual_outputs: Mapping[str, Any],
    expected_outputs: Mapping[str, Any],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    match_by_order: bool = True,
) -> list[CoreAIOutputComparison]:
    actual_arrays = {
        str(name): _output_to_numpy(value)
        for name, value in actual_outputs.items()
    }
    expected_arrays = {
        str(name): np.asarray(value)
        for name, value in expected_outputs.items()
    }
    actual_items = list(actual_arrays.items())
    comparisons: list[CoreAIOutputComparison] = []

    for index, (expected_name, expected) in enumerate(expected_arrays.items()):
        if expected_name in actual_arrays:
            actual_name: str | None = expected_name
            actual = actual_arrays[expected_name]
        elif match_by_order and index < len(actual_items):
            actual_name, actual = actual_items[index]
        else:
            comparisons.append(
                CoreAIOutputComparison(
                    expected_name=expected_name,
                    actual_name=None,
                    passed=False,
                    expected_shape=tuple(expected.shape),
                    actual_shape=None,
                    expected_dtype=str(expected.dtype),
                    actual_dtype=None,
                    message="missing output",
                )
            )
            continue

        comparison = _compare_array(
            actual_name=actual_name,
            actual=actual,
            expected_name=expected_name,
            expected=expected,
            rtol=rtol,
            atol=atol,
        )
        comparisons.append(comparison)

    return comparisons


async def validate_aimodel_outputs(
    asset_or_path: Any,
    inputs: Mapping[str, Any],
    expected_outputs: Mapping[str, Any],
    *,
    function_name: str = "main",
    specialization_options: Any | None = None,
    storage_kind: Any | str | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    match_by_order: bool = True,
) -> CoreAIValidationResult:
    runtime = await run_aimodel(
        asset_or_path,
        inputs,
        function_name=function_name,
        specialization_options=specialization_options,
        storage_kind=storage_kind,
    )
    comparisons = compare_coreai_outputs(
        runtime.outputs,
        expected_outputs,
        rtol=rtol,
        atol=atol,
        match_by_order=match_by_order,
    )
    return CoreAIValidationResult(runtime=runtime, comparisons=comparisons)


def validate_aimodel_outputs_sync(
    asset_or_path: Any,
    inputs: Mapping[str, Any],
    expected_outputs: Mapping[str, Any],
    *,
    function_name: str = "main",
    specialization_options: Any | None = None,
    storage_kind: Any | str | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    match_by_order: bool = True,
) -> CoreAIValidationResult:
    return _run_sync(
        validate_aimodel_outputs(
            asset_or_path,
            inputs,
            expected_outputs,
            function_name=function_name,
            specialization_options=specialization_options,
            storage_kind=storage_kind,
            rtol=rtol,
            atol=atol,
            match_by_order=match_by_order,
        )
    )


async def validate_converted_model(
    converted: Any,
    inputs: Mapping[str, Any] | None = None,
    expected_outputs: Mapping[str, Any] | None = None,
    *,
    function_name: str | None = None,
    specialization_options: Any | None = None,
    storage_kind: Any | str | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    match_by_order: bool = True,
) -> CoreAIValidationResult:
    resolved_expected = expected_outputs
    if resolved_expected is None:
        resolved_expected = converted.prepared.expected_outputs
    runtime = await run_converted_model(
        converted,
        inputs,
        function_name=function_name,
        specialization_options=specialization_options,
        storage_kind=storage_kind,
    )
    comparisons = compare_coreai_outputs(
        runtime.outputs,
        resolved_expected,
        rtol=rtol,
        atol=atol,
        match_by_order=match_by_order,
    )
    return CoreAIValidationResult(runtime=runtime, comparisons=comparisons)


def validate_converted_model_sync(
    converted: Any,
    inputs: Mapping[str, Any] | None = None,
    expected_outputs: Mapping[str, Any] | None = None,
    *,
    function_name: str | None = None,
    specialization_options: Any | None = None,
    storage_kind: Any | str | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    match_by_order: bool = True,
) -> CoreAIValidationResult:
    return _run_sync(
        validate_converted_model(
            converted,
            inputs,
            expected_outputs,
            function_name=function_name,
            specialization_options=specialization_options,
            storage_kind=storage_kind,
            rtol=rtol,
            atol=atol,
            match_by_order=match_by_order,
        )
    )


def _load_coreai_runtime() -> _CoreAIRuntimeBindings:
    try:
        from coreai.authoring import AIModelAsset  # noqa: PLC0415
        from coreai.runtime import (  # noqa: PLC0415
            ComputeUnitKind,
            NDArray,
            SpecializationOptions,
            StorageKind,
        )
    except Exception as exc:  # pragma: no cover - depends on installed wheel/OS
        raise CoreAIRuntimeUnavailableError(
            "coreai.runtime is not available. Install coreai-core with runtime "
            "support and run on a CoreAI-capable macOS/iOS runtime."
        ) from exc
    return _CoreAIRuntimeBindings(
        AIModelAsset=AIModelAsset,
        NDArray=NDArray,
        StorageKind=StorageKind,
        SpecializationOptions=SpecializationOptions,
        ComputeUnitKind=ComputeUnitKind,
    )


def _coerce_asset(asset_or_path: Any, AIModelAsset: Any) -> tuple[Any, Path | None]:
    if isinstance(asset_or_path, str | Path):
        path = Path(asset_or_path)
        return AIModelAsset.load(path), path
    if not hasattr(asset_or_path, "executable"):
        raise TypeError(
            "asset_or_path must be a .aimodel path or an AIModelAsset-like object "
            "with executable()."
        )
    raw_path = getattr(asset_or_path, "path", None)
    return asset_or_path, Path(raw_path) if raw_path is not None else None


def _resolve_storage_kind(storage_kind: Any | str | None, StorageKind: Any) -> Any | None:
    if storage_kind is None:
        return None
    if not isinstance(storage_kind, str):
        return storage_kind
    normalized = storage_kind.lower().replace("-", "_")
    for candidate in StorageKind:
        value = str(getattr(candidate, "value", "")).lower().replace("-", "_")
        name = str(getattr(candidate, "name", "")).lower().replace("-", "_")
        if normalized in {value, name}:
            return candidate
    raise ValueError(f"Unknown CoreAI storage kind: {storage_kind!r}")


def _to_ndarray(value: Any, NDArray: Any, storage_kind: Any | None) -> Any:
    if isinstance(value, NDArray):
        return value
    data = _runtime_input_data(value)
    if storage_kind is None:
        return NDArray(data)
    return NDArray(data=data, backing=storage_kind)


def _runtime_input_data(value: Any) -> Any:
    if isinstance(value, np.ndarray | list | tuple):
        return value
    module = type(value).__module__.split(".", maxsplit=1)[0]
    if module == "torch":
        return value
    try:
        return np.asarray(value)
    except Exception:
        return value


def _output_to_numpy(value: Any) -> np.ndarray:
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        return np.asarray(numpy_fn())
    return np.asarray(value)


def _compare_array(
    *,
    actual_name: str,
    actual: np.ndarray,
    expected_name: str,
    expected: np.ndarray,
    rtol: float,
    atol: float,
) -> CoreAIOutputComparison:
    if tuple(actual.shape) != tuple(expected.shape):
        return CoreAIOutputComparison(
            expected_name=expected_name,
            actual_name=actual_name,
            passed=False,
            expected_shape=tuple(expected.shape),
            actual_shape=tuple(actual.shape),
            expected_dtype=str(expected.dtype),
            actual_dtype=str(actual.dtype),
            message="shape mismatch",
        )

    max_abs_error = _max_abs_error(actual, expected)
    if _needs_allclose(actual, expected):
        passed = bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True))
        message = None if passed else "values differ"
    else:
        passed = bool(np.array_equal(actual, expected))
        message = None if passed else "values differ"

    return CoreAIOutputComparison(
        expected_name=expected_name,
        actual_name=actual_name,
        passed=passed,
        expected_shape=tuple(expected.shape),
        actual_shape=tuple(actual.shape),
        expected_dtype=str(expected.dtype),
        actual_dtype=str(actual.dtype),
        max_abs_error=max_abs_error,
        message=message,
    )


def _needs_allclose(actual: np.ndarray, expected: np.ndarray) -> bool:
    return _is_numeric_dtype(actual.dtype) or _is_numeric_dtype(expected.dtype)


def _max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float | None:
    if actual.size == 0:
        return 0.0
    if not (_is_numeric_dtype(actual.dtype) and _is_numeric_dtype(expected.dtype)):
        return None
    try:
        delta = actual.astype(np.complex128) - expected.astype(np.complex128)
        return float(np.max(np.abs(delta)))
    except Exception:
        return None


def _is_numeric_dtype(dtype: np.dtype) -> bool:
    if dtype == ml_dtypes.bfloat16:
        return True
    return np.dtype(dtype).kind in {"b", "i", "u", "f", "c"}


def _run_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "Cannot use a sync CoreAI runtime helper from a running event loop; "
        "await the async helper instead."
    )
