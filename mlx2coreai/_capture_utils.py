"""Array and output adaptation shared by callback and compatibility capture."""
from __future__ import annotations
from typing import Any
import numpy as np
import ml_dtypes
from .dtypes import capture_numpy_dtype as _numpy_dtype_to_ir
from .ir import TensorSpec


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if value is None:
        return tuple()
    return (int(value),)


def _constant_to_numpy(value: Any) -> np.ndarray:
    try:
        return np.asarray(value)
    except Exception:
        # MLX bf16 arrays can fail direct numpy conversion via buffer protocol.
        if hasattr(value, "astype"):
            try:
                import mlx.core as mx  # noqa: PLC0415

                if "bfloat16" in str(getattr(value, "dtype", "")).lower():
                    return np.asarray(value.astype(mx.float32)).astype(ml_dtypes.bfloat16)
                return np.asarray(value.astype(mx.float32))
            except Exception:
                pass
        raise


def _normalize_numpy_inputs(inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {name: _constant_to_numpy(value) for name, value in inputs.items()}


def _default_input_specs(inputs: dict[str, np.ndarray]) -> list[TensorSpec]:
    return [
        TensorSpec(
            name=name,
            shape=tuple(int(v) for v in array.shape),
            dtype=_numpy_dtype_to_ir(np.asarray(array).dtype),
        )
        for name, array in inputs.items()
    ]


def _normalize_outputs(outputs: Any) -> list[Any]:
    if isinstance(outputs, dict):
        return list(outputs.values())
    if isinstance(outputs, (tuple, list)):
        return list(outputs)
    return [outputs]
