"""Precision aliases and conversion policies shared by capture, emission, and runtime."""
from __future__ import annotations
from typing import Any
import ml_dtypes
import numpy as np


_DTYPE_ALIASES = {
    "half": "fp16",
    "float16": "fp16",
    "fp16": "fp16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float": "fp32",
    "float32": "fp32",
    "fp32": "fp32",
    "double": "fp64",
    "float64": "fp64",
    "fp64": "fp64",
    "int": "int32",
    "int32": "int32",
    "long": "int64",
    "int64": "int64",
    "bool": "bool",
    "complex64": "complex64",
}


def normalize_dtype(dtype: str) -> str:
    return _DTYPE_ALIASES.get(str(dtype).strip().lower(), str(dtype).strip().lower())


def execution_numpy_dtype(dtype: str) -> Any:
    dtype = normalize_dtype(dtype)
    if dtype == "fp16":
        return np.float16
    if dtype == "bf16":
        return ml_dtypes.bfloat16
    if dtype in {"fp32", "fp64"}:
        return np.float32
    if dtype in {"int32", "int64"}:
        return np.int32
    if dtype == "bool":
        return np.bool_
    if dtype == "complex64":
        return np.complex64
    raise ValueError(f"Unsupported dtype for constant: {dtype}")


def constant_array(value: Any, dtype_hint: str | None = None) -> tuple[np.ndarray, str | None]:
    arr = np.asarray(value)
    downcast: str | None = None
    if dtype_hint is not None:
        dtype_hint = normalize_dtype(dtype_hint)

    if arr.dtype == np.float64 or dtype_hint == "fp64":
        arr = arr.astype(np.float32)
        downcast = "fp64->fp32"
    elif arr.dtype == np.int64 or dtype_hint == "int64":
        if arr.size and (arr.min() < np.iinfo(np.int32).min or arr.max() > np.iinfo(np.int32).max):
            raise ValueError("int64 constant cannot be safely downcast to int32.")
        arr = arr.astype(np.int32)
        downcast = "int64->int32"
    elif dtype_hint == "bf16":
        arr = arr.astype(ml_dtypes.bfloat16)
    elif dtype_hint is not None:
        arr = arr.astype(execution_numpy_dtype(dtype_hint))
    # ascontiguousarray promotes scalar tensors to rank one.
    return np.ascontiguousarray(arr).reshape(arr.shape), downcast


def capture_numpy_dtype(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == ml_dtypes.bfloat16:
        return "bf16"
    if dtype == np.float16:
        return "fp16"
    if dtype == np.float32:
        return "fp32"
    if dtype == np.int32:
        return "int32"
    if dtype == np.int64:
        return "int64"
    if dtype == np.bool_:
        return "bool"
    if dtype == np.complex64:
        return "complex64"
    if np.issubdtype(dtype, np.floating):
        if dtype.itemsize <= np.dtype(np.float16).itemsize:
            return "fp16"
        return "fp32"
    if np.issubdtype(dtype, np.signedinteger):
        if dtype.itemsize <= np.dtype(np.int32).itemsize:
            return "int32"
        return "int64"
    if np.issubdtype(dtype, np.unsignedinteger):
        if dtype.itemsize <= np.dtype(np.uint32).itemsize:
            return "int32"
        return "int64"
    raise ValueError(f"Unsupported dtype for v0 translator: {dtype}")


def capture_mlx_dtype(dtype: Any) -> str:
    if isinstance(dtype, np.dtype):
        return capture_numpy_dtype(dtype)
    text = str(dtype).strip().lower()
    if text.endswith('complex64'):
        return 'complex64'
    if "bfloat16" in text or text.endswith("bf16"):
        return "bf16"
    if text.endswith("float16") or text.endswith("fp16"):
        return "fp16"
    if text.endswith("float32") or text.endswith("fp32"):
        return "fp32"
    if text.endswith("int32"):
        return "int32"
    if text.endswith("int64"):
        return "int64"
    if text.endswith("bool"):
        return "bool"
    raise ValueError(f"Unsupported MLX dtype for capture parser: {dtype}")


def runtime_numpy_dtype(dtype: Any) -> Any:
    text = str(dtype).strip().lower()
    for aliases, result in (
        (("bfloat16", "bf16"), ml_dtypes.bfloat16),
        (("float16", "fp16"), np.float16),
        (("float32", "fp32"), np.float32),
        (("int32",), np.int32),
    ):
        if any(alias in text for alias in aliases):
            return result
    raise ValueError(f"Unsupported runtime state dtype: {dtype!r}.")


def cache_numpy_dtype(dtype: str) -> Any:
    normalized = dtype.strip().lower()
    if normalized in {"fp32", "float32"}:
        return np.float32
    if normalized in {"fp16", "float16"}:
        return np.float16
    if normalized in {"bf16", "bfloat16"}:
        return ml_dtypes.bfloat16
    raise ValueError(f"Unsupported cache dtype: {dtype!r}.")


def cast_model_precision(model: Any, compute_precision: str) -> None:
    set_dtype = getattr(model, "set_dtype", None)
    if not callable(set_dtype):
        return
    try:
        import mlx.core as mx  # noqa: PLC0415
    except ImportError:
        return
    dtype = {
        "bf16": mx.bfloat16,
        "fp16": mx.float16,
        "fp32": mx.float32,
    }[compute_precision]
    predicate = getattr(model, "cast_predicate", None)
    if predicate is not None:
        set_dtype(dtype, predicate=predicate)
    else:
        set_dtype(dtype)
    # Do not capture lazy parameter casts on the first shape but materialized
    # weights on the probe shape: the two graphs must have the same structure.
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        mx.eval(parameters())


def floating_precision(dtype: Any) -> str | None:
    if dtype is None:
        return None
    text = str(dtype).strip().lower()
    if "bfloat16" in text or "bf16" in text:
        return "bf16"
    if "float16" in text or "fp16" in text:
        return "fp16"
    if "float32" in text or "fp32" in text:
        return "fp32"
    return None


def normalize_compute_precision(dtype: str) -> str:
    normalized = dtype.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"fp32", "float32"}:
        return "fp32"
    if normalized in {"fp16", "float16"}:
        return "fp16"
    if normalized in {"bf16", "bfloat16"}:
        return "bf16"
    raise ValueError(f"Unsupported compute/cache dtype: {dtype!r}.")
