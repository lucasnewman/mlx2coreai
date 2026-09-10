"""Stateless endpoint inference from prepared features; no MLX imports."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Request:
    input_features: np.ndarray
    threshold: float | None = None


async def run(session, request: Request):
    if session.bundle.manifest["recipe"] != "smart_turn":
        raise ValueError("Expected a SmartTurn recipe bundle.")
    metadata = session.bundle.metadata
    features = np.asarray(request.input_features)
    shape = tuple(metadata["feature_shape"])
    if features.ndim != 3 or features.shape[0] <= 0 or features.shape[1:] != shape:
        raise ValueError(f"Features must have shape [batch, {shape[0]}, {shape[1]}] with a positive batch.")
    if not np.issubdtype(features.dtype, np.floating) or not np.isfinite(features).all():
        raise ValueError("Features must be finite floating-point values.")
    threshold = metadata["threshold"] if request.threshold is None else request.threshold
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1].")
    features = features.astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Features must be representable as finite FP32 values.")
    outputs = await session.run("main", {"input_features": features}, readback=True)
    expected_shape = (len(features), 1)
    if any(value.shape != expected_shape or not np.isfinite(value).all() for value in outputs.values()):
        raise RuntimeError("SmartTurn produced invalid output shapes or nonfinite values.")
    if np.any(outputs["probability"] < 0) or np.any(outputs["probability"] > 1):
        raise RuntimeError("SmartTurn produced a probability outside [0, 1].")
    return {**outputs, "prediction": (outputs["probability"] > threshold).astype(np.int32)}
