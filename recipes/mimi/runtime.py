"""Runtime-only offline codec requests. Consecutive calls do not share history."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Request:
    component: str
    value: np.ndarray


async def run(session, request: Request):
    metadata = session.bundle.metadata
    value = np.asarray(request.value)
    if session.bundle.manifest["recipe"] != "mimi":
        raise ValueError("Expected a Mimi recipe bundle.")
    if request.component == "encode":
        size = metadata["samples_per_frame"]
        if value.ndim != 3 or value.shape[:2] != (1, 1) or value.shape[-1] <= 0 or value.shape[-1] % size:
            raise ValueError(f"Audio must have shape [1,1,N] with N a positive multiple of {size}.")
        if not np.isfinite(value).all():
            raise ValueError("Audio must be finite.")
        return (await session.run("encode", {"audio": value.astype(np.float32)}, readback=True))["codes"]
    if request.component == "decode":
        if (value.ndim != 3 or value.shape[:2] != (1, metadata["codebooks"]) or value.shape[-1] <= 0
                or not np.issubdtype(value.dtype, np.integer)
                or np.any(value < 0) or np.any(value >= metadata["codebook_size"])):
            raise ValueError("Codes must be integer [1,codebooks,frames] values inside the codebook range.")
        return (await session.run("decode", {"codes": value.astype(np.int32)}, readback=True))["audio"]
    raise ValueError(f"Unknown Mimi component: {request.component}")
