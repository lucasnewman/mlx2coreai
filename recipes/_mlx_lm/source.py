"""MLX-LM loading and capture inputs shared by language-model recipes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np



@dataclass(slots=True)
class MLXLMConversionInputs:
    input_ids: np.ndarray
    prompt: str | None
    token_count: int
    padded_token_count: int
    synthetic: bool = False

    def as_dict(self, *, input_name: str = "input_ids") -> dict[str, np.ndarray]:
        return {input_name: self.input_ids}


def load_mlx_lm_model(
    model_id: str,
    *,
    lazy_load: bool = False,
    revision: str | None = None,
    load_fn: Callable[..., tuple[Any, Any]] | None = None,
) -> tuple[Any, Any]:
    if load_fn is None:
        try:
            from mlx_lm import load as load_fn
        except ImportError as exc:  # pragma: no cover - depends on optional package install
            raise ImportError(
                "Language-model recipes require mlx-lm. Install it with "
                "`pip install mlx-lm` or pass a custom load_fn."
            ) from exc

    kwargs: dict[str, Any] = {"lazy": bool(lazy_load)}
    if revision is not None:
        kwargs["revision"] = revision
    return load_fn(model_id, **kwargs)


def build_mlx_lm_inputs(
    *,
    tokenizer: Any | None,
    prompt: str | None = None,
    sequence_length: int | None = None,
    batch_size: int = 1,
    input_ids: Any | None = None,
) -> MLXLMConversionInputs:
    if sequence_length is not None and sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    if input_ids is not None:
        array = np.asarray(input_ids, dtype=np.int32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError(f"input_ids must be rank 1 or 2, got shape {array.shape}.")
        return MLXLMConversionInputs(
            input_ids=array,
            prompt=prompt,
            token_count=int(array.shape[-1]),
            padded_token_count=0,
            synthetic=False,
        )

    synthetic = prompt is None
    token_ids = _synthesize_token_ids(tokenizer, sequence_length) if synthetic else _tokenize_prompt(tokenizer, prompt)
    if not token_ids:
        token_ids = [_fallback_token_id(tokenizer)]
    if sequence_length is None:
        token_count = len(token_ids)
        padded_token_count = 0
    else:
        token_count = min(len(token_ids), int(sequence_length))
        if len(token_ids) > sequence_length:
            token_ids = token_ids[:sequence_length]
        padded_token_count = max(0, int(sequence_length) - len(token_ids))
        if padded_token_count:
            token_ids = token_ids + [_pad_token_id(tokenizer, token_ids)] * padded_token_count
    array = np.asarray([token_ids], dtype=np.int32)
    if batch_size > 1:
        array = np.repeat(array, int(batch_size), axis=0)
    return MLXLMConversionInputs(
        input_ids=array,
        prompt=prompt,
        token_count=int(token_count),
        padded_token_count=int(padded_token_count),
        synthetic=synthetic,
    )


def _tokenize_prompt(tokenizer: Any | None, prompt: str | None) -> list[int]:
    if prompt is None:
        return []
    if tokenizer is None:
        return [_fallback_token_id(None)]
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return [_fallback_token_id(tokenizer)]
    encoded = encode(prompt)
    ids = getattr(encoded, "ids", encoded)
    return [int(token_id) for token_id in ids]


def _synthesize_token_ids(tokenizer: Any | None, sequence_length: int | None) -> list[int]:
    length = int(sequence_length) if sequence_length is not None else 1
    return [_fallback_token_id(tokenizer)] * length


def _pad_token_id(tokenizer: Any | None, token_ids: list[int]) -> int:
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return int(token_ids[-1]) if token_ids else _fallback_token_id(tokenizer)


def _fallback_token_id(tokenizer: Any | None) -> int:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return 0
