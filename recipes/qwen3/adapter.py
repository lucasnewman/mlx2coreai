"""Qwen3's full-attention layout uses the common bounded KV adapter."""
from recipes._mlx_lm.stateful import attention_layout

WORKAROUNDS = ["Use FP32 for close logit parity; BF16 has measured drift"]


def experimental(precision, cache_dtype=None):
    return cache_dtype is not None and cache_dtype != precision


def adapt(model):
    kind = getattr(getattr(model, "args", None), "model_type", None)
    if kind not in {"qwen3", "qwen3_moe"}:
        raise ValueError(f"Qwen3 recipe received model type {kind!r}.")
    return model


cache_layout = attention_layout


def warn_precision(precision):
    pass
