"""Qwen3 full-attention recipe. FP32 is the default correctness baseline."""
from recipes._mlx_lm.build import build_model
from recipes._mlx_lm.runtime import Request, run
from . import adapter


def build(source="mlx-community/Qwen3-0.6B-bf16", **options):
    return build_model(source, recipe="qwen3", adapter=adapter, **options)
