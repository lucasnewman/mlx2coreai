"""LFM2/2.5 short-convolution recipe; FP32 is the validated default."""
from recipes._mlx_lm.build import build_model
from recipes._mlx_lm.runtime import Request, run
from . import adapter


def build(source="LiquidAI/LFM2.5-2.6B-MLX-bf16", **options):
    return build_model(source, recipe="lfm2", adapter=adapter, **options)
