"""Qwen3.5 text-only recipe; native recurrence remains experimental."""
from recipes._mlx_lm.build import build_model
from recipes._mlx_lm.runtime import Request, run
from . import adapter


def build(source="Qwen/Qwen3.5-0.8B", *, compute_precision="auto", **options):
    return build_model(source, recipe="qwen35", adapter=adapter, compute_precision=compute_precision, **options)
