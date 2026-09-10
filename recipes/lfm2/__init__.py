"""LFM2/2.5 short-convolution recipe; FP32 is the correctness baseline."""
from recipes._mlx_lm.build import build_model
from recipes._mlx_lm.runtime import Request, run
from . import adapter


def build(source="LiquidAI/LFM2.5-2.6B-MLX-bf16", **options):
    plan = build_model(source, recipe="lfm2", adapter=adapter, **options)
    if plan.metadata['model_type'] == 'lfm2_moe':
        plan.metadata['experimental'] = True
        plan.metadata['experimental_reason'] = (
            'Deep MoE decoding fails native logit/state parity on the beta CoreAI runtime; '
            'full-checkpoint execution can also report Metal command-buffer errors.')
    return plan
