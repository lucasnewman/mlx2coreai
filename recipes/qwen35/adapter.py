"""Qwen3.5 text-only hybrid state layout and its current runtime warning."""
import warnings

from recipes._mlx_lm.stateful import _layers, attention_layout

WORKAROUNDS = ["Text decoder only; FP32/decomposed recurrence validated on b3; native recurrent state corrupts"]


def experimental(precision, cache_dtype=None):
    return True


def adapt(model):
    kind = getattr(getattr(model, "args", None), "model_type", None)
    if kind not in {"qwen3_5", "qwen3_5_moe"}:
        raise ValueError(f"Qwen3.5 recipe received model type {kind!r}.")
    return getattr(model, "language_model", model)


def cache_layout(model):
    layers = _layers(model)
    linear = tuple(bool(getattr(layer, "is_linear", False)) for layer in layers)
    shapes = {(layer.linear_attn.conv_kernel_size - 1, layer.linear_attn.conv_dim,
               layer.linear_attn.num_v_heads, layer.linear_attn.head_v_dim, layer.linear_attn.head_k_dim)
              for layer, active in zip(layers, linear) if active}
    if len(shapes) != 1:
        raise ValueError("Hybrid conversion requires uniform gated-delta state shapes.")
    dims = shapes.pop()
    return attention_layout(model, linear_layers=linear, conv_shape=dims[:2], recurrent_shape=dims[2:])


def warn_precision(precision):
    warnings.warn("Hybrid gated-delta export is experimental: native recurrence corrupts state on "
                  "coreai-core 1.0.0b3. Use FP32 with decomposed lowering and validate against MLX before use.",
                  RuntimeWarning, stacklevel=3)
