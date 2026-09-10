"""Legacy automatic selection; explicit recipes select their own adapters."""


def unwrap(model):
    kind = getattr(getattr(model, "args", None), "model_type", None)
    if kind in {"qwen3_5", "qwen3_5_moe"}:
        from recipes.qwen35.adapter import adapt
        return adapt(model)
    return model


def layout(model):
    from .stateful import _layers, attention_layout

    layers = _layers(model)
    linear = any(getattr(layer, "is_linear", False) for layer in layers)
    conv = any(getattr(layer, "is_attention_layer", None) is False
               and hasattr(getattr(layer, "conv", None), "L_cache") for layer in layers)
    if linear and conv:
        raise ValueError("Mixing gated-delta and short-convolution layer types is not supported.")
    if linear:
        from recipes.qwen35.adapter import cache_layout
        return cache_layout(model)
    if conv:
        from recipes.lfm2.adapter import cache_layout
        return cache_layout(model)
    return attention_layout(model)


def warn_precision(layout, precision):
    if layout.num_linear_layers:
        from recipes.qwen35.adapter import warn_precision
        warn_precision(precision)
    elif layout.num_short_conv_layers:
        from recipes.lfm2.adapter import warn_precision
        warn_precision(precision)
