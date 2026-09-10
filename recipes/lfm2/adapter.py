"""LFM2/2.5's short-convolution state layout and precision restrictions."""
import warnings

from recipes._mlx_lm.stateful import _ExportableRecurrentCache, _layers, attention_layout

WORKAROUNDS = ["FP32 is the correctness baseline; reduced precision is experimental",
               "Materialize convolution-history updates with gathers before packing state"]


class ArrayCache(_ExportableRecurrentCache):
    def __setitem__(self, index, value):
        import mlx.core as mx

        # Avoid a slice-view/reshape chain aliasing the live history buffer in
        # the beta runtime. The history length is fixed by the convolution.
        self.state[index] = mx.take(value, mx.arange(value.shape[1]), axis=1)


def experimental(precision, cache_dtype=None):
    return precision != "fp32" or cache_dtype not in (None, "fp32")


def adapt(model):
    kind = getattr(getattr(model, "args", None), "model_type", None)
    if kind != "lfm2":
        raise ValueError(f"LFM2 recipe received model type {kind!r}.")
    return model


def cache_layout(model):
    layers = _layers(model)
    conv = tuple(getattr(layer, "is_attention_layer", None) is False
                 and hasattr(getattr(layer, "conv", None), "L_cache") for layer in layers)
    shapes = {(layer.conv.L_cache - 1, layer.conv.args.hidden_size)
              for layer, active in zip(layers, conv) if active}
    if len(shapes) != 1 or any(size <= 0 for shape in shapes for size in shape):
        raise ValueError("Short-convolution conversion requires uniform, positive cache shapes.")
    return attention_layout(model, short_conv_layers=conv, conv_shape=shapes.pop())


def warn_precision(precision):
    if precision != "fp32":
        warnings.warn("Reduced-precision short-convolution export is experimental: validate against MLX. "
                      "macOS 27 build 26A428 can abort during FP16 compilation; use FP32 for correctness checks.",
                      RuntimeWarning, stacklevel=3)
