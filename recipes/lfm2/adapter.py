"""LFM2/2.5 short-convolution state, MoE dispatch, and precision restrictions."""
import warnings

from recipes._mlx_lm.stateful import _ExportableRecurrentCache, _layers, attention_layout

WORKAROUNDS = ["FP32 is the correctness baseline; reduced precision is experimental",
               "Materialize convolution-history updates with gathers before packing state",
               "When present, MoE uses shape-independent unsorted expert dispatch"]


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
    if kind not in {"lfm2", "lfm2_moe"}:
        raise ValueError(f"LFM2 recipe received model type {kind!r}.")
    if kind == 'lfm2_moe':
        import mlx.core as mx
        import mlx.nn as nn

        class ExportSwitchGLU(nn.Module):
            def __init__(self, source):
                super().__init__()
                self.gate_proj = source.gate_proj
                self.up_proj = source.up_proj
                self.down_proj = source.down_proj
                self.activation = source.activation

            def __call__(self, x, indices):
                # Sorting is an MLX locality optimization, not model semantics.
                # Avoid its token-count branch in the exported dynamic graph.
                x = mx.expand_dims(x, (-2, -3))
                gate = self.gate_proj(x, indices)
                up = self.up_proj(x, indices)
                # Expose tokens * selected_experts as a tensor dimension before
                # MLX generates the down projection's implicit batch indices.
                hidden = self.activation(up, gate).reshape(-1, 1, gate.shape[-1])
                return self.down_proj(hidden, indices.reshape(-1)).reshape(*indices.shape, -1)

        for layer in _layers(model):
            block = layer.feed_forward
            if hasattr(block, 'switch_mlp'):
                block.switch_mlp = ExportSwitchGLU(block.switch_mlp)
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
