"""FP32 stateful capture adapters for mlx-audio's Pocket TTS (optional dependency)."""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from mlx2coreai.kv_cache import LayeredKVCache, LayeredKVCacheState
from mlx2coreai.conversion import ConversionConfig
from mlx2coreai.ir import StateSpec
from mlx2coreai.signature import CaptureSignature, StateBinding


def kv_specs(layers, heads, head_dim):
    shape = (layers, 1, heads, -1, head_dim)
    return [StateSpec("keyCache", shape, "fp32", capacity_axis=3),
            StateSpec("valueCache", shape, "fp32", capacity_axis=3)]


def initial_state(specs, capacity):
    """Zero state plus the ordinary (non-mutating) position input for capture."""
    if capacity <= 0:
        raise ValueError("Cache capacity must be positive.")
    states = {spec.name: np.zeros(spec.resolved_shape(capacity),
                                 np.int32 if spec.dtype == "int32" else np.float32)
              for spec in specs}
    return {"position": np.zeros((1,), np.int32), **states}


def stateful_config(specs, inputs, probe, *, sequence_name, sequence_axis, output_count):
    axes = {sequence_name: [sequence_axis],
            **{spec.name: [spec.capacity_axis] for spec in specs if spec.capacity_axis is not None}}
    return ConversionConfig(
        optimize=True, capture_shapeless=True, dynamic_axes=axes, dynamic_probe_inputs=probe,
        signature=CaptureSignature(
            input_order=tuple(inputs), output_count=output_count + len(specs),
            states=tuple(StateBinding(spec, output_count + i) for i, spec in enumerate(specs)),
        ),
    )


def _caches(states, position, layers):
    state = LayeredKVCacheState(states["keyCache"], states["valueCache"])
    return [LayeredKVCache(state, layer_idx=i, offset=position[0]) for i in range(layers)]


def _cache_outputs(caches):
    import mlx.core as mx

    return mx.stack([c.keys for c in caches]), mx.stack([c.values for c in caches])


def backbone_specs(flow_lm, *, layer_index=None):
    t = flow_lm.transformer
    return kv_specs(t.num_layers if layer_index is None else 1, t.num_heads, t.head_dim)


def backbone_forward(flow_lm, embeddings, position, *, layer_index=None, **states):
    """Source transformer with tensor RoPE offsets and capacity-aware masking.

    The converter exports one layer per asset: the beta GPU runtime corrupts
    cache tails in the full six-layer graph at some dynamic capacities.
    Only the final layer applies the output norm and EOS head.
    """
    import mlx.core as mx
    import mlx.nn as nn

    layers = flow_lm.transformer.layers
    final = layer_index is None or layer_index == len(layers) - 1
    if layer_index is not None:
        if not 0 <= layer_index < len(layers):
            raise ValueError("Backbone layer index is out of range.")
        layers = [layers[layer_index]]
    caches = _caches(states, position, len(layers))
    x = embeddings
    for layer, cache in zip(layers, caches, strict=True):
        attn = layer.self_attn
        b, t, _ = x.shape
        qkv = attn.in_proj(layer.norm1(x)).reshape(b, t, 3, attn.num_heads, attn.head_dim)
        q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
        if attn.rope is not None:
            q = mx.fast.rope(q, dims=attn.head_dim, traditional=True,
                             base=attn.rope.max_period, scale=1.0, offset=position[0])
            k = mx.fast.rope(k, dims=attn.head_dim, traditional=True,
                             base=attn.rope.max_period, scale=1.0, offset=position[0])
        k, v = cache.update_and_fetch(k, v)
        allowed = (mx.arange(t) + position[0])[:, None] >= mx.arange(k.shape[2])[None, :]
        mask = mx.where(allowed, 0.0, -1e9)[None, None].astype(x.dtype)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
        y = attn.out_proj(y.transpose(0, 2, 1, 3).reshape(b, t, attn.embed_dim))
        x = x + layer._apply_scale(y, layer.layer_scale_1)
        y = layer.linear2(nn.gelu(layer.linear1(layer.norm2(x))))
        x = x + layer._apply_scale(y, layer.layer_scale_2)
    if final:
        x = flow_lm.out_norm(x) if flow_lm.out_norm is not None else x
        return (x, flow_lm.out_eos(x), *_cache_outputs(caches))
    return (x, *_cache_outputs(caches))


def sample_forward(flow_lm, hidden, noise, *, steps=1):
    """Host supplies temperature-scaled noise; return latent and next LM input."""
    from mlx_audio.tts.models.pocket_tts.flow_lm import lsd_decode

    if steps <= 0:
        raise ValueError("Flow decoding steps must be positive.")
    with _uncompiled_silu(flow_lm.flow_net):
        latent = lsd_decode(lambda s, t, x: flow_lm.flow_net(hidden, s, t, x), noise, steps)
    return latent, flow_lm.input_linear(latent[:, None, :])


@contextmanager
def _uncompiled_silu(module):
    """MLX 0.32.2 cannot export repeated compiled SiLU calls; preserve the math."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    class SiLU(nn.Module):
        def __call__(self, x):
            return x * mx.sigmoid(x)

    originals = [(name, child) for name, child in module.named_modules() if isinstance(child, nn.SiLU)]
    try:
        module.update_modules(tree_unflatten([(name, SiLU()) for name, _ in originals]))
        yield
    finally:
        module.update_modules(tree_unflatten(originals))


class StreamingDecoder:
    """Explicit KV, convolution-history, and overlap-add state; host position.

    Decoder convolutions have stride one; transposed convolutions emit complete
    frame-aligned chunks. No pending partial input frames or residuals are needed.
    State is functional during capture and bound to CoreAI mutable buffers later.
    """

    def __init__(self, model):
        from mlx_audio.codec.models.mimi.modules.conv import StreamableConv1d, StreamableConvTranspose1d

        self.model = model
        mimi = model.mimi
        cfg = mimi.decoder_transformer.transformer.cfg
        self.specs = kv_specs(cfg.num_layers, cfg.num_heads, cfg.head_dim)
        self.convolutions = []
        modules = ([] if mimi.upsample is None else list(mimi.upsample.named_modules()))
        modules += list(mimi.decoder.named_modules())
        for _, module in modules:
            if isinstance(module, StreamableConv1d):
                conv = module.conv.conv
                if conv._stride != 1 or not module._causal or module._pad_mode != "constant":
                    raise ValueError("Streaming decoder requires causal, constant-padded stride-one convolutions.")
                length = (module._ksize - 1) * conv._dilation
                channels = conv.weight.shape[-1] * conv._groups
                attr = "_prev_xs"
            elif isinstance(module, StreamableConvTranspose1d):
                if not module._causal:
                    raise ValueError("Streaming decoder requires causal transposed convolutions.")
                conv = module.convtr.convtr
                length = module._ksize - conv._stride
                channels = conv._out_channels
                attr = "_prev_ys"
            else:
                continue
            if length > 0:
                spec = StateSpec(f"convState{len(self.convolutions)}", (1, channels, length), "fp32")
                self.specs.append(spec)
                self.convolutions.append((module, attr, spec))

    def _conv_step(self, module, x, states, updated):
        import mlx.core as mx

        binding = next(((attr, spec) for m, attr, spec in self.convolutions if m is module), None)
        if binding is None:
            return module(x)
        attr, spec = binding
        length = spec.shape[-1]
        if attr == "_prev_xs":
            x = mx.concatenate([states[spec.name], x], axis=-1)
            updated[spec.name] = x[..., -length:]
            return module.conv(x)
        y = module.convtr(x)
        y = mx.concatenate([y[..., :length] + states[spec.name], y[..., length:]], axis=-1)
        tail = y[..., -length:]
        # Bias-free overlap makes zero initial state correct on the first call.
        if module.convtr.convtr.bias is not None:
            tail = tail - module.convtr.convtr.bias[None, :, None]
        updated[spec.name] = tail
        return y[..., :-length]

    def __call__(self, latent, position, **states):
        import mlx.core as mx
        import mlx.nn as nn

        mimi = self.model.mimi
        caches = _caches(states, position, mimi.decoder_transformer.transformer.cfg.num_layers)
        updated = {}
        conv = lambda module, x: self._conv_step(module, x, states, updated)
        x = latent * self.model.flow_lm.emb_std + self.model.flow_lm.emb_mean
        x = mimi.quantizer(x.transpose(0, 2, 1))
        if mimi.upsample is not None:
            x = conv(mimi.upsample.convtr, x)
        delta = (mx.arange(x.shape[-1]) + position[0])[:, None] - mx.arange(caches[0].keys.shape[2])[None, :]
        allowed = delta >= 0
        context = mimi.decoder_transformer.transformer.cfg.context
        if context:
            allowed = allowed & (delta < context)
        mask = mx.where(allowed, 0.0, -1e9)[None, None].astype(x.dtype)
        x = mimi.decoder_transformer(x, cache=caches, mask=mask)[0]
        x = conv(mimi.decoder.init_conv1d, x)
        for layer in mimi.decoder.layers:
            x = conv(layer.upsample, nn.elu(x))
            for residual in layer.residuals:
                skip = x if residual.shortcut is None else conv(residual.shortcut, x)
                for module in residual.block:
                    x = conv(module, nn.elu(x))
                x = x + skip
        audio = conv(mimi.decoder.final_conv1d, nn.elu(x))
        return (audio, *_cache_outputs(caches),
                *(updated[spec.name] for _, _, spec in self.convolutions))
