"""Capture adapters for mlx-audio's Mimi codec (optional dependency)."""
from __future__ import annotations


class _FullSequenceCache:
    """Offline attention sees only this call's keys, with no allocation padding."""

    def __init__(self):
        self.offset = 0

    def update_and_fetch(self, keys, values):
        self.offset += keys.shape[2]
        return keys, values


def offline_forward(model, component, value):
    """Equivalent to encode/decode after reset, without tracing cache allocation."""
    import mlx.core as mx

    if component == "encode":
        frame_size = int(model.sample_rate / model.frame_rate)
        if value.shape[-1] <= 0 or value.shape[-1] % frame_size:
            raise ValueError(f"Offline Mimi encoding requires a positive multiple of {frame_size} samples.")
        value = model.encoder(value)
        caches = [_FullSequenceCache() for _ in model.encoder_cache]
        value = model.encoder_transformer(value, cache=caches)[0]
        # Express causal edge padding as concat rather than tracing MLX's
        # temporary allocation and slice updates. Aligned frames need no right pad.
        downsample = model.downsample.conv
        left = downsample._ksize - downsample.conv.conv._stride
        value = mx.concatenate([value[..., :1]] * left + [value], axis=-1)
        return model.quantizer.encode(downsample.conv(value)).astype(mx.int32)
    if component == "decode":
        value = model.upsample(model.quantizer.decode(value))
        caches = [_FullSequenceCache() for _ in model.decoder_cache]
        value = model.decoder_transformer(value, cache=caches)[0]
        return model.decoder(value)
    raise ValueError(f"Unknown Mimi component: {component}")
