"""Offline Mimi conversion contracts; no source-model imports at module import."""
from pathlib import Path

import numpy as np

from mlx2coreai import ConversionConfig
from mlx2coreai.recipe import Build, Component
from .adapter import offline_forward


def load_source(source):
    import mlx.core as mx
    from mlx_audio.codec.models.mimi.mimi import Mimi, mimi_202407

    model = Mimi(mimi_202407(32))
    model.load_pytorch_weights(str(source), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def from_model(model, source, *, frames=(2, 3)):
    if len(frames) != 2 or min(frames) <= 0 or frames[0] == frames[1]:
        raise ValueError("Capture and probe require two different positive frame counts.")
    size = int(model.sample_rate / model.frame_rate)
    # Infer dimensions from the model so the same recipe supports tiny test models.
    books = len(model.quantizer.rvq_first.vq.layers) + len(model.quantizer.rvq_rest.vq.layers)
    rng = np.random.default_rng(52)
    components = {}
    for name, input_name, output_name in (("encode", "audio", "codes"), ("decode", "codes", "audio")):
        def example(length):
            if name == "encode":
                value = rng.normal(0, 0.1, (1, 1, length * size)).astype(np.float32)
            else:
                value = rng.integers(0, model.cfg.quantizer_bins, (1, books, length), dtype=np.int32)
            return {input_name: value}
        # Bind both the component and input name; conversion happens after this loop.
        def forward(*, component=name, argument=input_name, **inputs):
            return offline_forward(model, component, inputs[argument])
        components[name] = Component(forward, example(frames[0]), (output_name,),
            ConversionConfig(optimize=False, capture_shapeless=True, dynamic_axes={input_name: [2]},
                             dynamic_probe_inputs=example(frames[1])))
    return Build("mimi", components, metadata={
        "source": str(source), "precision": "fp32", "sample_rate": model.sample_rate,
        "frame_rate": model.frame_rate, "samples_per_frame": size, "codebooks": books,
        "codebook_size": model.cfg.quantizer_bins, "streaming": False,
        "workarounds": ["offline cache allocation removed", "causal edge padding as concatenation",
                        "authoring optimization disabled"],
    })


def build(source, *, frames=(2, 3)):
    return from_model(load_source(source), Path(source).resolve(), frames=frames)
