"""Pocket-specific source loading, component contracts, and capture workarounds."""
from functools import partial
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from mlx2coreai import ConversionConfig
from mlx2coreai.recipe import Build, Component
from .adapter import StreamingDecoder, backbone_forward, backbone_specs, initial_state, sample_forward, stateful_config


def load_source(source, revision=None):
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.models.pocket_tts.pocket_tts import Model

    path = Path(source)
    if not path.is_dir():
        path = Path(snapshot_download(source, revision=revision,
                    allow_patterns=["config.json", "model.safetensors", "*.model", "*.safetensors"]))
    model = Model(json.loads((path / "config.json").read_text()))
    # Voice prompts are also safetensors, but are not model parameters.
    weights = mx.load(str(path / "model.safetensors"))
    model.load_weights([(name, value.astype(mx.float32) if mx.issubdtype(value.dtype, mx.floating) else value)
                        for name, value in weights.items()], strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, path


def components(model, *, steps=1):
    """Also usable with tiny source models in checkpoint-free integration tests."""
    from mlx_audio.tts.models.pocket_tts.conditioners import TokenizedText

    if steps <= 0:
        raise ValueError("Flow steps must be positive.")
    flow = model.flow_lm
    rng = np.random.default_rng(101)
    random = lambda shape: rng.normal(0, 0.1, shape).astype(np.float32)
    result = {"conditioner": Component(
        lambda tokens: flow.conditioner(TokenizedText(tokens)), {"tokens": np.array([[1, 2, 3]], np.int32)},
        ("embeddings",), ConversionConfig(optimize=True, capture_shapeless=True, dynamic_axes={"tokens": [1]},
            dynamic_probe_inputs={"tokens": np.array([[1, 2, 3, 4, 5]], np.int32)}))}
    for index in range(flow.transformer.num_layers):
        specs = backbone_specs(flow, layer_index=index)
        def example(length, capacity):
            return {"embeddings": random((1, length, flow.dim)), **initial_state(specs, capacity)}
        inputs, probe = example(3, 32), example(5, 48)
        outputs = ("hidden", "eos") if index == flow.transformer.num_layers - 1 else ("hidden",)
        result[f"backbone{index}"] = Component(partial(backbone_forward, flow, layer_index=index), inputs, outputs,
            stateful_config(specs, inputs, probe, sequence_name="embeddings", sequence_axis=1, output_count=len(outputs)))
    result["sampler"] = Component(partial(sample_forward, flow, steps=steps),
        {"hidden": random((1, flow.dim)), "noise": random((1, flow.ldim))}, ("latent", "embedding"),
        ConversionConfig(optimize=False, capture_shapeless=True))
    decoder = StreamingDecoder(model)
    inputs = {"latent": random((1, 2, flow.ldim)), **initial_state(decoder.specs, 64)}
    probe = {"latent": random((1, 3, flow.ldim)), **initial_state(decoder.specs, 96)}
    result["decoder"] = Component(decoder, inputs, ("audio",), stateful_config(
        decoder.specs, inputs, probe, sequence_name="latent", sequence_axis=1, output_count=1))
    return result


def build(source="mlx-community/pocket-tts", *, revision=None, steps=1, voice="alba"):
    import mlx.core as mx
    from mlx_audio.tts.models.pocket_tts.utils import download_if_necessary, load_predefined_voice

    model, path = load_source(source, revision)
    flow = model.flow_lm
    voice_path = path / f"{voice}.safetensors"
    prompt = mx.load(str(voice_path))["audio_prompt"] if voice_path.exists() else load_predefined_voice(voice)
    conditioning = BytesIO()
    np.savez(conditioning, voice=np.asarray(prompt.astype(mx.float32)),
             bos=np.asarray(flow.input_linear(flow.bos_emb[None, None, :])))
    return Build("pocket_tts", components(model, steps=steps), resources={
        "tokenizer.model": Path(download_if_necessary(model.config.flow_lm.lookup_table.tokenizer_path)),
        "conditioning.npz": conditioning.getvalue(),
    }, metadata={
        "source": str(path), "precision": "fp32", "sample_rate": model.sample_rate,
        "frame_rate": model.mimi.frame_rate, "latent_dim": flow.ldim, "dim": flow.dim, "flow_steps": steps,
        "decoder_steps_per_frame": int(model.mimi.encoder_frame_rate / model.mimi.frame_rate),
        "voice": voice, "streaming": True,
        "backbone_components": [f"backbone{i}" for i in range(flow.transformer.num_layers)],
        "workarounds": ["per-layer backbone assets: beta packed-cache corruption",
                        "FP32 source execution", "uncompiled SiLU during capture",
                        "sampler authoring optimization disabled"],
    })
