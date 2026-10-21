"""Pocket TTS and Phonon source loading and component contracts."""
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
                    allow_patterns=["config.json", "*.model", "tokenizer*.json", "*.safetensors"]))
    config = json.loads((path / "config.json").read_text())
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("Pocket TTS conversion requires floating-point source weights.")
    model = Model({**config, "model_path": str(path.resolve())})
    # Voice prompts are also safetensors, but are not model parameters.
    weights = model.sanitize(mx.load(str(path / (config.get("weights_name") or "model.safetensors"))))
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


def _voice_prompt(model, path, voice):
    """Resolve bundled voices and project Gradium's speaker latents once."""
    import mlx.core as mx
    from mlx_audio.tts.models.pocket_tts.utils import download_if_necessary, load_predefined_voice

    voice = None if voice is None else str(voice)
    local_voice = path / f"{voice or 'alba'}.safetensors"
    if local_voice.is_file() and (voice is not None or not (path / "default-voice.safetensors").is_file()):
        selected = local_voice
    else:
        selected = model._resolve_audio_prompt(voice, None, False)
    if str(selected).endswith(".safetensors") or ".safetensors@" in str(selected):
        weights = mx.load(str(download_if_necessary(str(selected))))
        if "speaker_wavs" in weights:
            latent = weights["speaker_wavs"].astype(mx.float32)
            if latent.ndim == 2:
                latent = latent[None]
            prompt = latent.transpose(0, 2, 1) @ model.speaker_proj_weight.T
        else:
            key = "emb" if "emb" in weights else "audio_prompt"
            prompt = weights[key].astype(mx.float32)
    else:
        prompt = load_predefined_voice(str(selected)).astype(mx.float32)
    if prompt.ndim == 2:
        prompt = prompt[None]
    if prompt.ndim != 3 or prompt.shape[0] != 1 or prompt.shape[-1] != model.flow_lm.dim:
        raise ValueError(f"Voice embedding must have shape [1, T, {model.flow_lm.dim}], got {prompt.shape}.")
    return prompt, str(selected)


def build(source="mlx-community/pocket-tts", *, revision=None, steps=None, voice=None):
    import mlx.core as mx
    from mlx_audio.tts.models.pocket_tts.utils import download_if_necessary
    from .adapter import input_embedding

    model, path = load_source(source, revision)
    flow = model.flow_lm
    steps = model.lsd_decode_steps if steps is None else steps
    prompt, voice = _voice_prompt(model, path, voice)
    tokenizer = model.config.flow_lm.lookup_table
    tokenizer_name = "tokenizer.json" if tokenizer.tokenizer == "huggingface" else "tokenizer.model"
    conditioning = BytesIO()
    np.savez(conditioning, voice=np.asarray(prompt.astype(mx.float32)),
             bos=np.asarray(input_embedding(flow, flow.bos_emb[None, None, :])))
    return Build("pocket_tts", components(model, steps=steps), resources={
        tokenizer_name: Path(download_if_necessary(tokenizer.tokenizer_path)),
        "conditioning.npz": conditioning.getvalue(),
    }, metadata={
        "source": str(path), "precision": "fp32", "sample_rate": model.sample_rate,
        "frame_rate": model.mimi.frame_rate, "latent_dim": flow.ldim, "dim": flow.dim, "flow_steps": steps,
        "decoder_steps_per_frame": int(model.mimi.encoder_frame_rate / model.mimi.frame_rate),
        "voice": voice, "streaming": True,
        "model_type": model.config.model_type, "tokenizer_type": tokenizer.tokenizer,
        "temperature": model.temp, "eos_threshold": model.eos_threshold, "noise_clamp": model.noise_clamp,
        "backbone_components": [f"backbone{i}" for i in range(flow.transformer.num_layers)],
        "workarounds": ["per-layer backbone assets: beta packed-cache corruption",
                        "FP32 source execution", "uncompiled SiLU during capture",
                        "sampler authoring optimization disabled"],
    })
