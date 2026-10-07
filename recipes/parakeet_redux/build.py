"""FP32 offline Parakeet Redux conversion with dynamic mel sequence length."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from mlx2coreai import ConversionConfig
from mlx2coreai.recipe import Build, Component
from .adapter import prepare_model, encode, decode_step
from .runtime import encoder_inputs

DEFAULT_SOURCE = "moondream/parakeet-redux"


def load_source(source, *, revision=None):
    try:
        from mlx_audio.stt.models.parakeet.redux import ParakeetRedux
    except ImportError as exc:
        raise ImportError("Parakeet Redux requires mlx-audio >= 0.5.6.") from exc
    from mlx_audio.stt import load

    options = {} if revision is None else {"revision": revision}
    model = load(source, strict=True, **options)
    if not isinstance(model, ParakeetRedux):
        raise ValueError("Expected a Parakeet Redux source model.")
    return model


def from_model(model, source, *, revision=None, frames=(32, 49)):
    if (len(frames) != 2 or any(type(n) is not int or n < 3 for n in frames)
            or frames[0] == frames[1]):
        raise ValueError("Capture and probe require two different mel frame counts >= 3.")
    cfg = model.encoder_config
    if (not getattr(cfg, "mask_padding", False) or cfg.self_attention_model != "rel_pos"
            or cfg.subsampling != "dw_striding" or cfg.subsampling_factor not in (2, 4, 8)):
        raise ValueError("Expected the Redux masked relative-position Conformer encoder.")
    factor = int(cfg.subsampling_factor)
    # Distinguish ceil-divided subsampler dimensions from 2*S-1 PE dimensions.
    # With both captures at 8*k+1, for example, ceil(T/4) aliases 2*S-1;
    # two-point probing would then truncate an unseen 8*k+5 sequence.
    if {n % factor for n in frames} != {0, 1}:
        raise ValueError("Capture/probe need one frame count divisible by subsampling_factor and one with remainder 1.")
    if (frames[0] + factor - 1) // factor == (frames[1] + factor - 1) // factor:
        raise ValueError("Capture and probe must have different subsampled frame counts.")
    quantized_modules = prepare_model(model)
    rnn = model.decoder.prediction["dec_rnn"]
    metadata = {
        "source": str(Path(source).resolve()) if Path(source).is_dir() else str(source),
        "revision": revision, "precision": "fp32", "mel_bins": int(cfg.feat_in),
        "encoder_dim": int(cfg.d_model), "subsampling_factor": factor,
        "decoder_layers": int(rnn.num_layers), "decoder_hidden": int(rnn.hidden_size),
        "blank_id": int(model.blank_id), "durations": list(model.durations),
        "max_symbols": int(model.max_symbols), "processor": asdict(model.preprocessor_config),
        "frame_seconds": factor * model.preprocessor_config.hop_length / model.preprocessor_config.sample_rate,
        "dynamic_frames": True, "streaming": False,
        "quantized_modules_decompressed": quantized_modules,
        "preprocessing": "mlx-audio log_mel_spectrogram outside the CoreAI assets",
        "workarounds": ["ternary weights losslessly decompressed to FP32 before capture",
                        "relative positional embeddings supplied by the recipe runtime",
                        "authoring optimization disabled"],
    }
    if (metadata["blank_id"] != len(model.vocabulary) or not model.durations
            or any(type(d) is not int or d < 0 for d in model.durations)
            or metadata["max_symbols"] <= 0):
        raise ValueError("Invalid Redux vocabulary or TDT decoding configuration.")
    rng = np.random.default_rng(53)

    def example(n):
        mel = rng.normal(size=(1, n, cfg.feat_in)).astype(np.float32)
        mel[:, -1] = 0
        return encoder_inputs(mel, n - 1, metadata)

    components = {
        "encoder": Component(
            lambda mel, lengths, pos_emb, positions: encode(model, mel, lengths, pos_emb, positions),
            example(frames[0]), ("features", "lengths"),
            ConversionConfig(optimize=False, capture_shapeless=True,
                             dynamic_axes={"mel": [1], "pos_emb": [1], "positions": [1]},
                             dynamic_probe_inputs=example(frames[1])),
        ),
        "decoder_step": Component(
            lambda feature, current_token, hidden, cell: decode_step(model, feature, current_token, hidden, cell),
            {"feature": rng.normal(size=(1, 1, cfg.d_model)).astype(np.float32),
             "current_token": np.array([[model.blank_id]], np.int32),
             "hidden": np.zeros((rnn.num_layers, 1, rnn.hidden_size), np.float32),
             "cell": np.zeros((rnn.num_layers, 1, rnn.hidden_size), np.float32)},
            ("token_logits", "duration_logits", "hidden", "cell"),
            ConversionConfig(optimize=False),
        ),
    }
    return Build("parakeet_redux", components, metadata=metadata,
                 resources={"vocabulary.json": (json.dumps(model.vocabulary, ensure_ascii=False) + "\n").encode()})


def build(source=DEFAULT_SOURCE, *, revision=None, frames=(32, 49)):
    return from_model(load_source(source, revision=revision), source, revision=revision, frames=frames)
