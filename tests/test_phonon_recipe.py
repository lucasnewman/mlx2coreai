"""Phonon packaging and generation policy without downloaded checkpoints."""
import asyncio
from io import BytesIO
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx_audio")


def source_model(path, layers):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_audio.tts.models.pocket_tts.pocket_tts import Model
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel(
        {"[UNK]": 0, "Hello": 1, "world": 2, ".": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    config = {
        "model_type": "gradium_phonon_tts",
        "flow_lm": {"d_model": 16, "num_heads": 2, "num_layers": layers,
                    "dim_feedforward": 32, "max_period": 10000, "n_bins": 4,
                    "lut_dim": 16, "flow_dim": 16, "flow_depth": 2, "ldim": 4,
                    "time_rms_norm": "rms", "gelu": "erf"},
        "mimi": {"channels": 1, "sample_rate": 16, "frame_rate": 2,
                 "dimension": 8, "quantizer_dimension": 4, "quantizer_output_dimension": 8,
                 "n_filters": 2, "n_residual_layers": 1, "ratios": [2], "kernel_size": 3,
                 "last_kernel_size": 3, "residual_kernel_size": 3, "dilation_base": 2,
                 "compress": 2, "transformer_d_model": 8, "transformer_num_heads": 2,
                 "transformer_num_layers": 1, "transformer_layer_scale": 0.01,
                 "transformer_context": 16, "transformer_dim_feedforward": 16,
                 "encoder_enabled": False, "gelu": "erf"},
        "conditioners": [
            {"name": "num_speakers", "type": "lut", "lut": {"n_bins": 31, "dim": 4}},
            {"name": "duration_delta", "type": "continuous",
             "continuous": {"dim": 4, "scale_factor": 1000, "max_period": 10000}},
        ],
        "fuser": {"sum": ["num_speakers", "duration_delta"]},
        "temp": 0.8, "eos_threshold": 4.0, "noise_clamp": 0.1, "lsd_decode_steps": 2,
    }
    mx.random.seed(191 + layers)
    model = Model({**config, "model_path": str(path)})
    model.speaker_proj_weight = mx.arange(16 * 8, dtype=mx.float32).reshape(16, 8) * 0.01
    weights = {}
    for key, value in tree_flatten(model.parameters()):
        key = key.replace("flow_lm.conditioner.",
                          "flow_lm.condition_provider.conditioners.transcript_in_segment.")
        if value.ndim == 3 and key.endswith(".weight"):
            value = value.transpose(2, 0, 1) if ".convtr." in key and not key.startswith("mimi.upsample.") else value.transpose(0, 2, 1)
        weights[key] = value.astype(mx.bfloat16)
    weights["flow_lm.condition_provider.conditioners.duration_delta.learnt_padding"] = mx.zeros((1, 1, 16))
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    mx.save_safetensors(str(path / "default-voice.safetensors"),
                        {"speaker_wavs": mx.full((1, 8, 3), 0.5)})
    (path / "config.json").write_text(json.dumps(config))


@pytest.mark.parametrize("layers", [6, 9])
def test_phonon_build_packages_model_defaults_and_projected_voice(tmp_path, layers):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from recipes.pocket_tts.build import build, load_source

    source_model(tmp_path, layers)
    model, _ = load_source(tmp_path)
    assert model.mimi.encoder is None
    assert all(value.dtype == mx.float32 for _, value in tree_flatten(model.parameters()))
    plan = build(tmp_path)
    assert len(plan.metadata["backbone_components"]) == layers
    assert plan.metadata["tokenizer_type"] == "huggingface"
    assert (plan.metadata["temperature"], plan.metadata["eos_threshold"],
            plan.metadata["flow_steps"], plan.metadata["noise_clamp"]) == (0.8, 4.0, 2, 0.1)
    assert "tokenizer.json" in plan.resources
    with np.load(BytesIO(plan.resources["conditioning.npz"])) as conditioning:
        np.testing.assert_allclose(conditioning["voice"],
            np.asarray(mx.full((1, 3, 8), 0.5) @ model.speaker_proj_weight.T), atol=1e-6)
        expected = model.flow_lm.input_linear(model.flow_lm.bos_emb[None, None])
        expected += model.flow_lm.num_speakers(mx.array([1]))
        expected += model.flow_lm.duration_delta(mx.array([0.0]))
        np.testing.assert_allclose(conditioning["bos"], np.asarray(expected), atol=1e-6)


def test_phonon_default_voice_rejects_unknown_name(tmp_path):
    from recipes.pocket_tts.build import build

    source_model(tmp_path, 6)
    with pytest.raises(ValueError, match="Unknown voice"):
        build(tmp_path, voice="not-a-voice")


@pytest.mark.parametrize("override", [False, True])
def test_phonon_runtime_uses_model_sampling_settings(tmp_path, override):
    from recipes.pocket_tts import Request, run
    from recipes.pocket_tts.runtime import tokenize

    source_model(tmp_path, 6)
    np.savez(tmp_path / "conditioning.npz", voice=np.zeros((1, 2, 16), np.float32),
             bos=np.zeros((1, 1, 16), np.float32))
    metadata = {"backbone_components": ["backbone0"], "tokenizer_type": "huggingface",
                "temperature": 0.8, "eos_threshold": 4.0, "noise_clamp": 0.1,
                "latent_dim": 4, "decoder_steps_per_frame": 1, "sample_rate": 16}
    bundle = SimpleNamespace(metadata=metadata, manifest={"recipe": "pocket_tts"},
                             resource=lambda name: tmp_path / name, path=tmp_path)
    np.testing.assert_array_equal(tokenize(bundle, "Hello world."), [[1, 2, 3]])
    assert tokenize(bundle, "Hello world.").dtype == np.int32

    class Session:
        names = ("conditioner", "backbone0", "sampler", "decoder")

        def __init__(self):
            self.bundle = bundle
            self.noise = []

        def reset_state(self, capacities):
            pass

        async def run(self, name, inputs, **kwargs):
            if name == "conditioner":
                return {"embeddings": np.zeros((1, inputs["tokens"].shape[1], 16), np.float32)}
            if name == "backbone0":
                return {"hidden": inputs["embeddings"], "eos": np.zeros((1, 1, 1), np.float32)}
            if name == "sampler":
                self.noise.append(inputs["noise"])
                return {"latent": np.zeros((1, 4), np.float32), "embedding": np.zeros((1, 1, 16), np.float32)}
            return {"audio": np.zeros((1, 1, 8), np.float32)}

    async def check():
        session, report = Session(), {}
        options = {"temperature": 0.0, "eos_threshold": 10.0, "noise_clamp": 0.01} if override else {}
        request = Request(text="Hello world.", max_frames=3, frames_after_eos=0, **options)
        chunks = [chunk async for chunk in run(session, request, report=report)]
        assert len(chunks) == 3 and report["eos_step"] is None
        if override:
            assert report["temperature"] == 0 and report["eos_threshold"] == 10
            np.testing.assert_array_equal(np.stack(session.noise), 0)
        else:
            assert report["temperature"] == 0.8 and report["eos_threshold"] == 4
            assert np.max(np.abs(session.noise)) <= 0.1

    asyncio.run(check())


def test_sampling_cli_accepts_float_overrides(monkeypatch, tmp_path):
    from recipes.pocket_tts import cli

    async def generate(args):
        assert args.temperature == 0.8
        assert args.eos_threshold == 4.0
        assert args.noise_clamp == 0.1

    monkeypatch.setattr(cli, "generate", generate)
    cli.run_main([str(tmp_path), "--temperature", "0.8", "--eos-threshold", "4.0", "--noise-clamp", "0.1"])
