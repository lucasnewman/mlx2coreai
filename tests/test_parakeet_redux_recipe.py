"""Redux recipe contracts, greedy control flow, and checkpoint-free CoreAI parity."""
import asyncio
import importlib
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from recipes import parakeet_redux
from recipes.parakeet_redux.build import from_model
from recipes.parakeet_redux.runtime import encoder_inputs


METADATA = {"mel_bins": 8, "encoder_dim": 128, "subsampling_factor": 2,
            "decoder_layers": 2, "decoder_hidden": 16, "blank_id": 3,
            "durations": [0, 1, 2], "max_symbols": 3, "frame_seconds": 0.02}


class FakeSession:
    def __init__(self, tmp_path, decisions):
        path = tmp_path / "vocabulary.json"
        path.write_text(json.dumps(["▁hello", "world", "<unk>"]))
        self.bundle = SimpleNamespace(manifest={"recipe": "parakeet_redux"}, metadata=METADATA,
                                      resource=lambda name: path)
        self.decisions = iter(decisions)
        self.calls = []

    async def run(self, name, inputs, *, readback):
        self.calls.append((name, inputs))
        assert readback
        if name == "encoder":
            count = (inputs["mel"].shape[1] + 1) // 2
            return {"features": np.arange(count, dtype=np.float32)[None, :, None].repeat(128, axis=2),
                    "lengths": (inputs["lengths"] + 1) // 2}
        token, duration = next(self.decisions)
        return {"token_logits": np.eye(4, dtype=np.float32)[token],
                "duration_logits": np.eye(3, dtype=np.float32)[duration],
                "hidden": inputs["hidden"] + 1, "cell": inputs["cell"] + 2}


def test_greedy_blank_state_zero_duration_and_reset(tmp_path):
    async def check():
        # Zero-duration nonblank retries the same frame; blank/zero advances.
        session = FakeSession(tmp_path, [(0, 0), (3, 0), (2, 1), (1, 1)] * 2)
        for _ in range(2):
            report = {}
            result = await parakeet_redux.run(session, parakeet_redux.Request(np.zeros((7, 8))), report=report)
            assert result["text"] == "helloworld" and result["completed"]
            assert [t["id"] for t in result["tokens"]] == [0, 1]
            assert result["tokens"][1]["start"] == 0.04
            assert report == {"encoder_frames": 3, "decoder_steps": 4, "completed": True, "budget_exhausted": False}
            calls = session.calls[-4:]
            assert [c[1]["feature"][0, 0, 0] for c in calls] == [0, 0, 1, 2]
            assert [c[1]["current_token"][0, 0] for c in calls] == [3, 0, 0, 2]
            assert [c[1]["hidden"][0, 0, 0] for c in calls] == [0, 1, 1, 2]
    asyncio.run(check())


def test_bounded_zero_duration_decode(tmp_path):
    session = FakeSession(tmp_path, [(0, 0)] * 9)
    report = {}
    result = asyncio.run(parakeet_redux.run(session, parakeet_redux.Request(np.zeros((7, 8))), report=report))
    assert len(result["tokens"]) == 9 and not result["completed"]
    assert report["budget_exhausted"]


@pytest.mark.parametrize("mel", [np.zeros((1, 8)), np.zeros((2, 7, 8)), np.zeros((7, 9)),
                                    np.zeros((7, 8), np.int32), np.full((7, 8), np.nan)])
def test_invalid_mel(tmp_path, mel):
    with pytest.raises(ValueError, match="Mel"):
        asyncio.run(parakeet_redux.run(FakeSession(tmp_path, []), parakeet_redux.Request(mel)))


@pytest.mark.parametrize("length", [0, 1, 8, 2.5, True])
def test_invalid_length(tmp_path, length):
    with pytest.raises(ValueError, match="Length"):
        asyncio.run(parakeet_redux.run(FakeSession(tmp_path, []), parakeet_redux.Request(np.zeros((7, 8)), length)))


def test_runtime_imports_without_mlx():
    result = subprocess.run([sys.executable, "-c", "import recipes.parakeet_redux, sys; "
                             "assert not any(k == 'mlx' or k.startswith(('mlx.', 'mlx_audio')) for k in sys.modules)"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_build_forwards_source_revision_frames(monkeypatch):
    module = importlib.import_module("recipes.parakeet_redux.build")
    model = object()
    monkeypatch.setattr(module, "load_source", lambda source, **kw: model)
    monkeypatch.setattr(module, "from_model", lambda *a, **kw: (a, kw))
    assert parakeet_redux.build("local", revision="abc", frames=(17, 33)) == (
        (model, "local"), {"revision": "abc", "frames": (17, 33), "weight_format": "fp32"})


def test_validation_requires_full_precision_mlx(monkeypatch):
    from recipes.parakeet_redux.validation import validate

    monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    with pytest.raises(RuntimeError, match="MLX_ENABLE_TF32=0"):
        asyncio.run(validate(SimpleNamespace()))


def test_cli_convert(monkeypatch, tmp_path):
    from recipes.parakeet_redux import cli

    calls = []
    monkeypatch.setattr(cli, "build", lambda *a, **kw: calls.append((a, kw)) or "plan")
    monkeypatch.setattr(cli, "export", lambda *a: calls.append(a))
    cli.main(["convert", "local", "--revision", "abc", "--frames", "17", "33",
              "--weight-format", "uint2", "--output", str(tmp_path)])
    assert calls == [(("local",), {"revision": "abc", "frames": (17, 33), "weight_format": "uint2"}), ("plan", tmp_path)]


def test_wrong_bundle(tmp_path):
    session = FakeSession(tmp_path, [])
    session.bundle.manifest["recipe"] = "smart_turn"
    with pytest.raises(ValueError, match="Redux recipe bundle"):
        asyncio.run(parakeet_redux.run(session, parakeet_redux.Request(np.zeros((7, 8)))))


def test_invalid_decoder_output(tmp_path):
    class Session(FakeSession):
        async def run(self, name, inputs, **kwargs):
            output = await super().run(name, inputs, **kwargs)
            if name == "decoder_step":
                output["hidden"].fill(np.nan)
            return output

    with pytest.raises(RuntimeError, match="decoder produced"):
        asyncio.run(parakeet_redux.run(Session(tmp_path, [(0, 1)]), parakeet_redux.Request(np.zeros((7, 8)))))


def test_compression_rejects_unmatched_and_inexact_weights():
    from mlx2coreai.ir import Graph, Node
    from mlx2coreai import PackedLinearWeights

    codes = np.arange(128, dtype=np.uint32) % 3
    words = np.sum(codes.reshape(1, 8, 16) << (2 * np.arange(16, dtype=np.uint32)), axis=-1, dtype=np.uint32)
    module = SimpleNamespace(weight=words, scales=np.array([[0.25]], np.float32),
                             biases=np.array([[-0.25]], np.float32), bits=2, group_size=128)
    dense = codes.astype(np.float32)[None] * 0.25 - 0.25
    packed = PackedLinearWeights(require_all=True)
    with pytest.raises(ValueError, match="exactly reconstruct"):
        packed.add("weight", module, dense + 1)
    packed.add("weight", module, dense)
    graph = Graph([], [Node("constant", (), "weight", {"value": dense + 1})], ["weight"])
    with pytest.raises(ValueError, match="missed packed"):
        packed(graph)
    graph = Graph([], [Node("constant", (), "weight", {"value": dense})], ["weight"])
    transformed = packed(graph)
    transformed.validate()
    assert transformed.nodes[-1].op == "blockwise_shift_scale"
    assert transformed.nodes[0].attrs["dtype"] == "uint2"
    assert graph.nodes[0].op == "constant"


def test_comparison_tensor_errors():
    from recipes.parakeet_redux.comparison import tensor_error

    expected = np.array([3, 4], np.float32)
    row = tensor_error(expected + np.array([0, 1]), expected, atol=1e-4, rtol=1e-4)
    assert row["valid"] and not row["allclose"]
    assert row["max_abs_error"] == 1 and row["relative_l2"] == 0.2
    assert not tensor_error(np.zeros(3), expected, atol=1e-4, rtol=1e-4)["valid"]
    assert not tensor_error(expected * np.nan, expected, atol=1e-4, rtol=1e-4)["valid"]


def tiny_model(factor=2):
    pytest.importorskip("mlx_audio.stt.models.parakeet.redux")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_audio.stt.models.parakeet.parakeet import ParakeetTDT, ParakeetTDTArgs
    from mlx_audio.utils import from_dict

    mx.random.seed(54)
    model = ParakeetTDT(from_dict(ParakeetTDTArgs, {
        "preprocessor": {"sample_rate": 16000, "normalize": "per_feature", "window_size": 0.025,
                         "window_stride": 0.01, "window": "hann", "features": 8,
                         "n_fft": 512, "dither": 0, "normalize_valid_frames": True},
        "encoder": {"feat_in": 8, "n_layers": 1, "d_model": 128, "n_heads": 2,
                    "ff_expansion_factor": 2, "subsampling_factor": factor, "self_attention_model": "rel_pos",
                    "subsampling": "dw_striding", "conv_kernel_size": 3, "subsampling_conv_channels": 4,
                    "pos_emb_max_len": 64, "use_bias": False, "mask_padding": True},
        "decoder": {"blank_as_pad": True, "vocab_size": 3,
                    "prednet": {"pred_hidden": 16, "pred_rnn_layers": 2}},
        "joint": {"num_classes": 3, "vocabulary": ["▁hello", "world", "<unk>"], "num_extra_outputs": 3,
                  "jointnet": {"joint_hidden": 16, "activation": "relu", "encoder_hidden": 128, "pred_hidden": 16}},
        "decoding": {"model_type": "tdt", "durations": [0, 1, 2], "greedy": {"max_symbols": 3}},
    }))
    # Exercise the packed 2-bit conversion, without a download.
    nn.quantize(model, group_size=128, bits=2,
                class_predicate=lambda path, module: path.startswith("encoder.layers.") and isinstance(module, nn.Linear))
    # Use Redux's actual ternary codes and affine biases, rather than a new
    # lossy quantization of the random model's weights.
    for _, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            words = np.asarray(module.weight)
            shifts = 2 * np.arange(16, dtype=np.uint32)
            codes = np.minimum((words[..., None] >> shifts) & 3, 2)
            module.weight = mx.array(np.sum(codes << shifts, axis=-1, dtype=np.uint32))
            module.biases = -module.scales
    model.eval()
    return model


def test_reject_ambiguous_capture_shapes():
    model = tiny_model(8)
    with pytest.raises(ValueError, match="remainder 1"):
        from_model(model, "tiny-redux", frames=(33, 49))


@pytest.mark.parametrize("weight_format", ["uint4", "uint8"])
def test_tiny_generic_quantization_exports(tmp_path, weight_format):
    from coreai.runtime import SpecializationOptions
    from mlx2coreai.recipe import export

    plan = from_model(tiny_model(8), "tiny-redux", weight_format=weight_format)
    assert all(c.config.weight_quantization.bits == int(weight_format[-1]) for c in plan.components.values())
    report = {}
    bundle = export(plan, tmp_path / weight_format, quantization_report=report)
    assert report["encoder"]["quantized_weights"] >= 9
    assert report["decoder_step"]["quantized_weights"] > 0

    async def check():
        async with bundle.session(specialization_options=SpecializationOptions.cpu_only()) as session:
            inputs = encoder_inputs(np.zeros((1, 17, 8), np.float32), 16, bundle.metadata)
            encoded = await session.run("encoder", inputs, readback=True)
            assert encoded["features"].shape == (1, 3, 128)
            output = await session.run("decoder_step", {
                "feature": encoded["features"][:, :1], "current_token": np.array([[3]], np.int32),
                "hidden": np.zeros((2, 1, 16), np.float32), "cell": np.zeros((2, 1, 16), np.float32),
            }, readback=True)
            assert output["token_logits"].shape == (4,)
            assert all(np.isfinite(value).all() for value in output.values())

    asyncio.run(check())


@pytest.mark.parametrize("factor,frames", [(2, (8, 13)), (8, (32, 49))])
@pytest.mark.parametrize("weight_format", ["fp32", "uint2"])
def test_tiny_native_dynamic_parity(tmp_path, factor, frames, weight_format):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions
    from mlx2coreai.recipe import export, Bundle
    from recipes.parakeet_redux.adapter import decode_step

    model = tiny_model(factor)
    rng = np.random.default_rng(55)
    packed_mel = rng.normal(size=(1, 17, 8)).astype(np.float32)
    packed_features = np.asarray(model.encoder(mx.array(packed_mel))[0])
    plan = from_model(model, "tiny-redux", frames=frames, weight_format=weight_format)
    assert plan.metadata["quantized_modules_decompressed"] == 9
    np.testing.assert_allclose(np.asarray(model.encoder(mx.array(packed_mel))[0]), packed_features, atol=1e-5, rtol=1e-5)
    bundle = export(plan, tmp_path / "bundle", save_graphs=True)
    config = json.loads((bundle.path / "config.json").read_text())
    assert not (bundle.path / "manifest.json").exists()
    assert sorted(path.name for path in bundle.path.glob("*.aimodel")) == ["model.aimodel"]
    assert config["model"] == "model.aimodel"
    assert config["metadata"] == plan.runtime_metadata
    assert not {"source", "revision", "weight_format", "workarounds"} & config["metadata"].keys()
    assert {entry["entrypoint"] for entry in config["components"].values()} == {"encode", "decode"}
    assert all(set(entry) == {"entrypoint", "outputs"} for entry in config["components"].values())
    graph = json.loads((bundle.path / "encoder_graph.json").read_text())
    assert sum(node["op"] == "blockwise_shift_scale" for node in graph["nodes"]) == (9 if weight_format == "uint2" else 0)
    bundle = Bundle.open(bundle.path)

    async def check():
        async with bundle.session(specialization_options=SpecializationOptions.cpu_only()) as session:
            assert session._sessions["decoder_step"]._owner is session._sessions["encoder"]
            for n, length in ((frames[0], frames[0] - 1), (frames[1], frames[1] - 1),
                              (101, 100), (100, 91), (17, 13), (10, 9), (9, 8), (3, 2)):
                mel = rng.normal(size=(1, n, 8)).astype(np.float32)
                inputs = encoder_inputs(mel, length, bundle.metadata)
                expected, lengths = model.encoder(mx.array(mel), mx.array([length], mx.int32))
                actual = await session.run("encoder", inputs, readback=True)
                np.testing.assert_allclose(actual["features"], np.asarray(expected), atol=1e-4, rtol=1e-4)
                np.testing.assert_array_equal(actual["lengths"], np.asarray(lengths))
            hidden = rng.normal(size=(2, 1, 16)).astype(np.float32)
            cell = rng.normal(size=(2, 1, 16)).astype(np.float32)
            for token in (3, 0, 1):
                inputs = {"feature": rng.normal(size=(1, 1, 128)).astype(np.float32),
                          "current_token": np.array([[token]], np.int32), "hidden": hidden, "cell": cell}
                reference = decode_step(model, **{k: mx.array(v) for k, v in inputs.items()})
                actual = await session.run("decoder_step", inputs, readback=True)
                for name, expected in zip(("token_logits", "duration_logits", "hidden", "cell"), reference):
                    np.testing.assert_allclose(actual[name], np.asarray(expected), atol=1e-5, rtol=1e-5)
                hidden, cell = actual["hidden"], actual["cell"]
    asyncio.run(check())

    mel_path = tmp_path / "mel.npy"
    np.save(mel_path, packed_mel)
    result_path = tmp_path / "transcript.json"
    result = subprocess.run([sys.executable, "-c", "import runpy, sys; "
        "runpy.run_module('recipes.parakeet_redux', run_name='__main__'); "
        "assert not any(k == 'mlx' or k.startswith(('mlx.', 'mlx_audio')) for k in sys.modules)",
        "run", str(bundle.path), "--mel", str(mel_path), "--json-output", str(result_path)],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert isinstance(json.loads(result_path.read_text())["text"], str)
    if weight_format == "uint2" and factor == 8:
        reference = export(from_model(tiny_model(8), "tiny-redux", frames=frames), tmp_path / "fp32")
        def size(path):
            return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
        assert size(bundle.path / "model.aimodel") < size(reference.path / "model.aimodel")
        comparison_path = tmp_path / "comparison.json"
        result = subprocess.run([sys.executable, "-c", "import runpy, sys; "
            "runpy.run_module('recipes.parakeet_redux', run_name='__main__'); "
            "assert not any(k == 'mlx' or k.startswith(('mlx.', 'mlx_audio')) for k in sys.modules)",
            "compare", str(reference.path), str(bundle.path), "--mel", str(mel_path),
            "--output", str(comparison_path)], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        report = json.loads(comparison_path.read_text())
        assert report["passed"] and report["asset_size_reduction"] > 0
        assert report["candidate_asset_bytes"] == size(bundle.path / "model.aimodel")
        assert report["results"][0]["matching_timestamps"]
