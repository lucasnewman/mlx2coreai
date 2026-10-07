"""SmartTurn recipe contracts and native parity without downloaded weights."""
import asyncio
import importlib
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2coreai.recipe import Bundle, export
from recipes import smart_turn
from recipes.smart_turn.build import from_model


@pytest.mark.parametrize("tf32", [None, "1"])
def test_validation_requires_full_precision_mlx(monkeypatch, tf32):
    from recipes.smart_turn.validation import validate

    if tf32 is None:
        monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", tf32)
    with pytest.raises(RuntimeError, match="MLX_ENABLE_TF32=0"):
        asyncio.run(validate(SimpleNamespace()))


def test_tiny_smart_turn_dynamic_batch(tmp_path):
    pytest.importorskip("mlx_audio")
    import mlx.core as mx
    from coreai.runtime import ComputeUnitKind, SpecializationOptions
    from mlx_audio.vad.models.smart_turn.config import EncoderConfig, ModelConfig
    from mlx_audio.vad.models.smart_turn.smart_turn import Model

    mx.random.seed(41)
    model = Model(ModelConfig(dtype="float16", encoder_config=EncoderConfig(
        num_mel_bins=8, max_source_positions=16, d_model=16,
        encoder_attention_heads=2, encoder_layers=1, encoder_ffn_dim=32,
    )))
    model.set_dtype(mx.float16)
    plan = from_model(model, "tiny-smart-turn")
    assert model.dtype == mx.float32
    assert plan.metadata["feature_shape"] == [8, 32]
    assert plan.metadata["precision"] == "fp32"
    assert plan.components["main"].config.dynamic_axes == {"input_features": [0]}
    bundle = export(plan, tmp_path / "smart_turn")
    entry = bundle.manifest["components"]["main"]
    assert not entry["optimized"] and not entry["states"]
    assert list(entry["outputs"]) == ["logits", "probability"]
    bundle = Bundle.open(bundle.path)
    rng = np.random.default_rng(42)

    async def check(options):
        async with bundle.session(specialization_options=options) as session:
            for batch in (1, 2, 3, 1):
                features = rng.normal(size=(batch, 8, 32)).astype(np.float32)
                logits = model(mx.array(features), return_logits=True)
                expected = {"logits": np.asarray(logits), "probability": np.asarray(mx.sigmoid(logits))}
                actual = await smart_turn.run(session, smart_turn.Request(features))
                for name in expected:
                    np.testing.assert_allclose(actual[name], expected[name], atol=1e-5, rtol=1e-5)
                np.testing.assert_array_equal(actual["prediction"], expected["probability"] > bundle.metadata["threshold"])
                assert actual["prediction"].dtype == np.int32

    asyncio.run(check(SpecializationOptions.cpu_only()))
    if SpecializationOptions.is_supported():
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        asyncio.run(check(options))

    features_path = tmp_path / "features.npy"
    np.save(features_path, np.zeros((3, 8, 32), np.float32))
    result_path = tmp_path / "result.json"
    # A fresh CLI process must use only the bundle, never source model imports.
    result = subprocess.run([sys.executable, "-c",
        "import runpy, sys; runpy.run_module('recipes.smart_turn', run_name='__main__'); "
        "assert not any(k == 'mlx' or k.startswith(('mlx.', 'mlx_audio')) for k in sys.modules)",
        "run", str(bundle.path), "--features", str(features_path), "--threshold", "1",
        "--json-output", str(result_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads(result_path.read_text())
    assert report["prediction"] == [[0], [0], [0]]
    assert np.asarray(report["probability"]).shape == (3, 1)


class FakeSession:
    bundle = SimpleNamespace(manifest={"recipe": "smart_turn"},
                             metadata={"feature_shape": [8, 32], "threshold": 0.5})

    async def run(self, component, inputs, *, readback):
        assert component == "main" and readback
        assert inputs["input_features"].dtype == np.float32
        return {"logits": np.zeros((len(inputs["input_features"]), 1), np.float32),
                "probability": np.full((len(inputs["input_features"]), 1), 0.5, np.float32)}


@pytest.mark.parametrize("features", [
    np.zeros((8, 32), np.float32), np.zeros((0, 8, 32), np.float32),
    np.zeros((1, 8, 31), np.float32), np.zeros((1, 32, 8), np.float32),
    np.zeros((1, 8, 32), np.int32), np.zeros((1, 8, 32), np.complex64),
    np.full((1, 8, 32), np.nan), np.full((1, 8, 32), np.inf),
])
def test_reject_invalid_features(features):
    with pytest.raises(ValueError, match="Features"):
        asyncio.run(smart_turn.run(FakeSession(), smart_turn.Request(features)))


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf])
def test_reject_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="Threshold"):
        asyncio.run(smart_turn.run(FakeSession(), smart_turn.Request(np.zeros((1, 8, 32)), threshold)))


def test_threshold_is_strict_and_request_local():
    async def check():
        session = FakeSession()
        for threshold, expected in ((None, 0), (0.4, 1), (0.5, 0), (1, 0), (None, 0)):
            result = await smart_turn.run(session, smart_turn.Request(np.zeros((2, 8, 32)), threshold))
            np.testing.assert_array_equal(result["prediction"], np.full((2, 1), expected))
    asyncio.run(check())


def test_reject_wrong_recipe():
    session = SimpleNamespace(bundle=SimpleNamespace(manifest={"recipe": "mimi"}))
    with pytest.raises(ValueError, match="SmartTurn recipe"):
        asyncio.run(smart_turn.run(session, smart_turn.Request(np.zeros((1, 8, 32)))))


@pytest.mark.parametrize("probability", [np.nan, -0.1, 1.1])
def test_reject_invalid_runtime_outputs(probability):
    class Session(FakeSession):
        async def run(self, *args, **kwargs):
            return {"logits": np.zeros((1, 1)), "probability": np.full((1, 1), probability)}
    with pytest.raises(RuntimeError, match="SmartTurn produced"):
        asyncio.run(smart_turn.run(Session(), smart_turn.Request(np.zeros((1, 8, 32)))))


def test_build_forwards_source_and_revision(monkeypatch):
    module = importlib.import_module("recipes.smart_turn.build")
    calls = []
    source_model = object()
    monkeypatch.setattr(module, "load_source", lambda source, **kw: calls.append((source, kw)) or source_model)
    monkeypatch.setattr(module, "from_model", lambda model, source, **kw: (model, source, kw))
    assert smart_turn.build("checkpoint", revision="abc") == (source_model, "checkpoint", {"revision": "abc"})
    assert calls == [("checkpoint", {"revision": "abc"})]


def test_cli_convert_forwards_options(monkeypatch, tmp_path):
    from recipes.smart_turn import cli
    calls = []
    plan = object()
    monkeypatch.setattr(cli, "build", lambda source, **kw: calls.append((source, kw)) or plan)
    monkeypatch.setattr(cli, "export", lambda value, path: calls.append((value, path)))
    cli.main(["convert", "checkpoint", "--revision", "abc", "--output", str(tmp_path)])
    assert calls == [("checkpoint", {"revision": "abc"}), (plan, tmp_path)]


@pytest.mark.parametrize("flag,value", [("--atol", "0"), ("--rtol", "-1"), ("--atol", "nan")])
def test_cli_rejects_invalid_validation_limits(tmp_path, flag, value):
    from recipes.smart_turn.cli import main
    with pytest.raises(SystemExit) as error:
        main(["validate", "--output", str(tmp_path), flag, value])
    assert error.value.code == 2
