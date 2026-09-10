"""Recipe contracts and lifecycle, plus checkpoint-free real-model execution."""
import asyncio
from dataclasses import replace
from io import BytesIO
import subprocess
import sys

import numpy as np
import pytest

from mlx2coreai import ConversionConfig
from mlx2coreai.recipe import Build, Bundle, Component, export


def test_named_outputs_partial_build_and_failure(tmp_path):
    from coreai.runtime import SpecializationOptions

    x = np.ones((1, 3), np.float32)
    component = Component(lambda x: (x + 1, x * 2), {"x": x}, ("added", "doubled"),
        ConversionConfig(optimize=False, capture_shapeless=True, dynamic_axes={"x": [1]},
                         dynamic_probe_inputs={"x": np.ones((1, 5), np.float32)}))
    plan = Build("test", {"first": component, "second": component},
                 resources={"tokenizer.txt": b"fixture"}, metadata={"precision": "fp32"})
    bundle = export(plan, tmp_path / "bundle", only=["first"])
    assert list(bundle.manifest["components"]) == ["first"]
    bundle = export(plan, bundle.path, only=["second"])
    assert set(bundle.manifest["components"]) == {"first", "second"}
    assert bundle.resource("tokenizer.txt").read_bytes() == b"fixture"
    with pytest.raises(KeyError):
        bundle.resource("undeclared")
    events = []

    async def check():
        async with bundle.session(specialization_options=SpecializationOptions.cpu_only(),
                                  observer=lambda *event: events.append(event[1])) as session:
            session.reset_state({})
            x = np.arange(7, dtype=np.float32)[None]
            out = await session.run("first", {"x": x})
            assert not isinstance(out["added"], np.ndarray)
            np.testing.assert_array_equal(out["added"].numpy(), x + 1)
            out = await session.run("second", {"x": out["added"]}, readback=True)
            np.testing.assert_array_equal(out["doubled"], (x + 1) * 2)
            with pytest.raises(ValueError):
                session.reset_state({"first": 7})
        with pytest.raises(RuntimeError):
            await session.run("first", {"x": x})
    asyncio.run(check())
    assert events == ["first", "second"]
    before = (bundle.path / "manifest.json").read_bytes()
    plan.metadata = {"precision": "fp16"}
    with pytest.raises(ValueError, match="contract"):
        export(plan, bundle.path, only=["first"])
    plan.metadata = {"precision": "fp32"}
    component.outputs = ("wrong_count",)
    with pytest.raises(ValueError, match="public output"):
        export(plan, bundle.path)
    assert (bundle.path / "manifest.json").read_bytes() == before


def test_state_binding_not_last_output(tmp_path):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions
    from mlx2coreai import CaptureSignature, StateBinding, StateSpec

    component = Component(lambda x, total: (total + x, (total + x) * 2),
        {"x": np.ones(1, np.float32), "total": np.zeros(1, np.float32)}, ("doubled",),
        ConversionConfig(signature=CaptureSignature(output_count=2,
            states=(StateBinding(StateSpec("total", (1,), "fp32"), 0),))))
    unoptimized = replace(component, config=replace(component.config, optimize=False))
    with pytest.raises(ValueError, match="buffer promotion"):
        export(Build("sum", {"main": unoptimized}), tmp_path / "unoptimized")
    assert not (tmp_path / "unoptimized" / "manifest.json").exists()
    bundle = export(Build("sum", {"main": component}), tmp_path / "state")

    async def check():
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        async with bundle.session(specialization_options=options) as session:
            with pytest.raises(RuntimeError, match="reset_state"):
                await session.run("main", {"x": np.ones(1, np.float32)})
            with pytest.raises(ValueError):
                session.reset_state({})
            for _ in range(2):
                session.reset_state({"main": 1})
                for i in (1, 2, 3):
                    result = await session.run("main", {"x": np.ones(1, np.float32)}, readback=True)
                    np.testing.assert_array_equal(result["doubled"], [i * 2])
                    np.testing.assert_array_equal(session.snapshot_state("main")["total"], [i])
    asyncio.run(check())


def test_bundle_paths_and_runtime_imports(tmp_path):
    bundle = Bundle(tmp_path, {"resources": ["../escape"]})
    with pytest.raises(ValueError):
        bundle.resource("../escape")
    result = subprocess.run([sys.executable, "-c", "import sys; import recipes.mimi; import recipes.pocket_tts; "
        "assert not any(k.startswith('mlx_audio') for k in sys.modules); "
        "assert not any(k.startswith('recipes._mlx_lm') for k in sys.modules)"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_session_cleanup_on_load_and_observer_failure(tmp_path, monkeypatch):
    import mlx2coreai.recipe as recipe

    closed = []
    class FakeSession:
        def __init__(self, path, **kwargs):
            self.name = path.stem
        async def __aenter__(self):
            if self.name == "bad":
                raise RuntimeError("load failed")
            return self
        async def __aexit__(self, *exc):
            closed.append(self.name)
        async def run(self, inputs):
            return {"raw": object()}
    monkeypatch.setattr(recipe, "CoreAISession", FakeSession)
    entries = {name: {"asset": f"{name}.aimodel", "entrypoint": "main", "states": [],
                      "outputs": {"named": "raw"}} for name in ("good", "bad")}
    bundle = Bundle(tmp_path, {"components": entries})
    async def check():
        with pytest.raises(RuntimeError, match="load failed"):
            async with bundle.session():
                pass
        assert closed == ["good"]
        def fail(*args):
            raise ValueError("reference failed")
        with pytest.raises(ValueError, match="reference failed"):
            async with bundle.session(components=["good"], observer=fail) as session:
                await session.run("good", {})
        assert closed == ["good", "good"]
    asyncio.run(check())


def test_mimi_recipe(tmp_path):
    from coreai.runtime import SpecializationOptions
    import mlx.core as mx
    from recipes.mimi.build import from_model
    from recipes.mimi import Request, run
    from tests.test_mimi_conversion import tiny_mimi

    model = tiny_mimi()
    bundle = export(from_model(model, "tiny"), tmp_path / "mimi")

    async def check():
        async with bundle.session(specialization_options=SpecializationOptions.cpu_only()) as session:
            for length in (1, 5):
                audio = np.random.default_rng(length).normal(0, 0.1, (1, 1, length * 8)).astype(np.float32)
                codes = await run(session, Request("encode", audio))
                np.testing.assert_array_equal(codes, np.asarray(model.encode(mx.array(audio))))
                decoded = await run(session, Request("decode", codes))
                np.testing.assert_allclose(decoded, np.asarray(model.decode(mx.array(codes))), atol=1e-5, rtol=1e-4)
            with pytest.raises(ValueError, match="multiple"):
                await run(session, Request("encode", np.zeros((1, 1, 3), np.float32)))
            with pytest.raises(ValueError, match="Codes"):
                await run(session, Request("decode", np.full((1, 4, 1), 16, np.int32)))
    asyncio.run(check())


def test_pocket_recipe_stream_and_reset(tmp_path):
    sp = pytest.importorskip("sentencepiece")
    import mlx.core as mx
    import mlx.nn as nn
    from coreai.runtime import ComputeUnitKind, SpecializationOptions
    from recipes.pocket_tts.build import components
    from recipes.pocket_tts import Request, run
    from tests.test_pocket_tts import tiny_decoder

    model = tiny_decoder()
    class Conditioner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 16)
        def __call__(self, text):
            return self.embed(text.tokens)
    model.flow_lm.conditioner = Conditioner()
    mx.eval(model.flow_lm.parameters())
    tokenizer = BytesIO()
    sp.SentencePieceTrainer.train(sentence_iterator=iter(["Hello world.", "This is a recipe test."]),
        model_writer=tokenizer, vocab_size=32, hard_vocab_limit=False, minloglevel=2)
    conditioning = BytesIO()
    np.savez(conditioning, voice=np.ones((1, 3, 16), np.float32) * 0.1,
             bos=np.asarray(model.flow_lm.input_linear(model.flow_lm.bos_emb[None, None])))
    plan = Build("pocket_tts", components(model),
        {"tokenizer.model": tokenizer.getvalue(), "conditioning.npz": conditioning.getvalue()},
        {"backbone_components": ["backbone0", "backbone1"], "latent_dim": 4,
         "sample_rate": 24000, "decoder_steps_per_frame": 2})
    bundle = export(plan, tmp_path / "pocket")
    request = Request(text="Hello world.", max_frames=3, ignore_eos=True, prefill_chunk_size=2)

    async def check():
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        async with bundle.session(specialization_options=options, storage_kind="metal") as session:
            report = {}
            first = [value async for value in run(session, request, report=report)]
            assert len(first) == 3 and all(value.shape == (8,) for value in first)
            assert report["positions"]["decoder"] == 6
            assert np.isfinite(np.concatenate(first)).all()
            stream = run(session, request)
            await anext(stream)
            await stream.aclose()
            second = [value async for value in run(session, request)]
            np.testing.assert_array_equal(np.concatenate(first), np.concatenate(second))
    asyncio.run(check())
