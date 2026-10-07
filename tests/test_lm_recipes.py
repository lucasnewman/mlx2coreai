"""Shared recipe construction, stateful execution, and known-failure boundaries."""
import asyncio
from dataclasses import replace
import subprocess
import sys

import numpy as np
import pytest

from mlx2coreai.recipe import export
from recipes import qwen3, qwen35, lfm2
from recipes._mlx_lm.runtime import Request, run, sample
from recipes._mlx_lm.validation import Reference
from tests.test_mlx_lm_source import FakeTokenizer
from tests.test_lfm2_stateful import tiny_lfm2


@pytest.mark.parametrize("tf32", [None, "1"])
def test_reference_requires_full_precision_mlx(monkeypatch, tf32):
    from types import SimpleNamespace

    if tf32 is None:
        monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", tf32)
    with pytest.raises(RuntimeError, match="MLX_ENABLE_TF32=0"):
        Reference(SimpleNamespace(), qwen3.adapter)


def test_reference_reports_finite_metrics_for_corrupted_fp32_state():
    from types import SimpleNamespace

    reference = Reference(SimpleNamespace(metadata={}), qwen3.adapter, model=object())
    actual = np.full((4,), np.finfo(np.float32).max, np.float32)
    with pytest.raises(AssertionError, match="recurrentState"):
        reference.compare("recurrentState", actual, np.ones((4,), np.float32))
    assert np.isfinite(reference.checks["recurrentState"]["max_relative_l2"])


def tiny_qwen3():
    import mlx.core as mx
    from mlx_lm.models.qwen3 import Model, ModelArgs

    mx.random.seed(18)
    model = Model(ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=2,
        intermediate_size=128, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        rms_norm_eps=1e-6, vocab_size=128, max_position_embeddings=128,
        rope_theta=10000.0, tie_word_embeddings=True))
    model.eval()
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("family", [qwen3, lfm2])
def test_language_recipe_parity_and_request_reset(tmp_path, family):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    model = tiny_qwen3() if family is qwen3 else tiny_lfm2(two_attention_layers=True)
    build = family.build("tiny", max_context_length=32, load_fn=lambda *args, **kwargs: (model, FakeTokenizer()))
    assert build.components["main"].config.state_specs is None
    bundle = export(build, tmp_path / "lm")
    assert len(bundle.manifest["components"]["main"]["states"]) == (2 if family is qwen3 else 3)
    reference = Reference(bundle, family.adapter, model=model, max_abs_error=0.001, max_relative_l2=0.001)

    async def check():
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        async with bundle.session(specialization_options=options, observer=reference) as session:
            request = Request(token_ids=(1, 2, 3, 4, 5, 6), max_new_tokens=4, prefill_chunks=(2, 3),
                              ignore_eos=True, state_capacity=12)
            report = {}
            first = [token async for token in run(session, request, report=report)]
            assert len(first) == 4 and report["position"] == 9
            # An interrupted request must not contaminate the next request.
            stream = run(session, request)
            await anext(stream)
            await stream.aclose()
            second = [token async for token in run(session, replace(request, state_capacity=64))]
            assert first == second
            with pytest.raises(ValueError, match="capacity"):
                await anext(run(session, replace(request, state_capacity=2)))
            with pytest.raises(ValueError, match="integer token"):
                await anext(run(session, replace(request, token_ids=(1, 999))))
            # Immediate EOS stops without yielding or evaluating another token.
            bundle.metadata["eos_token_ids"] = first
            assert [token async for token in run(session, replace(request, ignore_eos=False))] == []
    asyncio.run(check())
    assert reference.checks["keyCache"]["calls"] > 0
    if family is lfm2:
        assert reference.checks["convState"]["calls"] > 0


def test_qwen35_recipe_contract_and_execution_guard(tmp_path):
    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    from types import SimpleNamespace

    model = TextModel(TextModelArgs(model_type="qwen3_5", hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        vocab_size=128, full_attention_interval=4, linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32, tie_word_embeddings=True))
    model.eval()
    mx.eval(model.parameters())
    wrapper = SimpleNamespace(args=SimpleNamespace(model_type="qwen3_5"), language_model=model)
    with pytest.warns(RuntimeWarning, match="experimental"):
        plan = qwen35.build("tiny", max_context_length=32, load_fn=lambda *a, **kw: (wrapper, FakeTokenizer()))
    states = plan.components["main"].config.signature.states
    assert [s.spec.name for s in states] == ["keyCache", "valueCache", "convState", "recurrentState"]
    assert states[-1].spec.dtype == "fp32"
    assert states[-1].spec.shape == (3, 1, 2, 32, 32)
    bundle = export(plan, tmp_path / "qwen35")
    assert bundle.metadata["experimental"]
    assert bundle.manifest["components"]["main"]["optimized"]
    # Guard runs before any native execution, so it can be tested without loading
    # an executable whose integrated gated-delta kernel is known to be incorrect.
    async def check():
        with pytest.raises(ValueError, match="experimental"):
            await anext(run(SimpleNamespace(bundle=bundle), Request(token_ids=(1, 2))))
    asyncio.run(check())
    assert qwen35.adapter.experimental("fp32")
    assert qwen35.adapter.experimental("bf16")
    assert lfm2.adapter.experimental("bf16") and not lfm2.adapter.experimental("fp32")


def test_family_mismatch_and_no_source_imports():
    from types import SimpleNamespace
    for family in (qwen3, qwen35, lfm2):
        with pytest.raises(ValueError, match="model type"):
            family.adapter.adapt(SimpleNamespace(args=SimpleNamespace(model_type="unrelated")))
    result = subprocess.run([sys.executable, "-c",
        "import sys; from recipes import qwen3, qwen35, lfm2; "
        "assert not any(x == 'mlx_lm' or x.startswith('mlx_lm.') for x in sys.modules)"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_seeded_sampling_and_top_k():
    logits = np.array([-5, -4, 1, 2], np.float32)
    request = Request(token_ids=(1,), temperature=0.8, top_k=2)
    sequences = []
    for _ in range(2):
        rng = np.random.default_rng(42)
        sequences.append([sample(logits, request, rng) for _ in range(20)])
    assert sequences[0] == sequences[1]
    assert set(sequences[0]) <= {2, 3}
    assert sample(logits, replace(request, temperature=0), np.random.default_rng()) == 3


def test_chat_template_requests_token_ids(monkeypatch):
    from types import SimpleNamespace
    from recipes._mlx_lm import runtime

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "hello"}]
            assert kwargs["return_dict"] is False
            return [1, 2, 3]

    class Session:
        bundle = SimpleNamespace(metadata={"vocab_size": 8, "eos_token_ids": []})

        def reset_state(self, capacities):
            assert capacities == {"main": 4}

        async def run(self, component, inputs, *, readback):
            np.testing.assert_array_equal(inputs["input_ids"], [[1, 2, 3]])
            return {"logits": np.arange(24).reshape(1, 3, 8)}

    monkeypatch.setattr(runtime, "load_tokenizer", lambda bundle: Tokenizer())

    async def check():
        assert [token async for token in run(Session(), Request(prompt="hello", chat=True,
                                                                max_new_tokens=1))] == [7]
    asyncio.run(check())


def test_cli_experimental_guard_precedes_loading(monkeypatch):
    from types import SimpleNamespace
    from recipes._mlx_lm import cli

    bundle = SimpleNamespace(manifest={"recipe": "qwen35"}, metadata={"experimental": True})
    monkeypatch.setattr(cli.Bundle, "open", lambda path: bundle)
    # Deliberately provide no source-loading or session API: neither may be
    # reached before the experimental opt-in is checked.
    with pytest.raises(ValueError, match="allow-experimental"):
        asyncio.run(cli.generate(qwen35, SimpleNamespace(bundle="unused", allow_experimental=False)))


def test_lfm_history_update_materializes_a_gather():
    import mlx.core as mx
    from mlx2coreai import capture_mlx_graph

    def forward(x, history):
        cache = lfm2.adapter.ArrayCache(history)
        cache[0] = mx.concatenate([cache[0], x], axis=1)[:, -2:, :]
        return cache[0]

    for length in (1, 3):
        x = np.arange(length * 4, dtype=np.float32).reshape(1, length, 4)
        history = np.full((1, 2, 4), -1, np.float32)
        captured = capture_mlx_graph(forward, {"x": x, "history": history}, capture_shapeless=True)
        assert any(node.op == "gather" for node in captured.graph.nodes)
        np.testing.assert_array_equal(next(iter(captured.expected_outputs.values())),
                                      np.concatenate([history, x], axis=1)[:, -2:])
