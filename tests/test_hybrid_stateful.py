from __future__ import annotations

import asyncio
import json

import ml_dtypes
import numpy as np
import pytest

from mlx2coreai.recipe import export
from recipes import qwen35
from recipes._mlx_lm.stateful import _CacheLayout, _make_state_specs, _stateful_inputs
from recipes._mlx_lm.source import build_mlx_lm_inputs
from tests.test_mlx_lm_source import FakeTokenizer


@pytest.mark.parametrize("implementation", ["native", "decomposed"])
def test_cli_forwards_gated_delta_implementation(monkeypatch, tmp_path, implementation):
    import sys
    from recipes._mlx_lm import cli

    configs, exports = [], []
    plan = object()

    def build(*args, **kwargs):
        configs.append(kwargs["config"])
        return plan

    monkeypatch.setattr(qwen35, "build", build)
    monkeypatch.setattr(cli, "export", lambda value, path: exports.append((value, path)))
    monkeypatch.setattr(sys, "argv", ["qwen35", "convert", "model", "--output", str(tmp_path),
                                    "--gated-delta-implementation", implementation])
    cli.main(qwen35)
    assert configs[0].gated_delta_implementation == implementation
    assert exports == [(plan, tmp_path)]


@pytest.mark.parametrize("implementation", [
    "native",
    "decomposed",
])
def test_live_gated_delta_net(tmp_path, monkeypatch, implementation):
    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs
    from coreai.runtime import SpecializationOptions, ComputeUnitKind
    from mlx2coreai import ConversionConfig, convert_mlx_to_coreai
    from recipes._mlx_lm.stateful import _ExportableRecurrentCache
    from mlx2coreai.runtime import run_aimodel_sync

    if not SpecializationOptions.is_supported():
        pytest.skip("requires macOS 27 OS runtime")
    mx.random.seed(8)
    layer = GatedDeltaNet(TextModelArgs(hidden_size=64, linear_num_value_heads=2, linear_num_key_heads=2,
                                       linear_key_head_dim=32, linear_value_head_dim=32))
    layer.eval()
    mx.eval(layer.parameters())
    import mlx_lm.models.qwen3_5 as qwen
    original = qwen.gated_delta_update
    intermediates = []
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        intermediates[:] = [*args[:3], result[0]]
        return result
    monkeypatch.setattr(qwen, "gated_delta_update", observed)
    def forward(x, conv, state):
        cache = _ExportableRecurrentCache(conv, state)
        y = layer(x, cache=cache)
        return y, cache[0], cache[1], *intermediates
    inputs = {"x": np.random.default_rng(0).normal(size=(1, 3, 64)).astype(np.float32),
              "conv": np.zeros((1, 3, 192), np.float32), "state": np.zeros((1, 2, 32, 32), np.float32)}
    converted = convert_mlx_to_coreai(forward, inputs,
                                     config=ConversionConfig(capture_shapeless=True, dynamic_axes={"x": [1]},
                                         gated_delta_implementation=implementation,
                                         dynamic_probe_inputs={**inputs, "x": np.pad(inputs["x"], ((0, 0), (0, 1), (0, 0)))}),
                                     output_path=tmp_path / "gdn.aimodel")
    opts = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    runtime = run_aimodel_sync(converted.asset, inputs, specialization_options=opts)
    actual = runtime.outputs
    expected = converted.prepared.expected_outputs
    for name in reversed(list(expected)):
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-3, atol=1e-7, err_msg=name)


def test_hybrid_state_shapes_and_probe_dtypes():
    layout = _CacheLayout(4, 2, 32, (True, True, True, False), (3, 192), (2, 32, 32))
    specs = _make_state_specs(layout, batch_size=1, max_context_length=16,
                             cache_dtype="bf16", key_cache_name="keys", value_cache_name="values")
    values = _stateful_inputs(build_mlx_lm_inputs(tokenizer=None, sequence_length=3),
                              state_specs=specs, input_name="input_ids", position_ids_name="position_ids",
                              position_length=3, cache_dtype="bf16", state_context_length=17,
                              context_state_names=("keys", "values"))
    assert values["keys"].shape == (1, 1, 2, 17, 32)
    assert values["convState"].shape == (3, 1, 3, 192)
    assert values["recurrentState"].shape == (3, 1, 2, 32, 32)
    assert values["keys"].dtype == values["convState"].dtype == ml_dtypes.bfloat16
    assert values["recurrentState"].dtype == np.float32


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="macOS 27 26A428 native gated delta fails hybrid state/logit parity")
def test_tiny_qwen35_dynamic_prefill_and_decode(tmp_path):
    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    from coreai.runtime import SpecializationOptions, ComputeUnitKind

    if not SpecializationOptions.is_supported():
        pytest.skip("requires macOS 27 OS runtime")
    mx.random.seed(8)
    model = TextModel(TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, vocab_size=128, full_attention_interval=4,
        linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=32,
        linear_value_head_dim=32, tie_word_embeddings=True,
    ))
    model.eval()
    mx.eval(model.parameters())
    plan = qwen35.build("tiny-hybrid", max_context_length=32,
                        load_fn=lambda *args, **kwargs: (model, FakeTokenizer()))
    bundle = export(plan, tmp_path / "hybrid", save_graphs=True)
    main = bundle.manifest["components"]["main"]
    assert len(main["states"]) == 4
    assert main["optimized"]
    graph = json.loads((bundle.path / "main_graph.json").read_text())
    assert sum(n["op"] == "gated_delta_update" for n in graph["nodes"]) == len(bundle.metadata["conv_layers"])

    async def check():
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        async with bundle.session(specialization_options=options) as session:
            for capacity, chunks in [(12, [3, 1, 2, 1]), (24, [1, 4, 1])]:
                session.reset_state({"main": capacity})
                cache = model.make_cache()
                offset = 0
                for count in chunks:
                    ids = np.arange(offset + 1, offset + count + 1, dtype=np.int32)
                    expected = np.asarray(model(mx.array(ids[None]), cache=cache))
                    actual = await session.run("main", {
                        "input_ids": ids[None],
                        "position_ids": np.arange(offset, offset + count, dtype=np.int32)[None],
                    }, readback=True)
                    logits = actual["logits"]
                    np.testing.assert_allclose(session.snapshot_state("main")["recurrentState"],
                                               np.stack([np.asarray(c[1]) for c in cache[:-1]]),
                                               rtol=3e-3, atol=3e-4)
                    np.testing.assert_allclose(logits, expected, rtol=2e-3, atol=3e-4)
                    offset += count
    asyncio.run(check())
