from __future__ import annotations

import asyncio

import ml_dtypes
import numpy as np
import pytest

from mlx2coreai import convert_mlx_lm_stateful
from mlx2coreai._convert_mlx_lm_stateful import _CacheLayout, _make_state_specs, _stateful_inputs
from mlx2coreai._convert_mlx_lm import build_mlx_lm_inputs
from tests.test_convert_mlx_lm import FakeTokenizer


@pytest.mark.parametrize("implementation", ["native", "decomposed"])
def test_cli_forwards_gated_delta_implementation(monkeypatch, tmp_path, implementation):
    from types import SimpleNamespace
    from mlx2coreai import cli

    configs = []
    def convert(*args, **kwargs):
        configs.append(kwargs["config"])
        return SimpleNamespace(bundle_path=tmp_path, asset_path=tmp_path / "model.aimodel",
            lowered=SimpleNamespace(entrypoint_names=["main"]), state_specs=[None] * 4,
            metadata={"mlx_lm_stateful": {"compute_precision": "bf16", "cache_dtype": "bf16"}},
            max_context_length=256)
    monkeypatch.setattr(cli, "convert_mlx_lm_stateful", convert)
    assert cli.main(["convert-mlx-lm-stateful", "model", "--output", str(tmp_path),
                     "--gated-delta-implementation", implementation]) == 0
    assert configs[0].gated_delta_implementation == implementation


@pytest.mark.parametrize("implementation", [
    "native",
    "decomposed",
])
def test_live_gated_delta_net(tmp_path, monkeypatch, implementation):
    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs
    from coreai.runtime import SpecializationOptions, ComputeUnitKind
    from mlx2coreai import ConversionConfig, convert_mlx_to_coreai
    from mlx2coreai._convert_mlx_lm_stateful import _ExportableRecurrentCache
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
    from coreai.runtime import NDArray, SpecializationOptions, ComputeUnitKind
    from mlx2coreai.runtime import allocate_state, run_main

    if not SpecializationOptions.is_supported():
        pytest.skip("requires macOS 27 OS runtime")
    mx.random.seed(8)
    model = TextModel(TextModelArgs(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, vocab_size=128, full_attention_interval=4,
        linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=32,
        linear_value_head_dim=32, tie_word_embeddings=True,
    ))
    model.eval()
    mx.eval(model.parameters())
    converted = convert_mlx_lm_stateful("tiny-hybrid", tmp_path / "hybrid", max_context_length=32,
                                       load_fn=lambda *args, **kwargs: (model, FakeTokenizer()))
    assert len(converted.state_specs) == 4
    assert converted.lowered.optimized
    assert sum(n.op == "gated_delta_update" for n in converted.main.normalized_graph.nodes) == 6

    async def check():
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        async with converted.asset.executable(specialization_options=options) as executable:
            fn = executable.load_function("main")
            for capacity, chunks in [(12, [3, 1, 2, 1]), (24, [1, 4, 1])]:
                state = allocate_state(fn, NDArray, state_capacity=capacity)
                cache = model.make_cache()
                offset = 0
                for count in chunks:
                    ids = np.arange(offset + 1, offset + count + 1, dtype=np.int32)
                    expected = np.asarray(model(mx.array(ids[None]), cache=cache))
                    actual = await run_main(fn, NDArray, ids, np.arange(offset, offset + count, dtype=np.int32),
                                            state, input_name="input_ids", position_ids_name="position_ids")
                    logits = actual[fn.desc.output_names[0]].numpy()
                    np.testing.assert_allclose(state["recurrentState"].numpy(),
                                               np.stack([np.asarray(c[1]) for c in cache[:-1]]),
                                               rtol=3e-3, atol=3e-4)
                    np.testing.assert_allclose(logits, expected, rtol=2e-3, atol=3e-4)
                    offset += count
    asyncio.run(check())
