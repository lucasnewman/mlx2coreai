from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import numpy as np
import pytest

from mlx2coreai import convert_mlx_lm_stateful
from mlx2coreai._convert_mlx_lm_stateful import _infer_cache_layout, _make_state_specs
from tests.test_convert_mlx_lm import FakeTokenizer


def tiny_lfm2(*, two_attention_layers=False):
    import mlx.core as mx
    from mlx_lm.models.lfm2 import Model, ModelArgs

    mx.random.seed(9)
    model = Model(ModelArgs(
        model_type="lfm2", vocab_size=128, hidden_size=64,
        num_hidden_layers=5 if two_attention_layers else 3,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
        norm_eps=1e-5, conv_bias=False, conv_L_cache=3, block_dim=64,
        block_ff_dim=128, block_multiple_of=64, block_ffn_dim_multiplier=1.0,
        block_auto_adjust_ff_dim=False, full_attn_idxs=[1, 3] if two_attention_layers else [1],
    ))
    model.eval()
    mx.eval(model.parameters())
    return model


def test_lfm2_cache_layout():
    layout = _infer_cache_layout(tiny_lfm2())
    assert layout.short_conv_layers == (True, False, True)
    assert layout.num_short_conv_layers == layout.num_conv_layers == 2
    assert layout.num_linear_layers == 0
    specs = _make_state_specs(layout, batch_size=1, max_context_length=32,
                             cache_dtype="bf16", key_cache_name="keys", value_cache_name="values")
    assert [(s.name, s.shape, s.dtype) for s in specs] == [
        ("keys", (1, 1, 1, 32, 32), "bf16"),
        ("values", (1, 1, 1, 32, 32), "bf16"),
        ("convState", (2, 1, 2, 64), "bf16"),
    ]


@pytest.mark.parametrize("cache_length", [1, 4])
def test_lfm2_rejects_invalid_conv_cache_layout(cache_length):
    model = tiny_lfm2()
    model.layers[0].conv.L_cache = cache_length
    with pytest.raises(ValueError, match="uniform, positive cache shapes"):
        _infer_cache_layout(model)


def test_lfm2_requires_attention_layer():
    model = tiny_lfm2()
    model.model.layers = [model.layers[0], model.layers[2]]
    with pytest.raises(ValueError, match="at least one full-attention layer"):
        _infer_cache_layout(model)


@pytest.mark.parametrize("precision", [
    "fp32",
    pytest.param("fp16", marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="macOS 27 26A428 FP16 LFM2 fails parity or aborts in MPSMemrefRegion",
    )),
    "bf16",
])
@pytest.mark.parametrize("two_attention_layers", [False, True])
def test_lfm2_dynamic_prefill_decode(tmp_path, precision, two_attention_layers, request):
    import mlx.core as mx
    from coreai.runtime import NDArray, SpecializationOptions, ComputeUnitKind
    from scripts.benchmark_aimodel_sampling import allocate_state, run_main

    if not SpecializationOptions.is_supported():
        pytest.skip("requires macOS 27 OS runtime")
    if precision == "bf16" and two_attention_layers:
        request.applymarker(pytest.mark.xfail(
            strict=True, raises=AssertionError,
            reason="Two-attention BF16 LFM2 exceeds the 0.03 convolution-state error bound",
        ))
    if precision == "fp16" and not os.environ.get("MLX2COREAI_FP16_TEST_CHILD"):
        # A native compiler abort cannot be caught by pytest in this process.
        result = subprocess.run(
            [sys.executable, "-m", "pytest", request.node.nodeid, "--runxfail", "-q", "-s",
             "--basetemp", str(tmp_path / "child")],
            env={**os.environ, "MLX2COREAI_FP16_TEST_CHILD": "1"},
            capture_output=True, text=True, timeout=180,
        )
        output = result.stdout + result.stderr
        if result.returncode and not any(message in output for message in (
            "MPSMemrefRegion", "Not equal to tolerance",
        )):
            raise RuntimeError(f"Unexpected FP16 subprocess failure: {output}")
        assert result.returncode == 0, output
        return
    model = tiny_lfm2(two_attention_layers=two_attention_layers)
    converted = convert_mlx_lm_stateful("tiny-lfm2", tmp_path / "lfm2", max_context_length=32,
        compute_precision=precision,
        load_fn=lambda *args, **kwargs: (model, FakeTokenizer()))
    assert len(converted.state_specs) == 3
    assert converted.lowered.optimized
    assert not any(n.op == "gated_delta_update" for n in converted.main.normalized_graph.nodes)
    assert converted.metadata["mlx_lm_stateful"]["num_attention_layers"] == (2 if two_attention_layers else 1)

    async def check():
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        async with converted.asset.executable(specialization_options=options) as executable:
            fn = executable.load_function("main")
            for capacity, chunks in [(12, [1, 2, 4, 1, 1]), (24, [5, 1, 3, 1]), (64, [21, 1, 5])]:
                state = allocate_state(fn, NDArray, state_capacity=capacity)
                cache = model.make_cache()
                offset = 0
                for count in chunks:
                    ids = np.arange(offset + 1, offset + count + 1, dtype=np.int32)
                    expected = np.asarray(model(mx.array(ids[None]), cache=cache).astype(mx.float32))
                    actual = await run_main(fn, NDArray, ids, np.arange(offset, offset + count, dtype=np.int32),
                        state, input_name="input_ids", position_ids_name="position_ids")
                    logits = actual[fn.desc.output_names[0]].numpy().astype(np.float32)
                    conv_indices = (0, 2, 4) if two_attention_layers else (0, 2)
                    expected_conv = np.stack([np.asarray(cache[i][0].astype(mx.float32)) for i in conv_indices])
                    actual_conv = state["convState"].numpy().astype(np.float32)
                    attention_indices = (1, 3) if two_attention_layers else (1,)
                    for state_name, cache_attr in (("keyCache", "keys"), ("valueCache", "values")):
                        actual_cache = state[state_name].numpy().astype(np.float32)
                        expected_cache = np.stack([
                            np.asarray(getattr(cache[i], cache_attr)[..., :offset + count, :].astype(mx.float32))
                            for i in attention_indices
                        ])
                        np.testing.assert_array_equal(actual_cache[..., offset + count:, :], 0)
                        valid_cache = actual_cache[..., :offset + count, :]
                        if precision == "fp32":
                            np.testing.assert_allclose(valid_cache, expected_cache, rtol=3e-4, atol=2e-5)
                        else:
                            error = valid_cache - expected_cache
                            assert np.isfinite(valid_cache).all()
                            assert np.linalg.norm(error) / np.linalg.norm(expected_cache) < 0.025
                    if precision == "bf16":
                        # Fused BF16 arithmetic differs between runtimes. Bound
                        # aggregate drift as well as individual outliers.
                        for actual_value, expected_value, max_error in [
                            (actual_conv, expected_conv, 0.03),
                            (logits, expected, 0.06),
                        ]:
                            assert np.isfinite(actual_value).all()
                            error = actual_value - expected_value
                            assert np.linalg.norm(error) / np.linalg.norm(expected_value) < 0.02
                            assert np.max(np.abs(error)) < max_error
                    elif precision == "fp16":
                        np.testing.assert_allclose(actual_conv, expected_conv, rtol=0.005, atol=0.006)
                        np.testing.assert_allclose(logits, expected, rtol=0.005, atol=0.01)
                    else:
                        np.testing.assert_allclose(actual_conv, expected_conv, rtol=3e-4, atol=2e-5)
                        np.testing.assert_allclose(logits, expected, rtol=3e-4, atol=2e-4)
                    offset += count
    asyncio.run(check())
