from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2coreai import convert_mlx_lm_stateful
from tests.test_convert_mlx_lm import FakeTokenizer


def test_layer_cache_preserves_storage_dtype():
    import mlx.core as mx
    from mlx2coreai._convert_mlx_lm_stateful import _ExportableLayeredKVCache, _LayeredKVCacheState

    state = _LayeredKVCacheState(
        mx.zeros((2, 1, 1, 8, 1), dtype=mx.bfloat16),
        mx.zeros((2, 1, 1, 8, 1), dtype=mx.bfloat16),
    )
    cache = _ExportableLayeredKVCache(state, layer_idx=1, offset=mx.array(3, mx.int32))
    update = mx.full((1, 1, 2, 1), 1.2345, dtype=mx.float32)
    keys, values = cache.update_and_fetch(update, update)
    assert keys.dtype == values.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        np.asarray(keys[..., 3:5, :].astype(mx.float32)),
        np.asarray(update.astype(mx.bfloat16).astype(mx.float32)),
    )


def test_packed_cache_reads_and_untouched_regions(tmp_path):
    import mlx.core as mx
    from coreai.runtime import ComputeUnitKind, NDArray, SpecializationOptions
    from mlx2coreai.runtime import run_main

    if not SpecializationOptions.is_supported():
        pytest.skip("requires macOS 27 OS runtime")

    class Model:
        args = SimpleNamespace(num_key_value_heads=1, head_dim=1)
        layers = [SimpleNamespace(self_attn=SimpleNamespace(n_kv_heads=1)) for _ in range(2)]

        def __call__(self, input_ids, *, cache):
            values = input_ids.astype(mx.float32)[:, None, :, None]
            reads = []
            for index, layer in enumerate(cache):
                k, v = layer.update_and_fetch(values + index * 10, values + index * 20)
                reads.append(mx.sum(k + v, axis=(1, 2, 3)))
            return mx.stack(reads)

    converted = convert_mlx_lm_stateful(
        "cache-regression", tmp_path / "cache", max_context_length=32,
        load_fn=lambda *args, **kwargs: (Model(), FakeTokenizer()),
    )
    assert converted.lowered.optimized

    async def check():
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        async with converted.asset.executable(specialization_options=options) as executable:
            fn = executable.load_function("main")
            for capacity in (10, 24):
                shape = (2, 1, 1, capacity, 1)
                initial = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                expected = {"keyCache": initial + 100, "valueCache": initial + 200}
                state = {name: NDArray(value.copy()) for name, value in expected.items()}
                offset = 3
                for count in (2, 1, 4):
                    ids = np.arange(offset, offset + count, dtype=np.int32)
                    for index in range(2):
                        expected["keyCache"][index, 0, 0, offset:offset + count, 0] = ids + index * 10
                        expected["valueCache"][index, 0, 0, offset:offset + count, 0] = ids + index * 20
                    actual = await run_main(
                        fn, NDArray, ids, ids, state,
                        input_name="input_ids", position_ids_name="position_ids",
                    )
                    np.testing.assert_array_equal(
                        actual[fn.desc.output_names[0]].numpy(),
                        (expected["keyCache"] + expected["valueCache"]).sum(axis=(2, 3, 4)),
                    )
                    for name in expected:
                        np.testing.assert_array_equal(state[name].numpy(), expected[name])
                    offset += count

    asyncio.run(check())
