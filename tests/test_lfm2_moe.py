import asyncio

import numpy as np
import pytest

from mlx2coreai.recipe import export
from recipes import lfm2
from tests.test_convert_mlx_lm import FakeTokenizer


def test_unresolved_expert_index_range_is_rejected():
    import mlx.core as mx
    from mlx2coreai import ConversionConfig
    from mlx2coreai.conversion import prepare_mlx_conversion

    make = lambda n: {'x': np.zeros((1, n, 4, 1, 3), np.float32)}
    fn = lambda x: mx.gather_mm(x, mx.ones((8, 3, 2)), rhs_indices=mx.zeros(x.shape[:-2], mx.int32))
    with pytest.raises(ValueError, match='Dynamic Arange'):
        prepare_mlx_conversion(fn, make(16), config=ConversionConfig(
            capture_shapeless=True, dynamic_axes={'x': [1]}, dynamic_probe_inputs=make(7)))


def tiny_moe(num_layers=3):
    import mlx.core as mx
    from mlx_lm.models.lfm2_moe import Model, ModelArgs

    mx.random.seed(36)
    model = Model(ModelArgs(model_type='lfm2_moe', vocab_size=128, hidden_size=32,
        intermediate_size=64, moe_intermediate_size=48, num_hidden_layers=num_layers,
        num_experts=8, num_experts_per_tok=4, norm_topk_prob=True,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
        use_expert_bias=True, num_dense_layers=1, norm_eps=1e-5,
        conv_bias=False, conv_L_cache=3,
        full_attn_idxs=[1] if num_layers == 3 else [2, 6, 10, 14, 18, 21]))
    model.eval()
    mx.eval(model.parameters())
    return model


def test_moe_dispatch_matches_native():
    import mlx.core as mx

    model = tiny_moe()
    block = model.layers[1].feed_forward
    data = [mx.array(np.random.default_rng(size).normal(size=(1, size, 32)).astype(np.float32))
            for size in [1, 3, 16, 32]]
    expected = [np.asarray(block(x)) for x in data]
    lfm2.adapter.adapt(model)
    for x, reference in zip(data, expected):
        np.testing.assert_allclose(np.asarray(block(x)), reference, rtol=1e-5, atol=1e-6)


def test_moe_block_dynamic(tmp_path):
    import mlx.core as mx
    from mlx2coreai import ConversionConfig, convert_mlx_to_coreai
    from mlx2coreai.runtime import run_aimodel_sync

    model = lfm2.adapter.adapt(tiny_moe())
    fn = model.layers[1].feed_forward
    def inputs(length):
        return {'x': np.random.default_rng(length).normal(size=(1, length, 32)).astype(np.float32)}
    config = ConversionConfig(capture_shapeless=True, dynamic_axes={'x': [1]},
                              dynamic_probe_inputs=inputs(7))
    converted = convert_mlx_to_coreai(fn, inputs(16), config=config, output_path=tmp_path / 'block.aimodel')
    for length in (1, 3, 16, 32):
        data = inputs(length)
        expected = np.asarray(fn(mx.array(data['x'])))
        actual = next(iter(run_aimodel_sync(converted.asset, data).outputs.values()))
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=2e-4, err_msg=f'length={length}')


def test_moe_adapted_stateful_matches_native():
    import mlx.core as mx

    native = tiny_moe(24)
    component = lfm2.build('tiny-moe', max_context_length=64,
        load_fn=lambda *a, **kw: (tiny_moe(24), FakeTokenizer())).components['main']
    names = [binding.spec.name for binding in component.config.signature.states]
    state = {name: mx.array(component.inputs[name]) for name in names}
    cache, position = native.make_cache(), 0
    for length in (3, 1, 16, 2):
        ids = mx.arange(position + 1, position + length + 1)[None]
        expected = np.asarray(native(ids, cache=cache))
        outputs = component.forward(input_ids=ids,
            position_ids=mx.arange(position, position + length)[None], **state)
        mx.eval(*outputs)
        np.testing.assert_allclose(np.asarray(outputs[0]), expected, rtol=2e-4, atol=3e-5)
        state = dict(zip(names, outputs[1:], strict=True))
        position += length


@pytest.mark.parametrize('num_layers', [3, pytest.param(24, marks=pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason='CoreAI beta: deep LFM2 MoE decoding diverges despite matching adapted MLX recurrence'))])
def test_moe_dynamic_stateful(tmp_path, num_layers):
    import mlx.core as mx
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    native = tiny_moe(num_layers)
    plan = lfm2.build('tiny-moe', max_context_length=32,
        load_fn=lambda *a, **kw: (tiny_moe(num_layers), FakeTokenizer()))
    bundle = export(plan, tmp_path / 'moe')

    async def check():
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        async with bundle.session(specialization_options=options) as session:
            for capacity, chunks in [(64, [3, 1, 16, 2]), (80, [32, 1, 5])]:
                session.reset_state({'main': capacity})
                cache = native.make_cache()
                position = 0
                for length in chunks:
                    ids = np.arange(position + 1, position + length + 1, dtype=np.int32)[None] % 128
                    expected = np.asarray(native(mx.array(ids), cache=cache))
                    outputs = await session.run('main', {'input_ids': ids,
                        'position_ids': np.arange(position, position + length, dtype=np.int32)[None]})
                    np.testing.assert_allclose(outputs['logits'].numpy(), expected, rtol=2e-4, atol=3e-5,
                                               err_msg=f'capacity={capacity}, position={position}, length={length}')
                    state = session.snapshot_state('main')
                    position += length
                    np.testing.assert_allclose(state['convState'], np.stack([
                        np.asarray(cache[i][0]) for i in bundle.metadata['conv_layers']]),
                                               rtol=2e-4, atol=3e-5)
                    for name, attr in [('keyCache', 'keys'), ('valueCache', 'values')]:
                        np.testing.assert_allclose(state[name][:, :, :, :position], np.stack([
                            np.asarray(getattr(cache[i], attr))[:, :, :position] for i in bundle.metadata['attention_layers']]),
                                                   rtol=2e-4, atol=3e-5)
                        np.testing.assert_array_equal(state[name][:, :, :, position:], 0)
    asyncio.run(check())
