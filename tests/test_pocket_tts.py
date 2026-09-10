"""Optional Pocket TTS integration tests with small, randomly initialized models."""
import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx_audio")

from mlx2coreai import ConversionConfig, CoreAISession, convert_mlx_to_coreai
from recipes.pocket_tts.adapter import (
    StreamingDecoder, backbone_forward, backbone_specs, initial_state, sample_forward, stateful_config,
)


def tiny_flow():
    pytest.importorskip("mlx_audio")
    import mlx.core as mx
    from mlx_audio.tts.models.pocket_tts.flow_lm import FlowLMModel
    from mlx_audio.tts.models.pocket_tts.transformer import StreamingTransformer
    from mlx_audio.tts.models.pocket_tts.mlp import SimpleMLPAdaLN

    mx.random.seed(91)
    flow = FlowLMModel(None, SimpleMLPAdaLN(4, 16, 4, 16, 2, num_time_conds=2),
                       StreamingTransformer(16, 2, 2, 32), dim=16, ldim=4)
    flow.eval()
    mx.eval(flow.parameters())
    return flow


@pytest.mark.parametrize("partitioned", [False, True])
def test_stateful_backbone_dynamic_cache_and_reset(tmp_path, partitioned):
    import mlx.core as mx
    from coreai.runtime import NDArray, SpecializationOptions, ComputeUnitKind

    flow = tiny_flow()
    rng = np.random.default_rng(92)
    converted = []
    for index in (range(flow.transformer.num_layers) if partitioned else [None]):
        specs = backbone_specs(flow, layer_index=index)
        inputs = {"embeddings": rng.normal(size=(1, 3, 16)).astype(np.float32), **initial_state(specs, 16)}
        probe = {"embeddings": rng.normal(size=(1, 5, 16)).astype(np.float32), **initial_state(specs, 24)}
        count = 2 if index is None or index == flow.transformer.num_layers - 1 else 1
        converted.append(convert_mlx_to_coreai(lambda **kw: backbone_forward(flow, layer_index=index, **kw), inputs,
            config=stateful_config(specs, inputs, probe, sequence_name="embeddings", sequence_axis=1, output_count=count),
            output_path=tmp_path / f"backbone{index}.aimodel"))

    async def check():
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        async with AsyncExitStack() as stack:
            sessions = [await stack.enter_async_context(CoreAISession(c.asset, specialization_options=options))
                        for c in converted]
            for capacity in (17, 25):
                caches = flow.make_cache()
                # Poison unwritten slots: these must remain intact and be masked.
                for session in sessions:
                    session.reset_state(state_capacity=capacity)
                    for name, value in session.state.items():
                        session.state[name] = NDArray(np.full(value.shape, 9.0, np.float32))
                offset = 0
                for length in (3, 1, 5, 2):
                    x = rng.normal(size=(1, length, 16)).astype(np.float32)
                    native = flow.out_norm(flow.transformer(mx.array(x), caches))
                    expected = [np.asarray(native), np.asarray(flow.out_eos(native))]
                    hidden = x
                    for session, component in zip(sessions, converted, strict=True):
                        actual = await session.run({"embeddings": hidden, "position": np.array([offset], np.int32)})
                        hidden = actual[component.prepared.normalized_graph.outputs[0]]
                    for name, reference in zip(converted[-1].prepared.normalized_graph.outputs[:2], expected):
                        np.testing.assert_allclose(actual[name].numpy(), reference, atol=1e-5, rtol=1e-4)
                    offset += length
                    snapshots = [s.snapshot_state() for s in sessions]
                    state = {key: np.concatenate([s[key] for s in snapshots]) for key in snapshots[0]}
                    for name, attr in (("keyCache", "keys"), ("valueCache", "values")):
                        for i, cache in enumerate(caches):
                            np.testing.assert_allclose(state[name][i, :, :, :offset],
                                np.asarray(getattr(cache, attr)[:, :, :offset]), atol=1e-5, rtol=1e-4)
                        np.testing.assert_array_equal(state[name][:, :, :, offset:], 9.0)
    asyncio.run(check())


@pytest.mark.parametrize("steps", [1, 4])
def test_flow_sampler(tmp_path, steps):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions
    from mlx_audio.tts.models.pocket_tts.flow_lm import lsd_decode
    from mlx2coreai.runtime import run_aimodel_sync

    flow = tiny_flow()
    rng = np.random.default_rng(93)
    inputs = {"hidden": rng.normal(size=(1, 16)).astype(np.float32),
              "noise": rng.normal(size=(1, 4)).astype(np.float32)}
    converted = convert_mlx_to_coreai(lambda **kw: sample_forward(flow, **kw, steps=steps), inputs,
        config=ConversionConfig(optimize=False, capture_shapeless=True), output_path=tmp_path / "flow.aimodel")
    native = lsd_decode(lambda s, t, x: flow.flow_net(mx.array(inputs["hidden"]), s, t, x),
                        mx.array(inputs["noise"]), steps)
    expected = [np.asarray(native), np.asarray(flow.input_linear(native[:, None]))]
    actual = run_aimodel_sync(converted.asset, inputs, specialization_options=SpecializationOptions.cpu_only()).outputs
    for name, reference in zip(converted.prepared.normalized_graph.outputs, expected):
        np.testing.assert_allclose(actual[name], reference, atol=1e-5, rtol=1e-4)


def tiny_decoder():
    import mlx.core as mx
    from mlx_audio.tts.models.pocket_tts.mimi import MimiAdapter, DummyQuantizer
    from mlx_audio.codec.models.mimi.mimi import mimi_202407
    from mlx_audio.codec.models.mimi.modules.seanet import SeanetDecoder, SeanetEncoder
    from mlx_audio.codec.models.mimi.modules.transformer import ProjectedTransformer
    from dataclasses import replace

    flow = tiny_flow()
    cfg = mimi_202407(4)
    seanet = replace(cfg.seanet, dimension=8, nfilters=4, ratios=[2, 2], pad_mode="constant")
    transformer = replace(cfg.transformer, d_model=8, num_heads=2, num_layers=2,
                          dim_feedforward=16, context=7)
    mimi = MimiAdapter(SeanetEncoder(seanet), SeanetDecoder(seanet), DummyQuantizer(4, 8),
        frame_rate=3000, encoder_frame_rate=6000, sample_rate=24000, channels=1,
        encoder_transformer=ProjectedTransformer(transformer, 8, [8]),
        decoder_transformer=ProjectedTransformer(transformer, 8, [8]))
    mimi.eval()
    mx.eval(mimi.parameters())
    return SimpleNamespace(flow_lm=flow, mimi=mimi)


def test_streaming_decoder_convolution_and_kv_state(tmp_path):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions, ComputeUnitKind

    model = tiny_decoder()
    adapter = StreamingDecoder(model)
    rng = np.random.default_rng(94)
    inputs = {"latent": rng.normal(size=(1, 2, 4)).astype(np.float32), **initial_state(adapter.specs, 24)}
    probe = {"latent": rng.normal(size=(1, 3, 4)).astype(np.float32), **initial_state(adapter.specs, 32)}
    converted = convert_mlx_to_coreai(adapter, inputs,
        config=stateful_config(adapter.specs, inputs, probe, sequence_name="latent", sequence_axis=1, output_count=1),
        output_path=tmp_path / "decoder.aimodel")

    async def check():
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        async with CoreAISession(converted.asset, specialization_options=options) as session:
            for capacity in (30, 40):
                session.reset_state(state_capacity=capacity)
                model.mimi.reset_state()
                offset = 0
                for length in (1, 3, 2, 5):
                    latent = rng.normal(size=(1, length, 4)).astype(np.float32)
                    quantized = model.mimi.quantizer((mx.array(latent) * model.flow_lm.emb_std
                                                     + model.flow_lm.emb_mean).transpose(0, 2, 1))
                    reference = np.asarray(model.mimi.decode_step(quantized)).copy()
                    actual = await session.run({"latent": latent, "position": np.array([offset], np.int32)})
                    np.testing.assert_allclose(actual[converted.prepared.normalized_graph.outputs[0]].numpy(),
                                               reference, atol=1e-5, rtol=1e-4)
                    offset += length * 2
                    state = session.snapshot_state()
                    for i, cache in enumerate(model.mimi.decoder_cache):
                        for name, attr in (("keyCache", "keys"), ("valueCache", "values")):
                            np.testing.assert_allclose(state[name][i, :, :, :offset],
                                np.asarray(getattr(cache, attr)[:, :, :offset]), atol=1e-5, rtol=1e-4)
                    for module, attr, spec in adapter.convolutions:
                        value = getattr(module, attr)
                        if attr == "_prev_ys" and module.convtr.convtr.bias is not None:
                            value = value - module.convtr.convtr.bias[None, :, None]
                        np.testing.assert_allclose(state[spec.name], np.asarray(value), atol=1e-5, rtol=1e-4)
    asyncio.run(check())
