"""Optional Mimi integration without external checkpoint downloads."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2coreai import ConversionConfig, convert_mlx_to_coreai
from recipes.mimi.adapter import offline_forward
from mlx2coreai.runtime import run_aimodel_sync


@pytest.mark.parametrize("tf32", [None, "1"])
def test_mimi_validation_rejects_reduced_precision_reference(monkeypatch, tf32):
    from recipes.mimi.validation import convert

    if tf32 is None:
        monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", tf32)
    with pytest.raises(RuntimeError, match="MLX_ENABLE_TF32=0"):
        asyncio.run(convert(SimpleNamespace()))


def tiny_mimi():
    pytest.importorskip("mlx_audio")
    import mlx.core as mx
    from mlx_audio.codec.models.mimi.mimi import Mimi, mimi_202407

    mx.random.seed(67)
    cfg = mimi_202407(4)
    cfg = replace(cfg, frame_rate=3000, quantizer_bins=16, quantizer_dim=4,
                  seanet=replace(cfg.seanet, dimension=8, nfilters=4, ratios=[2, 2]),
                  transformer=replace(cfg.transformer, d_model=8, num_heads=2,
                                      num_layers=1, dim_feedforward=16))
    model = Mimi(cfg)
    for rvq in (model.quantizer.rvq_first, model.quantizer.rvq_rest):
        for layer in rvq.vq.layers:
            book = layer.codebook
            book.embedding_sum = mx.random.normal(book.embedding_sum.shape)
            book.cluster_usage = mx.ones(book.cluster_usage.shape)
            book.update_in_place()
    model.eval()
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("component", ["encode", "decode"])
def test_tiny_mimi_dynamic_lengths(tmp_path, component):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions

    model = tiny_mimi()
    rng = np.random.default_rng(68)
    samples_per_frame = 8
    frames = (2, 3, 1, 5)
    values, references = [], []
    for length in frames:
        audio = mx.array(rng.normal(0, 0.1, (1, 1, length * samples_per_frame)).astype(np.float32))
        value = audio if component == "encode" else model.encode(audio).astype(mx.int32)
        native = getattr(model, component)(value)
        reference = np.asarray(native).copy()
        adapter = np.asarray(offline_forward(model, component, value))
        np.testing.assert_allclose(adapter, reference, rtol=1e-6, atol=1e-6)
        values.append(np.asarray(value).copy())
        references.append(reference)
    fn = lambda value: offline_forward(model, component, value)
    converted = convert_mlx_to_coreai(fn, {"value": values[0]},
        config=ConversionConfig(optimize=False, capture_shapeless=True, dynamic_axes={"value": [2]},
            dynamic_probe_inputs={"value": values[1]}), output_path=tmp_path / f"{component}.aimodel")
    for value, reference in zip(values, references, strict=True):
        actual = run_aimodel_sync(converted.asset, {"value": value},
                                 specialization_options=SpecializationOptions.cpu_only())
        output = next(iter(actual.outputs.values()))
        if component == "encode":
            np.testing.assert_array_equal(output, reference)
        else:
            assert np.max(np.abs(reference)) > 0
            np.testing.assert_allclose(output, reference, rtol=1e-4, atol=1e-5)
