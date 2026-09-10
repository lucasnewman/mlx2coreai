"""Optional mlx-audio integration test; no downloaded weights required."""
import numpy as np
import pytest

from mlx2coreai import ConversionConfig, convert_mlx_to_coreai
from mlx2coreai.runtime import run_aimodel_sync


def test_tiny_smart_turn_dynamic_batch(tmp_path):
    pytest.importorskip("mlx_audio")
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions
    from mlx_audio.vad.models.smart_turn.config import EncoderConfig, ModelConfig
    from mlx_audio.vad.models.smart_turn.smart_turn import Model

    mx.random.seed(41)
    model = Model(ModelConfig(encoder_config=EncoderConfig(
        num_mel_bins=8, max_source_positions=16, d_model=16,
        encoder_attention_heads=2, encoder_layers=1, encoder_ffn_dim=32,
    )))
    model.eval()
    mx.eval(model.parameters())
    rng = np.random.default_rng(42)
    inputs = {"input_features": rng.normal(size=(1, 8, 32)).astype(np.float32)}
    converted = convert_mlx_to_coreai(
        model, inputs,
        config=ConversionConfig(optimize=False, capture_shapeless=True,
            dynamic_axes={"input_features": [0]}, dynamic_probe_inputs={
                "input_features": rng.normal(size=(2, 8, 32)).astype(np.float32)}),
        output_path=tmp_path / "smart_turn.aimodel",
    )
    for batch in (1, 2, 3):
        features = rng.normal(size=(batch, 8, 32)).astype(np.float32)
        expected = np.asarray(model(mx.array(features)))
        actual = run_aimodel_sync(converted.asset, {"input_features": features},
                                 specialization_options=SpecializationOptions.cpu_only())
        np.testing.assert_allclose(next(iter(actual.outputs.values())), expected, atol=1e-5, rtol=1e-5)
