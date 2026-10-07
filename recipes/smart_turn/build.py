"""FP32 SmartTurn conversion with dynamic batches of fixed-length mel features."""
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mlx2coreai import ConversionConfig
from mlx2coreai.recipe import Build, Component


DEFAULT_SOURCE = "mlx-community/smart-turn-v3"


def load_source(source, *, revision=None):
    from mlx_audio.vad import load

    options = {} if revision is None else {"revision": revision}
    return load(source, strict=True, **options)


def from_model(model, source, *, revision=None):
    import mlx.core as mx

    if model.config.model_type != "smart_turn":
        raise ValueError("Expected a SmartTurn source model.")
    model.eval()
    model.set_dtype(mx.float32)
    model.dtype = mx.float32
    mx.eval(model.parameters())
    encoder = model.config.encoder_config
    processor = model.config.processor_config
    shape = (int(encoder.num_mel_bins), 2 * int(encoder.max_source_positions))
    if min(shape) <= 0 or not np.isfinite(processor.threshold) or not 0 <= processor.threshold <= 1:
        raise ValueError("SmartTurn requires positive feature dimensions and a threshold in [0, 1].")
    rng = np.random.default_rng(17)

    def forward(input_features):
        logits = model(input_features, return_logits=True)
        return logits, mx.sigmoid(logits)

    component = Component(
        forward,
        {"input_features": rng.normal(size=(1, *shape)).astype(np.float32)},
        ("logits", "probability"),
        ConversionConfig(
            # The beta optimizer folds matmul reshapes without reshaping its bias,
            # breaking batches larger than one.
            optimize=False, capture_shapeless=True, dynamic_axes={"input_features": [0]},
            dynamic_probe_inputs={"input_features": rng.normal(size=(2, *shape)).astype(np.float32)},
        ),
    )
    return Build("smart_turn", {"main": component}, metadata={
        "source": str(Path(source).resolve()) if Path(source).is_dir() else str(source),
        "revision": revision, "precision": "fp32", "feature_shape": list(shape),
        "threshold": float(processor.threshold), "dynamic_batch": True,
        "processor": asdict(processor),
        "preprocessing": "mlx-audio mel features, outside the CoreAI asset",
        "workarounds": ["authoring optimization disabled for dynamic-batch bias reshapes"],
    })


def build(source=DEFAULT_SOURCE, *, revision=None):
    return from_model(load_source(source, revision=revision), source, revision=revision)
