#!/usr/bin/env python3
"""Convert SmartTurn's feature-to-endpoint network and compare it against MLX.

Requires the optional mlx-audio package. Audio preprocessing stays in mlx-audio;
the asset accepts its native [batch, 80, 800] mel features, with dynamic batch.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx2coreai import ConversionConfig, CoreAISession, convert_mlx_to_coreai


def audio_cases(audio_paths):
    rng = np.random.default_rng(17)
    yield "silence_1s", np.zeros(16000, np.float32), 16000
    yield "noise_8s", rng.normal(0, 0.05, 128000).astype(np.float32), 16000
    time = np.arange(32000, dtype=np.float32) / 16000
    yield "tone_2s", (0.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32), 16000
    yield "resampled_noise_8khz", rng.normal(0, 0.05, 16000).astype(np.float32), 8000
    yield "cropped_noise_10s", rng.normal(0, 0.05, 160000).astype(np.float32), 16000
    for path in audio_paths:
        yield path.name, str(path), None


async def validate(args):
    import mlx.core as mx
    from coreai.runtime import ComputeUnitKind, SpecializationOptions
    from mlx_audio.vad import load

    model = load(args.model, strict=True)
    model.eval()
    cases = list(audio_cases(args.audio))
    features = [np.asarray(model.prepare_input_features(audio, sample_rate=sr))[None]
                for _, audio, sr in cases]

    def forward(input_features):
        logits = model(input_features, return_logits=True)
        return logits, mx.sigmoid(logits)

    args.output.mkdir(parents=True, exist_ok=True)
    converted = convert_mlx_to_coreai(
        forward, {"input_features": features[0]},
        config=ConversionConfig(
            # CoreAI b2 folds the surrounding reshapes into matmul without
            # reshaping its flattened bias; batches > 1 then fail at runtime.
            optimize=False,
            capture_shapeless=True, dynamic_axes={"input_features": [0]},
            dynamic_probe_inputs={"input_features": np.concatenate(features[:2])},
        ),
        output_path=args.output / "smart_turn.aimodel",
    )
    report = {"model": args.model, "conversion": converted.metadata,
              "input_shape": [None, *features[0].shape[1:]],
              "preprocessing": "unchanged mlx-audio, outside the CoreAI asset", "results": []}
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    batches = [(name, feature) for (name, _, _), feature in zip(cases, features, strict=True)]
    batches += [("batch_2", np.concatenate(features[:2])),
                ("batch_3_unseen", np.concatenate(features[:3]))]
    async with CoreAISession(converted.asset, specialization_options=options) as session:
        names = ("logits", "probability")
        output_names = session.function.desc.output_names
        assert len(output_names) == len(names)
        for name, feature in batches:
            expected = {key: np.asarray(value).astype(np.float32)
                        for key, value in zip(names, forward(mx.array(feature)), strict=True)}
            actual = await session.run({"input_features": feature})
            row = {"case": name, "batch_size": len(feature)}
            for key, output_name in zip(names, output_names, strict=True):
                reference = expected[key]
                value = np.asarray(actual[output_name].numpy(), dtype=np.float32)
                assert value.shape == reference.shape, (key, value.shape, reference.shape)
                assert np.isfinite(value).all() and np.isfinite(reference).all()
                np.testing.assert_allclose(value, reference, atol=args.atol, rtol=args.rtol)
                row[key] = {"mlx": reference.tolist(), "coreai": value.tolist(),
                            "max_abs_error": float(np.max(np.abs(value - reference)))}
            threshold = model.config.processor_config.threshold
            coreai_decisions = np.asarray(row["probability"]["coreai"]) > threshold
            mlx_decisions = expected["probability"] > threshold
            np.testing.assert_array_equal(coreai_decisions, mlx_decisions)
            row["matching_decisions"] = int(np.sum(coreai_decisions == mlx_decisions))
            report["results"].append(row)
            print(json.dumps(row), flush=True)
    for (name, audio, sr), row in zip(cases, report["results"]):
        endpoint = model.predict_endpoint(audio, sample_rate=sr)
        np.testing.assert_allclose(endpoint.probability, row["probability"]["mlx"][0][0],
                                   atol=1e-6, rtol=1e-6)
        assert endpoint.prediction == int(row["probability"]["coreai"][0][0] > threshold), name
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(batches)} cases; saved {args.output / 'validation.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/smart-turn-v3")
    parser.add_argument("--output", type=Path, default=Path("artifacts/refactor_smart_turn_v3"))
    parser.add_argument("--audio", type=Path, nargs="*", default=[])
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.atol <= 0 or args.rtol <= 0:
        parser.error("atol and rtol must be positive")
    asyncio.run(validate(args))


if __name__ == "__main__":
    main()
