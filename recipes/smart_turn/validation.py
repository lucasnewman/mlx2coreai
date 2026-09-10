"""Audio preprocessing and endpoint parity checks against the source model."""
from __future__ import annotations

import json

import numpy as np

from mlx2coreai.recipe import export
from .build import load_source, from_model
from .runtime import Request, run


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

    model = load_source(args.model, revision=args.revision)
    plan = from_model(model, args.model, revision=args.revision)
    cases = list(audio_cases(args.audio))
    features = [np.asarray(model.prepare_input_features(audio, sample_rate=sr))[None]
                for _, audio, sr in cases]

    bundle = export(plan, args.output)
    report = {"model": args.model, "conversion": bundle.manifest["components"]["main"],
              "input_shape": [None, *bundle.metadata["feature_shape"]],
              "preprocessing": bundle.metadata["preprocessing"], "results": []}
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    batches = [(name, feature) for (name, _, _), feature in zip(cases, features, strict=True)]
    batches += [("batch_2", np.concatenate(features[:2])),
                ("batch_3_unseen", np.concatenate(features[:3]))]
    async with bundle.session(specialization_options=options) as session:
        names = ("logits", "probability")
        for name, feature in batches:
            logits = model(mx.array(feature), return_logits=True)
            expected = {"logits": np.asarray(logits), "probability": np.asarray(mx.sigmoid(logits))}
            actual = await run(session, Request(feature))
            row = {"case": name, "batch_size": len(feature)}
            for key in names:
                reference = expected[key]
                value = actual[key]
                assert value.shape == reference.shape, (key, value.shape, reference.shape)
                assert np.isfinite(value).all() and np.isfinite(reference).all()
                np.testing.assert_allclose(value, reference, atol=args.atol, rtol=args.rtol)
                row[key] = {"mlx": reference.tolist(), "coreai": value.tolist(),
                            "max_abs_error": float(np.max(np.abs(value - reference)))}
            threshold = model.config.processor_config.threshold
            coreai_decisions = actual["prediction"]
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
    return report
