#!/usr/bin/env python3
"""Convert and validate Mimi's offline encode or decode function from mlx-audio."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from mlx2coreai.recipe import export
from .adapter import offline_forward
from .build import load_source, from_model
from .runtime import Request, run


async def convert(args):
    import mlx.core as mx
    from coreai.runtime import ComputeUnitKind, SpecializationOptions
    model = load_source(args.weights)
    rng = np.random.default_rng(52)
    frames = [int(n) for n in args.frames.split(",")]
    if len(frames) < 2 or any(n <= 0 for n in frames) or frames[0] == frames[1]:
        raise ValueError("Provide at least two different positive frame counts.")
    samples_per_frame = int(model.sample_rate / model.frame_rate)
    if args.component == "encode":
        name = "audio"
        fn = lambda audio: model.encode(audio).astype(mx.int32)
        inputs = [rng.normal(0, 0.1, (1, 1, f * samples_per_frame)).astype(np.float32) for f in frames]
    else:
        name = "codes"
        fn = lambda codes: model.decode(codes)
        inputs = []
        for f in frames:
            audio = rng.normal(0, 0.1, (1, 1, f * samples_per_frame)).astype(np.float32)
            inputs.append(np.asarray(model.encode(mx.array(audio))).astype(np.int32))
    if args.audio:
        from mlx_audio.audio_io import read
        from scipy.signal import resample_poly
        import math

        for path in args.audio:
            audio, sample_rate = read(str(path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            divisor = math.gcd(int(sample_rate), int(model.sample_rate))
            audio = resample_poly(audio, int(model.sample_rate) // divisor, int(sample_rate) // divisor)
            # Keep the parity check small; this is not a dataset quality benchmark.
            audio = np.asarray(audio[:int(model.sample_rate)], dtype=np.float32)
            audio = np.pad(audio, (0, (-len(audio)) % samples_per_frame))[None, None]
            inputs.append(audio if args.component == "encode" else np.asarray(model.encode(mx.array(audio))).astype(np.int32))
    references = [np.asarray(fn(mx.array(value))).copy() for value in inputs]
    capture_fn = lambda **kwargs: offline_forward(model, args.component, kwargs[name])
    for value, reference in zip(inputs, references, strict=True):
        np.testing.assert_allclose(np.asarray(capture_fn(**{name: mx.array(value)})), reference,
                                   atol=1e-6, rtol=1e-6)
    print(f"Loaded Mimi; converting {args.component}, shapes {[v.shape for v in inputs]}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = from_model(model, args.weights.resolve(), frames=frames[:2])
    bundle = export(plan, args.output, only=[args.component], save_graphs=True)
    entry = bundle.manifest["components"][args.component]
    print(f"Converted {entry['nodes']} nodes", flush=True)
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    report = {"component": args.component, "weights": str(args.weights), "audio_files": [str(p) for p in args.audio],
              "sample_rate": model.sample_rate, "frame_rate": model.frame_rate,
              "codebooks": 32, "codebook_size": model.cfg.quantizer_bins,
              "streaming": False, "samples_per_frame": samples_per_frame,
              "conversion": entry, "results": []}
    async with bundle.session(components=[args.component], specialization_options=options) as session:
        for value, expected in zip(inputs, references, strict=True):
            actual = await run(session, Request(args.component, value))
            assert actual.shape == expected.shape, (actual.shape, expected.shape)
            assert np.isfinite(actual).all() and np.isfinite(expected).all()
            error = actual.astype(np.float32) - expected.astype(np.float32)
            row = {"input_shape": list(value.shape), "output_shape": list(actual.shape),
                   "max_abs_error": float(np.max(np.abs(error))),
                   "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12)),
                   "equal_fraction": float(np.mean(actual == expected))}
            print(json.dumps(row), flush=True)
            report["results"].append(row)
            if args.component == "encode":
                np.testing.assert_array_equal(actual, expected)
            else:
                assert row["max_abs_error"] < 1e-3 and row["relative_l2"] < 1e-3, row
    (args.output / f"{args.component}_validation.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--component", choices=["encode", "decode"], required=True)
    parser.add_argument("--frames", default="2,3,5")
    parser.add_argument("--audio", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path("artifacts/mimi_fp32"))
    asyncio.run(convert(parser.parse_args()))


if __name__ == "__main__":
    main()
