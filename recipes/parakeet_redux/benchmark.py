"""Measure completed encoder calls with reusable inputs and output readback."""
import json
from importlib.metadata import version
import platform
import subprocess
from time import perf_counter

import numpy as np

from mlx2coreai.recipe import Bundle
from .cli import prepare_audio, specialization_options, specialization_report
from .runtime import encoder_inputs


def latency_stats(seconds, audio_seconds):
    samples = np.asarray(seconds, np.float64)
    median = float(np.median(samples))
    return {"samples_ms": (samples * 1000).tolist(), "median_ms": median * 1000,
            "mean_ms": float(samples.mean()) * 1000, "p90_ms": float(np.percentile(samples, 90)) * 1000,
            "min_ms": float(samples.min()) * 1000, "max_ms": float(samples.max()) * 1000,
            "real_time_factor": median / audio_seconds, "audio_seconds_per_second": audio_seconds / median}


async def benchmark(args):
    from coreai.runtime import NDArray

    bundle = Bundle.open(args.bundle)
    if bundle.manifest["recipe"] != "parakeet_redux":
        raise ValueError("Expected a Parakeet Redux recipe bundle.")
    mel = (np.load(args.mel, allow_pickle=False) if args.mel else
           prepare_audio(args.audio, bundle.metadata["processor"]))
    if mel.ndim == 2:
        mel = mel[None]
    if (mel.ndim != 3 or mel.shape[0] != 1 or mel.shape[1] < 3
            or mel.shape[2] != bundle.metadata["mel_bins"] or not np.issubdtype(mel.dtype, np.floating)
            or not np.isfinite(mel).all()):
        raise ValueError("Mel must contain finite floating-point features with shape [1, time >= 3, mel_bins].")
    length = mel.shape[1] - 1 if args.length is None else args.length
    if not 2 <= length <= mel.shape[1]:
        raise ValueError("Length must be between 2 and the mel frame count.")
    inputs = encoder_inputs(mel.astype(np.float32), length, bundle.metadata)
    # Public NDArrays bypass per-call host input allocation in CoreAISession.
    buffers = {name: NDArray(value) for name, value in inputs.items()}
    metadata = bundle.metadata
    factor = metadata["subsampling_factor"]
    valid = (length + factor - 1) // factor
    count = (inputs["mel"].shape[1] + factor - 1) // factor
    audio_seconds = length * metadata["frame_seconds"] / factor
    report = {
        "bundle": str(bundle.path.resolve()), "chip": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "macos": platform.mac_ver()[0], "coreai_version": version("coreai-core"),
        "mel_frames": inputs["mel"].shape[1], "valid_mel_frames": length,
        "encoder_frames": count, "valid_encoder_frames": valid, "audio_seconds": audio_seconds,
        "encoder_dtype": metadata.get("encoder_dtype", "fp32"),
        "warmup": args.warmup, "iterations": args.iterations,
        "scope": "Serial encoder calls with preallocated inputs and output readback; excludes preprocessing, input preparation, decoder, and specialization.",
        "results": [],
    }
    for unit in args.compute_unit:
        start = perf_counter()
        async with bundle.session(components=["encoder"], specialization_options=specialization_options(unit)) as session:
            opened = perf_counter() - start
            start = perf_counter()
            output = await session.run("encoder", buffers, readback=True)
            first = perf_counter() - start
            for _ in range(args.warmup):
                output = await session.run("encoder", buffers, readback=True)
            samples = []
            for _ in range(args.iterations):
                start = perf_counter()
                output = await session.run("encoder", buffers, readback=True)
                samples.append(perf_counter() - start)
            if (output["features"].shape != (1, count, metadata["encoder_dim"])
                    or not np.isfinite(output["features"]).all()
                    or output["lengths"].shape != (1,) or output["lengths"][0] != valid):
                raise RuntimeError("Benchmark encoder produced invalid features or lengths.")
            row = {"compute_unit_preference": unit, "session_open_seconds": opened,
                   "first_call_ms": first * 1000, **latency_stats(samples, audio_seconds)}
        if args.placement:
            # Debug instrumentation and its specialization are outside timing.
            async with bundle.session(components=["encoder"], specialization_options=specialization_options(unit, debug=True)) as session:
                await session.run("encoder", buffers, readback=True)
                row["placement"] = specialization_report(session)
        report["results"].append(row)
        print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report
