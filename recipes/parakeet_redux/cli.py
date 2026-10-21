"""Build, transcribe, and validate the Parakeet Redux recipe."""
import argparse
import asyncio
import json
import os
from pathlib import Path

import numpy as np

from mlx2coreai.recipe import Bundle, export
from .build import DEFAULT_SOURCE, build
from .runtime import Request, run


def specialization_options(compute_unit="gpu", *, debug=False):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    kind = ComputeUnitKind.neural_engine() if compute_unit == "neural-engine" else ComputeUnitKind.gpu()
    options = SpecializationOptions.from_preferred_compute_unit_kind(kind)
    return options.with_debug(enabled=True) if debug else options


def specialization_report(session):
    """Inspect beta SDK placement metadata, rather than infer placement from success."""
    from collections import Counter

    # AIModel's debug metadata currently has no public Python accessor.
    executable = session._sessions["encoder"]._executable
    debug = json.loads(executable._debug_infos)
    report = []
    for entry in debug:
        placements = []
        for op in entry.get("operations", []):
            history = [item["value"]["string"]["_0"] for item in op.get("metadatas", [])
                       if item["key"] == "residency"]
            # Debug locations retain placement before optimization as well.
            if history:
                placements.append(history[-1])
        report.append({"identifier": entry.get("identifier"), "residency": dict(Counter(placements))})
    return report


def prepare_audio(path, processor):
    """Optional mlx-audio preprocessing; never loads the source checkpoint."""
    import mlx.core as mx
    from mlx_audio.stt.models.parakeet.audio import PreprocessArgs, log_mel_spectrogram
    from mlx_audio.stt.utils import load_audio

    if "normalize_valid_frames" not in PreprocessArgs.__dataclass_fields__:
        raise ImportError("Redux audio preprocessing requires mlx-audio >= 0.5.6.")
    config = PreprocessArgs(**processor)
    audio = load_audio(str(path), sr=config.sample_rate, dtype=mx.float32)
    return np.asarray(log_mel_spectrogram(audio, config))


async def transcribe(args):
    bundle = Bundle.open(args.bundle)
    if bundle.manifest["recipe"] != "parakeet_redux":
        raise ValueError("Expected a Parakeet Redux recipe bundle.")
    mel = (np.load(args.mel, allow_pickle=False) if args.mel else
           prepare_audio(args.audio, bundle.metadata["processor"]))
    options = specialization_options(args.compute_unit, debug=args.specialization_report is not None)
    report = {}
    async with bundle.session(specialization_options=options) as session:
        result = await run(session, Request(mel, args.length), report=report)
        if args.specialization_report:
            args.specialization_report.parent.mkdir(parents=True, exist_ok=True)
            args.specialization_report.write_text(json.dumps(specialization_report(session), indent=2) + "\n")
    text = json.dumps({**result, "report": report}, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="Build one encoder/TDT decoder package and runtime config.")
    convert.add_argument("source", nargs="?", default=DEFAULT_SOURCE)
    convert.add_argument("--revision")
    convert.add_argument("--weight-format", choices=("fp32", "uint2", "uint4", "uint8"), default="fp32")
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--frames", type=int, nargs=2, default=(32, 49), metavar=("CAPTURE", "PROBE"))
    convert.add_argument("--neural-engine-frames", type=int,
                         help="Experimental dense FP16 encoder with a fixed mel frame count (multiple of 8, <= 4096).")
    generate = commands.add_parser("run", help="Transcribe audio or prepared mel features.")
    generate.add_argument("bundle", type=Path)
    inputs = generate.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio", type=Path)
    inputs.add_argument("--mel", type=Path, help="FP32 .npy with shape [time, mel_bins] or [1, time, mel_bins].")
    generate.add_argument("--length", type=int, help="Valid mel frames; defaults to time - 1.")
    generate.add_argument("--json-output", type=Path)
    generate.add_argument("--compute-unit", choices=("gpu", "neural-engine"), default="gpu")
    generate.add_argument("--specialization-report", type=Path, help="Save backend placement diagnostics.")
    validate = commands.add_parser("validate", help="Build and compare features, steps, transcripts, and timestamps against MLX.")
    validate.add_argument("--model", default=DEFAULT_SOURCE)
    validate.add_argument("--revision")
    validate.add_argument("--weight-format", choices=("fp32", "uint2", "uint4", "uint8"), default="fp32")
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--frames", type=int, nargs=2, default=(32, 49))
    validate.add_argument("--audio", type=Path, nargs="*", default=[])
    validate.add_argument("--atol", type=float, default=1e-3)
    validate.add_argument("--rtol", type=float, default=1e-3)
    compare = commands.add_parser("compare", help="Compare an existing candidate bundle directly against FP32.")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--audio", type=Path, nargs="*", default=[])
    compare.add_argument("--mel", type=Path, nargs="*", default=[])
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--atol", type=float, default=1e-4)
    compare.add_argument("--rtol", type=float, default=1e-4)
    compare.add_argument("--compute-unit", choices=("gpu", "neural-engine"), default="gpu",
                         help="Candidate backend preference; the reference uses GPU.")
    compare.add_argument("--specialization-report", type=Path)
    compare.add_argument("--decisions-only", action="store_true",
                         help="Require identical tokens, timestamps and decisions; report numerical errors without requiring allclose.")
    timing = commands.add_parser("benchmark-encoder", help="Measure warmed encoder latency with preallocated inputs and output readback.")
    timing.add_argument("bundle", type=Path)
    timing_inputs = timing.add_mutually_exclusive_group(required=True)
    timing_inputs.add_argument("--mel", type=Path)
    timing_inputs.add_argument("--audio", type=Path)
    timing.add_argument("--length", type=int)
    timing.add_argument("--compute-unit", choices=("gpu", "neural-engine"), nargs="+", default=["gpu", "neural-engine"])
    timing.add_argument("--warmup", type=int, default=3)
    timing.add_argument("--iterations", type=int, default=20)
    timing.add_argument("--placement", action="store_true", help="Collect backend placement in a separate untimed debug session.")
    timing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "convert":
        export(build(args.source, revision=args.revision, frames=tuple(args.frames),
                     weight_format=args.weight_format,
                     **({"neural_engine_frames": args.neural_engine_frames} if args.neural_engine_frames is not None else {})), args.output)
    elif args.command == "run":
        asyncio.run(transcribe(args))
    elif args.command == "benchmark-encoder":
        if args.warmup < 0 or args.iterations < 1:
            parser.error("warmup must be nonnegative and iterations must be positive")
        from .benchmark import benchmark
        asyncio.run(benchmark(args))
    else:
        if not np.isfinite([args.atol, args.rtol]).all() or args.atol <= 0 or args.rtol <= 0:
            parser.error("atol and rtol must be finite and positive")
        if args.command == "compare":
            from .comparison import compare as compare_bundles
            asyncio.run(compare_bundles(args))
        else:
            os.environ["MLX_ENABLE_TF32"] = "0"
            from .validation import validate as validate_model
            asyncio.run(validate_model(args))
