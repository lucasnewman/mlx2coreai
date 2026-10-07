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
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    bundle = Bundle.open(args.bundle)
    if bundle.manifest["recipe"] != "parakeet_redux":
        raise ValueError("Expected a Parakeet Redux recipe bundle.")
    mel = (np.load(args.mel, allow_pickle=False) if args.mel else
           prepare_audio(args.audio, bundle.metadata["processor"]))
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    report = {}
    async with bundle.session(specialization_options=options) as session:
        result = await run(session, Request(mel, args.length), report=report)
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
    generate = commands.add_parser("run", help="Transcribe audio or prepared mel features.")
    generate.add_argument("bundle", type=Path)
    inputs = generate.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio", type=Path)
    inputs.add_argument("--mel", type=Path, help="FP32 .npy with shape [time, mel_bins] or [1, time, mel_bins].")
    generate.add_argument("--length", type=int, help="Valid mel frames; defaults to time - 1.")
    generate.add_argument("--json-output", type=Path)
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
    args = parser.parse_args(argv)
    if args.command == "convert":
        export(build(args.source, revision=args.revision, frames=tuple(args.frames),
                     weight_format=args.weight_format), args.output)
    elif args.command == "run":
        asyncio.run(transcribe(args))
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
