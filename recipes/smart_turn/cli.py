"""Build, run, and validate the SmartTurn recipe."""
import argparse
import asyncio
import json
import os
from pathlib import Path

import numpy as np

from mlx2coreai.recipe import Bundle, export
from .build import DEFAULT_SOURCE, build
from .runtime import Request, run


async def generate(args):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    bundle = Bundle.open(args.bundle)
    if bundle.manifest["recipe"] != "smart_turn":
        raise ValueError("Expected a SmartTurn recipe bundle.")
    features = np.load(args.features, allow_pickle=False)
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    async with bundle.session(specialization_options=options) as session:
        outputs = await run(session, Request(features, threshold=args.threshold))
    report = {name: value.tolist() for name, value in outputs.items()}
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="Build a SmartTurn recipe bundle.")
    convert.add_argument("source", nargs="?", default=DEFAULT_SOURCE)
    convert.add_argument("--revision")
    convert.add_argument("--output", type=Path, required=True)
    generate_parser = commands.add_parser("run", help="Detect endpoints from a .npy feature batch.")
    generate_parser.add_argument("bundle", type=Path)
    generate_parser.add_argument("--features", type=Path, required=True)
    generate_parser.add_argument("--threshold", type=float)
    generate_parser.add_argument("--json-output", type=Path)
    validate = commands.add_parser("validate", help="Build and compare against MLX, including audio preprocessing.")
    validate.add_argument("--model", default=DEFAULT_SOURCE)
    validate.add_argument("--revision")
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--audio", type=Path, nargs="*", default=[])
    validate.add_argument("--atol", type=float, default=1e-4)
    validate.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if args.command == "convert":
        export(build(args.source, revision=args.revision), args.output)
    elif args.command == "run":
        asyncio.run(generate(args))
    else:
        if not np.isfinite([args.atol, args.rtol]).all() or args.atol <= 0 or args.rtol <= 0:
            parser.error("atol and rtol must be finite and positive")
        os.environ["MLX_ENABLE_TF32"] = "0"
        from .validation import validate as validate_model
        asyncio.run(validate_model(args))
