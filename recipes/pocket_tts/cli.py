"""Command-line adapters; model build and execution are independently reusable."""
import argparse
import asyncio
from dataclasses import fields
import json
from pathlib import Path
import wave

import numpy as np

from mlx2coreai.recipe import Bundle, export
from .build import build
from .runtime import Request, run


def convert_main():
    parser = argparse.ArgumentParser(description="Convert the Pocket TTS recipe.")
    parser.add_argument("--model", default="mlx-community/pocket-tts")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pocket_tts_fp32"))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--voice", default="alba")
    parser.add_argument("--component", choices=["all", "conditioner", "backbone", "sampler", "decoder"], default="all")
    args = parser.parse_args()
    plan = build(args.model, revision=args.revision, steps=args.steps, voice=args.voice)
    only = None if args.component == "all" else (
        plan.metadata["backbone_components"] if args.component == "backbone" else [args.component])
    export(plan, args.output, only=only)


async def generate(args):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    bundle = Bundle.open(args.bundle)
    reference = None
    if args.validate_mlx:
        from .validation import Reference
        reference = Reference(bundle, source=args.source)
    request = Request(**{field.name: getattr(args, field.name) for field in fields(Request)})
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    report, chunks = {}, []
    async with bundle.session(specialization_options=options, storage_kind="metal", observer=reference) as session:
        async for chunk in run(session, request, report=report):
            chunks.append(chunk)
            if (len(chunks) - 1) % 10 == 0:
                print(f"frame={len(chunks)} eos_logit={report['eos_logit']:.4f}", flush=True)
    audio = np.concatenate(chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(bundle.metadata["sample_rate"])
        output.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    report["timing_includes_mlx_validation"] = reference is not None
    report["checks"] = {} if reference is None else reference.checks
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2), flush=True)
    if reference is not None:
        print(f"Validated {sum(v['calls'] for v in reference.checks.values())} tensor comparisons", flush=True)


def run_main():
    parser = argparse.ArgumentParser(description="Generate audio using a Pocket TTS recipe bundle.")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/pocket_tts_fp32/speech.wav"))
    for field in fields(Request):
        name = "--" + field.name.replace("_", "-")
        if isinstance(field.default, bool):
            parser.add_argument(name, action="store_true")
        else:
            parser.add_argument(name, default=field.default, type=int if field.default is None else type(field.default))
    parser.add_argument("--validate-mlx", action="store_true")
    parser.add_argument("--source")
    asyncio.run(generate(parser.parse_args()))
