"""Shared CLI used explicitly by each language-model recipe."""
import argparse
import asyncio
import json
from pathlib import Path

from mlx2coreai import ConversionConfig
from mlx2coreai.recipe import Bundle, export
from .runtime import Request, load_tokenizer, run


async def generate(recipe, args):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    bundle = Bundle.open(args.bundle)
    if bundle.manifest["recipe"] != recipe.__name__.split(".")[-1]:
        raise ValueError("Bundle belongs to a different recipe.")
    # Check before specialization: known beta failures can abort during loading,
    # not just during the first inference call.
    if bundle.metadata.get("experimental") and not args.allow_experimental:
        raise ValueError("Experimental bundle: --allow-experimental is required for diagnostic execution.")
    observer = None
    if args.validate_mlx:
        from .validation import Reference
        observer = Reference(bundle, recipe.adapter, source=args.source,
            max_abs_error=args.max_abs_error, max_relative_l2=args.max_relative_l2)
    request = Request(prompt=args.prompt, max_new_tokens=args.max_new_tokens, chat=args.chat,
        state_capacity=args.state_capacity, prefill_chunk_size=args.prefill_chunk_size,
        prefill_chunks=tuple(int(x) for x in args.prefill_chunks.split(",") if x),
        temperature=args.temperature, top_k=args.top_k, seed=args.seed, ignore_eos=args.ignore_eos,
        allow_experimental=args.allow_experimental)
    tokenizer = load_tokenizer(bundle)
    report = {}
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    async with bundle.session(specialization_options=options, storage_kind="metal", observer=observer) as session:
        tokens = [token async for token in run(session, request, report=report)]
    report["text"] = tokenizer.decode(tokens)
    report["timing_includes_mlx_validation"] = observer is not None
    if observer is not None:
        report.update(checks=observer.checks, calls=observer.calls)
    print(json.dumps(report, indent=2), flush=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")


def main(recipe):
    parser = argparse.ArgumentParser(description=f"Build and run the {recipe.__name__} recipe.")
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert")
    convert.add_argument("source", nargs="?")
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--revision")
    convert.add_argument("--compute-precision", choices=["auto", "fp32", "fp16", "bf16"])
    convert.add_argument("--cache-dtype", choices=["fp32", "fp16", "bf16"])
    convert.add_argument("--max-context-length", type=int, default=256)
    convert.add_argument("--gated-delta-implementation", choices=["native", "decomposed"], default="native")
    convert.add_argument("--no-optimize", action="store_true")
    generate_parser = commands.add_parser("run")
    generate_parser.add_argument("bundle", type=Path)
    generate_parser.add_argument("--prompt", default="What is the capital of France? Answer in one short sentence.")
    generate_parser.add_argument("--chat", action="store_true")
    generate_parser.add_argument("--max-new-tokens", type=int, default=32)
    generate_parser.add_argument("--state-capacity", type=int)
    generate_parser.add_argument("--prefill-chunk-size", type=int, default=128)
    generate_parser.add_argument("--prefill-chunks", default="")
    generate_parser.add_argument("--temperature", type=float, default=0.0)
    generate_parser.add_argument("--top-k", type=int, default=0)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--ignore-eos", action="store_true")
    generate_parser.add_argument("--allow-experimental", action="store_true")
    generate_parser.add_argument("--validate-mlx", action="store_true")
    generate_parser.add_argument("--source", help="Override validation checkpoint path.")
    generate_parser.add_argument("--max-abs-error", type=float, default=0.01)
    generate_parser.add_argument("--max-relative-l2", type=float, default=0.01)
    generate_parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.command == "convert":
        options = {"revision": args.revision, "cache_dtype": args.cache_dtype,
            "max_context_length": args.max_context_length,
            "config": ConversionConfig(optimize=not args.no_optimize,
                                       gated_delta_implementation=args.gated_delta_implementation)}
        if args.compute_precision is not None:
            options["compute_precision"] = args.compute_precision
        plan = recipe.build(**options) if args.source is None else recipe.build(args.source, **options)
        export(plan, args.output)
    else:
        asyncio.run(generate(recipe, args))
