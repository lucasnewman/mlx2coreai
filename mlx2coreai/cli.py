from __future__ import annotations

import argparse
from pathlib import Path

from ._convert_mlx_lm import convert_mlx_lm
from .conversion import ConversionConfig
from .op_coverage import write_coverage_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlx2coreai")
    subparsers = parser.add_subparsers(dest="command")
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a saved .aimodel bundle.")
    inspect_parser.add_argument("path", type=Path)
    ops_parser = subparsers.add_parser("ops", help="Generate an op coverage report.")
    ops_parser.add_argument("--output", type=Path, default=Path("docs/op_coverage.md"))
    ops_parser.add_argument("--json-output", type=Path, default=Path("docs/op_coverage.json"))
    ops_parser.add_argument("--model-zoo-module", default="tests.model_zoo")
    ops_parser.add_argument("--validate-assets", action="store_true")
    lm_parser = subparsers.add_parser(
        "convert-mlx-lm",
        help="Load an mlx-lm model and save a CoreAI .aimodel asset.",
    )
    lm_parser.add_argument("model_id")
    lm_parser.add_argument("--output", type=Path, required=True)
    lm_parser.add_argument("--prompt", default=None)
    lm_parser.add_argument(
        "--sequence-length",
        "--seq-len",
        type=int,
        default=None,
        help="Optional capture sequence length. Defaults to the prompt token length, or 1 for synthesized inputs.",
    )
    lm_parser.add_argument("--batch-size", type=int, default=1)
    lm_parser.add_argument("--revision", default=None)
    lm_parser.add_argument("--lazy-load", action="store_true")
    lm_parser.add_argument("--dot-output", type=Path, default=None)
    lm_parser.add_argument("--no-optimize", action="store_true")
    lm_parser.add_argument(
        "--dynamic-sequence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    lm_parser.add_argument(
        "--externalize-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    lm_parser.add_argument("--external-weight-threshold", type=int, default=10)
    lm_parser.add_argument(
        "--capture-is-training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    lm_parser.add_argument(
        "--allow-unknown-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)

    if args.command == "inspect":
        path = args.path
        if not path.exists():
            raise FileNotFoundError(path)
        for child in sorted(path.iterdir()):
            print(child.name)
        return 0

    if args.command == "ops":
        payload = write_coverage_report(
            output_path=args.output,
            json_output_path=args.json_output,
            model_zoo_module=args.model_zoo_module,
            validate_assets=args.validate_assets,
        )
        zoo = payload.get("model_zoo")
        if zoo is None:
            print(f"Wrote {args.output} without model zoo coverage.")
        else:
            print(
                f"Wrote {args.output}: {zoo['unique_source_ops']} ops, "
                f"{zoo['node_count']} nodes, asset validation={zoo['asset_validation_passed']}."
        )
        return 0

    if args.command == "convert-mlx-lm":
        converted = convert_mlx_lm(
            args.model_id,
            args.output,
            prompt=args.prompt,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            revision=args.revision,
            lazy_load=args.lazy_load,
            dynamic_sequence=bool(args.dynamic_sequence),
            dot_output_path=args.dot_output,
            config=ConversionConfig(
                allow_unknown_sources=bool(args.allow_unknown_sources),
                capture_is_training=bool(args.capture_is_training),
                optimize=not bool(args.no_optimize),
                externalize_weights=bool(args.externalize_weights),
                external_weight_threshold=int(args.external_weight_threshold),
            ),
        )
        print(f"Wrote {converted.asset_path}")
        print(f"Nodes: {len(converted.prepared.normalized_graph.nodes)}")
        resource_count = sum(1 for entry in converted.weight_manifest if entry.get("storage") == "resource")
        inline_count = sum(1 for entry in converted.weight_manifest if entry.get("storage") == "inline")
        print(
            f"Weights: {len(converted.weight_manifest)} constants "
            f"({resource_count} resource, {inline_count} inline)"
        )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
