from __future__ import annotations

import argparse
from pathlib import Path

from .op_coverage import write_coverage_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlx2coreai")
    subparsers = parser.add_subparsers(dest="command")
    inspect_parser = subparsers.add_parser("inspect", help="List files in a recipe bundle or .aimodel asset.")
    inspect_parser.add_argument("path", type=Path)
    ops_parser = subparsers.add_parser("ops", help="Generate an op coverage report.")
    ops_parser.add_argument("--output", type=Path, default=Path("dev/op_coverage.md"))
    ops_parser.add_argument("--json-output", type=Path, default=Path("dev/op_coverage.json"))
    ops_parser.add_argument("--model-zoo-module", default="tests.model_zoo")
    ops_parser.add_argument("--validate-assets", action="store_true")
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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
