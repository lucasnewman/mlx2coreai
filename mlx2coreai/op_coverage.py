from __future__ import annotations

import argparse
import importlib
import json
import tempfile
from collections.abc import Sequence
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .conversion import ConversionConfig, lower_graph_to_coreai
from .op_registry import SUPPORTED_MLX_TO_COREAI_OPS, coreai_op_for_mlx, normalize_mlx_op_name
from .reporting import collect_versions


@dataclass(slots=True)
class ModelOpCoverage:
    module: str
    name: str
    node_count: int
    unique_ops: list[str]
    asset_validated: bool | None
    validation_error: str | None = None


def generate_op_coverage(
    *,
    model_zoo_module: str = "tests.model_zoo",
    coverage_modules: Sequence[str] | None = None,
    validate_assets: bool = False,
) -> dict[str, Any]:
    registry_ops = sorted(SUPPORTED_MLX_TO_COREAI_OPS)
    lowering_keys = sorted(set(SUPPORTED_MLX_TO_COREAI_OPS.values()))
    module_names = list(coverage_modules) if coverage_modules is not None else [model_zoo_module, "tests.coverage_zoo"]
    modules = [(name, module) for name in module_names if (module := _load_optional_module(name)) is not None]

    payload: dict[str, Any] = {
        "schema_version": "mlx2coreai.op_coverage.v1",
        "versions": collect_versions(),
        "registry": {
            "supported_source_op_names": len(registry_ops),
            "lowering_keys": len(lowering_keys),
            "ops": [
                {
                    "op": op,
                    "lowering": SUPPORTED_MLX_TO_COREAI_OPS[op],
                }
                for op in registry_ops
            ],
        },
        "model_zoo": None,
        "notes": [
            "Coverage is asset-generation coverage, not runtime numerical parity.",
            "Runtime parity requires the macOS / iOS 27+ CoreAI execution stack.",
            "General transposed convolution uses a named composite fallback when the beta CoreAI asset writer rejects native conv_transpose IR; the vendored 1x1 stride-1 case lowers without that fallback.",
        ],
    }

    if not modules:
        return payload

    op_counts: Counter[str] = Counter()
    lowering_counts: Counter[str] = Counter()
    models_by_op: dict[str, set[str]] = defaultdict(set)
    models: list[ModelOpCoverage] = []
    validation_dir = Path(tempfile.mkdtemp(prefix="mlx2coreai_op_coverage_")) if validate_assets else None

    for module_name, module in modules:
        for model_name in module.available_model_names():
            spec = module.get_model_spec(model_name, seed=0)
            _record_model_coverage(
                module_name=module_name,
                spec=spec,
                op_counts=op_counts,
                lowering_counts=lowering_counts,
                models_by_op=models_by_op,
                models=models,
                validate_assets=validate_assets,
                validation_dir=validation_dir,
            )

    exercised_ops = sorted(op_counts)
    unexercised_registry_ops = sorted(set(registry_ops) - set(exercised_ops))
    unsupported_exercised_ops = sorted(op for op in exercised_ops if coreai_op_for_mlx(op) is None)
    payload["model_zoo"] = {
        "module": ", ".join(module_names),
        "modules": module_names,
        "loaded_modules": [name for name, _ in modules],
        "model_count": len(models),
        "node_count": int(sum(model.node_count for model in models)),
        "unique_source_ops": len(exercised_ops),
        "unique_lowering_keys": len(lowering_counts),
        "asset_validation_enabled": bool(validate_assets),
        "asset_validation_passed": (
            all(model.asset_validated for model in models)
            if validate_assets
            else None
        ),
        "unsupported_exercised_ops": unsupported_exercised_ops,
        "unexercised_registry_ops": unexercised_registry_ops,
        "ops": [
            {
                "op": op,
                "lowering": coreai_op_for_mlx(op),
                "node_count": int(op_counts[op]),
                "models": sorted(models_by_op[op]),
            }
            for op in exercised_ops
        ],
        "lowering_keys": [
            {
                "lowering": lowering,
                "node_count": int(lowering_counts[lowering]),
            }
            for lowering in sorted(lowering_counts)
        ],
        "models": [
            {
                "module": model.module,
                "name": model.name,
                "node_count": model.node_count,
                "unique_ops": model.unique_ops,
                "asset_validated": model.asset_validated,
                "validation_error": model.validation_error,
            }
            for model in models
        ],
    }
    return payload


def _record_model_coverage(
    *,
    module_name: str,
    spec: Any,
    op_counts: Counter[str],
    lowering_counts: Counter[str],
    models_by_op: dict[str, set[str]],
    models: list[ModelOpCoverage],
    validate_assets: bool,
    validation_dir: Path | None,
) -> None:
    model_ops: Counter[str] = Counter(normalize_mlx_op_name(node.op) for node in spec.graph.nodes)
    for op, count in model_ops.items():
        lowering = coreai_op_for_mlx(op)
        op_counts[op] += count
        if lowering is not None:
            lowering_counts[lowering] += count
        models_by_op[op].add(spec.name)

    asset_validated: bool | None = None
    validation_error: str | None = None
    if validate_assets and validation_dir is not None:
        try:
            lowered = lower_graph_to_coreai(spec.graph, config=ConversionConfig(optimize=False))
            asset_path = validation_dir / f"{spec.name}.aimodel"
            lowered.program.save_asset(asset_path)
            asset_validated = (asset_path / "main.mlirb").exists()
        except Exception as exc:  # pragma: no cover - exercised by failures only
            asset_validated = False
            validation_error = f"{type(exc).__name__}: {exc}"

    models.append(
        ModelOpCoverage(
            module=module_name,
            name=spec.name,
            node_count=len(spec.graph.nodes),
            unique_ops=sorted(model_ops),
            asset_validated=asset_validated,
            validation_error=validation_error,
        )
    )


def render_markdown(payload: dict[str, Any]) -> str:
    registry = payload["registry"]
    zoo = payload.get("model_zoo")
    lines = [
        "# mlx2coreai Op Coverage",
        "",
        "Coverage type: CoreAI asset generation. This does not imply runtime numerical parity.",
        "",
        "## Summary",
        "",
        f"- Supported source op names in registry: {registry['supported_source_op_names']}",
        f"- Distinct lowering keys in registry: {registry['lowering_keys']}",
    ]
    if zoo is None:
        lines.extend(["- Model zoo: unavailable", ""])
    else:
        validation = zoo["asset_validation_passed"]
        validation_text = "not run" if validation is None else ("passed" if validation else "failed")
        lines.extend(
            [
                f"- Coverage modules: `{zoo['module']}`",
                f"- Coverage graphs: {zoo['model_count']}",
                f"- Coverage graph nodes: {zoo['node_count']}",
                f"- Unique source ops exercised: {zoo['unique_source_ops']}",
                f"- Unique lowering keys exercised: {zoo['unique_lowering_keys']}",
                f"- Asset validation: {validation_text}",
                "",
            ]
        )
        if zoo["unsupported_exercised_ops"]:
            lines.extend(
                [
                    "## Unsupported Exercised Ops",
                    "",
                    ", ".join(f"`{op}`" for op in zoo["unsupported_exercised_ops"]),
                    "",
                ]
            )
        lines.extend(
            [
                "## Exercised Ops",
                "",
                "| Op | Lowering | Nodes | Models |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for row in zoo["ops"]:
            models = ", ".join(f"`{name}`" for name in row["models"])
            lines.append(
                f"| `{row['op']}` | `{row['lowering']}` | {row['node_count']} | {models} |"
            )
        lines.extend(
            [
                "",
                "## Coverage Graph Assets",
                "",
                "| Module | Graph | Nodes | Unique Ops | Asset |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in zoo["models"]:
            asset = row["asset_validated"]
            asset_text = "not run" if asset is None else ("passed" if asset else f"failed: {row['validation_error']}")
            lines.append(
                f"| `{row['module']}` | `{row['name']}` | {row['node_count']} | {len(row['unique_ops'])} | {asset_text} |"
            )
        lines.extend(
            [
                "",
                "## Unexercised Registry Ops",
                "",
                ", ".join(f"`{op}`" for op in zoo["unexercised_registry_ops"]) or "None",
                "",
            ]
        )

    lines.extend(["## Notes", ""])
    lines.extend(f"- {note}" for note in payload["notes"])
    lines.append("")
    return "\n".join(lines)


def write_coverage_report(
    *,
    output_path: Path,
    json_output_path: Path | None,
    model_zoo_module: str = "tests.model_zoo",
    validate_assets: bool = False,
) -> dict[str, Any]:
    payload = generate_op_coverage(
        model_zoo_module=model_zoo_module,
        validate_assets=validate_assets,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(payload), encoding="utf-8")
    if json_output_path is not None:
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _load_optional_module(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mlx2coreai.op_coverage")
    parser.add_argument("--output", type=Path, default=Path("docs/op_coverage.md"))
    parser.add_argument("--json-output", type=Path, default=Path("docs/op_coverage.json"))
    parser.add_argument("--model-zoo-module", default="tests.model_zoo")
    parser.add_argument("--validate-assets", action="store_true")
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
