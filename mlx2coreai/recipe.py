"""Model-agnostic component packaging and request-scoped execution.

Recipes are ordinary Python modules outside the conversion core. A bundle is
data, not executable Python: loading one never imports a recipe or source model.
"""
from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping

from .conversion import ConversionConfig, prepare_mlx_conversion, convert_prepared_mlx_to_coreai
from .runtime import CoreAISession
from .quantization import WeightQuantization


@dataclass
class Component:
    forward: Callable
    inputs: Mapping[str, Any]
    outputs: tuple[str, ...]
    config: ConversionConfig = field(default_factory=ConversionConfig)


@dataclass
class Build:
    recipe: str
    components: Mapping[str, Component]
    resources: Mapping[str, Path | bytes] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional runtime-only config; build diagnostics remain on the Build object.
    runtime_metadata: dict[str, Any] | None = None
    # Combine stateless components as named functions in one package.
    asset_name: str | None = None


def _path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Expected a relative bundle path, got {name!r}.")
    result = root / relative
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Bundle path escapes its root: {name!r}.")
    return result


def export(build: Build, output: str | Path, *, only=None, save_graphs=False,
           quantization: WeightQuantization | None = None, quantization_report: dict | None = None) -> Bundle:
    """Convert selected components and publish a data-only runtime descriptor.

    Partial builds must have identical recipe metadata. Conversion is staged
    before touching the existing bundle; a failed publication invalidates the
    manifest rather than leaving apparently usable mixed assets.
    """
    output = Path(output)
    selected = list(build.components if only is None else only)
    if not selected or len(set(selected)) != len(selected) or set(selected) - build.components.keys():
        raise ValueError("Select nonempty, unique component names from the build.")
    for name in build.components:
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid component name: {name!r}")
    if build.asset_name is not None:
        _path(output, build.asset_name)
        if Path(build.asset_name).name != build.asset_name or not build.asset_name.endswith(".aimodel"):
            raise ValueError("Combined asset must be a simple relative .aimodel name.")
        if set(selected) != set(build.components):
            raise ValueError("Combined assets require rebuilding all components.")
        policies = {(c.config.optimize, c.config.gated_delta_implementation,
                     c.config.externalize_weights, c.config.external_weight_threshold,
                     c.config.min_runtime_target) for c in build.components.values()}
        if len(policies) != 1:
            raise ValueError("Combined components must share their CoreAI lowering policy.")
        if any(c.config.constant_inputs or c.config.state_specs or
               (c.config.signature and c.config.signature.states) for c in build.components.values()):
            raise ValueError("Combined components currently require captured constants and explicit state inputs.")
        entrypoints = [c.config.entrypoint_name for c in build.components.values()]
        if len(set(entrypoints)) != len(entrypoints):
            raise ValueError("Combined components need unique entrypoint names.")
    manifest = {"bundle_format": "mlx2coreai.recipe", "format_version": 1,
                "recipe": build.recipe, "metadata": build.metadata,
                "resources": list(build.resources), "components": {}}
    manifest_path = output / ("config.json" if build.runtime_metadata is not None else "manifest.json")
    old_assets = set()
    if (output / "config.json").exists() or (output / "manifest.json").exists():
        previous = Bundle.open(output).manifest
        if not manifest_path.exists() and set(selected) != set(build.components):
            raise ValueError("Changing the bundle descriptor format requires rebuilding all components.")
        old_assets = {entry["asset"] for entry in previous["components"].values()}
        for name in old_assets:
            _path(output, name)
    if manifest_path.exists() and set(selected) != set(build.components):
        contract = dict(manifest, metadata=build.runtime_metadata if build.runtime_metadata is not None else build.metadata)
        if any(previous[key] != contract[key] for key in ("recipe", "metadata", "resources")):
            raise ValueError("Partial rebuild changes the recipe contract; rebuild all components.")
        manifest["components"] = dict(previous["components"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="recipe-", dir=output.parent) as directory:
        staging = Path(directory)
        entries = []
        for name in selected:
            component = build.components[name]
            config = (replace(component.config, weight_quantization=quantization)
                      if quantization is not None else component.config)
            if component.config.state_specs is not None:
                raise ValueError("Recipe components declare state through CaptureSignature, not state_specs.")
            bindings = component.config.signature.states if component.config.signature else ()
            if bindings and not component.config.optimize:
                raise ValueError("Mutable-state components require optimization for CoreAI buffer promotion.")
            print(f"Converting {name}", flush=True)
            prepared = prepare_mlx_conversion(component.forward, component.inputs, config=config)
            if quantization_report is not None and prepared.quantization_report is not None:
                quantization_report[name] = prepared.quantization_report
            state_indices = {binding.output_index for binding in bindings}
            public = [value for i, value in enumerate(prepared.normalized_graph.outputs) if i not in state_indices]
            if (len(public) != len(component.outputs)
                    or any(not isinstance(value, str) or not value for value in component.outputs)
                    or len(set(component.outputs)) != len(component.outputs)):
                raise ValueError(f"{name}: declare one unique name per public output.")
            if build.asset_name is None:
                converted = convert_prepared_mlx_to_coreai(prepared, output_path=staging / f"{name}.aimodel")
                optimized = converted.lowered.optimized
                if bindings and not optimized:
                    raise ValueError("CoreAI buffer promotion was skipped for a mutable-state component.")
                del converted
            else:
                from .lower_to_coreai import CoreAIGraphEntry

                entries.append(CoreAIGraphEntry(component.config.entrypoint_name,
                                               prepared.analysis or prepared.normalized_graph,
                                               public_input_names=set(prepared.normalized_inputs)))
                optimized = False
            if save_graphs:
                (staging / f"{name}_graph.json").write_text(json.dumps(prepared.normalized_graph.to_dict(), indent=2) + "\n")
            manifest["components"][name] = {
                "asset": build.asset_name or f"{name}.aimodel", "outputs": dict(zip(component.outputs, public, strict=True)),
                "states": [binding.spec.to_dict() for binding in bindings],
                "entrypoint": component.config.entrypoint_name,
                "nodes": len(prepared.normalized_graph.nodes), "optimized": optimized,
            }
            del prepared
        if entries:
            from .lower_to_coreai import CoreAILoweringConfig, build_coreai_programs, save_coreai_program

            config = build.components[selected[0]].config
            lowered = build_coreai_programs(entries, config=CoreAILoweringConfig(
                optimize=config.optimize, gated_delta_implementation=config.gated_delta_implementation,
                externalize_weights=config.externalize_weights,
                external_weight_threshold=config.external_weight_threshold))
            save_coreai_program(lowered.program, staging / build.asset_name)
            for entry in manifest["components"].values():
                entry["optimized"] = lowered.optimized
            del lowered, entries
        for name, resource in build.resources.items():
            target = _path(staging, name)
            reserved = {"manifest.json", "config.json", build.asset_name, *(f"{key}.aimodel" for key in build.components),
                        *(f"{key}_graph.json" for key in build.components)}
            if Path(name).parts[0] in reserved or target.exists():
                raise ValueError(f"Resource collides with a generated file: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(resource, bytes):
                target.write_bytes(resource)
            else:
                shutil.copyfile(resource, target)
        output.mkdir(parents=True, exist_ok=True)
        manifest_path.unlink(missing_ok=True)
        # Remove an obsolete descriptor when changing the packaging format.
        (output / ("manifest.json" if manifest_path.name == "config.json" else "config.json")).unlink(missing_ok=True)
        for path in staging.iterdir():
            destination = _path(output, path.name)
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
            shutil.move(str(path), destination)
        new_assets = {entry["asset"] for entry in manifest["components"].values()}
        for name in old_assets - new_assets - build.resources.keys():
            obsolete = _path(output, name)
            if obsolete.is_dir():
                shutil.rmtree(obsolete)
            else:
                obsolete.unlink(missing_ok=True)
        if build.runtime_metadata is not None:
            manifest = {"format_version": 1, "recipe": build.recipe,
                        **({"model": build.asset_name} if build.asset_name is not None else {}),
                        "metadata": build.runtime_metadata, "resources": list(build.resources),
                        "components": {name: {key: value for key, value in entry.items()
                                              if key in ("asset", "entrypoint", "outputs") or
                                              (key == "states" and value)}
                                       for name, entry in manifest["components"].items()}}
            if build.asset_name is not None:
                for entry in manifest["components"].values():
                    del entry["asset"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return Bundle.open(output)


@dataclass(frozen=True)
class Bundle:
    path: Path
    manifest: dict[str, Any]

    @classmethod
    def open(cls, path: str | Path) -> Bundle:
        path = Path(path)
        if (path / "config.json").exists():
            manifest = json.loads((path / "config.json").read_text())
            manifest["bundle_format"] = "mlx2coreai.recipe"
            for entry in manifest["components"].values():
                if "model" in manifest:
                    entry["asset"] = manifest["model"]
                entry.setdefault("states", [])
        else:
            manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("bundle_format") != "mlx2coreai.recipe" or manifest.get("format_version") != 1:
            raise ValueError("Unsupported recipe bundle; rebuild it with the recipe exporter.")
        return cls(path, manifest)

    @property
    def metadata(self):
        return self.manifest["metadata"]

    def resource(self, name: str) -> Path:
        if name not in self.manifest["resources"]:
            raise KeyError(f"Undeclared resource: {name}")
        return _path(self.path, name)

    def session(self, *, components=None, specialization_options=None, storage_kind=None, observer=None):
        return BundleSession(self, components, specialization_options, storage_kind, observer)


class BundleSession:
    """Own component sessions; calls and requests within one session are serial.

    Observers receive raw inputs/outputs after each call and may explicitly
    snapshot state. Normal execution never copies state back to the host.
    """

    def __init__(self, bundle, components, specialization_options, storage_kind, observer):
        self.bundle = bundle
        self.names = list(bundle.manifest["components"] if components is None else components)
        self.options = specialization_options
        self.storage = storage_kind
        self.observer = observer
        self._sessions = {}
        self._stack = None

    async def __aenter__(self):
        if self._stack is not None:
            raise RuntimeError("Bundle session is already open.")
        entries = self.bundle.manifest["components"]
        if not self.names or len(set(self.names)) != len(self.names) or set(self.names) - entries.keys():
            raise ValueError("Missing or duplicate components; finish conversion first.")
        stack = AsyncExitStack()
        try:
            assets = {}
            for name in self.names:
                entry = entries[name]
                asset_path = _path(self.bundle.path, entry["asset"])
                owner = assets.get(asset_path)
                session = (owner.function_session(entry["entrypoint"]) if owner else CoreAISession(
                    asset_path, function_name=entry["entrypoint"],
                    specialization_options=self.options, storage_kind=self.storage))
                self._sessions[name] = await stack.enter_async_context(session)
                assets.setdefault(asset_path, session)
        except BaseException:
            await stack.aclose()
            self._sessions.clear()
            raise
        self._stack = stack
        return self

    async def __aexit__(self, *exc):
        stack, self._stack = self._stack, None
        self._sessions.clear()
        if stack is not None:
            return await stack.__aexit__(*exc)

    def _require_open(self):
        if self._stack is None:
            raise RuntimeError("Use the bundle session inside 'async with'.")

    def reset_state(self, capacities: Mapping[str, int]):
        self._require_open()
        required = {name for name in self.names if self.bundle.manifest["components"][name]["states"]}
        if set(capacities) != required or any(count <= 0 for count in capacities.values()):
            raise ValueError("Supply a positive capacity for every stateful component, and no others.")
        for name, capacity in capacities.items():
            self._sessions[name].reset_state(state_capacity=capacity)

    def snapshot_state(self, component: str):
        self._require_open()
        return self._sessions[component].snapshot_state()

    async def run(self, component: str, inputs: Mapping[str, Any], *, readback=False):
        self._require_open()
        if self.bundle.manifest["components"][component]["states"] and not self._sessions[component].state:
            raise RuntimeError("Call reset_state with request capacities before running a stateful component.")
        result = await self._sessions[component].run(inputs)
        outputs = {name: result[value] for name, value in self.bundle.manifest["components"][component]["outputs"].items()}
        if self.observer is not None:
            self.observer(self, component, inputs, outputs)
        return {name: value.numpy().copy() for name, value in outputs.items()} if readback else outputs
