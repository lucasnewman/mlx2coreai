"""Reproduce CoreAI b3 GPU payload retention without a model checkpoint.

Compare --input-kind positions with --input-kind matrix and --device cpu.
Use --exporter coreai-torch for an independent PyTorch conversion, and
--load-asset to test only the CoreAI runtime in a fresh process.
The output directory receives an asset, JSON measurements, and vmmap snapshots.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coreai.authoring import AIModelAsset
from coreai.runtime import ComputeUnitKind, NDArray, SpecializationOptions, StorageKind, _use_os_coreai


def save_operation_summary(program, destination):
    operations = []
    def visit(operation):
        operations.append({"op": operation.name,
                           "operands": [str(value.type) for value in operation.operands],
                           "results": [str(value.type) for value in operation.results]})
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child.operation)
    with program._module._mlir_module.context:
        visit(program._module._mlir_module.operation)
    destination.write_text(json.dumps(operations, indent=2) + "\n")


def build_mlx_asset(args, weight):
    from mlx2coreai.conversion import ConversionConfig, lower_graph_to_coreai
    from mlx2coreai.ir import Graph, Node, TensorSpec
    nodes = [Node("constant", (), "weight", attrs={"value": weight}),
             Node("transpose", ("weight",), "weight_t", attrs={"perm": [1, 0]})]
    if args.input_kind == "positions":
        inputs = [TensorSpec("query_positions", (-1,), "int32"),
                  TensorSpec("key_positions", (args.dim,), "int32")]
        nodes.extend([
            Node("expanddims", ("query_positions",), "query_grid", attrs={"axes": [1]}),
            Node("expanddims", ("key_positions",), "key_grid", attrs={"axes": [0]}),
            Node("greaterequal", ("query_grid", "key_grid"), "allowed"),
            Node("cast", ("allowed",), "matrix", attrs={"dtype": "fp32"}),
        ])
    else:
        inputs = [TensorSpec("matrix", (-1, args.dim), "fp32")]
    nodes.append(Node("matmul", ("matrix", "weight_t"), "output"))
    lowered = lower_graph_to_coreai(Graph(inputs, nodes, ["output"]),
                                    config=ConversionConfig(optimize=True))
    asset = args.output / "reproducer.aimodel"
    save_operation_summary(lowered.program, args.output / "coreai_ops.json")
    lowered.program.save_asset(asset)
    return asset, None


def build_torch_asset(args, weight):
    import torch
    from coreai_torch import TorchConverter, get_decomp_table

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.from_numpy(weight))

    class PositionModel(Model):
        def forward(self, query_positions, key_positions):
            matrix = (query_positions[:, None] >= key_positions[None, :]).to(torch.float32)
            return matrix @ self.weight.T

    class MatrixModel(Model):
        def forward(self, matrix):
            return matrix @ self.weight.T

    model = (PositionModel() if args.input_kind == "positions" else MatrixModel()).eval()
    length = torch.export.Dim("query_length", min=1, max=max(1024, *args.lengths))
    if args.input_kind == "positions":
        inputs = (torch.arange(3, dtype=torch.int32), torch.arange(args.dim, dtype=torch.int32))
        dynamic_shapes = ({0: length}, None)
        names = ("query_positions", "key_positions")
    else:
        inputs = (torch.ones((3, args.dim), dtype=torch.float32),)
        dynamic_shapes = ({0: length},)
        names = ("matrix",)
    # This is the documented ExportedProgram conversion path. No custom lowering.
    exported = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
    exported = exported.run_decompositions(get_decomp_table())
    (args.output / "torch_export.txt").write_text(str(exported.graph_module.graph))
    converter = TorchConverter().add_exported_program(exported, input_names=names, output_names=("output",))
    program = converter.to_coreai()
    save_operation_summary(program, args.output / "coreai_ops.json")
    asset = args.output / "reproducer.aimodel"
    program.save_asset(asset)
    converter.clear()
    def eager(inputs):
        with torch.no_grad():
            return model(*(torch.from_numpy(inputs[name]) for name in names)).numpy().copy()
    return asset, eager


@asynccontextmanager
async def open_function(asset_path, options, runtime):
    if runtime == "native":
        asset = AIModelAsset.load(asset_path)
        async with asset.executable(specialization_options=options) as executable:
            yield executable.load_function("main")
    else:
        from mlx2coreai.runtime import CoreAISession
        async with CoreAISession(asset_path, specialization_options=options, storage_kind="bytes") as session:
            yield session.function


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    weight = np.random.default_rng(41).standard_normal(
        (args.output_dim, args.dim), dtype=np.float32
    ) * np.float32(0.01)
    build = build_torch_asset if args.exporter == "coreai-torch" else build_mlx_asset
    asset, eager = (args.load_asset, None) if args.load_asset else build(args, weight)
    gc.collect()
    pid = os.getpid()
    versions = {}
    for name in ("coreai-core", "coreai-torch", "torch", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {"options": {**vars(args), "output": str(args.output.resolve()),
                          "load_asset": str(args.load_asset.resolve()) if args.load_asset else None},
              "pid": pid, "weight_bytes": weight.nbytes,
              "weight_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
              "versions": versions, "python": platform.python_version(),
              "os": platform.platform(), "uses_os_coreai_framework": bool(_use_os_coreai),
              "imported_modules": {name: name in sys.modules for name in ("torch", "coreai_torch", "mlx.core", "mlx2coreai")},
              "events": []}

    def sample(event, **extra):
        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             check=True, capture_output=True, text=True)
        mapped = subprocess.run(["vmmap", "-w", str(pid)],
                                check=True, capture_output=True, text=True).stdout
        (args.output / f"{event}.vmmap.txt").write_text(mapped)
        sizes, files = [], []
        for line in mapped.splitlines():
            if line.startswith("mapped file ") and "/payload-" in line:
                address = re.search(r"([0-9a-f]+)-([0-9a-f]+)", line)
                if address:
                    sizes.append(int(address[2], 16) - int(address[1], 16))
                    files.append(Path(line.split()[-1]))
        row = {"event": event, "rss_bytes": int(rss.stdout.strip()) * 1024,
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "physical_footprint": next((line.split(":", 1)[1].strip()
                   for line in mapped.splitlines() if line.startswith("Physical footprint:")), None),
               "payload_mapped_bytes": sum(sizes), "payload_count": len(sizes),
               "existing_payload_files": sum(file.exists() for file in files), **extra}
        report["events"].append(row)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    sample("asset_selected" if args.load_asset else "converted")
    options = (SpecializationOptions.cpu_only() if args.device == "cpu" else
               SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu()))
    for cycle in range(args.cycles):
        async with open_function(asset, options, args.runtime) as function:
            function_ref = weakref.ref(function)
            sample(f"cycle{cycle}_open")
            for step, count in enumerate(args.lengths):
                query = np.arange(count, dtype=np.int32)
                keys = np.arange(args.dim, dtype=np.int32)
                if args.vary_values:
                    query += step * 7
                    if step % 2:
                        keys = keys[::-1].copy()
                matrix = (query[:, None] >= keys[None, :]).astype(np.float32)
                inputs = ({"query_positions": query, "key_positions": keys}
                          if args.input_kind == "positions" else {"matrix": matrix})
                expected = matrix @ weight.T
                native_inputs = {name: NDArray(data=value, backing=StorageKind(args.storage)) for name, value in inputs.items()}
                output = await function(inputs=native_inputs)
                actual = output["output"].numpy().copy()
                np.testing.assert_equal(actual.shape, expected.shape)
                error = float(np.max(np.abs(actual - expected)))
                np.testing.assert_allclose(actual, expected, atol=0.001, rtol=0.001)
                torch_error = None
                if eager is not None:
                    torch_output = eager(inputs)
                    torch_error = float(np.max(np.abs(actual - torch_output)))
                    np.testing.assert_allclose(actual, torch_output, atol=0.001, rtol=0.001)
                    del torch_output
                np.save(args.output / f"cycle{cycle}_step{step}.npy", actual)
                del output, actual, expected, native_inputs, inputs, matrix
                gc.collect()
                sample(f"cycle{cycle}_step{step}", query_length=count, max_abs_error=error,
                       max_abs_error_torch=torch_error)
        del function
        gc.collect()
        sample(f"cycle{cycle}_closed", function_wrapper_alive=function_ref() is not None)
    report["complete"] = True
    report["imported_modules_after_inference"] = {
        name: name in sys.modules for name in ("torch", "coreai_torch", "mlx.core", "mlx2coreai")
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-kind", choices=["positions", "matrix"], default="positions")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--exporter", choices=["mlx2coreai", "coreai-torch"], default="mlx2coreai")
    parser.add_argument("--runtime", choices=["native", "wrapper"], default="native")
    parser.add_argument("--storage", choices=["bytes", "metal"], default="bytes")
    parser.add_argument("--load-asset", type=Path, help="Run an existing asset without importing either exporter.")
    parser.add_argument("--vary-values", action="store_true", help="Change positions while repeating shapes to check value-independent caching.")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--output-dim", type=int, default=10752)
    parser.add_argument("--lengths", type=lambda text: [int(x) for x in text.split(",")], default=[1, 3, 5, 1])
    parser.add_argument("--cycles", type=int, default=1)
    args = parser.parse_args()
    if not args.lengths or min(args.dim, args.output_dim, args.cycles, *args.lengths) <= 0:
        parser.error("Dimensions, lengths, and cycles must be positive.")
    if args.load_asset and not args.load_asset.exists():
        parser.error("The supplied asset does not exist.")
    asyncio.run(main(args))
