from __future__ import annotations

import numpy as np
import pytest

from mlx2coreai.ir import Graph, Node, TensorSpec
from mlx2coreai.lower_to_coreai import CoreAILoweringConfig, build_coreai_program
from mlx2coreai.passes import infer_graph_specs
from mlx2coreai.runtime import run_aimodel_sync


def delta_graph(dtype="fp32", sequence=-1, key_dim=32, value_dim=16):
    shapes = [(1, sequence, 2, key_dim), (1, sequence, 2, key_dim), (1, sequence, 2, value_dim),
              (1, sequence, 2), (1, sequence, 2), (1, 2, value_dim, key_dim)]
    names = ("q", "k", "v", "decay", "beta", "state")
    specs = [TensorSpec(name, shape, "fp32" if name in {"decay", "state"} else dtype)
             for name, shape in zip(names, shapes, strict=True)]
    return Graph(specs, [Node("gated_delta_update", names, "y", {"output_index": 0}),
                         Node("gated_delta_update", names, "s", {"output_index": 1})], ["y", "s"])


def reference(inputs):
    q, k, v, decay, beta, state = [np.asarray(inputs[n], dtype=np.float32) for n in
                                  ("q", "k", "v", "decay", "beta", "state")]
    state = state.copy()
    ys = []
    for t in range(q.shape[1]):
        state *= decay[:, t, :, None, None]
        memory = np.sum(state * k[:, t, :, None, :], axis=-1)
        delta = (v[:, t] - memory) * beta[:, t, :, None]
        state += k[:, t, :, None, :] * delta[..., None]
        ys.append(np.sum(state * q[:, t, :, None, :], axis=-1))
    return np.stack(ys, axis=1), state


def test_gated_delta_infers_both_outputs():
    specs = infer_graph_specs(delta_graph("bf16"))
    assert specs["y"].shape == (1, -1, 2, 16)
    assert specs["y"].dtype == "bf16"
    assert specs["s"].shape == (1, 2, 16, 32)
    assert specs["s"].dtype == "fp32"


@pytest.mark.parametrize("sequence", [1, 3])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("dims", [(32, 16), (32, 32), (128, 128)])
@pytest.mark.parametrize("implementation", ["native", "decomposed"])
def test_gated_delta_dynamic_runtime(tmp_path, sequence, device, dims, implementation):
    from coreai.runtime import SpecializationOptions

    graph = delta_graph(key_dim=dims[0], value_dim=dims[1])
    graph.nodes.append(Node("rmsnorm", ("y",), "norm", attrs={"eps": 1e-6}))
    graph.outputs.append("norm")
    lowered = build_coreai_program(graph, config=CoreAILoweringConfig(
        optimize=True, gated_delta_implementation=implementation))
    path = tmp_path / "delta.aimodel"
    lowered.program.save_asset(path)
    rng = np.random.default_rng(3)
    inputs = {spec.name: rng.normal(0, 0.1, tuple(sequence if d < 0 else d for d in spec.shape)).astype(np.float32)
              for spec in graph.inputs}
    inputs["decay"] = np.exp(-np.abs(inputs["decay"]))
    inputs["beta"] = 1 / (1 + np.exp(-inputs["beta"]))
    options = SpecializationOptions.cpu_only() if SpecializationOptions.is_supported() else None
    if device == "gpu":
        if not SpecializationOptions.is_supported():
            pytest.skip("GPU specialization requires the OS runtime")
        from coreai.runtime import ComputeUnitKind

        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    actual = run_aimodel_sync(path, inputs, specialization_options=options).outputs
    y, s = reference(inputs)
    np.testing.assert_allclose(actual["y"], y, rtol=3e-4, atol=2e-6)
    np.testing.assert_allclose(actual["s"], s, rtol=3e-4, atol=2e-6)


def test_live_kernel_capture_and_runtime(tmp_path):
    from coreai.runtime import SpecializationOptions
    from mlx_lm.models.gated_delta import gated_delta_kernel
    from mlx2coreai.conversion import ConversionConfig, convert_mlx_to_coreai

    graph = delta_graph(sequence=3)
    rng = np.random.default_rng(5)
    inputs = {spec.name: rng.normal(0, 0.1, spec.shape).astype(np.float32) for spec in graph.inputs}
    inputs["decay"] = np.exp(-np.abs(inputs["decay"]))
    def forward(**kwargs):
        return gated_delta_kernel(*[kwargs[spec.name] for spec in graph.inputs])
    converted = convert_mlx_to_coreai(
        forward, inputs, config=ConversionConfig(capture_shapeless=True),
        output_path=tmp_path / "captured_delta.aimodel",
    )
    prepared = converted.prepared
    nodes = [n for n in prepared.normalized_graph.nodes if n.op == "gated_delta_update"]
    assert len(nodes) == 1
    assert len(nodes[0].outputs) == 2
    assert "output_index" not in nodes[0].attrs
    assert all(len(n.inputs) == 6 for n in nodes)
    expected = reference(inputs)
    actual = run_aimodel_sync(converted.asset, inputs,
                             specialization_options=SpecializationOptions.cpu_only()).outputs
    for name, reference_output in zip(prepared.normalized_graph.outputs, expected, strict=True):
        np.testing.assert_allclose(prepared.expected_outputs[name], reference_output, rtol=3e-4, atol=2e-6)
        np.testing.assert_allclose(actual[name], reference_output, rtol=3e-4, atol=2e-6)
