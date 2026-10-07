import numpy as np
import pytest

from mlx2coreai import ConversionConfig, convert_mlx_to_coreai
from mlx2coreai.runtime import run_aimodel_sync


@pytest.mark.parametrize("axes", [(-1,), (0, 2), (0, 1, 2)])
@pytest.mark.parametrize("ddof", [0, 1])
def test_dynamic_variance_and_square(tmp_path, axes, ddof):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions

    rng = np.random.default_rng(103)
    base = rng.normal(size=(2, 3, 4)).astype(np.float32)
    converted = convert_mlx_to_coreai(lambda x: (mx.var(x, axis=axes, ddof=ddof), mx.square(x)), {"x": base},
        config=ConversionConfig(optimize=False, capture_shapeless=True, dynamic_axes={"x": [0, 2]},
            dynamic_probe_inputs={"x": np.ones((5, 3, 7), np.float32)}),
        output_path=tmp_path / "variance.aimodel")
    counts = [n for n in converted.prepared.normalized_graph.nodes if n.op == "number_of_elements"]
    assert counts and all(n.attrs["dtype"] == "fp32" for n in counts)
    for shape in ((2, 3, 4), (6, 3, 9)):
        x = rng.normal(size=shape).astype(np.float32)
        actual = run_aimodel_sync(converted.asset, {"x": x},
                                 specialization_options=SpecializationOptions.cpu_only()).outputs
        expected = [np.var(x, axis=axes, ddof=ddof), np.square(x)]
        for name, reference in zip(converted.prepared.normalized_graph.outputs, expected, strict=True):
            np.testing.assert_allclose(actual[name], reference, atol=2e-6, rtol=2e-6)


def test_sqrt_and_rsqrt_capture_flags(tmp_path):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions

    x = np.array([0.01, 0.25, 1.0, 4.0, 100.0], np.float32)
    converted = convert_mlx_to_coreai(lambda x: (mx.sqrt(x), mx.rsqrt(x)), {"x": x},
        config=ConversionConfig(optimize=False), output_path=tmp_path / "roots.aimodel")
    nodes = [n for n in converted.prepared.normalized_graph.nodes if n.op == "sqrt"]
    assert {n.attrs["inverted"] for n in nodes} == {False, True}
    actual = run_aimodel_sync(converted.asset, {"x": x},
                             specialization_options=SpecializationOptions.cpu_only()).outputs
    for name, reference in zip(converted.prepared.normalized_graph.outputs, (np.sqrt(x), 1 / np.sqrt(x)), strict=True):
        np.testing.assert_allclose(actual[name], reference, atol=1e-6, rtol=1e-6)
