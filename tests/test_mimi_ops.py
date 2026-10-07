import numpy as np
import pytest

from mlx2coreai import ConversionConfig, convert_mlx_to_coreai
from mlx2coreai.runtime import run_aimodel_sync


@pytest.mark.parametrize("groups", [1, 2])
def test_captured_conv_transpose1d(tmp_path, groups):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions

    rng = np.random.default_rng(48)
    weight = rng.normal(size=(4 if groups == 1 else 2, 4, 2 // groups)).astype(np.float32)
    fn = lambda x: mx.conv_transpose1d(x, mx.array(weight), stride=2, padding=1, groups=groups)
    converted = convert_mlx_to_coreai(fn, {"x": rng.normal(size=(1, 4, 2)).astype(np.float32)},
        config=ConversionConfig(optimize=False, capture_shapeless=True, dynamic_axes={"x": [1]},
            dynamic_probe_inputs={"x": np.ones((1, 7, 2), np.float32)}),
        output_path=tmp_path / "conv.aimodel")
    for length in (4, 7, 9):
        x = rng.normal(size=(1, length, 2)).astype(np.float32)
        expected = np.asarray(fn(mx.array(x)))
        actual = run_aimodel_sync(converted.asset, {"x": x}, specialization_options=SpecializationOptions.cpu_only())
        np.testing.assert_allclose(next(iter(actual.outputs.values())), expected, atol=3e-6, rtol=3e-6)


def test_captured_pad(tmp_path):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions

    x = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
    padding = ((0, 0), (2, 1), (1, 0))
    c = convert_mlx_to_coreai(lambda x: mx.pad(x, padding, constant_values=0.75), {"x": x},
                            output_path=tmp_path / "pad.aimodel")
    actual = run_aimodel_sync(c.asset, {"x": x}, specialization_options=SpecializationOptions.cpu_only())
    np.testing.assert_array_equal(next(iter(actual.outputs.values())), np.pad(x, padding, constant_values=0.75))


@pytest.mark.parametrize("kind", ["argmin", "argmax", "trim", "rope", "edge_pad"])
def test_mimi_capture_primitives(tmp_path, kind):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions

    rng = np.random.default_rng(60)
    x = rng.normal(size=(1, 2, 4, 8)).astype(np.float32)
    if kind in ("argmin", "argmax"):
        fn = lambda x: getattr(mx, kind)(x, axis=-1)
    elif kind == "trim":
        fn = lambda x: x[:, :, :-1, :]
    elif kind == "rope":
        fn = lambda x: mx.fast.rope(x, dims=8, traditional=True, base=10000, scale=1, offset=3)
    else:
        fn = lambda x: mx.pad(x, ((0, 0), (0, 0), (0, 0), (2, 0)), mode="edge")
    c = convert_mlx_to_coreai(fn, {"x": x},
        config=ConversionConfig(optimize=False, capture_shapeless=True,
            dynamic_axes={"x": [2]}, dynamic_probe_inputs={"x": np.ones((1, 2, 7, 8), np.float32)}),
        output_path=tmp_path / f"{kind}.aimodel")
    for length in (4, 7, 9):
        value = rng.normal(size=(1, 2, length, 8)).astype(np.float32)
        expected = np.asarray(fn(mx.array(value)))
        actual = run_aimodel_sync(c.asset, {"x": value}, specialization_options=SpecializationOptions.cpu_only())
        np.testing.assert_allclose(next(iter(actual.outputs.values())), expected, atol=3e-6, rtol=3e-6)
