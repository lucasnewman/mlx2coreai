import ml_dtypes
import numpy as np
import pytest

from mlx2coreai.dtypes import (
    cache_numpy_dtype, capture_mlx_dtype, capture_numpy_dtype,
    constant_array, execution_numpy_dtype, normalize_dtype, runtime_numpy_dtype,
)


@pytest.mark.parametrize("dtype,name", [(np.float16, "fp16"), (np.float32, "fp32"), (ml_dtypes.bfloat16, "bf16")])
def test_precision_agrees_across_boundaries(dtype, name):
    assert normalize_dtype(str(np.dtype(dtype))) == name
    assert capture_numpy_dtype(np.dtype(dtype)) == name
    assert capture_mlx_dtype("mlx.core." + str(np.dtype(dtype))) == name
    assert cache_numpy_dtype(name) == runtime_numpy_dtype(name) == execution_numpy_dtype(name) == dtype
    value = np.array([1.25, -0.5], dtype=dtype)
    actual, downcast = constant_array(value, dtype_hint=name)
    assert actual.dtype == value.dtype
    assert downcast is None
    np.testing.assert_array_equal(actual, value)


@pytest.mark.parametrize("dtype,hint,target,description", [
    (np.float64, "fp64", np.float32, "fp64->fp32"),
    (np.int64, "int64", np.int32, "int64->int32"),
])
def test_execution_narrowing_remains_explicit(dtype, hint, target, description):
    actual, downcast = constant_array(np.array([1, 2], dtype=dtype), hint)
    assert actual.dtype == execution_numpy_dtype(hint) == target
    assert downcast == description


def test_integer_narrowing_rejects_overflow():
    with pytest.raises(ValueError, match="safely downcast"):
        constant_array(np.array([2**40], dtype=np.int64))


@pytest.mark.parametrize('dtype', [np.int32, np.float16, np.float32, np.complex64])
def test_scalar_constant_rank(dtype):
    value, _ = constant_array(np.array(1, dtype))
    assert value.shape == ()
    assert value.dtype == dtype


@pytest.mark.parametrize('hint', ['int8', 'uint8', 'fp16', 'bf16', 'fp32'])
def test_explicit_constant_dtype_takes_precedence(hint):
    value, _ = constant_array([1, 2, 3], hint)
    assert value.dtype == execution_numpy_dtype(hint)
