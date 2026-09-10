"""Native MLX capture -> CoreAI asset -> numerical runtime checks."""
import numpy as np
import pytest

from mlx2coreai import ConversionConfig, convert_mlx_to_coreai
from mlx2coreai.runtime import run_aimodel_sync


def check(tmp_path, fn, inputs, *, optimize=True, atol=2e-5, rtol=2e-5, config=None, cpu_only=True):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions

    expected = fn(**{name: mx.array(x) for name, x in inputs.items()})
    expected = list(expected) if isinstance(expected, (list, tuple)) else [expected]
    expected = [np.asarray(x) for x in expected]
    converted = convert_mlx_to_coreai(fn, inputs,
        config=config or ConversionConfig(optimize=optimize), output_path=tmp_path / 'op.aimodel')
    actual = run_aimodel_sync(converted.asset, inputs,
        specialization_options=SpecializationOptions.cpu_only() if cpu_only else None).outputs
    for name, ref in zip(converted.prepared.normalized_graph.outputs, expected, strict=True):
        np.testing.assert_allclose(actual[name], ref, atol=atol, rtol=rtol)
    return actual, expected


@pytest.mark.parametrize('optimize', [False, True])
@pytest.mark.parametrize('op', ['floor', 'ceil', 'round', 'sign', 'arccosh', 'arcsinh', 'cosh', 'sinh', 'tan'])
def test_elementwise(tmp_path, op, optimize):
    import mlx.core as mx

    x = np.array([-3.5, -2.5, -0.5, -0.0, 0, 0.5, 1.5, 2.5, 3.5], np.float32)
    if op == 'arccosh':
        x = np.abs(x) + 1
    check(tmp_path, lambda x: getattr(mx, op)(x), {'x': x}, optimize=optimize)


@pytest.mark.parametrize('op', ['floor', 'ceil', 'round', 'sign'])
def test_rounding_special_values(tmp_path, op):
    import mlx.core as mx
    check(tmp_path, lambda x: getattr(mx, op)(x),
          {'x': np.array([-np.inf, -0.0, 0.0, np.inf, np.nan], np.float32)})


@pytest.mark.parametrize('op', ['floor', 'ceil', 'round', 'sign', 'arcsinh', 'cosh', 'sinh', 'tan'])
def test_elementwise_fp16(tmp_path, op):
    import mlx.core as mx
    check(tmp_path, lambda x: getattr(mx, op)(x),
          {'x': np.array([-1.5, -0.0, 0.5, 1.0], np.float16)}, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize('op', ['floor', 'ceil', 'round', 'sign'])
def test_integer_elementwise(tmp_path, op):
    import mlx.core as mx
    check(tmp_path, lambda x: getattr(mx, op)(x), {'x': np.array([-3, 0, 4], np.int32)}, atol=0, rtol=0)


@pytest.mark.parametrize('operator', [False, True])
@pytest.mark.parametrize('dtype', [np.int32, np.bool_])
def test_bitwise_invert(tmp_path, operator, dtype):
    import mlx.core as mx
    original = mx.bitwise_invert, mx.array.__invert__
    fn = (lambda x: ~x) if operator else (lambda x: mx.bitwise_invert(x))
    check(tmp_path, fn, {'x': np.array([-3, 0, 4], dtype)}, atol=0, rtol=0)
    assert (mx.bitwise_invert, mx.array.__invert__) == original


def test_capture_patches_restore_on_failure():
    import mlx.core as mx
    from mlx2coreai.from_mlx import capture_graph_from_mlx_function
    original = mx.contiguous, mx.bitwise_invert, mx.array.__invert__
    def fail(x):
        raise RuntimeError('capture failed')
    with pytest.raises(RuntimeError, match='capture failed'):
        capture_graph_from_mlx_function(None, {'x': np.ones(2, np.int32)}, fail)
    assert (mx.contiguous, mx.bitwise_invert, mx.array.__invert__) == original


@pytest.mark.parametrize('op', ['logical_and', 'logical_or', 'logical_not'])
def test_logical(tmp_path, op):
    import mlx.core as mx
    fn = (lambda x: mx.logical_not(x)) if op == 'logical_not' else (lambda x: getattr(mx, op)(x, x.T))
    check(tmp_path, fn, {'x': np.array([[0, -2], [1, 0]], np.float32)}, atol=0, rtol=0)


@pytest.mark.parametrize('axis', [0, 1, -1])
def test_gather_axis(tmp_path, axis):
    import mlx.core as mx
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    shape = list(x.shape)
    shape[axis] = 2
    indices = np.random.default_rng(4).integers(0, x.shape[axis], shape, dtype=np.int32)
    check(tmp_path, lambda x, i: mx.take_along_axis(x, i, axis=axis), {'x': x, 'i': indices}, atol=0, rtol=0)


@pytest.mark.parametrize('optimize', [False, True])
def test_atan2_edges(tmp_path, optimize):
    import mlx.core as mx
    values = np.array([-np.inf, -2, -0.0, 0.0, 2, np.inf, np.nan], np.float32)
    x, y = np.meshgrid(values, values)
    actual, expected = check(tmp_path, lambda x, y: mx.arctan2(y, x), {'x': x, 'y': y}, optimize=optimize)
    result = next(iter(actual.values()))
    zero = expected[0] == 0
    np.testing.assert_array_equal(np.signbit(result[zero]), np.signbit(expected[0][zero]))


def test_advanced_gather(tmp_path):
    import mlx.core as mx
    x = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    check(tmp_path, lambda x, i, j: x[i, :, j],
          {'x': x, 'i': np.array([0, -1], np.int32), 'j': np.array([1, -2], np.int32)}, atol=0, rtol=0)


@pytest.mark.parametrize('op', ['cumsum', 'cumprod', 'cummin', 'cummax'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('inclusive', [False, True])
@pytest.mark.parametrize('dtype', [np.float32, np.int32])
def test_scan(tmp_path, op, reverse, inclusive, dtype):
    import mlx.core as mx
    x = np.array([[[2, -1], [0, 3], [4, 2]]], dtype=dtype)
    check(tmp_path, lambda x: getattr(mx, op)(x, axis=1, reverse=reverse, inclusive=inclusive), {'x': x})


@pytest.mark.parametrize('axis', [0, 1, -1])
@pytest.mark.parametrize('op', ['sort', 'argsort'])
def test_sort(tmp_path, axis, op):
    import mlx.core as mx
    x = np.random.default_rng(41).permutation(60).astype(np.float32).reshape(3, 4, 5)
    check(tmp_path, lambda x: getattr(mx, op)(x, axis=axis), {'x': x}, atol=0, rtol=0)


@pytest.mark.parametrize('mode', ['edge', 'reflect'])
@pytest.mark.parametrize('optimize', [False, True])
def test_padding(tmp_path, mode, optimize):
    import mlx.core as mx
    x = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    check(tmp_path, lambda x: mx.pad(x, ((0, 0), (2, 1), (1, 2)), mode=mode), {'x': x}, optimize=optimize, atol=0, rtol=0)


@pytest.mark.parametrize('start,stop,step', [(None, None, -1), (3, None, -2), (2, 0, -1)])
def test_reverse_slice(tmp_path, start, stop, step):
    x = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    check(tmp_path, lambda x: x[:, start:stop:step, ::-1], {'x': x}, atol=0, rtol=0)


@pytest.mark.parametrize('mode', ['add', 'multiply', 'maximum', 'minimum', 'set'])
@pytest.mark.parametrize('multi_axis', [False, True])
def test_scatter(tmp_path, mode, multi_axis):
    import mlx.core as mx
    x = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    i = np.array([1, -1, 1] if mode != 'set' else [0, 1, 2], np.int32)
    def fn(x, i):
        index = (i, mx.array([1, 2, 1])) if multi_axis else (slice(None), i)
        if mode == 'set':
            x[index] = 2.0
            return x
        return getattr(x.at[index], mode)(2.0)
    check(tmp_path, fn, {'x': x, 'i': i}, atol=0, rtol=0)


@pytest.mark.parametrize('axis', [0, 1, -1])
def test_scatter_axis(tmp_path, axis):
    import mlx.core as mx
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    shape = list(x.shape)
    shape[axis] = 1
    i = np.full(shape, -1, np.int32)
    check(tmp_path, lambda x, i: mx.put_along_axis(x, i, mx.array(7), axis=axis), {'x': x, 'i': i}, atol=0, rtol=0)


@pytest.mark.parametrize('mode', ['lhs', 'rhs', 'both'])
@pytest.mark.parametrize('optimize', [False, True])
def test_gather_mm(tmp_path, mode, optimize):
    import mlx.core as mx
    rng = np.random.default_rng(80)
    a = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)
    b = rng.normal(size=(2, 3, 5, 7)).astype(np.float32)
    i = np.arange(6, dtype=np.int32)[::-1].reshape(2, 3).copy()
    fn = lambda a, b, i: mx.gather_mm(a, b, lhs_indices=i if mode != 'rhs' else None,
                                    rhs_indices=i if mode != 'lhs' else None)
    # The beta's fused gather_mm executes on the default GPU backend, not CPU-only.
    check(tmp_path, fn, {'a': a, 'b': b, 'i': i}, optimize=optimize, cpu_only=False)


@pytest.mark.parametrize('kind', ['max', 'avg'])
@pytest.mark.parametrize('rank', [1, 2, 3])
def test_overlapping_pool(tmp_path, kind, rank):
    import mlx.nn as nn
    x = np.random.default_rng(21).normal(size=(2, *([5] * rank), 3)).astype(np.float32)
    pool = getattr(nn, ('Max' if kind == 'max' else 'Avg') + f'Pool{rank}d')(3, stride=1)
    check(tmp_path, lambda x: pool(x), {'x': x})


@pytest.mark.parametrize('shape,strides,offset', [((3, 2), (4, 1), 2), ((3, 4), (0, 1), 0), ((2, 3), (3, -1), 2)])
def test_as_strided(tmp_path, shape, strides, offset):
    import mlx.core as mx
    check(tmp_path, lambda x: mx.as_strided(x, shape=shape, strides=strides, offset=offset),
          {'x': np.arange(32, dtype=np.float32)}, atol=0, rtol=0)


@pytest.mark.parametrize('mode,scale', [('nearest', 1.5), ('linear', 1.5), ('linear', 2.0)])
def test_upsample(tmp_path, mode, scale):
    import mlx.nn as nn
    upsample = nn.Upsample(scale, mode=mode)
    x = np.random.default_rng(2).normal(size=(2, 4, 5, 3)).astype(np.float32)
    check(tmp_path, lambda x: upsample(x), {'x': x})


def test_float_arange(tmp_path):
    import mlx.core as mx
    check(tmp_path, lambda x: x + mx.arange(-0.75, 1.0, 0.25, dtype=mx.float32),
          {'x': np.arange(7, dtype=np.float32)}, atol=0, rtol=0)


def test_unresolved_dynamic_strides_are_rejected(tmp_path):
    import mlx.core as mx
    make = lambda size: {'x': np.zeros((size,), np.float32)}
    config = ConversionConfig(capture_shapeless=True, dynamic_axes={'x': [0]}, dynamic_probe_inputs=make(7))
    with pytest.raises(ValueError, match='Dynamic AsStrided'):
        convert_mlx_to_coreai(lambda x: mx.as_strided(x, (x.shape[0] // 2, 2), (2, 1)), make(5), config=config,
                             output_path=tmp_path / 'unresolved.aimodel')


@pytest.mark.parametrize('op', ['partition', 'argpartition', 'topk'])
def test_partition_contract(tmp_path, op):
    import mlx.core as mx
    from coreai.runtime import SpecializationOptions
    x = np.random.default_rng(23).permutation(60).astype(np.float32).reshape(3, 4, 5)
    fn = lambda x: getattr(mx, op)(x, 2, axis=-1)
    converted = convert_mlx_to_coreai(fn, {'x': x}, output_path=tmp_path / 'partition.aimodel')
    result = next(iter(run_aimodel_sync(converted.asset, {'x': x},
        specialization_options=SpecializationOptions.cpu_only()).outputs.values()))
    if op == 'argpartition':
        result = np.take_along_axis(x, result, axis=-1)
    sorted_x = np.sort(x, axis=-1)
    if op == 'topk':
        np.testing.assert_array_equal(np.sort(result, axis=-1), sorted_x[..., -2:])
    else:
        np.testing.assert_array_equal(np.sort(result, axis=-1), sorted_x)
        np.testing.assert_array_equal(result[..., 2], sorted_x[..., 2])
        assert np.all(result[..., :2] <= result[..., 2:3])
        assert np.all(result[..., 3:] >= result[..., 2:3])


@pytest.mark.parametrize('kind', ['scan', 'sort', 'gather', 'scatter', 'pool_batch', 'gather_mm', 'gather_mm_singleton'])
def test_dynamic_extended_ops(tmp_path, kind):
    import mlx.core as mx
    import mlx.nn as nn
    from coreai.runtime import SpecializationOptions
    if kind == 'pool_batch':
        fn = lambda x: nn.AvgPool1d(3, stride=1)(x)
        make = lambda length: {'x': np.arange(length * 5 * 4, dtype=np.float32).reshape(length, 5, 4)}
        axes = {'x': [0]}
    elif kind == 'gather_mm':
        fn = lambda x: mx.gather_mm(x, mx.ones((3, 4, 2)), rhs_indices=mx.zeros((x.shape[0],), mx.int32))
        make = lambda length: {'x': np.arange(length * 2 * 4, dtype=np.float32).reshape(length, 2, 4)}
        axes = {'x': [0]}
    elif kind == 'gather_mm_singleton':
        fn = lambda x: mx.gather_mm(x, mx.ones((3, 4, 2)),
            rhs_indices=mx.zeros((1, x.shape[1], 2), mx.int32)).squeeze(-2)
        make = lambda length: {'x': np.arange(length * 4, dtype=np.float32).reshape(1, length, 1, 1, 4)}
        axes = {'x': [1]}
    else:
        make = lambda length: {'x': np.arange(2 * length * 4, dtype=np.float32).reshape(2, length, 4)}
        axes = {'x': [1]}
        fn = {
            'scan': lambda x: mx.cumsum(x, axis=1, reverse=True, inclusive=False),
            'sort': lambda x: mx.sort(-x, axis=1),
            'gather': lambda x: mx.take_along_axis(x, mx.full((2, 1, 4), -1, mx.int32), axis=1),
            'scatter': lambda x: x.at[mx.array([0, 1])].add(2),
        }[kind]
    config = ConversionConfig(capture_shapeless=True, dynamic_axes=axes, dynamic_probe_inputs=make(5))
    converted = convert_mlx_to_coreai(fn, make(3), config=config, output_path=tmp_path / 'dynamic.aimodel')
    for length in (2, 3, 5, 7):
        inputs = make(length)
        expected = np.asarray(fn(**{name: mx.array(value) for name, value in inputs.items()}))
        result = next(iter(run_aimodel_sync(converted.asset, inputs,
            specialization_options=None if kind.startswith('gather_mm') else SpecializationOptions.cpu_only()).outputs.values()))
        np.testing.assert_allclose(result, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('op', ['real', 'imag', 'conjugate', 'abs', 'add', 'multiply', 'real_view', 'bool_cast'])
def test_complex(tmp_path, op):
    import mlx.core as mx
    x = np.array([[1 + 2j, 3 - 4j], [-2 - 3j, 0 + 1j]], np.complex64)
    fn = {'add': lambda x: x + mx.array(2 + 3j, mx.complex64),
          'multiply': lambda x: x * x,
          'bool_cast': lambda x: x.astype(mx.bool_),
          'real_view': lambda x: x.view(mx.float32)}.get(op, lambda x: getattr(mx, op)(x))
    check(tmp_path, fn, {'x': x}, cpu_only=False)


@pytest.mark.parametrize('view', [False, True])
def test_real_to_complex(tmp_path, view):
    import mlx.core as mx
    fn = (lambda x: x.view(mx.complex64)) if view else (lambda x: x.astype(mx.complex64))
    check(tmp_path, fn, {'x': np.arange(12, dtype=np.float32).reshape(3, 4)}, cpu_only=False)


def test_complex_abs_special(tmp_path):
    import mlx.core as mx
    x = np.array([0j, complex(np.inf, 1), complex(np.inf, np.nan), complex(1e30, 1e30)], np.complex64)
    check(tmp_path, lambda x: mx.abs(x), {'x': x}, cpu_only=False)


@pytest.mark.parametrize('optimize', [False, True])
def test_scalar_take_gpu(tmp_path, optimize):
    import mlx.core as mx
    check(tmp_path, lambda x: mx.take(x, mx.array(2, mx.int32)),
          {'x': np.arange(4, dtype=np.float32)}, optimize=optimize, cpu_only=False)
