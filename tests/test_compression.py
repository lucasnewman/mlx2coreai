"""Numerical compression contracts independent of MLX's packed weight format."""
import numpy as np
import pytest
from coreai.runtime import SpecializationOptions

from mlx2coreai import ConversionConfig, convert_mlx_to_coreai
from mlx2coreai.conversion import lower_graph_to_coreai
from mlx2coreai.ir import Graph, Node, TensorSpec
from mlx2coreai.runtime import run_aimodel_sync
from mlx2coreai.dtypes import capture_numpy_dtype
from mlx2coreai._compression import PACKED_INTS, pack_integer_constant


def execute(tmp_path, nodes, inputs, outputs, optimize, *, constants=(), shapes=None, cpu_only=True):
    specs = [TensorSpec(name, (shapes or {}).get(name, value.shape), capture_numpy_dtype(value.dtype))
             for name, value in inputs.items() if name not in constants]
    nodes = [Node('constant', (), name, {'value': inputs[name]}) for name in constants] + nodes
    lowered = lower_graph_to_coreai(Graph(specs, nodes, outputs),
        config=ConversionConfig(optimize=optimize, external_weight_threshold=0))
    asset = lowered.program.save_asset(tmp_path / 'compression.aimodel')
    actual = run_aimodel_sync(asset, {k: v for k, v in inputs.items() if k not in constants},
        specialization_options=SpecializationOptions.cpu_only() if cpu_only else None).outputs
    return actual


@pytest.mark.parametrize('dtype', [np.int8, np.uint8])
@pytest.mark.parametrize('axis', [None, 0, -1])
@pytest.mark.parametrize('optimize', [False, True])
def test_affine_roundtrip(tmp_path, dtype, axis, optimize):
    x = np.array([[-1000, -2.5, -1.5], [0.5, 1.5, 2.5], [4, 1000, 2000]], np.float32)
    scale = np.array(0.5, np.float32) if axis is None else np.array([0.5, 1, 2], np.float32)
    offset1 = np.full(scale.shape, 3, dtype)
    offset2 = np.full(scale.shape, -0.25, np.float32)
    nodes = [Node('affine_quantize', ('x', 'scale', 'offset1', 'offset2'), 'q', {'axis': axis}),
             Node('affine_dequantize', ('q', 'scale', 'offset1', 'offset2'), 'out', {'axis': axis})]
    actual = execute(tmp_path, nodes, locals_to_inputs(x, scale, offset1, offset2), ['q', 'out'], optimize)
    shape = () if axis is None else ((3, 1) if axis == 0 else (1, 3))
    scale, offset1, offset2 = [value.reshape(shape) for value in (scale, offset1, offset2)]
    limits = np.iinfo(dtype)
    expected = np.clip(np.rint((x - offset2) / scale) + offset1, limits.min, limits.max).astype(dtype)
    assert actual['q'].dtype == dtype
    np.testing.assert_array_equal(actual['q'], expected)
    np.testing.assert_allclose(actual['out'], (expected.astype(np.float32) - offset1) * scale + offset2)


def locals_to_inputs(x, scale, offset1, offset2):
    return {'x': x, 'scale': scale, 'offset1': offset1, 'offset2': offset2}


@pytest.mark.parametrize('dtype', [np.int8, np.uint8])
@pytest.mark.parametrize('optimize', [False, True])
def test_blockwise(tmp_path, dtype, optimize):
    x = np.arange(24, dtype=dtype).reshape(4, 6)
    scale = np.array([[0.5, 1, 2], [3, 4, 5]], np.float32)
    offset1 = np.full((2, 3), 4, dtype)
    offset2 = np.arange(6, dtype=np.float32).reshape(2, 3)
    nodes = [Node('blockwise_shift_scale', ('x', 'scale', 'offset1', 'offset2'), 'out')]
    actual = execute(tmp_path, nodes, locals_to_inputs(x, scale, offset1, offset2), ['out'], optimize,
                     constants=('scale', 'offset1', 'offset2'))
    scale, offset1, offset2 = [value.repeat(2, 0).repeat(2, 1) for value in (scale, offset1, offset2)]
    np.testing.assert_allclose(actual['out'], (x.astype(np.float32) - offset1) * scale + offset2)


@pytest.mark.parametrize('vector_size', [1, 3])
@pytest.mark.parametrize('optimize', [False, True])
def test_lut(tmp_path, vector_size, optimize):
    indices = np.array([[1, 4], [255, 2], [128, 0], [3, 254]], np.uint8)
    lut = np.arange(2 * 256 * vector_size, dtype=np.float32).reshape(2, 1, 256, vector_size) / 4
    node = Node('lut_to_dense', ('indices', 'lut'), 'out', {'axis': -1})
    actual = execute(tmp_path, [node], {'indices': indices, 'lut': lut}, ['out'], optimize, constants=('lut',))
    expected = np.stack([lut[row // 2, 0, indices[row]].reshape(-1) for row in range(4)])
    np.testing.assert_array_equal(actual['out'], expected)


@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('optimize', [False, True])
def test_sparse(tmp_path, empty, optimize):
    mask = np.array([[False, True, False], [True, False, True]]) & (not empty)
    values = np.array([] if empty else [10, 20, 30], np.float32)
    actual = execute(tmp_path, [Node('constant', (), 'mask', {'value': mask, 'dtype': 'uint1'}),
        Node('sparse_to_dense', ('values', 'mask'), 'out')], {'values': values}, ['out'], optimize)
    expected = np.zeros(mask.shape, np.float32)
    expected[mask] = values
    np.testing.assert_array_equal(actual['out'], expected)


@pytest.mark.parametrize('dtype', [np.int8, np.uint8])
def test_byte_capture(tmp_path, dtype):
    import mlx.core as mx
    x = np.array([0, 1, 2, 127], dtype)
    converted = convert_mlx_to_coreai(lambda x: mx.bitwise_xor(x, mx.array(3, x.dtype)), {'x': x},
        output_path=tmp_path / 'byte.aimodel')
    actual = next(iter(run_aimodel_sync(converted.asset, {'x': x},
        specialization_options=SpecializationOptions.cpu_only()).outputs.values()))
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, x ^ np.array(3, dtype))


@pytest.mark.parametrize('dtype', ['uint1', 'uint2', 'uint3', 'uint4', 'uint6'])
@pytest.mark.parametrize('optimize', [False, True])
def test_packed_lut(tmp_path, dtype, optimize):
    width = PACKED_INTS[dtype]
    count = 2 ** width
    indices = (np.arange(10) % count).astype(np.uint8).reshape(2, 5)
    indices[-1, -1] = count - 1
    lut = np.linspace(-4, 4, 2 * count, dtype=np.float32).reshape(2, 1, count, 1)
    nodes = [Node('constant', (), 'indices', {'value': indices, 'dtype': dtype}),
             Node('lut_to_dense', ('indices', 'lut'), 'out', {'axis': 0})]
    actual = execute(tmp_path, nodes, {'lut': lut}, ['out'], optimize, constants=('lut',))
    expected = np.stack([lut[row, 0, indices[row], 0] for row in range(2)])
    np.testing.assert_array_equal(actual['out'], expected)


@pytest.mark.parametrize('dtype', ['int2', 'int4', 'uint2', 'uint4'])
@pytest.mark.parametrize('optimize', [False, True])
def test_packed_blockwise(tmp_path, dtype, optimize):
    bits = PACKED_INTS[dtype]
    signed = dtype.startswith('int')
    lower = -(2 ** (bits - 1)) if signed else 0
    values = (np.arange(10) % 2 ** bits + lower).astype(np.int8 if signed else np.uint8).reshape(2, 5)
    offset = np.array([[lower]], values.dtype)
    scale = np.array([[0.25]], np.float32)
    bias = np.array([[-2]], np.float32)
    nodes = [Node('constant', (), 'values', {'value': values, 'dtype': dtype}),
             Node('constant', (), 'offset', {'value': offset, 'dtype': dtype}),
             Node('blockwise_shift_scale', ('values', 'scale', 'offset', 'bias'), 'out')]
    actual = execute(tmp_path, nodes, {'scale': scale, 'bias': bias}, ['out'], optimize, constants=('scale', 'bias'))
    np.testing.assert_array_equal(actual['out'], (values.astype(np.float32) - lower) * scale + bias)


@pytest.mark.parametrize('dtype', list(PACKED_INTS))
def test_packing_contract(dtype):
    bits = PACKED_INTS[dtype]
    lower = -(2 ** (bits - 1)) if dtype.startswith('int') else 0
    upper = lower + 2 ** bits - 1
    values = np.array([lower, 0, upper, 1, lower])
    packed = pack_integer_constant(values, dtype)
    assert packed.nbytes == (values.size * bits + 7) // 8
    bitstream = int.from_bytes(packed.tobytes(), 'little')
    for index, value in enumerate(values):
        assert (bitstream >> (index * bits)) & (2 ** bits - 1) == int(value) & (2 ** bits - 1)
    with pytest.raises(ValueError, match='require integers'):
        pack_integer_constant(np.array([upper + 1]), dtype)
    with pytest.raises(ValueError, match='require integers'):
        pack_integer_constant(np.array([0.5]), dtype)


@pytest.mark.parametrize('dtype', ['int4', 'uint4'])
@pytest.mark.parametrize('optimize', [False, True])
def test_lowbit_affine(tmp_path, dtype, optimize):
    bits = PACKED_INTS[dtype]
    lower = -(2 ** (bits - 1)) if dtype.startswith('int') else 0
    upper = lower + 2 ** bits - 1
    x = np.array([-100, -3.5, -2.5, -0.5, 0.5, 1.5, 2.5, 100], np.float32)
    nodes = [Node('constant', (), 'offset', {'value': np.array(1), 'dtype': dtype}),
        Node('affine_quantize', ('x', 'scale', 'offset', 'bias'), 'q'),
        Node('cast', ('q',), 'numeric_q', {'dtype': 'fp32'}),
        Node('affine_dequantize', ('q', 'scale', 'offset', 'bias'), 'out')]
    actual = execute(tmp_path, nodes, {'x': x, 'scale': np.array(1, np.float32), 'bias': np.array(0, np.float32)},
                     ['numeric_q', 'out'], optimize, cpu_only=False)
    expected = np.clip(np.rint(x) + 1, lower, upper)
    np.testing.assert_array_equal(actual['numeric_q'], expected)
    np.testing.assert_array_equal(actual['out'], expected - 1)


@pytest.mark.parametrize('floating', [np.float16, np.float32])
@pytest.mark.parametrize('optimize', [False, True])
def test_dynamic_affine(tmp_path, floating, optimize):
    dtype = capture_numpy_dtype(np.dtype(floating))
    specs = [TensorSpec('x', (-1, 3), dtype), TensorSpec('scale', (3,), dtype),
             TensorSpec('offset1', (3,), 'int8'), TensorSpec('offset2', (3,), dtype)]
    nodes = [Node('affine_quantize', ('x', 'scale', 'offset1', 'offset2'), 'q', {'axis': -1}),
             Node('affine_dequantize', ('q', 'scale', 'offset1', 'offset2'), 'out', {'axis': -1})]
    lowered = lower_graph_to_coreai(Graph(specs, nodes, ['q', 'out']), config=ConversionConfig(optimize=optimize))
    asset = lowered.program.save_asset(tmp_path / 'dynamic_affine.aimodel')
    for length in (1, 3, 7):
        x = (np.arange(length * 3).reshape(length, 3) / 2 - 5).astype(floating)
        scale = np.array([0.5, 1, 2], floating)
        offset1 = np.array([-3, 0, 3], np.int8)
        offset2 = np.array([0, 0.25, -1], floating)
        actual = run_aimodel_sync(asset, locals_to_inputs(x, scale, offset1, offset2),
            specialization_options=SpecializationOptions.cpu_only()).outputs
        expected = np.clip(np.rint((x - offset2) / scale) + offset1, -128, 127).astype(np.int8)
        np.testing.assert_array_equal(actual['q'], expected)
        np.testing.assert_array_equal(actual['out'], (expected.astype(floating) - offset1) * scale + offset2)


def test_packed_public_inputs_rejected():
    graph = Graph([TensorSpec('packed', (4,), 'uint4')], [], ['packed'])
    with pytest.raises(ValueError, match='not public runtime inputs'):
        lower_graph_to_coreai(graph)


def test_packed_public_outputs_rejected():
    graph = Graph([], [Node('constant', (), 'packed', {'value': np.array([1, 2]), 'dtype': 'uint4'})], ['packed'])
    with pytest.raises(ValueError, match='not public runtime outputs'):
        lower_graph_to_coreai(graph)


def test_affine_parameter_shapes_rejected():
    specs = [TensorSpec('x', (2, 3), 'fp32'), TensorSpec('scale', (2,), 'fp32'),
             TensorSpec('offset', (2,), 'int8'), TensorSpec('bias', (2,), 'fp32')]
    graph = Graph(specs, [Node('affine_quantize', ('x', 'scale', 'offset', 'bias'), 'q', {'axis': 1})], ['q'])
    with pytest.raises(ValueError, match='channel dimension'):
        lower_graph_to_coreai(graph)
