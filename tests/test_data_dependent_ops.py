"""Synthetic IR coverage where MLX has no public capture operation."""
import numpy as np
import pytest

from mlx2coreai.conversion import ConversionConfig, lower_graph_to_coreai
from mlx2coreai.ir import Graph, Node, TensorSpec
from mlx2coreai.runtime import run_aimodel_sync


@pytest.mark.parametrize('optimize', [False, True])
def test_nonzero_runtime_extent(tmp_path, optimize):
    graph = Graph([TensorSpec('x', (2, 3), 'fp32')], [Node('nonzero', ('x',), 'out')], ['out'])
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=optimize))
    asset = lowered.program.save_asset(tmp_path / 'nonzero.aimodel')
    for x in [np.zeros((2, 3), np.float32), np.eye(2, 3, dtype=np.float32), np.ones((2, 3), np.float32)]:
        actual = run_aimodel_sync(asset, {'x': x}).outputs['out']
        np.testing.assert_array_equal(actual, np.argwhere(x))


@pytest.mark.parametrize('all_false', [False, True])
@pytest.mark.parametrize('optimize', [False, True])
def test_masked_scatter(tmp_path, all_false, optimize):
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    mask = np.array([[False, not all_false, not all_false]])
    source = np.arange(4 if not all_false else 0, dtype=np.float32) + 10
    graph = Graph([TensorSpec('x', x.shape, 'fp32'), TensorSpec('mask', mask.shape, 'bool'), TensorSpec('source', source.shape, 'fp32')],
        [Node('masked_scatter', ('x', 'mask', 'source'), 'out')], ['out'])
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=optimize))
    asset = lowered.program.save_asset(tmp_path / 'masked.aimodel')
    result = run_aimodel_sync(asset, {'x': x, 'mask': mask, 'source': source}).outputs['out']
    expected = x.copy()
    expected[np.broadcast_to(mask, x.shape)] = source
    np.testing.assert_array_equal(result, expected)


def test_trunc_synthetic(tmp_path):
    x = np.array([-np.inf, -2.5, -0.25, -0.0, 0.0, 0.25, 2.5, np.inf, np.nan], np.float32)
    graph = Graph([TensorSpec('x', x.shape, 'fp32')], [Node('trunc', ('x',), 'out')], ['out'])
    lowered = lower_graph_to_coreai(graph)
    asset = lowered.program.save_asset(tmp_path / 'trunc.aimodel')
    result = run_aimodel_sync(asset, {'x': x}).outputs['out']
    np.testing.assert_allclose(result, np.trunc(x), atol=0, rtol=0)
    zeros = np.trunc(x) == 0
    np.testing.assert_array_equal(np.signbit(result[zeros]), np.signbit(np.trunc(x)[zeros]))


@pytest.mark.parametrize('op', ['complex', 'polar'])
@pytest.mark.parametrize('optimize', [False, True])
def test_complex_construction(tmp_path, op, optimize):
    graph = Graph([TensorSpec('a', (-1, 1), 'fp32'), TensorSpec('b', (1, 3), 'fp32')],
                  [Node(op, ('a', 'b'), 'out')], ['out'])
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=optimize))
    asset = lowered.program.save_asset(tmp_path / 'construct.aimodel')
    for length in (1, 2, 5):
        a = np.arange(length, dtype=np.float32).reshape(length, 1) + 0.5
        b = np.array([[-1, 0, 1.5]], np.float32)
        expected = a + 1j * b if op == 'complex' else a * np.exp(1j * b)
        actual = run_aimodel_sync(asset, {'a': a, 'b': b}).outputs['out']
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('dtype', ['int32', 'complex64'])
def test_complex_construction_requires_real_inputs(dtype):
    graph = Graph([TensorSpec('a', (2,), dtype), TensorSpec('b', (2,), 'fp32')],
                  [Node('complex', ('a', 'b'), 'out')], ['out'])
    with pytest.raises(ValueError, match='real floating-point'):
        lower_graph_to_coreai(graph)
