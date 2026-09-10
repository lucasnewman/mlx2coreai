"""Explicit subgraph IR tests; MLX does not capture Python control flow."""
import json

import numpy as np
import pytest
from coreai.runtime import SpecializationOptions

from mlx2coreai.conversion import ConversionConfig, lower_graph_to_coreai
from mlx2coreai.ir import Graph, Node, TensorSpec
from mlx2coreai.runtime import run_aimodel_sync


def conditional_graph(shape=(3,)):
    inputs = [TensorSpec('x', shape, 'fp32')]
    then = Graph(inputs, [Node('add', ('x', 'x'), 'y')], ['y', 'x'])
    otherwise = Graph(inputs, [Node('negative', ('x',), 'y')], ['y', 'x'])
    return Graph([TensorSpec('predicate', (), 'bool'), *inputs],
        [Node('cond', ('predicate', 'x'), outputs=('out', 'unchanged'), attrs={'then': then, 'else': otherwise})],
        ['out', 'unchanged'])


def loop_graph():
    inputs = [TensorSpec('i', (), 'int32'), TensorSpec('x', (3,), 'fp32'), TensorSpec('limit', (), 'int32')]
    condition = Graph(inputs, [Node('less', ('i', 'limit'), 'test')], ['test'])
    branches = conditional_graph()
    body = Graph(inputs, [
        Node('constant', (), 'one', {'value': np.array(1, np.int32)}),
        Node('add', ('i', 'one'), 'next_i'),
        Node('less', ('next_i', 'limit'), 'test'),
        Node('cond', ('test', 'x'), outputs=('next_x', 'old_x'), attrs=branches.nodes[0].attrs),
    ], ['next_i', 'next_x'])
    return Graph(inputs, [Node('while_loop', ('i', 'x', 'limit'), outputs=('out_i', 'out'),
        attrs={'condition': condition, 'body': body, 'carried_count': 2})], ['out_i', 'out'])


@pytest.mark.parametrize('optimize', [False, True])
@pytest.mark.parametrize('dynamic', [False, True])
def test_cond_runtime(tmp_path, optimize, dynamic):
    graph = conditional_graph((-1,) if dynamic else (3,))
    json.dumps(graph.to_dict())
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=optimize))
    asset = lowered.program.save_asset(tmp_path / 'cond.aimodel')
    for size in ([1, 3, 7] if dynamic else [3]):
        x = np.arange(size, dtype=np.float32)
        for predicate in [False, True]:
            outputs = run_aimodel_sync(asset, {'predicate': np.array(predicate), 'x': x},
                specialization_options=SpecializationOptions.cpu_only()).outputs
            np.testing.assert_array_equal(outputs['out'], x * (2 if predicate else -1))
            np.testing.assert_array_equal(outputs['unchanged'], x)


@pytest.mark.parametrize('optimize', [False, True])
def test_loop_with_nested_cond(tmp_path, optimize):
    graph = loop_graph()
    json.dumps(graph.to_dict())
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=optimize))
    asset = lowered.program.save_asset(tmp_path / 'loop.aimodel')
    x = np.arange(3, dtype=np.float32)
    for limit in [0, 1, 4, 7]:
        outputs = run_aimodel_sync(asset, {'i': np.array(0, np.int32), 'x': x, 'limit': np.array(limit, np.int32)},
            specialization_options=SpecializationOptions.cpu_only()).outputs
        np.testing.assert_array_equal(outputs['out_i'], limit)
        np.testing.assert_array_equal(outputs['out'], x if limit == 0 else -x * 2 ** (limit - 1))


def test_branch_types_rejected():
    graph = conditional_graph()
    graph.nodes[0].attrs['else'].inputs[0] = TensorSpec('x', (4,), 'fp32')
    with pytest.raises(ValueError, match='input types must match'):
        lower_graph_to_coreai(graph)


def test_loop_carried_type_rejected():
    graph = loop_graph()
    graph.nodes[0].attrs['body'].outputs = ['next_i']
    with pytest.raises(ValueError, match='preserve carried tensor types'):
        lower_graph_to_coreai(graph)


@pytest.mark.parametrize('optimize', [False, True])
def test_composite_in_branch(tmp_path, optimize):
    graph = conditional_graph()
    graph.nodes[0].attrs['then'] = Graph([TensorSpec('x', (3,), 'fp32')], [
        Node('constant', (), 'scale', {'value': np.array([1, 2, 3], np.float32)}),
        Node('rmsnorm', ('x', 'scale'), 'normalized', {'axes': [-1], 'eps': 1e-5}),
    ], ['normalized', 'x'])
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=optimize, external_weight_threshold=0))
    asset = lowered.program.save_asset(tmp_path / 'nested_composite.aimodel')
    x = np.array([1, 2, 3], np.float32)
    outputs = run_aimodel_sync(asset, {'predicate': np.array(True), 'x': x},
        specialization_options=SpecializationOptions.cpu_only()).outputs
    np.testing.assert_allclose(outputs['out'], x * x / np.sqrt(np.mean(x * x) + 1e-5), rtol=1e-5)
