import numpy as np
import pytest

from coreai.runtime import SpecializationOptions
from mlx2coreai import ConversionConfig, convert_mlx_to_coreai
from mlx2coreai.runtime import run_aimodel_sync
from mlx2coreai.conversion import lower_graph_to_coreai
from mlx2coreai.ir import Graph, Node, TensorSpec


@pytest.mark.parametrize('dimensions', [1, 2, 3])
@pytest.mark.parametrize('average', [False, True])
@pytest.mark.parametrize('optimize', [False, True])
def test_dynamic_pooling(tmp_path, dimensions, average, optimize):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.nn.layers import pooling

    original = pooling._sliding_windows
    model = getattr(nn, ('Avg' if average else 'Max') + f'Pool{dimensions}d')(3, stride=2, padding=1)
    rng = np.random.default_rng(4)
    def inputs(size, batch=2):
        return {'x': rng.normal(size=(batch, *[size + axis for axis in range(dimensions)], 2)).astype(np.float32)}
    converted = convert_mlx_to_coreai(lambda x: model(x), inputs(5),
        config=ConversionConfig(optimize=optimize, capture_shapeless=True,
            dynamic_axes={'x': list(range(dimensions + 1))}, dynamic_probe_inputs=inputs(8, 3)),
        output_path=tmp_path / 'pool.aimodel')
    assert pooling._sliding_windows is original
    for size in [3, 6, 9, 12]:
        data = inputs(size, 1 if size % 2 else 3)
        expected = np.asarray(model(mx.array(data['x'])))
        actual = next(iter(run_aimodel_sync(converted.asset, data,
            specialization_options=SpecializationOptions.cpu_only()).outputs.values()))
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_pooling_patch_restored_on_error():
    import mlx.core as mx
    from mlx.nn.layers import pooling
    from mlx2coreai._capture_compat import export_compatibility

    original = pooling._sliding_windows
    with pytest.raises(RuntimeError, match='capture failed'):
        with export_compatibility(mx):
            assert pooling._sliding_windows is not original
            raise RuntimeError('capture failed')
    assert pooling._sliding_windows is original


@pytest.mark.parametrize('kernel,stride', [((2, 2), (2, 2)), ((2, 3), (1, 2)), ((1, 1), (1, 1))])
@pytest.mark.parametrize('average', [False, True])
def test_pooling_window_variants_gpu(tmp_path, kernel, stride, average):
    import mlx.core as mx
    import mlx.nn as nn
    model = (nn.AvgPool2d if average else nn.MaxPool2d)(kernel, stride=stride)
    rng = np.random.default_rng(18)
    def inputs(size):
        return {'x': rng.normal(size=(2, size, size + 1, 2)).astype(np.float16)}
    converted = convert_mlx_to_coreai(lambda x: model(x), inputs(5),
        config=ConversionConfig(capture_shapeless=True, dynamic_axes={'x': [1, 2]}, dynamic_probe_inputs=inputs(8)),
        output_path=tmp_path / 'pool_variant.aimodel')
    for size in (4, 7, 10):
        data = inputs(size)
        expected = np.asarray(model(mx.array(data['x'])))
        actual = next(iter(run_aimodel_sync(converted.asset, data).outputs.values()))
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize('output_size', [(2,), (2, 3), (None, 2), (2, 1, 2), (None, None)])
@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('optimize', [False, True])
def test_adaptive_average_ir(tmp_path, output_size, dynamic, optimize):
    dimensions = len(output_size)
    shape = (2, *[5 + axis for axis in range(dimensions)], 2)
    spec = TensorSpec('x', (-1, *[-1] * dimensions, 2) if dynamic else shape, 'fp32')
    graph = Graph([spec], [Node('adaptive_avg_pool', ('x',), 'out', {'output_size': output_size})], ['out'])
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=optimize))
    asset = lowered.program.save_asset(tmp_path / 'adaptive.aimodel')
    for size in ([3, 5, 8] if dynamic else [5]):
        x = np.random.default_rng(size).normal(size=(2, *[size + axis for axis in range(dimensions)], 2)).astype(np.float32)
        spatial = tuple(dim if count is None else count for dim, count in zip(x.shape[1:-1], output_size))
        expected = np.empty((2, *spatial, 2), np.float32)
        for index in np.ndindex(spatial):
            slices = [slice(None)]
            for axis, (position, count) in enumerate(zip(index, spatial), 1):
                extent = x.shape[axis]
                slices.append(slice(position * extent // count, ((position + 1) * extent + count - 1) // count))
            slices.append(slice(None))
            expected[(slice(None), *index, slice(None))] = x[tuple(slices)].mean(axis=tuple(range(1, dimensions + 1)))
        actual = run_aimodel_sync(asset, {'x': x}, specialization_options=SpecializationOptions.cpu_only()).outputs['out']
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_adaptive_requires_float():
    graph = Graph([TensorSpec('x', (1, 5, 2), 'int32')],
                  [Node('adaptive_avg_pool', ('x',), 'out', {'output_size': [2]})], ['out'])
    with pytest.raises(ValueError, match='floating-point input'):
        lower_graph_to_coreai(graph)
