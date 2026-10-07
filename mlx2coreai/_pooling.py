"""Adaptive pooling with runtime-computed boundaries and channels-last layout."""
import numpy as np
from coreai._compiler.dialects import coreai

from .ir import TensorType


def infer_adaptive_average(node, inputs):
    if len(inputs) != 1:
        raise ValueError('adaptive_avg_pool requires one tensor.')
    if inputs[0].dtype not in {'fp16', 'bf16', 'fp32', 'fp64'}:
        raise ValueError('adaptive_avg_pool requires floating-point input.')
    shape = inputs[0].shape
    output = node.attrs.get('output_size')
    if not isinstance(output, (list, tuple)) or not 1 <= len(output) <= 3:
        raise ValueError('adaptive_avg_pool output_size must contain 1-3 spatial sizes.')
    if any(size is not None and (not isinstance(size, int) or size <= 0) for size in output):
        raise ValueError('Adaptive output sizes must be positive integers or None.')
    if shape is None:
        return TensorType(None, inputs[0].dtype)
    if len(shape) != len(output) + 2:
        raise ValueError('adaptive_avg_pool expects batch, spatial dimensions, channels.')
    return TensorType((shape[0], *(dim if size is None else size for dim, size in zip(shape[1:-1], output)), shape[-1]), inputs[0].dtype)


def emit_adaptive_average(context, node):
    from .lower_to_coreai import _dim_1d_from_value, _mixed_shape_operand

    infer_adaptive_average(node, [context.inferred[name] for name in node.inputs])
    x = context.env[node.inputs[0]]
    output = node.attrs['output_size']
    dims = [int(size) if int(size) >= 0 else _dim_1d_from_value(x, axis)
            for axis, size in enumerate(x.type.shape)]
    axes = [axis for axis, size in enumerate(output, 1) if size is not None]
    if not axes:
        return x

    def boundary(dim, numerator, denominator, ceil=False):
        extra = denominator - 1 if ceil else 0
        if isinstance(dim, int):
            return (dim * numerator + extra) // denominator
        value = coreai.broadcasting_mul(dim, np.int32(numerator))
        value = coreai.broadcasting_add(value, np.int32(extra))
        return coreai.broadcasting_floor_divide(value, np.int32(denominator))

    def build(axis, starts, ends):
        if axis > len(output):
            window = coreai.slice_(x, _mixed_shape_operand(starts), _mixed_shape_operand(ends),
                                   np.ones(x.type.rank, np.int32))
            return coreai.reduce_mean(window, np.array(axes, np.int32))
        size = output[axis - 1]
        if size is None:
            return build(axis + 1, starts, ends)
        cells = []
        for index in range(size):
            begin, end = list(starts), list(ends)
            begin[axis] = boundary(dims[axis], index, size)
            end[axis] = boundary(dims[axis], index + 1, size, ceil=True)
            cells.append(build(axis + 1, begin, end))
        return cells[0] if len(cells) == 1 else coreai.concat(axis, cells)

    return build(1, [0] * x.type.rank, dims)
