"""Explicit compression IR; MLX's packed Quantize is a different operation."""
import numpy as np
from coreai._compiler.dialects import coreai

from .ir import TensorType


PACKED_INTS = {'int2': 2, 'int4': 4, 'uint1': 1, 'uint2': 2, 'uint3': 3, 'uint4': 4, 'uint6': 6}


def pack_integer_constant(value, dtype):
    """Row-major, least-significant bit first, with zero trailing padding."""
    values = np.asarray(value)
    width = PACKED_INTS[dtype]
    signed = dtype.startswith('int')
    lower = -(1 << (width - 1)) if signed else 0
    upper = (1 << (width - int(signed))) - 1
    if values.dtype.kind not in 'biu' or (values.size and (values.min() < lower or values.max() > upper)):
        raise ValueError(f'{dtype} constants require integers in [{lower}, {upper}].')
    encoded = values.astype(np.uint8).reshape(-1)
    bits = ((encoded[:, None] >> np.arange(width, dtype=np.uint8)) & 1).reshape(-1)
    return np.packbits(bits, bitorder='little')


def infer_affine(node, inputs):
    if len(inputs) != 4:
        raise ValueError(f'{node.op} requires data, scale, offset1, offset2.')
    if node.op == 'affine_quantize' and inputs[2].dtype not in {'int4', 'uint4', 'int8', 'uint8'}:
        raise ValueError('affine_quantize supports int4/uint4/int8/uint8 output storage.')
    data, scale, offset1, offset2 = inputs
    quantize = node.op == 'affine_quantize'
    if data.dtype != (offset2.dtype if quantize else offset1.dtype) or scale.dtype != offset2.dtype:
        raise ValueError('Affine data/offset and scale/bias dtypes must match their respective domains.')
    if data.shape is not None and scale.shape is not None:
        if not data.shape or scale.shape != offset1.shape or scale.shape != offset2.shape:
            raise ValueError('Affine parameters must have identical shapes and data must have rank >= 1.')
        if node.op == 'blockwise_shift_scale':
            if len(scale.shape) != len(data.shape) or any(
                size == 0 or (dim >= 0 and size > 0 and dim % size != 0)
                for dim, size in zip(data.shape, scale.shape)
            ):
                raise ValueError('Blockwise parameter dimensions must divide data dimensions.')
        elif node.attrs.get('axis') is None:
            if scale.shape:
                raise ValueError('Per-tensor affine parameters must be scalars.')
        else:
            axis = int(node.attrs['axis'])
            if not -len(data.shape) <= axis < len(data.shape):
                raise ValueError('Affine quantization axis is out of range.')
            dim = data.shape[axis]
            if len(scale.shape) > 1 or (scale.shape and dim >= 0 and scale.shape[0] >= 0 and scale.shape[0] != dim):
                raise ValueError('Per-axis affine parameters must match the channel dimension.')
    return TensorType(inputs[0].shape, inputs[2 if node.op == 'affine_quantize' else 3].dtype)


def infer_sparse(node, inputs):
    if len(inputs) != 2 or inputs[1].dtype != 'uint1':
        raise ValueError('sparse_to_dense requires values and a packed uint1 mask.')
    return TensorType(inputs[1].shape, inputs[0].dtype)


def infer_lut(node, inputs):
    if len(inputs) != 2:
        raise ValueError('lut_to_dense requires indices and a lookup table.')
    shape, table = inputs[0].shape, inputs[1].shape
    if shape is None or table is None:
        return TensorType(None, inputs[1].dtype)
    if len(table) != len(shape) + 2:
        raise ValueError('LUT rank must equal indices rank plus two.')
    axis = int(node.attrs.get('axis', 0))
    if not -len(shape) <= axis < len(shape):
        raise ValueError('LUT expansion axis is out of range.')
    result = list(shape)
    result[axis] = result[axis] * table[-1] if result[axis] >= 0 else -1
    return TensorType(tuple(result), inputs[1].dtype)


def emit_affine(context, node, *, quantize=False, blockwise=False):
    infer_affine(node, [context.inferred[name] for name in node.inputs])
    if len(node.inputs) != 4:
        raise ValueError(f'{node.op} requires data, scale, offset1, offset2.')
    data, scale, offset1, offset2 = [context.env[name] for name in node.inputs]
    if blockwise:
        return coreai.blockwise_shift_scale(data, scale, offset1, offset2)
    rank = data.type.rank
    axis = node.attrs.get('axis')
    if axis is None:
        axis_value = np.arange(rank, dtype=np.int32)
    else:
        axis = int(axis)
        if not -rank <= axis < rank:
            raise ValueError('Affine quantization axis is out of range.')
        axis_value = np.array(axis % rank, np.int32)
    shape = [1] * rank
    if axis is not None:
        shape[axis] = -1

    def expanded(value):
        return coreai.reshape(value, shape) if value.type.rank else value

    if quantize:
        # The beta native kernel rounds after adding offset1, changing ties
        # for odd zero points. Its FP16 per-axis offset broadcasting is also
        # incorrect. Normalize explicitly; leave saturation/packing native.
        data = coreai.round_(coreai.broadcasting_divide(coreai.broadcasting_sub(data, expanded(offset2)), expanded(scale)))
        data = coreai.broadcasting_add(coreai.cast(data, np.float32), expanded(coreai.cast(offset1, np.float32)))
        zero = context._constant(node.output + '_zero_point', np.array(0, np.int8),
                                 dtype=context.inferred[node.inputs[2]].dtype)
        return coreai.quantize(data, np.float32(1), zero, np.float32(0), np.arange(rank, dtype=np.int32))
    if axis is None:
        return coreai.dequantize(data, scale, offset1, offset2, axis_value)
    dtype = offset2.type.element_type
    centered = coreai.broadcasting_sub(coreai.cast(data, dtype), expanded(coreai.cast(offset1, dtype)))
    return coreai.broadcasting_add(coreai.broadcasting_mul(centered, expanded(scale)), expanded(offset2))


def emit_lut(context, node):
    indices, table = [context.env[name] for name in node.inputs]
    infer_lut(node, [context.inferred[name] for name in node.inputs])
    axis = int(node.attrs.get('axis', 0)) % indices.type.rank
    return coreai.lut_to_dense(indices, table, np.array(axis, np.int16))


def emit_sparse(context, node):
    infer_sparse(node, [context.inferred[name] for name in node.inputs])
    values, mask = [context.env[name] for name in node.inputs]
    return coreai.sparse_with_bitmask_to_dense(coreai.build_sparse_with_bitmask(values, mask))
