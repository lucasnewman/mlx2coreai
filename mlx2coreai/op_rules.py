"""Operation definitions: source aliases, decoding, lowering, and optional typing.

Backend methods are resolved when emitting, so importing the registry does not
need a CoreAI compiler context. Numerical test references remain independent.
"""
from __future__ import annotations
import math
from coreai._compiler.dialects import coreai
from dataclasses import dataclass
from functools import partial
from typing import Callable
from . import _op_attrs as attrs, _type_inference as types
from . import _elementwise as elementwise
from . import _control_flow as control_flow


@dataclass(frozen=True)
class OpRule:
    key: str
    aliases: tuple[str, ...]
    lower: Callable
    decode: Callable | None = None
    infer: Callable | None = None


def _lower(method, *args, **kwargs):
    def emit(context, node):
        return getattr(context, method)(node, *args, **kwargs)
    return emit


def less_equal(lhs, rhs):
    return coreai.broadcasting_or(coreai.broadcasting_greater(rhs, lhs), coreai.broadcasting_equal(lhs, rhs))


_RULES = (
    OpRule('cond', ('cond',), _lower('_emit_cond'), None, control_flow.infer_cond),
    OpRule('while_loop', ('while_loop',), _lower('_emit_while'), None, control_flow.infer_while),
    OpRule('matmul', ('matmul',), _lower('_emit_matmul'), None, types.infer_matmul),
    OpRule('add', ('add',), _lower('_emit_binary', fn=coreai.broadcasting_add), None, types.infer_binary),
    OpRule('maximum', ('maximum',), _lower('_emit_binary', fn=coreai.broadcasting_maximum), None, types.infer_binary),
    OpRule('minimum', ('minimum',), _lower('_emit_binary', fn=coreai.broadcasting_minimum), None, types.infer_binary),
    OpRule('sub', ('subtract', 'sub'), _lower('_emit_binary', fn=coreai.broadcasting_sub), None, types.infer_binary),
    OpRule('mul', ('multiply', 'mul'), _lower('_emit_binary', fn=coreai.broadcasting_mul), None, types.infer_binary),
    OpRule('real_div', ('divide', 'real_div'), _lower('_emit_binary', fn=coreai.broadcasting_divide), None, types.infer_binary),
    OpRule('pow', ('power', 'pow'), _lower('_emit_binary', fn=coreai.broadcasting_pow), None, types.infer_binary),
    OpRule('inverse', ('reciprocal', 'inverse'), _lower('_emit_inverse'), None, None),
    OpRule('mod', ('remainder', 'mod'), _lower('_emit_binary', fn=coreai.broadcasting_modulo), None, types.infer_binary),
    OpRule('reduce', ('reduce',), _lower('_lower_generic_reduce'), attrs.decode_reduce, types.infer_reduce),
    OpRule('reduce_sum', ('sum', 'reduce_sum'), _lower('_emit_reduction', fn=coreai.reduce_sum), attrs.decode_reduction, types.infer_reduction),
    OpRule('reduce_mean', ('mean', 'reduce_mean'), _lower('_emit_reduction', fn=coreai.reduce_mean), attrs.decode_reduction, types.infer_reduction),
    OpRule('reduce_min', ('min', 'reduce_min'), _lower('_emit_reduction', fn=coreai.reduce_min), attrs.decode_reduction, types.infer_reduction),
    OpRule('reduce_max', ('max', 'reduce_max'), _lower('_emit_reduction', fn=coreai.reduce_max), attrs.decode_reduction, types.infer_reduction),
    OpRule('reduce_prod', ('prod', 'reduce_prod'), _lower('_emit_reduction', fn=coreai.reduce_product), attrs.decode_reduction, types.infer_reduction),
    OpRule('reduce_argmax', ('argmax', 'reduce_argmax'), _lower('_emit_arg_reduction', op='reduce_argmax'), attrs.decode_arg_reduction, types.infer_arg_reduction),
    OpRule('reduce_argmin', ('argmin', 'reduce_argmin'), _lower('_emit_arg_reduction', op='reduce_argmin'), attrs.decode_arg_reduction, types.infer_arg_reduction),
    OpRule('argreduce', ('argreduce',), _lower('_emit_argreduce'), attrs.decode_argreduce, types.infer_arg_reduction),
    OpRule('flatten', ('flatten',), _lower('_emit_reshape'), attrs.decode_shape, types.infer_reshape),
    OpRule('unflatten', ('unflatten',), _lower('_emit_reshape'), attrs.decode_shape, types.infer_reshape),
    OpRule('reshape', ('reshape',), _lower('_emit_reshape'), attrs.decode_shape, types.infer_reshape),
    OpRule('transpose', ('transpose',), _lower('_emit_transpose'), attrs.decode_transpose, types.infer_transpose),
    OpRule('atleast_1d', ('atleast_1d',), _lower('_lower_expand', 'atleast_1d'), None, None),
    OpRule('atleast_2d', ('atleast_2d',), _lower('_lower_expand', 'atleast_2d'), None, None),
    OpRule('atleast_3d', ('atleast_3d',), _lower('_lower_expand', 'atleast_3d'), None, None),
    OpRule('moveaxis', ('moveaxis',), _lower('_lower_moveaxis'), attrs.decode_moveaxis, types.infer_moveaxis),
    OpRule('swapaxes', ('swapaxes',), _lower('_lower_swapaxes'), attrs.decode_swapaxes, types.infer_swapaxes),
    OpRule('slice_by_index', ('slice', 'slice_by_index'), _lower('_emit_slice_by_index'), attrs.decode_slice, types.infer_slice_by_index),
    OpRule('slice_update', ('slice_update', 'sliceupdate'), _lower('_lower_slice_update'), attrs.decode_slice, None),
    OpRule('dynamic_slice_update', ('dynamic_slice_update', 'dynamicsliceupdate'), _lower('_lower_dynamic_slice_update'), attrs.decode_dynamic_slice_update, None),
    OpRule('gather', ('take', 'gather'), _lower('_lower_gather'), attrs.decode_gather, types.infer_gather),
    OpRule('gather_along_axis', ('take_along_axis', 'gatheraxis'), _lower('_emit_gather_along_axis'), attrs.decode_gather_along_axis, types.infer_gather_along_axis),
    OpRule('logical_and', ('logicaland', 'logical_and'), _lower('_emit_binary', fn=coreai.broadcasting_and), None, types.infer_comparison),
    OpRule('logical_or', ('logicalor', 'logical_or'), _lower('_emit_binary', fn=coreai.broadcasting_or), None, types.infer_comparison),
    OpRule('logical_not', ('logicalnot', 'logical_not'), _lower('_emit_unary', fn=coreai.not_), None, types.infer_elementwise_bool),
    OpRule('floor', ('floor',), _lower('_emit_unary', fn=elementwise.floor), None, types.infer_passthrough),
    OpRule('ceil', ('ceil',), _lower('_emit_unary', fn=elementwise.ceil), None, types.infer_passthrough),
    OpRule('round', ('round',), _lower('_emit_unary', fn=elementwise.round_), None, types.infer_passthrough),
    OpRule('sign', ('sign',), _lower('_emit_unary', fn=elementwise.sign), None, types.infer_passthrough),
    OpRule('trunc', ('trunc',), _lower('_emit_unary', fn=elementwise.trunc), None, types.infer_passthrough),
    OpRule('acosh', ('arccosh',), _lower('_emit_unary', fn=coreai.acosh), None, types.infer_passthrough),
    OpRule('asinh', ('arcsinh',), _lower('_emit_unary', fn=coreai.asinh), None, types.infer_passthrough),
    OpRule('cosh', ('cosh',), _lower('_emit_unary', fn=coreai.cosh), None, types.infer_passthrough),
    OpRule('sinh', ('sinh',), _lower('_emit_unary', fn=coreai.sinh), None, types.infer_passthrough),
    OpRule('tan', ('tan',), _lower('_emit_unary', fn=coreai.tan), None, types.infer_passthrough),
    OpRule('atan2', ('arctan2',), _lower('_emit_binary', fn=elementwise.atan2), None, types.infer_binary),
    OpRule('scan', ('scan',), _lower('_emit_scan'), attrs.decode_scan, types.infer_passthrough),
    OpRule('sort', ('sort', 'partition'), _lower('_emit_sort', indices=False), attrs.decode_sort, types.infer_passthrough),
    OpRule('argsort', ('argsort', 'argpartition'), _lower('_emit_sort', indices=True), attrs.decode_sort, types.infer_sort_indices),
    OpRule('scatter', ('scatter',), _lower('_emit_scatter'), attrs.decode_scatter, types.infer_passthrough),
    OpRule('scatter_axis', ('scatteraxis',), _lower('_emit_scatter_axis'), attrs.decode_scatter, types.infer_passthrough),
    OpRule('gather_mm', ('gathermm', 'gather_mm'), _lower('_emit_gather_mm'), None, None),
    OpRule('as_strided', ('asstrided', 'as_strided'), _lower('_emit_as_strided'), attrs.decode_as_strided, types.infer_reshape),
    OpRule('nonzero', ('nonzero',), _lower('_emit_nonzero'), None, types.infer_nonzero),
    OpRule('masked_scatter', ('masked_scatter',), _lower('_emit_masked_scatter'), None, types.infer_passthrough),
    OpRule('squeeze', ('squeeze',), _lower('_emit_squeeze'), attrs.decode_squeeze, None),
    OpRule('zeros', ('zeros',), _lower('_emit_fill', op='zeros'), None, types.infer_fill),
    OpRule('ones', ('ones',), _lower('_emit_fill', op='ones'), None, types.infer_fill),
    OpRule('full', ('full',), _lower('_emit_fill', op='full'), None, types.infer_fill),
    OpRule('zeros_like', ('zeros_like',), _lower('_emit_zeros_like', op='zeros_like'), None, types.infer_passthrough),
    OpRule('ones_like', ('ones_like',), _lower('_emit_zeros_like', op='ones_like'), None, types.infer_passthrough),
    OpRule('full_like', ('full_like',), _lower('_emit_zeros_like', op='full_like'), None, types.infer_passthrough),
    OpRule('arange', ('arange',), _lower('_emit_arange'), attrs.decode_arange, types.infer_arange),
    OpRule('linspace', ('linspace',), _lower('_lower_linspace'), attrs.decode_linspace, types.infer_linspace),
    OpRule('select', ('where', 'select'), _lower('_emit_select'), None, types.infer_select),
    OpRule('greater', ('greater',), _lower('_emit_binary', fn=coreai.broadcasting_greater), None, types.infer_comparison),
    OpRule('greater_equal', ('greaterequal', 'greater_equal'), _lower('_emit_binary', fn=lambda x, y: coreai.broadcasting_or(coreai.broadcasting_greater(x, y), coreai.broadcasting_equal(x, y))), None, types.infer_comparison),
    OpRule('less', ('less',), _lower('_emit_binary', fn=lambda x, y: coreai.broadcasting_greater(y, x)), None, types.infer_comparison),
    OpRule('less_equal', ('lessequal', 'less_equal'), _lower('_emit_binary', fn=less_equal), None, types.infer_comparison),
    OpRule('equal', ('equal',), _lower('_emit_binary', fn=coreai.broadcasting_equal), None, types.infer_comparison),
    OpRule('not_equal', ('not_equal', 'notequal'), _lower('_emit_binary', fn=coreai.broadcasting_not_equal), None, types.infer_comparison),
    OpRule('exp', ('exp',), _lower('_emit_unary', fn=coreai.exp), None, types.infer_passthrough),
    OpRule('log', ('log',), _lower('_emit_unary', fn=coreai.log), None, types.infer_passthrough),
    OpRule('sqrt', ('sqrt',), _lower('_emit_sqrt'), attrs.decode_sqrt, types.infer_passthrough),
    OpRule('square', ('square',), _lower('_emit_unary', fn=lambda x: coreai.broadcasting_mul(x, x)), None, types.infer_passthrough),
    OpRule('rsqrt', ('rsqrt',), _lower('_emit_unary', fn=coreai.rsqrt), None, types.infer_passthrough),
    OpRule('abs', ('abs',), _lower('_emit_unary', fn=elementwise.abs_), None, types.infer_passthrough),
    OpRule('split', ('split',), _lower('_lower_split'), attrs.decode_split, types.infer_split),
    OpRule('expand_dims', ('expanddims', 'expand_dims'), _lower('_lower_expand', 'expand_dims'), attrs.decode_expand_dims, types.infer_expand_dims),
    OpRule('bitwisebinary', ('bitwisebinary',), _lower('_lower_bitwise_binary'), attrs.decode_bitwisebinary, types.infer_bitwisebinary),
    OpRule('scaled_dot_product_attention', ('scaled_dot_product_attention', 'scaleddotproductattention'), _lower('_lower_sdpa'), attrs.decode_scaled_dot_product_attention, types.infer_scaled_dot_product_attention),
    OpRule('gated_delta_update', ('gated_delta_update',), _lower('_emit_gated_delta_update'), None, types.infer_gated_delta_update),
    OpRule('rope', ('rope',), _lower('_lower_rope'), attrs.decode_rope, types.infer_passthrough),
    OpRule('softmax', ('softmax',), _lower('_emit_softmax'), attrs.decode_softmax, types.infer_passthrough),
    OpRule('sigmoid', ('sigmoid',), _lower('_emit_unary', fn=coreai.sigmoid), None, types.infer_passthrough),
    OpRule('silu', ('silu',), _lower('_emit_unary', fn=coreai.silu), None, types.infer_passthrough),
    OpRule('gelu', ('gelu',), _lower('_emit_unary', fn=coreai.gelu), None, types.infer_passthrough),
    OpRule('tanh', ('tanh',), _lower('_emit_unary', fn=coreai.tanh), None, types.infer_passthrough),
    OpRule('sin', ('sin',), _lower('_emit_unary', fn=coreai.sin), None, types.infer_passthrough),
    OpRule('cos', ('cos',), _lower('_emit_unary', fn=coreai.cos), None, types.infer_passthrough),
    OpRule('erf', ('erf',), _lower('_emit_unary', fn=coreai.erf), None, types.infer_passthrough),
    OpRule('layernorm', ('layernorm',), _lower('_lower_layernorm'), attrs.decode_layernorm, types.infer_passthrough),
    OpRule('rmsnorm', ('rmsnorm',), _lower('_lower_rmsnorm'), attrs.decode_rmsnorm, types.infer_passthrough),
    OpRule('cast', ('astype', 'cast'), _lower('_emit_cast'), attrs.decode_cast, types.infer_cast),
    OpRule('real', ('real',), _lower('_emit_unary', fn=coreai.real_part), None, types.infer_complex_component),
    OpRule('imag', ('imag',), _lower('_emit_unary', fn=coreai.imaginary_part), None, types.infer_complex_component),
    OpRule('conjugate', ('conjugate',), _lower('_emit_conjugate'), None, types.infer_passthrough),
    OpRule('view', ('view',), _lower('_emit_view'), attrs.decode_cast, None),
    OpRule('number_of_elements', ('number_of_elements',), _lower('_emit_number_of_elements'), attrs.decode_number_of_elements, types.infer_number_of_elements),
    OpRule('identity', ('stop_gradient', 'copy', 'contiguous'), _lower('_emit_identity'), None, types.infer_passthrough),
    OpRule('addmm', ('addmm',), _lower('_emit_addmm'), attrs.decode_addmm, None),
    OpRule('broadcast_to', ('broadcast', 'broadcast_to'), _lower('_emit_broadcast_to'), attrs.decode_shape, types.infer_broadcast_to),
    OpRule('broadcast_arrays', ('broadcast_arrays',), _lower('_lower_broadcast_arrays'), None, types.infer_broadcast_arrays),
    OpRule('broadcast_axes', ('broadcast_axes',), _lower('_lower_broadcast_axes'), attrs.decode_broadcast_axes, types.infer_broadcast_axes),
    OpRule('outer', ('outer',), _lower('_emit_outer'), None, None),
    OpRule('inner', ('inner',), _lower('_emit_inner'), None, None),
    OpRule('tensordot', ('tensordot',), _lower('_lower_tensordot'), None, None),
    OpRule('isclose', ('isclose',), _lower('_lower_isclose'), None, types.infer_elementwise_bool),
    OpRule('allclose', ('allclose',), _lower('_emit_allclose'), None, types.infer_scalar_bool),
    OpRule('nan_to_num', ('nan_to_num',), _lower('_lower_nan_to_num'), None, None),
    OpRule('diag', ('diag',), _lower('_lower_diag'), None, types.infer_diag),
    OpRule('diagonal', ('diagonal',), _lower('_lower_diagonal'), None, types.infer_diagonal),
    OpRule('trace', ('trace',), _lower('_lower_trace'), None, types.infer_trace),
    OpRule('tri', ('tri',), _lower('_lower_tri'), None, types.infer_matrix_shape),
    OpRule('tril', ('tril',), _lower('_emit_triangular', op='tril'), None, types.infer_passthrough),
    OpRule('triu', ('triu',), _lower('_emit_triangular', op='triu'), None, types.infer_passthrough),
    OpRule('all', ('all',), _lower('_emit_bool_reduction', op='all'), attrs.decode_reduction, types.infer_all),
    OpRule('any', ('any',), _lower('_emit_bool_reduction', op='any'), attrs.decode_reduction, types.infer_all),
    OpRule('array_equal', ('array_equal',), _lower('_lower_array_equal'), None, types.infer_scalar_bool),
    OpRule('isnan', ('isnan',), _lower('_emit_unary', fn=lambda x: coreai.broadcasting_not_equal(x, x)), None, types.infer_elementwise_bool),
    OpRule('isinf', ('isinf',), _lower('_emit_unary', fn=lambda x: coreai.broadcasting_equal(coreai.abs_(x), float('inf'))), None, types.infer_elementwise_bool),
    OpRule('isfinite', ('isfinite',), _lower('_emit_unary', fn=lambda x: coreai.not_(coreai.broadcasting_equal(coreai.abs_(x), float('inf')))), None, types.infer_elementwise_bool),
    OpRule('isneginf', ('isneginf',), _lower('_emit_unary', fn=lambda x: coreai.broadcasting_equal(x, float('-inf'))), None, types.infer_elementwise_bool),
    OpRule('isposinf', ('isposinf',), _lower('_emit_unary', fn=lambda x: coreai.broadcasting_equal(x, float('inf'))), None, types.infer_elementwise_bool),
    OpRule('eye', ('eye',), _lower('_lower_eye'), None, types.infer_matrix_shape),
    OpRule('meshgrid', ('meshgrid',), _lower('_lower_meshgrid'), None, types.infer_meshgrid),
    OpRule('kron', ('kron',), _lower('_lower_kron'), None, types.infer_kron),
    OpRule('logaddexp', ('logaddexp',), _lower('_emit_logaddexp'), None, types.infer_binary),
    OpRule('concat', ('concatenate',), _lower('_emit_concat'), attrs.decode_concat, types.infer_concat),
    OpRule('acos', ('arccos',), _lower('_emit_unary', fn=coreai.acos), None, types.infer_passthrough),
    OpRule('asin', ('arcsin',), _lower('_emit_unary', fn=coreai.asin), None, types.infer_passthrough),
    OpRule('atan', ('arctan',), _lower('_emit_unary', fn=coreai.atan), None, types.infer_passthrough),
    OpRule('atanh', ('arctanh',), _lower('_emit_unary', fn=coreai.atanh), None, types.infer_passthrough),
    OpRule('negative', ('negative',), _lower('_emit_negative'), None, types.infer_passthrough),
    OpRule('degrees', ('degrees',), _lower('_emit_unary', fn=lambda x: coreai.broadcasting_mul(x, 180.0 / math.pi)), None, types.infer_passthrough),
    OpRule('radians', ('radians',), _lower('_emit_unary', fn=lambda x: coreai.broadcasting_mul(x, math.pi / 180.0)), None, types.infer_passthrough),
    OpRule('expm1', ('expm1',), _lower('_emit_unary', fn=lambda x: coreai.broadcasting_sub(coreai.exp(x), 1.0)), None, types.infer_passthrough),
    OpRule('log1p', ('log1p',), _lower('_emit_unary', fn=lambda x: coreai.log(coreai.broadcasting_add(x, 1.0))), None, types.infer_passthrough),
    OpRule('log2', ('log2',), _lower('_emit_unary', fn=lambda x: coreai.broadcasting_divide(coreai.log(x), math.log(2.0))), None, types.infer_passthrough),
    OpRule('log10', ('log10',), _lower('_emit_unary', fn=lambda x: coreai.broadcasting_divide(coreai.log(x), math.log(10.0))), None, types.infer_passthrough),
    OpRule('reduce_log_sum_exp', ('logsumexp',), _lower('_emit_reduction', fn=lambda x, axes: coreai.log(coreai.reduce_sum(coreai.exp(x), axes))), attrs.decode_reduction, types.infer_reduction),
    OpRule('floor_div', ('floor_divide', 'floor_div'), _lower('_emit_floor_div'), None, types.infer_binary),
    OpRule('var', ('var',), _lower('_emit_variance', op='var'), attrs.decode_reduction, None),
    OpRule('std', ('std',), _lower('_emit_variance', op='std'), attrs.decode_reduction, None),
    OpRule('divmod', ('divmod',), _lower('_emit_divmod'), None, None),
    OpRule('conv', ('conv1d', 'conv2d', 'conv3d', 'conv_general', 'convolution'), _lower('_lower_conv', transpose=False), attrs.decode_conv, partial(types.infer_conv, op='conv')),
    OpRule('conv_transpose', ('conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d'), _lower('_lower_conv', transpose=True), None, partial(types.infer_conv, op='conv_transpose')),
    OpRule('const', ('const', 'constant'), _lower('_emit_const'), None, types.infer_const),
    OpRule('pad', ('pad',), _lower('_emit_pad'), attrs.decode_pad, types.infer_pad),
    OpRule('read_state', ('read_state',), _lower('_emit_read_state'), None, None),
    OpRule('write_state', ('write_state',), _lower('_emit_write_state'), None, None),
    OpRule('state_update_masked', ('state_update_masked',), _lower('_emit_state_update_masked'), None, None),
)

OPERATIONS = {rule.key: rule for rule in _RULES}
