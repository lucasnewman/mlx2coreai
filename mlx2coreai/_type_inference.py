"""Conservative type rules for manual IR and compiler-created values."""
from __future__ import annotations
from collections.abc import Sequence
import math
from typing import Any
import numpy as np
from .dtypes import normalize_dtype as _normalize_input_dtype
from .ir import Node, TensorType, is_dynamic_dim_ref

InferredTensorSpec = TensorType


def _promote_dtype(lhs: str | None, rhs: str | None) -> str | None:
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    rank = {"bool": 0, "int32": 1, "int64": 2, "fp16": 3, "bf16": 4, "fp32": 5, "fp64": 6, "complex64": 7}
    lhs_n = _normalize_input_dtype(lhs)
    rhs_n = _normalize_input_dtype(rhs)
    if lhs_n == rhs_n:
        return lhs_n
    if lhs_n in {'int8', 'uint8'} or rhs_n in {'int8', 'uint8'}:
        lhs_n = 'int32' if lhs_n in {'int8', 'uint8'} else lhs_n
        rhs_n = 'int32' if rhs_n in {'int8', 'uint8'} else rhs_n
    if lhs_n not in rank or rhs_n not in rank:
        return lhs_n if lhs_n == rhs_n else lhs_n
    return lhs_n if rank[lhs_n] >= rank[rhs_n] else rhs_n


def _infer_const_spec(node: Node) -> InferredTensorSpec:
    if "value" not in node.attrs:
        return InferredTensorSpec(shape=None, dtype=_normalize_input_dtype(str(node.attrs.get("dtype", "fp32"))))
    arr = np.asarray(node.attrs["value"])
    if node.attrs.get('dtype') is not None:
        return TensorType(tuple(arr.shape), _normalize_input_dtype(node.attrs['dtype']))
    dtype = str(arr.dtype).lower()
    if dtype in {'complex64', 'int8', 'uint8'}:
        out_dtype = dtype
    elif "bfloat16" in dtype:
        out_dtype = "bf16"
    elif "float16" in dtype:
        out_dtype = "fp16"
    elif "float64" in dtype:
        out_dtype = "fp64"
    elif "float" in dtype:
        out_dtype = "fp32"
    elif "int64" in dtype:
        out_dtype = "int64"
    elif "int" in dtype:
        out_dtype = "int32"
    elif "bool" in dtype:
        out_dtype = "bool"
    else:
        out_dtype = _normalize_input_dtype(str(node.attrs.get("dtype", "fp32")))
    return InferredTensorSpec(shape=tuple(int(v) for v in arr.shape), dtype=out_dtype)


def _infer_broadcast_shape(
    lhs: tuple[int, ...] | None,
    rhs: tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    if lhs is None or rhs is None:
        return None
    out_rank = max(len(lhs), len(rhs))
    out: list[int] = []
    for axis in range(out_rank):
        li = axis - (out_rank - len(lhs))
        ri = axis - (out_rank - len(rhs))
        ld = lhs[li] if li >= 0 else 1
        rd = rhs[ri] if ri >= 0 else 1
        if ld == rd:
            out.append(rd)
        elif ld < 0 or rd < 0:
            if ld == 1:
                out.append(rd)
            elif rd == 1:
                out.append(ld)
            else:
                out.append(-1)
        elif ld == 1:
            out.append(rd)
        elif rd == 1:
            out.append(ld)
        else:
            return None
    return tuple(out)


def infer_broadcast_axes_shape(
    shapes: Sequence[tuple[int, ...] | None],
    ignore_axes: Sequence[int],
) -> tuple[int, ...] | None:
    if not shapes:
        raise ValueError("broadcast_axes requires at least one input.")
    axes = [int(axis) for axis in ignore_axes]
    if axes != sorted(set(axes)) or any(axis >= 0 for axis in axes):
        raise ValueError("broadcast_axes ignore_axes must be sorted, unique negative axes.")
    if any(shape is None for shape in shapes):
        return None

    shape: tuple[int, ...] = ()
    for input_shape in shapes:
        assert input_shape is not None
        rank = len(input_shape)
        if any(axis < -rank for axis in axes):
            raise ValueError(f"broadcast_axes ignore_axes {axes} are out of range for rank {rank}.")
        reduced = tuple(dim for axis, dim in enumerate(input_shape) if axis - rank not in axes)
        broadcast = _infer_broadcast_shape(shape, reduced)
        if broadcast is None:
            raise ValueError(f"broadcast_axes inputs have incompatible non-ignored dimensions: {shapes}.")
        shape = broadcast

    # Ignored dimensions are removed before broadcasting, then restored from input 0.
    result = list(shape)
    output_rank = len(result) + len(axes)
    for axis in axes:
        result.insert(output_rank + axis, shapes[0][axis])
    return tuple(result)


def _shape_from_attr(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    out: list[int] = []
    for dim in value:
        if is_dynamic_dim_ref(dim):
            out.append(-1)
        else:
            out.append(int(dim))
    return tuple(out)


def _static_product(shape: tuple[int, ...]) -> int | None:
    if any(int(dim) < 0 for dim in shape):
        return None
    return int(math.prod(shape))


def _normalize_axes_for_rank(axes_raw: Any, rank: int) -> list[int]:
    if axes_raw is None:
        return list(range(rank))
    if isinstance(axes_raw, int):
        axes = [int(axes_raw)]
    elif isinstance(axes_raw, (list, tuple)):
        axes = [int(v) for v in axes_raw]
    else:
        return list(range(rank))

    norm: list[int] = []
    for axis in axes:
        a = axis + rank if axis < 0 else axis
        if 0 <= a < rank and a not in norm:
            norm.append(a)
    return norm


def _reduced_shape(shape: tuple[int, ...] | None, axes_raw: Any, keep_dims: bool) -> tuple[int, ...] | None:
    if shape is None:
        return None
    rank = len(shape)
    axes = _normalize_axes_for_rank(axes_raw, rank)
    if keep_dims:
        return tuple(1 if i in axes else d for i, d in enumerate(shape))
    return tuple(d for i, d in enumerate(shape) if i not in axes)


def _normalize_axis(axis: int, rank: int) -> int | None:
    axis = int(axis)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return None
    return axis


def _diagonal_len(dim1: int, dim2: int, offset: int) -> int | None:
    row_start = max(-int(offset), 0)
    col_start = max(int(offset), 0)
    out = min(int(dim1) - row_start, int(dim2) - col_start)
    return out if out > 0 else None


def _conv_spatial_output(
    input_spatial: tuple[int, ...],
    kernel_spatial: tuple[int, ...],
    strides: list[int],
    dilations: list[int],
    padding: list[int],
    *,
    transpose: bool,
    output_padding: list[int] | None = None,
) -> tuple[int, ...]:
    out: list[int] = []
    output_padding = output_padding or [0] * len(input_spatial)
    for i, input_size in enumerate(input_spatial):
        kernel = int(kernel_spatial[i])
        stride = int(strides[i])
        dilation = int(dilations[i])
        before = int(padding[2 * i])
        after = int(padding[2 * i + 1])
        if transpose:
            dim = (int(input_size) - 1) * stride - before - after + dilation * (kernel - 1) + int(output_padding[i]) + 1
        else:
            dim = math.floor((int(input_size) + before + after - dilation * (kernel - 1) - 1) / stride + 1)
        out.append(int(dim))
    return tuple(out)


def _as_int_list_attr(value: Any, count: int, default: int = 1) -> list[int] | None:
    if value is None:
        return [int(default)] * count
    if isinstance(value, int):
        return [int(value)] * count
    if not isinstance(value, (list, tuple)):
        return None
    out = [int(v) for v in value]
    if len(out) == 1:
        return out * count
    if len(out) != count:
        return None
    return out


def _padding_attr(node: Node, spatial: int, input_spatial: tuple[int, ...], kernel_spatial: tuple[int, ...], strides: list[int], dilations: list[int], *, transpose: bool) -> list[int] | None:
    raw_pad = node.attrs.get("pad", node.attrs.get("padding", node.attrs.get("pads")))
    pad_type = str(node.attrs.get("pad_type", "custom" if raw_pad is not None else "valid")).strip().lower()
    if pad_type == "valid":
        return [0] * (2 * spatial)
    if raw_pad is not None:
        if isinstance(raw_pad, int):
            return [int(raw_pad), int(raw_pad)] * spatial
        if not isinstance(raw_pad, (list, tuple)):
            return None
        parsed = [int(v) for v in raw_pad]
        if len(parsed) == spatial:
            return [v for item in parsed for v in (item, item)]
        if len(parsed) == 2 * spatial:
            return parsed
        return None
    if pad_type not in {"same", "same_lower"} or transpose:
        return None
    padding: list[int] = []
    for i in range(spatial):
        output_size = math.ceil(int(input_spatial[i]) / int(strides[i]))
        needed = max(
            0,
            (output_size - 1) * int(strides[i])
            + int(dilations[i]) * (int(kernel_spatial[i]) - 1)
            + 1
            - int(input_spatial[i]),
        )
        before = needed // 2
        after = needed - before
        if pad_type == "same_lower":
            before, after = after, before
        padding.extend([before, after])
    return padding


def unknown_spec(input_specs):
    return InferredTensorSpec(shape=None, dtype=input_specs[0].dtype if input_specs else None)


def infer_const(node, input_specs):
    return _infer_const_spec(node)


def infer_passthrough(node, input_specs):
    return input_specs[0] if input_specs else InferredTensorSpec(shape=None, dtype=None)


def infer_binary(node, input_specs):
    if not (len(input_specs) == 2):
        return unknown_spec(input_specs)
    return InferredTensorSpec(
        shape=_infer_broadcast_shape(input_specs[0].shape, input_specs[1].shape),
        dtype=_promote_dtype(input_specs[0].dtype, input_specs[1].dtype),
    )


def infer_comparison(node, input_specs):
    if not (len(input_specs) == 2):
        return unknown_spec(input_specs)
    return InferredTensorSpec(
        shape=_infer_broadcast_shape(input_specs[0].shape, input_specs[1].shape),
        dtype="bool",
    )


def infer_bitwisebinary(node, input_specs):
    if not (len(input_specs) == 2):
        return unknown_spec(input_specs)
    return InferredTensorSpec(
        shape=_infer_broadcast_shape(input_specs[0].shape, input_specs[1].shape),
        dtype="bool",
    )


def infer_matmul(node, input_specs):
    if not (len(input_specs) == 2):
        return unknown_spec(input_specs)
    x_shape = input_specs[0].shape
    y_shape = input_specs[1].shape
    out_shape: tuple[int, ...] | None = None
    if x_shape is not None and y_shape is not None and len(x_shape) == 2 and len(y_shape) == 2:
        out_shape = (x_shape[0], y_shape[1])
    return InferredTensorSpec(shape=out_shape, dtype=_promote_dtype(input_specs[0].dtype, input_specs[1].dtype))


def infer_gated_delta_update(node, input_specs):
    if not (len(input_specs) == 6):
        return unknown_spec(input_specs)
    if int(node.attrs.get("output_index", 0)) == 1:
        return input_specs[5]
    return InferredTensorSpec(shape=input_specs[2].shape, dtype=input_specs[0].dtype)


def infer_sort_indices(node, input_specs):
    return InferredTensorSpec(shape=input_specs[0].shape, dtype="int32")


def infer_complex_component(node, input_specs):
    return InferredTensorSpec(shape=input_specs[0].shape, dtype="fp32")


def infer_complex_construction(node, input_specs):
    return TensorType(_infer_broadcast_shape(input_specs[0].shape, input_specs[1].shape), 'complex64')


def infer_nonzero(node, input_specs):
    shape = input_specs[0].shape
    return InferredTensorSpec(shape=(-1, len(shape)) if shape is not None else None, dtype='int32')


def infer_scaled_dot_product_attention(node, input_specs):
    if not (len(input_specs) >= 3):
        return unknown_spec(input_specs)
    q_shape = input_specs[0].shape
    v_shape = input_specs[2].shape
    out_shape: tuple[int, ...] | None = None
    if (
        q_shape is not None
        and v_shape is not None
        and len(q_shape) >= 3
        and len(v_shape) >= 3
    ):
        out_shape = tuple(list(q_shape[:-1]) + [v_shape[-1]])
    return InferredTensorSpec(shape=out_shape, dtype=input_specs[0].dtype)


def infer_reduce(node, input_specs):
    src = input_specs[0] if input_specs else InferredTensorSpec(shape=None, dtype=None)
    keep_dims = bool(node.attrs.get("keep_dims", True))
    mode = int(node.attrs.get("mode", 2))
    out_dtype = "bool" if mode in {0, 1} else src.dtype
    return InferredTensorSpec(
        shape=_reduced_shape(src.shape, node.attrs.get("axes"), keep_dims),
        dtype=out_dtype,
    )


def infer_reduction(node, input_specs):
    src = input_specs[0] if input_specs else InferredTensorSpec(shape=None, dtype=None)
    keep_dims = bool(node.attrs.get("keep_dims", False))
    return InferredTensorSpec(
        shape=_reduced_shape(src.shape, node.attrs.get("axes"), keep_dims),
        dtype=src.dtype,
    )


def infer_arg_reduction(node, input_specs):
    src = input_specs[0] if input_specs else InferredTensorSpec(shape=None, dtype=None)
    keep_dims = bool(node.attrs.get("keep_dims", False))
    axis = node.attrs.get("axis")
    if axis is None and "axes" in node.attrs:
        axes = node.attrs.get("axes")
        axis = axes[0] if isinstance(axes, (list, tuple)) and axes else None
    return InferredTensorSpec(
        shape=_reduced_shape(src.shape, axis, keep_dims),
        dtype="int32",
    )


def infer_reshape(node, input_specs):
    out_shape = _shape_from_attr(node.attrs.get("shape"))
    if out_shape is not None:
        if -1 in out_shape and input_specs:
            src_shape = input_specs[0].shape
            if src_shape is not None:
                known = 1
                unknown_count = 0
                for dim in out_shape:
                    if dim == -1:
                        unknown_count += 1
                    else:
                        known *= dim
                if unknown_count == 1 and known != 0:
                    total = _static_product(src_shape)
                    if total is not None:
                        fill = total // known if total % known == 0 else -1
                        out_shape = tuple(fill if dim == -1 else dim for dim in out_shape)
        return InferredTensorSpec(shape=out_shape, dtype=input_specs[0].dtype if input_specs else None)
    return InferredTensorSpec(shape=None, dtype=input_specs[0].dtype if input_specs else None)


def infer_transpose(node, input_specs):
    if not (input_specs):
        return unknown_spec(input_specs)
    src = input_specs[0]
    perm_attr = node.attrs.get("perm")
    if src.shape is None or not isinstance(perm_attr, (list, tuple)):
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    perm = [int(v) for v in perm_attr]
    if len(src.shape) != len(perm):
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    try:
        out_shape = tuple(src.shape[idx] for idx in perm)
    except IndexError:
        out_shape = None
    return InferredTensorSpec(shape=out_shape, dtype=src.dtype)


def infer_moveaxis(node, input_specs):
    if not (input_specs):
        return unknown_spec(input_specs)
    src = input_specs[0]
    if src.shape is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    rank = len(src.shape)
    source_raw = node.attrs.get("source")
    dest_raw = node.attrs.get("destination")
    if source_raw is None or dest_raw is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    sources = [source_raw] if isinstance(source_raw, int) else list(source_raw)
    destinations = [dest_raw] if isinstance(dest_raw, int) else list(dest_raw)
    if len(sources) != len(destinations):
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    norm_sources = [_normalize_axis(int(axis), rank) for axis in sources]
    norm_destinations = [_normalize_axis(int(axis), rank) for axis in destinations]
    if any(axis is None for axis in norm_sources + norm_destinations):
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    perm = [axis for axis in range(rank) if axis not in norm_sources]
    for destination, source in sorted(zip(norm_destinations, norm_sources, strict=True)):  # type: ignore[arg-type]
        perm.insert(int(destination), int(source))
    return InferredTensorSpec(shape=tuple(src.shape[axis] for axis in perm), dtype=src.dtype)


def infer_swapaxes(node, input_specs):
    if not (input_specs):
        return unknown_spec(input_specs)
    src = input_specs[0]
    if src.shape is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    axis1 = _normalize_axis(int(node.attrs.get("axis1", 0)), len(src.shape))
    axis2 = _normalize_axis(int(node.attrs.get("axis2", 1)), len(src.shape))
    if axis1 is None or axis2 is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    shape = list(src.shape)
    shape[axis1], shape[axis2] = shape[axis2], shape[axis1]
    return InferredTensorSpec(shape=tuple(shape), dtype=src.dtype)


def infer_expand_dims(node, input_specs):
    if not (input_specs):
        return unknown_spec(input_specs)
    src = input_specs[0]
    if src.shape is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    axes_raw = node.attrs.get("axes", node.attrs.get("axis"))
    if isinstance(axes_raw, int):
        axes = [int(axes_raw)]
    elif isinstance(axes_raw, (list, tuple)):
        axes = [int(v) for v in axes_raw]
    else:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    if not axes:
        return InferredTensorSpec(shape=src.shape, dtype=src.dtype)
    rank = len(src.shape)
    # Normalize against output rank as expand_dims allows insertion at rank and below.
    out_rank = rank + len(axes)
    norm_axes: list[int] = []
    for axis in axes:
        a = axis + out_rank if axis < 0 else axis
        if a < 0 or a >= out_rank:
            return InferredTensorSpec(shape=None, dtype=src.dtype)
        norm_axes.append(a)
    if len(set(norm_axes)) != len(norm_axes):
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    out_shape = list(src.shape)
    for axis in sorted(norm_axes):
        out_shape.insert(axis, 1)
    return InferredTensorSpec(shape=tuple(out_shape), dtype=src.dtype)


def infer_concat(node, input_specs):
    if not input_specs:
        return InferredTensorSpec(shape=None, dtype=None)
    axis = int(node.attrs.get("axis", 0))
    dtype = input_specs[0].dtype
    shapes = [spec.shape for spec in input_specs]
    if any(shape is None for shape in shapes):
        return InferredTensorSpec(shape=None, dtype=dtype)
    shape0 = list(shapes[0])  # type: ignore[index]
    rank = len(shape0)
    axis = axis + rank if axis < 0 else axis
    if axis < 0 or axis >= rank:
        return InferredTensorSpec(shape=None, dtype=dtype)
    total = 0
    dynamic_axis = False
    for shape in shapes:  # type: ignore[assignment]
        assert shape is not None
        if len(shape) != rank:
            return InferredTensorSpec(shape=None, dtype=dtype)
        for i, dim in enumerate(shape):
            if i != axis and dim != shape0[i] and dim >= 0 and shape0[i] >= 0:
                return InferredTensorSpec(shape=None, dtype=dtype)
        if int(shape[axis]) < 0:
            dynamic_axis = True
        else:
            total += shape[axis]
    shape0[axis] = -1 if dynamic_axis else total
    return InferredTensorSpec(shape=tuple(shape0), dtype=dtype)


def infer_split(node, input_specs):
    if not (input_specs):
        return unknown_spec(input_specs)
    src = input_specs[0]
    if src.shape is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    axis = int(node.attrs.get("axis", 0))
    rank = len(src.shape)
    axis = axis + rank if axis < 0 else axis
    if axis < 0 or axis >= rank:
        return InferredTensorSpec(shape=None, dtype=src.dtype)

    axis_dim = int(src.shape[axis])
    output_index = int(node.attrs.get("output_index", 0))
    split_sizes = node.attrs.get("split_sizes")
    sizes: list[int] | None = None
    if isinstance(split_sizes, (list, tuple)):
        sizes = [int(v) for v in split_sizes]
    else:
        split_indices = node.attrs.get("split_indices")
        if isinstance(split_indices, (list, tuple)):
            indices = [int(v) for v in split_indices]
            prev = 0
            sizes = []
            for idx in indices:
                if idx < prev or idx > axis_dim:
                    return InferredTensorSpec(shape=None, dtype=src.dtype)
                sizes.append(idx - prev)
                prev = idx
            sizes.append(axis_dim - prev)
        else:
            num_splits_raw = node.attrs.get("num_splits", node.attrs.get("num_outputs"))
            if num_splits_raw is not None:
                num_splits = int(num_splits_raw)
                if num_splits <= 0 or axis_dim % num_splits != 0:
                    return InferredTensorSpec(shape=None, dtype=src.dtype)
                sizes = [axis_dim // num_splits] * num_splits

    if sizes is None or output_index < 0 or output_index >= len(sizes):
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    out_shape = list(src.shape)
    out_shape[axis] = int(sizes[output_index])
    return InferredTensorSpec(shape=tuple(out_shape), dtype=src.dtype)


def infer_cast(node, input_specs):
    dtype = node.attrs.get("dtype")
    if isinstance(dtype, str):
        return InferredTensorSpec(
            shape=input_specs[0].shape if input_specs else None,
            dtype=_normalize_input_dtype(dtype),
        )
    return InferredTensorSpec(shape=input_specs[0].shape if input_specs else None, dtype=None)


def infer_slice_by_index(node, input_specs):
    if not (input_specs):
        return unknown_spec(input_specs)
    src = input_specs[0]
    if src.shape is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    rank = len(src.shape)
    begin = list(node.attrs.get("begin", [0] * rank))
    end = list(node.attrs.get("end", list(src.shape)))
    stride = list(node.attrs.get("stride", [1] * rank))
    begin += [0] * (rank - len(begin))
    end += list(src.shape[len(end):])
    stride += [1] * (rank - len(stride))
    out = []
    for b, e, s, dim in zip(begin, end, stride, src.shape, strict=True):
        if is_dynamic_dim_ref(b) or is_dynamic_dim_ref(e) or int(dim) < 0:
            out.append(-1)
            continue
        b = max(0, int(b) + int(dim) if int(b) < 0 else int(b))
        e = min(int(dim), int(e) + int(dim) if int(e) < 0 else int(e))
        s = int(s)
        if s == 0:
            return InferredTensorSpec(shape=None, dtype=src.dtype)
        out.append(max(0, math.ceil((e - b) / s)))
    return InferredTensorSpec(shape=tuple(out), dtype=src.dtype)


def infer_gather(node, input_specs):
    if not (len(input_specs) == 2):
        return unknown_spec(input_specs)
    src, idx = input_specs[0], input_specs[1]
    captured_shape = _shape_from_attr(node.attrs.get("shape"))
    if captured_shape is not None:
        return InferredTensorSpec(shape=captured_shape, dtype=src.dtype)
    if src.shape is None or idx.shape is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    axis = _normalize_axis(int(node.attrs.get("axis", 0)), len(src.shape))
    if axis is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    if len(idx.shape) == 0:
        out_shape = tuple(dim for i, dim in enumerate(src.shape) if i != axis)
    elif len(idx.shape) == 1:
        out = list(src.shape)
        out[axis] = idx.shape[0]
        out_shape = tuple(out)
    else:
        out_shape = tuple(src.shape[:axis] + idx.shape + src.shape[axis + 1 :])
    return InferredTensorSpec(shape=out_shape, dtype=src.dtype)


def infer_gather_along_axis(node, input_specs):
    if not (len(input_specs) == 2):
        return unknown_spec(input_specs)
    return InferredTensorSpec(shape=input_specs[1].shape, dtype=input_specs[0].dtype)


def infer_fill(node, input_specs):
    shape = _shape_from_attr(node.attrs.get("shape"))
    dtype = _normalize_input_dtype(str(node.attrs.get("dtype", "fp32")))
    return InferredTensorSpec(shape=shape, dtype=dtype)


def infer_number_of_elements(node, input_specs):
    return InferredTensorSpec(shape=tuple(), dtype=_normalize_input_dtype(node.attrs.get("dtype", "int32")))


def infer_arange(node, input_specs):
    dtype = node.attrs.get('dtype', 'int32')
    start = float(node.attrs.get("start", 0))
    end = node.attrs.get("end")
    step = float(node.attrs.get("step", 1))
    if end is None or step == 0:
        return InferredTensorSpec(shape=None, dtype=dtype)
    if is_dynamic_dim_ref(end):
        return InferredTensorSpec(shape=(-1,), dtype=dtype)
    n = max(0, int(math.ceil((float(end) - start) / step)))
    return InferredTensorSpec(shape=(n,), dtype=dtype)


def infer_linspace(node, input_specs):
    num = int(node.attrs.get("num", 50))
    dtype = _normalize_input_dtype(str(node.attrs.get("dtype", "fp32")))
    return InferredTensorSpec(shape=(max(0, num),), dtype=dtype)


def infer_select(node, input_specs):
    if not (len(input_specs) == 3):
        return unknown_spec(input_specs)
    return InferredTensorSpec(
        shape=_infer_broadcast_shape(input_specs[1].shape, input_specs[2].shape),
        dtype=_promote_dtype(input_specs[1].dtype, input_specs[2].dtype),
    )


def infer_all(node, input_specs):
    if not (input_specs):
        return unknown_spec(input_specs)
    keep_dims = bool(node.attrs.get("keep_dims", False))
    return InferredTensorSpec(
        shape=_reduced_shape(input_specs[0].shape, node.attrs.get("axes"), keep_dims),
        dtype="bool",
    )


def infer_scalar_bool(node, input_specs):
    return InferredTensorSpec(shape=tuple(), dtype="bool")


def infer_elementwise_bool(node, input_specs):
    if not (input_specs):
        return unknown_spec(input_specs)
    return InferredTensorSpec(shape=input_specs[0].shape, dtype="bool")


def infer_diag(node, input_specs):
    if not (input_specs):
        return unknown_spec(input_specs)
    src = input_specs[0]
    if src.shape is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    offset = int(node.attrs.get("k", node.attrs.get("offset", 0)))
    if len(src.shape) == 1:
        size = int(src.shape[0]) + abs(offset)
        return InferredTensorSpec(shape=(size, size), dtype=src.dtype)
    if len(src.shape) == 2:
        diag_len = _diagonal_len(src.shape[0], src.shape[1], offset)
        return InferredTensorSpec(shape=(diag_len,) if diag_len is not None else None, dtype=src.dtype)
    return InferredTensorSpec(shape=None, dtype=src.dtype)


def infer_diagonal(node, input_specs):
    if not (input_specs):
        return unknown_spec(input_specs)
    src = input_specs[0]
    if src.shape is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    rank = len(src.shape)
    axis1 = _normalize_axis(int(node.attrs.get("axis1", 0)), rank)
    axis2 = _normalize_axis(int(node.attrs.get("axis2", 1)), rank)
    if axis1 is None or axis2 is None or axis1 == axis2:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    diag_len = _diagonal_len(src.shape[axis1], src.shape[axis2], int(node.attrs.get("offset", node.attrs.get("k", 0))))
    if diag_len is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    out_shape = tuple(src.shape[i] for i in range(rank) if i not in {axis1, axis2}) + (diag_len,)
    return InferredTensorSpec(shape=out_shape, dtype=src.dtype)


def infer_trace(node, input_specs):
    if not (input_specs):
        return unknown_spec(input_specs)
    diag = infer_diagonal(Node("diagonal", node.inputs, node.output, attrs=dict(node.attrs)), input_specs)
    return InferredTensorSpec(shape=_reduced_shape(diag.shape, [-1], False), dtype=diag.dtype)


def infer_matrix_shape(node, input_specs):
    n = int(node.attrs.get("n", 0))
    m = int(node.attrs.get("m", n))
    return InferredTensorSpec(shape=(n, m), dtype=_normalize_input_dtype(str(node.attrs.get("dtype", "fp32"))))


def infer_broadcast_to(node, input_specs):
    if not (input_specs):
        return unknown_spec(input_specs)
    shape = _shape_from_attr(node.attrs.get("shape"))
    return InferredTensorSpec(shape=shape, dtype=input_specs[0].dtype)


def infer_pad(node, input_specs):
    source = input_specs[0]
    padding = node.attrs["padding"]
    shape = None if source.shape is None else tuple(
        dim + padding[2 * axis] + padding[2 * axis + 1] if dim >= 0 else -1
        for axis, dim in enumerate(source.shape))
    return InferredTensorSpec(shape, source.dtype)


def infer_broadcast_arrays(node, input_specs):
    if not (input_specs):
        return unknown_spec(input_specs)
    shapes = [spec.shape for spec in input_specs]
    out_shape = None
    for shape in shapes:
        out_shape = _infer_broadcast_shape(out_shape, shape) if out_shape is not None else shape
    index = int(node.attrs.get("output_index", 0))
    return InferredTensorSpec(shape=out_shape, dtype=input_specs[index].dtype)


def infer_broadcast_axes(node, input_specs):
    shape = infer_broadcast_axes_shape(
        [spec.shape for spec in input_specs], node.attrs.get("ignore_axes", [])
    )
    return InferredTensorSpec(shape=shape, dtype=input_specs[0].dtype)


def infer_meshgrid(node, input_specs):
    if not (input_specs):
        return unknown_spec(input_specs)
    dims: list[int] = []
    for spec in input_specs:
        if spec.shape is None or len(spec.shape) != 1:
            return InferredTensorSpec(shape=None, dtype=input_specs[0].dtype)
        dims.append(int(spec.shape[0]))
    input_index = int(node.attrs.get("input_index", node.attrs.get("output_index", 0)))
    indexing = str(node.attrs.get("indexing", "xy")).strip().lower()
    if input_index < 0 or input_index >= len(dims):
        return InferredTensorSpec(shape=None, dtype=input_specs[0].dtype)
    if indexing == "xy" and len(dims) >= 2:
        out_shape = (dims[1], dims[0], *dims[2:])
    elif indexing == "ij" or len(dims) < 2:
        out_shape = tuple(dims)
    else:
        out_shape = None
    return InferredTensorSpec(shape=out_shape, dtype=input_specs[input_index].dtype)


def infer_kron(node, input_specs):
    if not (len(input_specs) == 2):
        return unknown_spec(input_specs)
    lhs, rhs = input_specs
    if lhs.shape is None or rhs.shape is None:
        return InferredTensorSpec(shape=None, dtype=_promote_dtype(lhs.dtype, rhs.dtype))
    rank = max(len(lhs.shape), len(rhs.shape))
    lhs_shape = (1,) * (rank - len(lhs.shape)) + lhs.shape
    rhs_shape = (1,) * (rank - len(rhs.shape)) + rhs.shape
    return InferredTensorSpec(
        shape=tuple(int(a) * int(b) for a, b in zip(lhs_shape, rhs_shape, strict=True)),
        dtype=_promote_dtype(lhs.dtype, rhs.dtype),
    )


def infer_conv(node, input_specs, *, op):
    if not (input_specs):
        return unknown_spec(input_specs)
    src = input_specs[0]
    weight = input_specs[1] if len(input_specs) > 1 else InferredTensorSpec(shape=None, dtype=None)
    if src.shape is None or weight.shape is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    channels_last = bool(node.attrs.get("channels_last", False))
    if channels_last:
        src = InferredTensorSpec(shape=(src.shape[0], src.shape[-1], *src.shape[1:-1]), dtype=src.dtype)
        weight = InferredTensorSpec(shape=(weight.shape[0], weight.shape[-1], *weight.shape[1:-1]), dtype=weight.dtype)
    spatial = len(src.shape) - 2
    if spatial not in {1, 2, 3} or len(weight.shape) != spatial + 2:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    strides = _as_int_list_attr(node.attrs.get("strides", node.attrs.get("stride")), spatial, 1)
    dilations = _as_int_list_attr(node.attrs.get("dilations", node.attrs.get("dilation")), spatial, 1)
    if strides is None or dilations is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    input_dilations = _as_int_list_attr(node.attrs.get("input_dilations"), spatial, 1)
    if input_dilations is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    src = InferredTensorSpec((src.shape[0], src.shape[1], *(
        (dim - 1) * dilation + 1 if dim >= 0 else -1
        for dim, dilation in zip(src.shape[2:], input_dilations, strict=True))), src.dtype)
    padding = _padding_attr(node, spatial, src.shape[2:], weight.shape[2:], strides, dilations, transpose=(op == "conv_transpose"))
    if padding is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    groups = int(node.attrs.get("groups", 1))
    output_padding = _as_int_list_attr(node.attrs.get("output_pad", node.attrs.get("output_padding")), spatial, 0)
    if output_padding is None:
        return InferredTensorSpec(shape=None, dtype=src.dtype)
    spatial_out = _conv_spatial_output(
        src.shape[2:],
        weight.shape[2:],
        strides,
        dilations,
        padding,
        transpose=(op == "conv_transpose"),
        output_padding=output_padding,
    )
    channels = int(weight.shape[1]) * groups if op == "conv_transpose" else int(weight.shape[0])
    shape = (int(src.shape[0]), *spatial_out, channels) if channels_last else (int(src.shape[0]), channels, *spatial_out)
    return InferredTensorSpec(shape=shape, dtype=src.dtype)
