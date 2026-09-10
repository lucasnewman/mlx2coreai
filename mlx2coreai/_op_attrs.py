"""MLX primitive argument codecs, selected by the operation registry."""
from __future__ import annotations
from typing import Any
import numpy as np


def _int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [int(v) for v in value]
    return [int(value)]


def decode_addmm(op, arguments, output_shape, output_dtype):
    # MLX's primitive stores A, B, C, unlike the public addmm(C, A, B) API.
    return {"input_order": "abc",
            "alpha": float(arguments[0]) if arguments else 1.0,
            "beta": float(arguments[1]) if len(arguments) > 1 else 1.0}


def decode_number_of_elements(op, arguments, output_shape, output_dtype):
    result = {"dtype": output_dtype or "int32"}
    if arguments:
        result["axes"] = _int_list(arguments[0])
    if len(arguments) > 1:
        result["inverted"] = bool(arguments[1])
    return result


def decode_sqrt(op, arguments, output_shape, output_dtype):
    return {"inverted": bool(arguments[0])} if arguments else {}


def decode_reduce(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "reduce":
        if arguments:
            attrs["mode"] = int(arguments[0])
        if len(arguments) >= 2:
            axes = _int_list(arguments[1])
            if axes is not None:
                attrs["axes"] = axes
        # MLX callback Reduce currently emits rank-preserving reductions.
        attrs["keep_dims"] = bool(arguments[2]) if len(arguments) >= 3 else True
    return attrs


def decode_reduction(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op in {"sum", "mean", "min", "max", "prod", "all", "any", "var", "std", "logsumexp"}:
        if arguments:
            axes = _int_list(arguments[0])
            if axes is not None:
                attrs["axes"] = axes
        if len(arguments) >= 2:
            attrs["keep_dims"] = bool(arguments[1])
        if op in {"var", "std"} and len(arguments) >= 3:
            attrs["ddof"] = int(arguments[2])
    return attrs


def decode_arg_reduction(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op in {"argmax", "argmin"}:
        if arguments:
            attrs["axis"] = int(arguments[0])
        if len(arguments) >= 2:
            attrs["keep_dims"] = bool(arguments[1])
    return attrs


def decode_argreduce(op, arguments, output_shape, output_dtype):
    mode, axis = map(int, arguments)
    if mode not in (0, 1):
        raise ValueError(f"Unknown MLX ArgReduce mode: {mode}")
    return {"mode": mode, "axis": axis, "keep_dims": True}


def decode_shape(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op in {"reshape", "flatten", "unflatten", "broadcast", "broadcast_to"} and output_shape is not None:
        attrs["shape"] = list(output_shape)
    return attrs


def decode_transpose(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "transpose" and arguments:
        perm = _int_list(arguments[0])
        if perm is not None:
            attrs["perm"] = perm
    return attrs


def decode_moveaxis(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "moveaxis" and len(arguments) >= 2:
        attrs["source"] = int(arguments[0])
        attrs["destination"] = int(arguments[1])
    return attrs


def decode_swapaxes(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "swapaxes" and len(arguments) >= 2:
        attrs["axis1"] = int(arguments[0])
        attrs["axis2"] = int(arguments[1])
    return attrs


def decode_slice(op, arguments, output_shape, output_dtype):
    if op in {"sliceupdate", "slice_update"} and len(arguments) == 4:
        if int(arguments[0]) != 4:
            raise ValueError(f"Unsupported MLX SliceUpdate mode: {arguments[0]}")
        arguments = arguments[1:]
    attrs: dict[str, Any] = {}
    if op in {"slice", "slice_update", "sliceupdate"}:
        if len(arguments) >= 1:
            begin = _int_list(arguments[0])
            if begin is not None:
                attrs["begin"] = begin
        if len(arguments) >= 2:
            end = _int_list(arguments[1])
            if end is not None:
                attrs["end"] = end
        if len(arguments) >= 3:
            stride = _int_list(arguments[2])
            if stride is not None:
                attrs["stride"] = stride
    return attrs


def decode_dynamic_slice_update(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op in {"dynamic_slice_update", "dynamicsliceupdate"} and arguments:
        axes = _int_list(arguments[0])
        if axes is not None:
            attrs["axes"] = axes
    return attrs


def decode_gather(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op in {"take", "take_along_axis"} and arguments:
        attrs["axis"] = int(arguments[0])
    if op == "gather" and arguments:
        axes = _int_list(arguments[0])
        if axes:
            attrs["axis"] = int(axes[0])
        if len(arguments) >= 2:
            slice_shape = _int_list(arguments[1])
            if slice_shape is not None:
                attrs["slice_shape"] = slice_shape
        if output_shape is not None:
            attrs["shape"] = list(output_shape)
    return attrs


def decode_gather_along_axis(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op in {"take", "take_along_axis"} and arguments:
        attrs["axis"] = int(arguments[0])
    return attrs


def decode_squeeze(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "squeeze" and arguments:
        axes = _int_list(arguments[0])
        if axes is not None:
            attrs["axes"] = axes
    return attrs


def decode_arange(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "arange" and len(arguments) >= 3:
        attrs["start"] = int(arguments[0])
        attrs["end"] = int(arguments[1])
        attrs["step"] = int(arguments[2])
    return attrs


def decode_linspace(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "linspace" and len(arguments) >= 3:
        attrs["start"] = float(arguments[0])
        attrs["stop"] = float(arguments[1])
        attrs["num"] = int(arguments[2])
        if len(arguments) >= 4:
            attrs["endpoint"] = bool(arguments[3])
    return attrs


def decode_split(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "split" and arguments:
        split_arg = arguments[0]
        if isinstance(split_arg, (list, tuple)):
            attrs["split_indices"] = [int(v) for v in split_arg]
        else:
            attrs["num_splits"] = int(split_arg)
        if len(arguments) >= 2:
            axis = _int_list(arguments[1])
            if axis:
                attrs["axis"] = int(axis[0])
        else:
            attrs["axis"] = 0
    return attrs


def decode_expand_dims(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "expanddims" and arguments:
        axes = _int_list(arguments[0])
        if axes is not None:
            attrs["axes"] = axes
    return attrs


def decode_bitwisebinary(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "bitwisebinary" and arguments:
        attrs["mode"] = int(arguments[0])
    return attrs


def decode_scaled_dot_product_attention(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "scaled_dot_product_attention" and arguments:
        # MLX fast primitive state is [scale, do_causal, has_sinks, output_logsumexp].
        # Some serializers may preserve placeholder nulls; filter them first.
        state = [arg for arg in arguments if arg is not None]
        if len(state) >= 4:
            attrs["scale"] = float(state[0])
            attrs["do_causal"] = bool(state[1])
            attrs["has_sinks"] = bool(state[2])
            attrs["output_logsumexp"] = bool(state[3])
    return attrs


def decode_rope(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "rope" and arguments:
        # Primitive state: [dims, traditional, base, scale, ...].
        state = [arg for arg in arguments if arg is not None]
        if len(state) >= 1:
            attrs["dims"] = int(state[0])
        if len(state) >= 2:
            attrs["traditional"] = bool(state[1])
        if len(state) >= 3:
            attrs["base"] = float(state[2])
        if len(state) >= 4:
            attrs["scale"] = float(state[3])
    return attrs


def decode_softmax(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "softmax" and arguments:
        # MLX export can emit [precise] when axis is defaulted to -1.
        first = arguments[0]
        if isinstance(first, bool):
            attrs["precise"] = bool(first)
        else:
            attrs["axis"] = int(first)
            if len(arguments) >= 2 and isinstance(arguments[1], bool):
                attrs["precise"] = bool(arguments[1])
    return attrs


def decode_rmsnorm(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "rmsnorm" and arguments:
        # Primitive state: [eps]
        attrs["eps"] = float(arguments[0])
    return attrs


def decode_layernorm(op, arguments, output_shape, output_dtype):
    return {"axes": [-1], "eps": float(arguments[0]) if arguments else 1e-5}


def decode_cast(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op in {"astype", "cast"} and output_dtype is not None:
        attrs["dtype"] = output_dtype
    return attrs


def decode_broadcast_axes(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "broadcast_axes":
        attrs["ignore_axes"] = _int_list(arguments[0]) if arguments else []
    return attrs


def decode_concat(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op in {"concatenate", "concat"} and arguments:
        attrs["axis"] = int(arguments[0])
    return attrs


def decode_conv(op, arguments, output_shape, output_dtype):
    attrs: dict[str, Any] = {}
    if op == "convolution" and arguments:
        attrs["channels_last"] = True
        strides = _int_list(arguments[0])
        if strides is not None:
            attrs["strides"] = strides

        if len(arguments) >= 2:
            pad_lo = _int_list(arguments[1])
            if pad_lo is not None:
                attrs["padding"] = pad_lo
                attrs["pad_type"] = "custom"

        if len(arguments) >= 3:
            pad_hi = _int_list(arguments[2])
            if pad_hi is not None:
                pad_lo = attrs.get("padding")
                if isinstance(pad_lo, list) and len(pad_lo) == len(pad_hi):
                    attrs["padding"] = [int(v) for pair in zip(pad_lo, pad_hi) for v in pair]

        if len(arguments) >= 4:
            dilations = _int_list(arguments[3])
            if dilations is not None:
                attrs["dilations"] = dilations

        if len(arguments) >= 5:
            attrs["input_dilations"] = _int_list(arguments[4])

        if len(arguments) >= 6:
            try:
                attrs["groups"] = int(arguments[5])
            except Exception:
                pass

        if len(arguments) >= 7 and isinstance(arguments[6], (bool, np.bool_)):
            attrs["flip"] = bool(arguments[6])
    return attrs


def decode_pad(op, arguments, output_shape, output_dtype):
    axes, low, high = arguments
    padding = [0] * (2 * len(output_shape))
    for axis, before, after in zip(axes, low, high, strict=True):
        padding[2 * int(axis):2 * int(axis) + 2] = [int(before), int(after)]
    return {"padding": padding}
