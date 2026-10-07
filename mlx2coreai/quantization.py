"""Build-time weight compression; runtime tensors retain their floating dtype.

The baseline quantizes constant RHS matrices of matmul/addmm projections.
MLX imports are confined to the optional checkpoint preparation helper.
"""
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
import hashlib

import numpy as np

from .ir import Graph, Node


@dataclass(frozen=True)
class WeightQuantization:
    bits: int = 4
    group_size: int = 128
    # Match captured constant names or source labels, rather than module paths.
    exclude: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.bits) is not int or self.bits not in (4, 8):
            raise ValueError("Weight quantization supports 4 or 8 bits.")
        if type(self.group_size) is not int or self.group_size <= 0:
            raise ValueError("group_size must be a positive integer.")
        if isinstance(self.exclude, str) or any(not isinstance(p, str) for p in self.exclude):
            raise ValueError("exclude must be a sequence of glob patterns.")
        object.__setattr__(self, "exclude", tuple(self.exclude))

    def to_dict(self):
        return {"bits": self.bits, "group_size": self.group_size, "exclude": list(self.exclude)}


@dataclass
class PackedLinearWeight:
    name: str
    codes: np.ndarray
    scales: np.ndarray
    biases: np.ndarray
    bits: int

    def dense(self, *, fused=False):
        repeats = tuple(d // s for d, s in zip(self.codes.shape, self.scales.shape, strict=True))
        scale, bias = self.scales, self.biases
        for axis, count in enumerate(repeats):
            scale, bias = scale.repeat(count, axis), bias.repeat(count, axis)
        if fused:
            return (self.codes.astype(np.float64) * scale.astype(np.float64) + bias.astype(np.float64)).astype(np.float32)
        return self.codes.astype(np.float32) * scale + bias


def _key(value):
    value = np.ascontiguousarray(value)
    return value.shape, value.dtype.str, hashlib.sha256(value).digest()


def _compressed_nodes(node, weight, occupied):
    prefix = node.output + "__packed"
    while any(prefix + suffix in occupied for suffix in ("_codes", "_scale", "_zero", "_bias")):
        prefix += "_"
    names = tuple(prefix + suffix for suffix in ("_codes", "_scale", "_zero", "_bias"))
    occupied.update(names)
    dtype = f"uint{weight.bits}"
    arrays = (weight.codes, weight.scales, np.zeros(weight.scales.shape, np.uint8), weight.biases)
    nodes = [Node("constant", (), name, {"value": data, "dtype": dt}, source=f"weight:{weight.name}")
             for name, data, dt in zip(names, arrays, (dtype, "fp32", dtype, "fp32"), strict=True)]
    nodes.append(Node("blockwise_shift_scale", names, node.output, source=node.source))
    return nodes


def _occupied(graph):
    return {spec.name for spec in graph.inputs} | {name for node in graph.nodes for name in node.outputs}


def _linear_constants(graph):
    """Find a constant projection matrix and its contraction axis through transposes."""
    producers = {name: node for node in graph.nodes for name in node.outputs}
    selected = {}
    for node in graph.nodes:
        if node.op == "matmul" and len(node.inputs) == 2:
            lhs, rhs = node.inputs
        elif node.op == "addmm" and len(node.inputs) == 3:
            lhs, rhs = node.inputs[:2] if node.attrs.get("input_order", "cab") == "abc" else node.inputs[1:]
        else:
            continue
        # Constant-only products are not runtime linear projections.
        if producers.get(lhs) is not None and producers[lhs].op in ("const", "constant", "literal"):
            continue
        axis = 0  # A rank-two RHS contracts its first dimension.
        producer = producers.get(rhs)
        while producer is not None:
            if producer.op == "transpose":
                perm = producer.attrs.get("perm")
                if perm is None or sorted(perm) != [0, 1]:
                    producer = None
                    break
                axis = perm[axis]
            elif producer.op == "broadcast_axes" and set(producer.attrs.get("ignore_axes", [])) == {-2, -1}:
                # MLX matmul broadcasts only the leading batch dimensions;
                # the first operand supplies data, the rest supply shape.
                pass
            elif producer.op == "broadcast":
                pass
            else:
                break
            producer = producers.get(producer.inputs[0])
        if producer is not None and producer.op in ("const", "constant", "literal"):
            value = producer.attrs.get("value")
            if isinstance(value, np.ndarray) and value.ndim == 2:
                selected.setdefault(producer.output, set()).add(axis)
    return selected


def quantize_linear_weights(graph: Graph, policy: WeightQuantization):
    """Return a new graph and coverage/error diagnostics for FP32 projection weights.

    Shared constants are reconstructed once and remain shared by every consumer.
    Nondivisible groups and non-FP32 weights are skipped explicitly in the report.
    """
    selected, occupied = _linear_constants(graph), _occupied(graph)
    nodes, rows = [], []
    for node in graph.nodes:
        axes = selected.get(node.output)
        if axes is None:
            nodes.append(node)
            continue
        value = node.attrs["value"]
        row = {"name": node.output, "source": node.source, "shape": list(value.shape)}
        reason = None
        if any(fnmatchcase(node.output, p) or fnmatchcase(node.source or "", p) for p in policy.exclude):
            reason = "excluded"
        elif len(axes) != 1:
            reason = "multiple contraction axes"
        elif value.dtype != np.float32:
            reason = "requires FP32 weights"
        elif not all(value.shape):
            reason = "empty matrix"
        else:
            axis = next(iter(axes))
            if value.shape[axis] == 0 or value.shape[axis] % policy.group_size:
                reason = "contraction dimension must be divisible by group_size"
        if reason is not None:
            row.update(quantized=False, reason=reason)
            nodes.append(node)
        else:
            if not np.isfinite(value).all():
                raise ValueError(f"Linear weight {node.output} contains nonfinite values.")
            # Group independently per output channel, along the input dimension.
            matrix = value if axis == 1 else value.T
            groups = matrix.astype(np.float64).reshape(matrix.shape[0], -1, policy.group_size)
            lo, hi = groups.min(-1), groups.max(-1)
            scale = np.where(hi == lo, 1.0, (hi - lo) / (2**policy.bits - 1)).astype(np.float32)
            bias = lo.astype(np.float32)
            if not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError(f"Linear weight {node.output} has an unrepresentable group scale.")
            codes = np.clip(np.rint((groups - bias[..., None]) / scale[..., None]),
                            0, 2**policy.bits - 1).astype(np.uint8).reshape(matrix.shape)
            weight = PackedLinearWeight(node.output, codes if axis == 1 else codes.T,
                                        scale if axis == 1 else scale.T,
                                        bias if axis == 1 else bias.T, policy.bits)
            restored = weight.dense()
            if not np.isfinite(restored).all():
                raise ValueError(f"Linear weight {node.output} cannot reconstruct finite FP32 values.")
            nodes.extend(_compressed_nodes(node, weight, occupied))
            row.update(quantized=True, axis=axis, original_bytes=value.nbytes,
                       compressed_bytes=(value.size * policy.bits + 7) // 8 + scale.nbytes + bias.nbytes
                       + (scale.size * policy.bits + 7) // 8,
                       max_abs_error=float(np.max(np.abs(restored.astype(np.float64) - value))))
        rows.append(row)
    report = {"policy": policy.to_dict(), "selected_weights": len(rows),
              "quantized_weights": sum(row["quantized"] for row in rows), "weights": rows}
    result = replace(graph, nodes=nodes)
    result.validate()
    return result, report


class PackedLinearWeights:
    """Preserve affine MLX 2/4/8-bit Linear weights as exact packed constants."""

    def __init__(self, *, require_all=False):
        self.weights = {}
        self._transposed = {}
        self.module_count = 0
        self.require_all = require_all

    def add(self, name, module, dense):
        bits, group_size = module.bits, module.group_size
        if bits not in (2, 4, 8) or getattr(module, "mode", "affine") != "affine":
            raise ValueError("Packed Linear preservation supports affine 2/4/8-bit weights.")
        if type(group_size) is not int or group_size <= 0:
            raise ValueError("Packed Linear group_size must be a positive integer.")
        dense = np.asarray(dense)
        if dense.dtype != np.float32 or dense.ndim != 2 or not all(dense.shape) or dense.shape[1] % group_size:
            raise ValueError("Packed Linear preservation requires grouped FP32 matrix weights.")
        words = np.asarray(module.weight)
        if words.dtype != np.uint32 or words.size * (32 // bits) != dense.size:
            raise ValueError("Packed Linear codes do not match the dense matrix shape.")
        codes = ((words[..., None] >> (bits * np.arange(32 // bits, dtype=np.uint32))) & (2**bits - 1))
        weight = PackedLinearWeight(name, codes.reshape(dense.shape).astype(np.uint8),
                                    np.asarray(module.scales).astype(np.float32),
                                    np.asarray(module.biases).astype(np.float32), bits)
        if weight.scales.shape != (dense.shape[0], dense.shape[1] // group_size) or weight.biases.shape != weight.scales.shape:
            raise ValueError("Packed Linear scale/bias shapes do not match the groups.")
        # MLX uses fused multiply-add for ordinary affine checkpoints. CoreAI
        # backends may fuse or split these FP32 operations; retain the original
        # codes/parameters and accept only either exact arithmetic convention.
        if not np.array_equal(weight.dense(), dense) and not np.array_equal(weight.dense(fused=True), dense):
            raise ValueError(f"Packed weight {name} does not exactly reconstruct the FP32 weight.")
        self.weights[_key(dense)] = weight
        self._transposed[_key(dense.T)] = replace(weight, codes=weight.codes.T,
                                                scales=weight.scales.T, biases=weight.biases.T)

    def __call__(self, graph):
        nodes, used, occupied = [], set(), _occupied(graph)
        # Some captures fold the transpose into the constant itself.
        matches = dict(self._transposed)
        matches.update(self.weights)
        shapes = {key[:2] for key in matches}
        for node in graph.nodes:
            value = node.attrs.get("value") if node.op in ("const", "constant", "literal") else None
            key = (_key(value) if isinstance(value, np.ndarray) and (value.shape, value.dtype.str) in shapes else None)
            weight = matches.get(key)
            if weight is None:
                nodes.append(node)
            else:
                used.add(weight.name)
                nodes.extend(_compressed_nodes(node, weight, occupied))
        missing = {weight.name for weight in self.weights.values()} - used
        if self.require_all and missing:
            raise ValueError(f"Capture missed packed Linear weights: {sorted(missing)}")
        result = replace(graph, nodes=nodes)
        result.validate()
        return result


def prepare_quantized_linears(model, *, preserve=True, require_all=False):
    """Materialize affine quantized MLX Linears for capture, optionally preserving storage.

    Mutates only QuantizedLinear modules. Evaluate/cast the model to FP32 first.
    The returned registry is usable as ConversionConfig.graph_transform.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    if isinstance(model, nn.QuantizedLinear):
        raise ValueError("Prepare a containing Module rather than a root QuantizedLinear.")
    packed = PackedLinearWeights(require_all=require_all)
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.QuantizedLinear):
            continue
        if getattr(module, "mode", "affine") != "affine" or module.bits not in (2, 4, 8):
            raise ValueError("Linear preparation supports affine 2/4/8-bit checkpoints.")
        weight = mx.dequantize(module.weight, module.scales, module.biases,
                              group_size=module.group_size, bits=module.bits).astype(mx.float32)
        dense = nn.Linear(weight.shape[1], weight.shape[0], bias="bias" in module)
        dense.weight = weight
        if "bias" in module:
            dense.bias = module.bias
        mx.eval(dense.parameters())
        if preserve:
            packed.add(name, module, np.asarray(weight))
        replacements.append((name, dense))
    model.update_modules(tree_unflatten(replacements))
    packed.module_count = len(replacements)
    return packed
