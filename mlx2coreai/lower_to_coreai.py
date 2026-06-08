from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable

import ml_dtypes
import numpy as np

from coreai._compiler.dialects import coreai
from coreai._compiler.ir import (
    ArrayAttr,
    BF16Type,
    DenseResourceElementsAttr,
    DictAttr,
    F16Type,
    F32Type,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    RankedTensorType,
    StringAttr,
    Type,
    Value,
)
from coreai.authoring import AIProgram, Context
from coreai._compiler.types import TensorSpec as CoreAITensorSpec

from ._composite_declaration import generate_composite_decl
from .ir import Graph, Node, StateSpec, TensorSpec, is_dynamic_dim_ref
from .op_registry import coreai_op_for_mlx, ensure_supported
from .passes import infer_graph_specs, normalize_graph


_DTYPE_ALIASES = {
    "half": "fp16",
    "float16": "fp16",
    "fp16": "fp16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float": "fp32",
    "float32": "fp32",
    "fp32": "fp32",
    "double": "fp64",
    "float64": "fp64",
    "fp64": "fp64",
    "int": "int32",
    "int32": "int32",
    "long": "int64",
    "int64": "int64",
    "bool": "bool",
}


@dataclass(slots=True)
class WeightInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    source: str
    storage: str = "inline"
    nbytes: int = 0
    resource_name: str | None = None
    external_weight_threshold: int | None = None
    downcast: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "shape": [int(v) for v in self.shape],
            "dtype": self.dtype,
            "source": self.source,
            "storage": self.storage,
            "nbytes": int(self.nbytes),
        }
        if self.resource_name is not None:
            payload["resource_name"] = self.resource_name
        if self.external_weight_threshold is not None:
            payload["external_weight_threshold"] = int(self.external_weight_threshold)
        if self.downcast is not None:
            payload["downcast"] = self.downcast
        return payload


@dataclass(slots=True)
class CoreAILoweringConfig:
    entrypoint_name: str = "main"
    optimize: bool = True
    state_specs: list[StateSpec] | None = None
    constant_inputs: Mapping[str, Any] | None = None
    public_input_names: set[str] | None = None
    externalize_weights: bool = True
    external_weight_threshold: int = 10


@dataclass(slots=True)
class LoweredCoreAIProgram:
    program: AIProgram
    graph: Graph
    public_inputs: list[TensorSpec]
    weight_manifest: list[WeightInfo] = field(default_factory=list)
    unresolved_extra_inputs: list[str] = field(default_factory=list)
    optimized: bool = False
    optimization_skip_reason: str | None = None


def _normalize_dtype(dtype: str) -> str:
    return _DTYPE_ALIASES.get(str(dtype).strip().lower(), str(dtype).strip().lower())


def _element_type(dtype: str) -> Type:
    dtype = _normalize_dtype(dtype)
    if dtype == "fp16":
        return F16Type.get()
    if dtype == "bf16":
        return BF16Type.get()
    if dtype in {"fp32", "fp64"}:
        return F32Type.get()
    if dtype == "int32":
        return IntegerType.get_signed(32)
    if dtype == "int64":
        # CoreAI can represent si64 in MLIR, but the runtime stack is
        # generally <=32-bit oriented. Input types are narrowed to match the
        # conversion policy used for constants.
        return IntegerType.get_signed(32)
    if dtype == "bool":
        return IntegerType.get_signless(1)
    raise ValueError(f"Unsupported CoreAI dtype: {dtype}")


def _tensor_type(spec: TensorSpec | StateSpec) -> RankedTensorType:
    shape = [
        int(dim) if int(dim) >= 0 else RankedTensorType.get_dynamic_size()
        for dim in spec.shape
    ]
    return RankedTensorType.get(shape, _element_type(spec.dtype))


def _np_dtype_for_ir(dtype: str) -> Any:
    dtype = _normalize_dtype(dtype)
    if dtype == "fp16":
        return np.float16
    if dtype == "bf16":
        return ml_dtypes.bfloat16
    if dtype in {"fp32", "fp64"}:
        return np.float32
    if dtype in {"int32", "int64"}:
        return np.int32
    if dtype == "bool":
        return np.bool_
    raise ValueError(f"Unsupported dtype for constant: {dtype}")


def _array_to_coreai(value: Any, dtype_hint: str | None = None) -> tuple[np.ndarray, str | None]:
    arr = np.asarray(value)
    downcast: str | None = None
    if dtype_hint is not None:
        dtype_hint = _normalize_dtype(dtype_hint)

    if arr.dtype == np.float64 or dtype_hint == "fp64":
        arr = arr.astype(np.float32)
        downcast = "fp64->fp32"
    elif arr.dtype == np.int64 or dtype_hint == "int64":
        if arr.size and (arr.min() < np.iinfo(np.int32).min or arr.max() > np.iinfo(np.int32).max):
            raise ValueError("int64 constant cannot be safely downcast to int32.")
        arr = arr.astype(np.int32)
        downcast = "int64->int32"
    elif dtype_hint == "bf16":
        arr = arr.astype(ml_dtypes.bfloat16)
    elif dtype_hint is not None:
        arr = arr.astype(_np_dtype_for_ir(dtype_hint))
    return np.ascontiguousarray(arr), downcast


def _static_shape(value: Value) -> list[int]:
    shape = list(value.type.shape)
    if any(int(dim) < 0 for dim in shape):
        raise ValueError(f"Expected static shape, got {shape}.")
    return [int(dim) for dim in shape]


def _rank(value: Value) -> int:
    return int(value.type.rank)


def _axes(value: Any, *, rank: int | None = None) -> list[int]:
    if value is None:
        return list(range(rank or 0))
    if isinstance(value, int):
        raw = [value]
    else:
        raw = [int(v) for v in value]
    if rank is None:
        return raw
    out: list[int] = []
    for axis in raw:
        axis = int(axis)
        if axis < 0:
            axis += rank
        out.append(axis)
    return out


def _shrink_if_needed(value: Value, axes: Sequence[int], keep_dims: bool) -> Value:
    if keep_dims or not axes:
        return value
    return coreai.shrink_dims(value, list(axes))


def _as_shape_value(shape: Sequence[int]) -> np.ndarray:
    return np.asarray([int(v) for v in shape], dtype=np.int32)


def _as_tile_value(repeats: Sequence[int]) -> np.ndarray:
    return np.asarray([int(v) for v in repeats], dtype=np.uint32)


def _shape_has_runtime_dims(shape: Sequence[Any]) -> bool:
    return any(is_dynamic_dim_ref(dim) for dim in shape)


def _static_or_dynamic_shape(value: Value) -> list[int]:
    return [int(dim) for dim in value.type.shape]


def _dim_1d_from_value(value: Value, axis: int, *, dtype: Any = np.int32) -> Value:
    rank = _rank(value)
    axis = axis + rank if axis < 0 else axis
    static_dim = int(value.type.shape[axis])
    if static_dim >= 0:
        return coreai.constant(np.asarray([static_dim], dtype=dtype))
    dim = coreai.slice_(coreai.get_shape(value), [axis], [axis + 1], [1])
    return coreai.cast(dim, dtype=dtype)


def _dim_scalar_from_value(value: Value, axis: int, *, dtype: Any = np.int32) -> Value:
    return coreai.shrink_dims(_dim_1d_from_value(value, axis, dtype=dtype), [0])


def _mixed_shape_operand(parts: Sequence[Any], *, dtype: Any = np.int32) -> Value | np.ndarray:
    if not any(isinstance(part, Value) for part in parts):
        return np.asarray([int(part) for part in parts], dtype=dtype)
    operands: list[Value] = []
    for part in parts:
        if isinstance(part, Value):
            operands.append(coreai.cast(part, dtype=dtype))
        else:
            operands.append(coreai.constant(np.asarray([int(part)], dtype=dtype)))
    return coreai.concat(0, operands) if len(operands) > 1 else operands[0]


def _value_shape_operand(
    value: Value,
    *,
    overrides: Mapping[int, Any] | None = None,
    dtype: Any = np.int32,
) -> Value | np.ndarray:
    rank = _rank(value)
    overrides = dict(overrides or {})
    parts: list[Any] = []
    for axis in range(rank):
        if axis in overrides:
            parts.append(overrides[axis])
        else:
            static_dim = int(value.type.shape[axis])
            parts.append(static_dim if static_dim >= 0 else _dim_1d_from_value(value, axis, dtype=dtype))
    return _mixed_shape_operand(parts, dtype=dtype)


def _ranked_tensor_type(shape: Sequence[Any], element_type: Type) -> RankedTensorType:
    dims = [
        RankedTensorType.get_dynamic_size()
        if isinstance(dim, Value) or is_dynamic_dim_ref(dim) or int(dim) < 0
        else int(dim)
        for dim in shape
    ]
    return RankedTensorType.get(dims, element_type)


def _reshape_with_mixed_shape(value: Value, shape: Sequence[Any]) -> Value:
    shape_operand = _mixed_shape_operand(shape)
    if not isinstance(shape_operand, Value):
        shape_operand = coreai.constant(shape_operand)
    return coreai.ReshapeOp(
        value,
        shape_operand,
        results=[_ranked_tensor_type(shape, value.type.element_type)],
    ).result


def _reshape_like(value: Value, exemplar: Value) -> Value:
    shape: list[Any] = []
    for axis, dim in enumerate(exemplar.type.shape):
        dim = int(dim)
        shape.append(dim if dim >= 0 else _dim_1d_from_value(exemplar, axis))
    shape_operand = _mixed_shape_operand(shape)
    if not isinstance(shape_operand, Value):
        shape_operand = coreai.constant(shape_operand)
    return coreai.ReshapeOp(
        value,
        shape_operand,
        results=[exemplar.type],
    ).result


def _resource_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    if not cleaned:
        cleaned = "constant"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return f"mlx2coreai_{cleaned}"


def _is_bool_value(value: Value) -> bool:
    return str(value.type.element_type) in {"i1", "!coreai.bool"}


def _zero_for_value(value: Value) -> Any:
    return False if _is_bool_value(value) else coreai.constant(0, dtype=value.type.element_type)


def _normalize_axis(axis: int, rank: int) -> int:
    axis = int(axis)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"Axis {axis} is out of range for rank {rank}.")
    return axis


def _as_int_list(value: Any, count: int, *, default: int = 1) -> list[int]:
    if value is None:
        return [int(default)] * count
    if isinstance(value, int):
        return [int(value)] * count
    out = [int(v) for v in value]
    if len(out) == 1:
        return out * count
    if len(out) != count:
        raise ValueError(f"Expected {count} values, got {out}.")
    return out


class CoreAILowerer:
    def __init__(self, config: CoreAILoweringConfig | None = None) -> None:
        self.config = config or CoreAILoweringConfig()
        self.context = Context()
        self.module: Module | None = None
        self.location: Location | None = None
        self.current_graph: coreai.GraphOp | None = None
        self.env: dict[str, Value] = {}
        self.inferred: dict[str, Any] = {}
        self.weight_manifest: list[WeightInfo] = []
        self.unresolved_extra_inputs: list[str] = []
        self._private_graph_counter = 0

    def lower(self, graph: Graph) -> LoweredCoreAIProgram:
        graph = normalize_graph(graph)
        graph.validate()
        ensure_supported(graph)
        self.inferred = infer_graph_specs(graph)

        with self.context:
            self.location = Location.unknown(self.context._mlir_context)
            with self.location:
                self.module = Module.create()
                with InsertionPoint(self.module.body):
                    public_inputs = self._public_inputs(graph)
                    input_names = [spec.name for spec in public_inputs]
                    input_types = [_tensor_type(spec) for spec in public_inputs]
                    graph_op = coreai.GraphOp(
                        name=self.config.entrypoint_name,
                        input_types=input_types,
                        result_types=[],
                        input_names=input_names,
                        loc=self.location,
                    )
                    self.current_graph = graph_op
                    with graph_op.block:
                        self._seed_inputs(graph_op, public_inputs, graph)
                        for node in graph.nodes:
                            self.env[node.output] = self._lower_node(node)
                        outputs = OrderedDict((name, self.env[name]) for name in graph.outputs)
                        graph_op.set_outputs_spec_from_dict(outputs)
                    self._mark_mutable_buffers(graph_op, public_inputs, graph)

        assert self.module is not None
        program = AIProgram._from_mlir_module(self.module)
        optimization_skip_reason = _optimization_skip_reason(graph) if self.config.optimize else None
        optimized = bool(self.config.optimize and optimization_skip_reason is None)
        if optimized:
            program.optimize()
        return LoweredCoreAIProgram(
            program=program,
            graph=graph,
            public_inputs=public_inputs,
            weight_manifest=list(self.weight_manifest),
            unresolved_extra_inputs=list(self.unresolved_extra_inputs),
            optimized=optimized,
            optimization_skip_reason=optimization_skip_reason,
        )

    def _public_inputs(self, graph: Graph) -> list[TensorSpec]:
        constants = self.config.constant_inputs or {}
        public_names = self.config.public_input_names
        out: list[TensorSpec] = []
        for spec in graph.inputs:
            if self.config.externalize_weights and spec.name in constants:
                continue
            if public_names is not None and spec.name not in public_names:
                self.unresolved_extra_inputs.append(spec.name)
            out.append(spec)
        return out

    def _seed_inputs(self, graph_op: coreai.GraphOp, public_inputs: list[TensorSpec], graph: Graph) -> None:
        for idx, spec in enumerate(public_inputs):
            self.env[spec.name] = graph_op.arguments[idx]
        constants = self.config.constant_inputs or {}
        for spec in graph.inputs:
            if spec.name in self.env:
                continue
            if spec.name in constants:
                self.env[spec.name] = self._constant(
                    spec.name,
                    constants[spec.name],
                    dtype=spec.dtype,
                    source="externalized_input",
                )

    def _constant(self, name: str, value: Any, *, dtype: str | None = None, source: str = "constant") -> Value:
        arr, downcast = _array_to_coreai(value, dtype)
        storage = "resource" if self._should_use_resource_constant(arr) else "inline"
        resource_name = _resource_name(name) if storage == "resource" else None
        self.weight_manifest.append(
            WeightInfo(
                name=name,
                shape=tuple(int(v) for v in arr.shape),
                dtype=str(arr.dtype),
                source=source,
                storage=storage,
                nbytes=int(arr.nbytes),
                resource_name=resource_name,
                external_weight_threshold=int(self.config.external_weight_threshold),
                downcast=downcast,
            )
        )
        if storage == "resource":
            return self._resource_constant(arr, resource_name or name)
        return coreai.constant(arr)

    def _should_use_resource_constant(self, arr: np.ndarray) -> bool:
        if not self.config.externalize_weights:
            return False
        threshold = int(self.config.external_weight_threshold)
        if threshold < 0:
            return False
        if int(arr.size) < threshold:
            return False
        if arr.dtype == np.bool_:
            return False
        return bool(np.issubdtype(arr.dtype, np.number) or arr.dtype == ml_dtypes.bfloat16)

    def _resource_constant(self, arr: np.ndarray, resource_name: str) -> Value:
        tensor_type = CoreAITensorSpec(list(arr.shape), arr.dtype.type)._to_mlir_type()
        attr = DenseResourceElementsAttr.get_from_buffer(
            np.ascontiguousarray(arr),
            resource_name,
            tensor_type,
        )
        return coreai.ConstantOp(value=attr, loc=self.location).result

    def _dynamic_dim_1d(self, ref: Mapping[str, Any], *, dtype: Any = np.int32) -> Value:
        source = str(ref["source"])
        axis = int(ref["axis"])
        if source not in self.env:
            raise ValueError(f"Dynamic shape references unknown source '{source}'.")
        return _dim_1d_from_value(self.env[source], axis, dtype=dtype)

    def _dynamic_dim_scalar(self, ref: Mapping[str, Any], *, dtype: Any = np.int32) -> Value:
        return coreai.shrink_dims(self._dynamic_dim_1d(ref, dtype=dtype), [0])

    def _shape_operand(self, shape: Sequence[Any], *, dtype: Any = np.int32) -> Value | np.ndarray:
        if not _shape_has_runtime_dims(shape):
            return np.asarray([int(dim) for dim in shape], dtype=dtype)
        parts: list[Value] = []
        for dim in shape:
            if is_dynamic_dim_ref(dim):
                parts.append(self._dynamic_dim_1d(dim, dtype=dtype))
            else:
                parts.append(coreai.constant(np.asarray([int(dim)], dtype=dtype)))
        return coreai.concat(0, parts) if len(parts) > 1 else parts[0]

    def _reshape(self, value: Value, shape: Sequence[Any]) -> Value:
        shape_operand = self._shape_operand(shape)
        if not isinstance(shape_operand, Value):
            shape_operand = coreai.constant(shape_operand)
        return coreai.ReshapeOp(
            value,
            shape_operand,
            results=[_ranked_tensor_type(shape, value.type.element_type)],
        ).result

    def _scalar_operand(self, value: Any, *, dtype: Any = np.int32) -> Value:
        if is_dynamic_dim_ref(value):
            return self._dynamic_dim_scalar(value, dtype=dtype)
        return coreai.constant(int(value), dtype=dtype)

    def _index_operand(
        self,
        value: Any,
        rank: int,
        default: int | None,
        *,
        x: Value | None = None,
    ) -> Value | np.ndarray:
        if value is None:
            if default is None:
                if x is None:
                    raise ValueError("Cannot infer slice end without input shape.")
                return coreai.cast(coreai.get_shape(x), dtype=np.int32)
            return np.asarray([int(default)] * rank, dtype=np.int32)

        raw = list(value) if isinstance(value, (list, tuple)) else [value]
        if len(raw) < rank:
            if default is None and x is not None:
                for axis in range(len(raw), rank):
                    raw.append(_dim_1d_from_value(x, axis, dtype=np.int32))
            else:
                raw.extend([int(default or 0)] * (rank - len(raw)))

        parts: list[Any] = []
        for item in raw[:rank]:
            if is_dynamic_dim_ref(item):
                parts.append(self._dynamic_dim_1d(item, dtype=np.int32))
            else:
                parts.append(item)
        return _mixed_shape_operand(parts, dtype=np.int32)

    def _mark_mutable_buffers(self, graph_op: coreai.GraphOp, public_inputs: list[TensorSpec], graph: Graph) -> None:
        state_specs = {spec.name: spec for spec in (self.config.state_specs or [])}
        if not state_specs:
            return
        output_by_state: dict[str, str] = {}
        for node in graph.nodes:
            op = coreai_op_for_mlx(node.op)
            if op in {"write_state", "state_update_masked"} and node.inputs:
                state_name = node.inputs[0]
                if state_name in state_specs:
                    output_by_state[state_name] = node.output
        if not output_by_state:
            return

        existing_arg_attrs = list(graph_op.arg_attrs) if graph_op.arg_attrs else []
        arg_attrs = []
        for idx, spec in enumerate(public_inputs):
            attrs: dict[str, Any] = {}
            if idx < len(existing_arg_attrs):
                for named_attr in existing_arg_attrs[idx]:
                    attrs[named_attr.name] = named_attr.attr
            if spec.name in output_by_state:
                attrs["MutableBuffers.buffer_mutation"] = StringAttr.get(output_by_state[spec.name])
            arg_attrs.append(DictAttr.get(attrs))
        graph_op.arg_attrs = ArrayAttr.get(arg_attrs)

    def _emit_private_composite(
        self,
        *,
        node: Node,
        composite_name: str,
        input_values: list[Value],
        input_names: list[str],
        attrs: dict[str, Any],
        body: Callable[[list[Value]], Value],
        result_type: Type | None = None,
    ) -> Value:
        assert self.module is not None
        assert self.location is not None
        assert self.current_graph is not None
        self._private_graph_counter += 1
        graph_name = f"__mlx2coreai_{composite_name}_{self._private_graph_counter}_{node.output}"
        graph_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in graph_name)
        composite_decl = generate_composite_decl(
            self.module.context,
            composite_name,
            input_names,
            ["output"],
            dict(attrs),
        )
        with InsertionPoint(self.module.body):
            private = coreai.GraphOp(
                name=graph_name,
                input_types=[value.type for value in input_values],
                result_types=[],
                input_names=input_names,
                private=True,
                no_inline=True,
                composite_decl=composite_decl,
                loc=self.location,
            )
            with private.block:
                result = body(list(private.arguments))
                private.set_outputs_spec_from_dict(OrderedDict([("output", result)]))
                result_type = result.type if result_type is None else result_type
        with self.current_graph.block:
            [out] = coreai.invoke(
                results=[result_type or input_values[0].type],
                callee=graph_name,
                operands=input_values,
                loc=self.location,
            )
        return out

    def _lower_node(self, node: Node) -> Value:
        op = coreai_op_for_mlx(node.op)
        if op is None:
            raise ValueError(f"Unsupported MLX op: {node.op}")
        if op == "const":
            if "value" not in node.attrs:
                raise ValueError(f"constant node '{node.output}' requires 'value'.")
            return self._constant(node.output, node.attrs["value"], dtype=node.attrs.get("dtype"), source="constant_node")
        if op == "identity":
            return self.env[node.inputs[0]]
        if op == "read_state":
            return self.env[node.inputs[0]]
        if op == "write_state":
            return self.env[node.inputs[1]]
        if op == "state_update_masked":
            state = self.env[node.inputs[0]]
            value = self.env[node.inputs[1]]
            if len(node.inputs) < 3:
                return value
            mask = self.env[node.inputs[2]]
            return coreai.broadcasting_where(mask, value, state)

        if op in _BINARY_OPS:
            x, y = self.env[node.inputs[0]], self.env[node.inputs[1]]
            return _BINARY_OPS[op](x, y)
        if op == "bitwisebinary":
            return self._lower_bitwise_binary(node)
        if op == "inverse":
            x = self.env[node.inputs[0]]
            eps = float(node.attrs.get("epsilon", 0.0))
            denom = coreai.broadcasting_add(x, eps) if eps else x
            return coreai.broadcasting_divide(coreai.constant(1.0, dtype=x.type.element_type), denom)
        if op == "matmul":
            return coreai.broadcasting_batch_matmul(self.env[node.inputs[0]], self.env[node.inputs[1]])
        if op == "addmm":
            bias, x, y = [self.env[name] for name in node.inputs[:3]]
            return coreai.broadcasting_add(bias, coreai.broadcasting_batch_matmul(x, y))

        if op == "softmax":
            axis = int(node.attrs.get("axis", -1))
            if axis < 0:
                axis += _rank(self.env[node.inputs[0]])
            return coreai.softmax(self.env[node.inputs[0]], axis)
        if op in _UNARY_OPS:
            return _UNARY_OPS[op](self.env[node.inputs[0]])
        if op == "negative":
            return coreai.broadcasting_mul(self.env[node.inputs[0]], -1.0)
        if op == "floor_div":
            return coreai.broadcasting_floor_divide(self.env[node.inputs[0]], self.env[node.inputs[1]])

        if op == "reduce":
            return self._lower_generic_reduce(node)
        if op in {"all", "any"}:
            return self._lower_bool_reduce(node, any_mode=(op == "any"))
        if op in _REDUCE_OPS:
            return self._lower_reduce(node, _REDUCE_OPS[op])
        if op in {"reduce_argmax", "reduce_argmin"}:
            return self._lower_arg_reduce(node, is_min=(op == "reduce_argmin"))

        if op in {"reshape", "flatten", "unflatten"}:
            shape = node.attrs.get("shape")
            if shape is None:
                shape = self.inferred.get(node.output).shape
            return self._reshape(self.env[node.inputs[0]], shape)
        if op == "moveaxis":
            return self._lower_moveaxis(node)
        if op == "swapaxes":
            return self._lower_swapaxes(node)
        if op == "transpose":
            x = self.env[node.inputs[0]]
            perm = node.attrs.get("perm")
            if perm is None:
                perm = list(reversed(range(_rank(x))))
            return coreai.transpose(x, np.asarray([int(v) for v in perm], dtype=np.uint32))
        if op in {"expand_dims", "atleast_1d", "atleast_2d", "atleast_3d"}:
            return self._lower_expand(node, op)
        if op == "squeeze":
            x = self.env[node.inputs[0]]
            axes = node.attrs.get("axes")
            if axes is None:
                shape = list(x.type.shape)
                axes = [idx for idx, dim in enumerate(shape) if int(dim) == 1]
            return coreai.shrink_dims(x, _axes(axes, rank=_rank(x)))
        if op == "slice_by_index":
            x = self.env[node.inputs[0]]
            rank = _rank(x)
            begin = self._index_operand(node.attrs.get("begin"), rank, 0)
            end = self._index_operand(node.attrs.get("end"), rank, None, x=x)
            stride = self._index_operand(node.attrs.get("stride"), rank, 1)
            return coreai.slice_(x, begin, end, stride)
        if op == "slice_update":
            return self._lower_slice_update(node)
        if op == "concat":
            axis = int(node.attrs.get("axis", 0))
            return coreai.concat(axis, [self.env[name] for name in node.inputs])
        if op == "split":
            return self._lower_split(node)
        if op == "broadcast_to":
            shape = node.attrs.get("shape")
            if shape is None:
                shape = self.inferred.get(node.output).shape
            return coreai.broadcast_to(self.env[node.inputs[0]], self._shape_operand(shape))
        if op == "broadcast_arrays":
            return self._lower_broadcast_arrays(node)

        if op == "gather":
            return self._lower_gather(node)
        if op == "gather_along_axis":
            x = self.env[node.inputs[0]]
            indices = self.env[node.inputs[1]]
            axis = _normalize_axis(int(node.attrs.get("axis", 0)), _rank(x))
            return coreai.gather_along_axis(x, indices, np.asarray(axis, dtype=np.int32))
        if op == "select":
            return coreai.broadcasting_where(
                self.env[node.inputs[0]], self.env[node.inputs[1]], self.env[node.inputs[2]]
            )
        if op == "cast":
            dtype = _normalize_dtype(str(node.attrs.get("dtype", "fp32")))
            return coreai.cast(self.env[node.inputs[0]], _element_type(dtype))
        if op == "number_of_elements":
            shape = coreai.cast(coreai.get_shape(self.env[node.inputs[0]]), IntegerType.get_signed(32))
            return coreai.reduce_product(shape, [0])

        if op in {"zeros", "ones", "full"}:
            if op == "full" and len(node.inputs) == 1 and "shape" not in node.attrs and "value" not in node.attrs:
                return self.env[node.inputs[0]]
            return self._lower_fill(node, op)
        if op in {"zeros_like", "ones_like", "full_like"}:
            x = self.env[node.inputs[0]]
            value = 0.0 if op == "zeros_like" else 1.0
            if op == "full_like":
                value = float(node.attrs.get("value", 0.0))
            return coreai.broadcasting_add(coreai.broadcasting_mul(x, 0.0), value)
        if op == "arange":
            start = node.attrs.get("start", 0)
            stop = node.attrs.get("stop", node.attrs.get("end", 0))
            step = node.attrs.get("step", 1)
            if is_dynamic_dim_ref(start) or is_dynamic_dim_ref(stop) or is_dynamic_dim_ref(step):
                dyn = RankedTensorType.get_dynamic_size()
                return coreai.RangeOp(
                    self._scalar_operand(start),
                    self._scalar_operand(stop),
                    self._scalar_operand(step),
                    results=[RankedTensorType.get([dyn], IntegerType.get_signed(32))],
                ).result
            return coreai.range_(
                coreai.constant(int(start), dtype=np.int32),
                coreai.constant(int(stop), dtype=np.int32),
                coreai.constant(int(step), dtype=np.int32),
            )
        if op == "linspace":
            return self._lower_linspace(node)

        if op == "rmsnorm":
            return self._lower_rmsnorm(node)
        if op == "layernorm":
            return self._lower_layernorm(node)
        if op == "scaled_dot_product_attention":
            return self._lower_sdpa(node)
        if op == "rope":
            return self._lower_rope(node)

        if op == "outer":
            x = coreai.expand_dims(self.env[node.inputs[0]], [_rank(self.env[node.inputs[0]])])
            y = coreai.expand_dims(self.env[node.inputs[1]], [0])
            return coreai.broadcasting_mul(x, y)
        if op == "inner":
            x, y = self.env[node.inputs[0]], self.env[node.inputs[1]]
            prod = coreai.broadcasting_mul(x, y)
            return coreai.reduce_sum(prod, [_rank(prod) - 1])
        if op == "tensordot":
            return self._lower_tensordot(node)
        if op == "logaddexp":
            x, y = self.env[node.inputs[0]], self.env[node.inputs[1]]
            m = coreai.broadcasting_maximum(x, y)
            return coreai.broadcasting_add(
                m,
                coreai.log(
                    coreai.broadcasting_add(
                        coreai.exp(coreai.broadcasting_sub(x, m)),
                        coreai.exp(coreai.broadcasting_sub(y, m)),
                    )
                ),
            )
        if op in {"var", "std"}:
            return self._lower_var_std(node, compute_std=(op == "std"))
        if op == "array_equal":
            return self._lower_array_equal(node)
        if op == "isclose":
            return self._lower_isclose(node)
        if op == "allclose":
            close = self._lower_isclose(node)
            reduced = coreai.all_(close, list(range(_rank(close))))
            return coreai.shrink_dims(reduced, list(range(_rank(reduced)))) if _rank(reduced) else reduced
        if op == "nan_to_num":
            return self._lower_nan_to_num(node)
        if op == "divmod":
            x, y = self.env[node.inputs[0]], self.env[node.inputs[1]]
            q = coreai.broadcasting_floor_divide(x, y)
            r = coreai.broadcasting_modulo(x, y)
            which = str(node.attrs.get("output", node.attrs.get("which", ""))).strip().lower()
            output_index = int(node.attrs.get("output_index", 1 if which in {"remainder", "rem", "mod"} else 0))
            return r if output_index == 1 or which in {"remainder", "rem", "mod"} else q

        if op == "diag":
            return self._lower_diag(node)
        if op == "diagonal":
            return self._lower_diagonal(node)
        if op == "trace":
            return self._lower_trace(node)
        if op == "tri":
            return self._lower_tri(node)
        if op in {"tril", "triu"}:
            return self._lower_triangular_band(node, lower=(op == "tril"))
        if op == "eye":
            return self._lower_eye(node)
        if op == "meshgrid":
            return self._lower_meshgrid(node)
        if op == "kron":
            return self._lower_kron(node)
        if op == "conv":
            return self._lower_conv(node, transpose=False)
        if op == "conv_transpose":
            return self._lower_conv(node, transpose=True)

        raise ValueError(f"CoreAI lowering for op '{op}' is not implemented yet.")

    def _lower_arg_reduce(self, node: Node, *, is_min: bool) -> Value:
        x = self.env[node.inputs[0]]
        axis = node.attrs.get("axis")
        axes = node.attrs.get("axes")
        if axis is None and axes is not None:
            axis = int(list(axes)[0])
        if axis is None:
            axis = 0
        axis = _normalize_axis(int(axis), _rank(x))
        source = coreai.broadcasting_mul(x, -1.0) if is_min else x
        result = coreai.argmax(source, np.asarray(axis, dtype=np.int32))
        result = coreai.cast(result, IntegerType.get_signed(32))
        return _shrink_if_needed(result, [axis], bool(node.attrs.get("keep_dims", False)))

    def _lower_bitwise_binary(self, node: Node) -> Value:
        x, y = self.env[node.inputs[0]], self.env[node.inputs[1]]
        raw_mode = node.attrs.get("mode", node.attrs.get("op", node.attrs.get("kind", "and")))
        if isinstance(raw_mode, int):
            mode = {0: "and", 1: "or", 2: "xor"}.get(int(raw_mode), "and")
        else:
            mode = str(raw_mode).strip().lower()
        if mode in {"and", "bitwise_and"}:
            return coreai.broadcasting_and(x, y) if _is_bool_value(x) else coreai.broadcasting_bitwise_and(x, y)
        if mode in {"or", "bitwise_or"}:
            return coreai.broadcasting_or(x, y) if _is_bool_value(x) else coreai.broadcasting_bitwise_or(x, y)
        if mode in {"xor", "bitwise_xor"}:
            return coreai.broadcasting_xor(x, y) if _is_bool_value(x) else coreai.broadcasting_bitwise_xor(x, y)
        raise ValueError(f"Unsupported bitwisebinary mode for node '{node.output}': {raw_mode!r}.")

    def _lower_reduce(self, node: Node, fn: Callable[[Value, list[int]], Value]) -> Value:
        x = self.env[node.inputs[0]]
        axes = _axes(node.attrs.get("axes"), rank=_rank(x))
        result = fn(x, axes)
        return _shrink_if_needed(result, axes, bool(node.attrs.get("keep_dims", False)))

    def _lower_generic_reduce(self, node: Node) -> Value:
        mode = int(node.attrs.get("mode", 2))
        if mode == 0:
            return self._lower_reduce(node, lambda x, axes: coreai.all_(x, axes))
        if mode == 1:
            return self._lower_reduce(node, lambda x, axes: coreai.any_(x, axes))
        if mode == 2:
            return self._lower_reduce(node, coreai.reduce_sum)
        if mode == 3:
            return self._lower_reduce(node, coreai.reduce_product)
        if mode == 4:
            return self._lower_reduce(node, coreai.reduce_min)
        if mode == 5:
            return self._lower_reduce(node, coreai.reduce_max)
        raise ValueError(f"Unsupported reduce mode={mode} for node {node.output}.")

    def _lower_expand(self, node: Node, op: str) -> Value:
        x = self.env[node.inputs[0]]
        if op == "expand_dims":
            return coreai.expand_dims(x, _axes(node.attrs.get("axes"), rank=_rank(x) + 1))
        if _rank(x) >= int(op[-2]):
            return x
        target = int(op[-2])
        shape = [1] * (target - _rank(x))
        for axis, dim in enumerate(x.type.shape):
            dim = int(dim)
            shape.append(dim if dim >= 0 else _dim_1d_from_value(x, axis))
        return _reshape_with_mixed_shape(x, shape)

    def _lower_moveaxis(self, node: Node) -> Value:
        x = self.env[node.inputs[0]]
        rank = _rank(x)
        source_raw = node.attrs.get("source")
        dest_raw = node.attrs.get("destination")
        if source_raw is None or dest_raw is None:
            raise ValueError(f"moveaxis node '{node.output}' requires source and destination attrs.")
        sources = [source_raw] if isinstance(source_raw, int) else list(source_raw)
        destinations = [dest_raw] if isinstance(dest_raw, int) else list(dest_raw)
        if len(sources) != len(destinations):
            raise ValueError(f"moveaxis node '{node.output}' source/destination lengths differ.")
        norm_sources = [_normalize_axis(int(axis), rank) for axis in sources]
        norm_destinations = [_normalize_axis(int(axis), rank) for axis in destinations]
        if len(set(norm_sources)) != len(norm_sources):
            raise ValueError(f"moveaxis node '{node.output}' has duplicate source axes.")
        perm = [axis for axis in range(rank) if axis not in norm_sources]
        for destination, source in sorted(zip(norm_destinations, norm_sources, strict=True)):
            perm.insert(destination, source)
        return coreai.transpose(x, np.asarray(perm, dtype=np.uint32))

    def _lower_swapaxes(self, node: Node) -> Value:
        x = self.env[node.inputs[0]]
        rank = _rank(x)
        axis1 = _normalize_axis(int(node.attrs.get("axis1")), rank)
        axis2 = _normalize_axis(int(node.attrs.get("axis2")), rank)
        perm = list(range(rank))
        perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
        return coreai.transpose(x, np.asarray(perm, dtype=np.uint32))

    def _lower_split(self, node: Node) -> Value:
        x = self.env[node.inputs[0]]
        axis = _normalize_axis(int(node.attrs.get("axis", 0)), _rank(x))
        output_index = int(node.attrs.get("output_index", 0))
        if "split_indices" in node.attrs:
            indices = [int(v) for v in node.attrs["split_indices"]]
            dim = int(x.type.shape[axis])
            sizes = [indices[0], *[b - a for a, b in zip(indices, indices[1:])], dim - indices[-1]]
        else:
            n = int(node.attrs.get("num_splits", 1))
            dim = int(x.type.shape[axis])
            sizes = [dim // n] * n
        begin = [0] * _rank(x)
        end = _static_shape(x)
        begin[axis] = sum(sizes[:output_index])
        end[axis] = begin[axis] + sizes[output_index]
        return coreai.slice_(x, begin, end, [1] * _rank(x))

    def _lower_slice_update(self, node: Node) -> Value:
        x = self.env[node.inputs[0]]
        update = self.env[node.inputs[1]]
        rank = _rank(x)
        begin = _pad_index(node.attrs.get("begin"), rank, 0)
        end = _pad_index(node.attrs.get("end"), rank, None, x=x)
        stride = _pad_index(node.attrs.get("stride"), rank, 1)
        if any(step <= 0 for step in stride):
            raise ValueError(f"slice_update node '{node.output}' only supports positive strides.")
        update_shape = _static_shape(update)
        expected_shape = [
            max(0, math.ceil((finish - start) / step))
            for start, finish, step in zip(begin, end, stride, strict=True)
        ]
        if update_shape != expected_shape:
            raise ValueError(
                f"slice_update node '{node.output}' update shape {update_shape} does not match slice shape {expected_shape}."
            )
        target_indices = []
        for index in np.ndindex(*update_shape):
            target_indices.append([begin[axis] + index[axis] * stride[axis] for axis in range(rank)])
        indices = np.asarray(target_indices, dtype=np.int32)
        flat_update = coreai.reshape(update, _as_shape_value([len(target_indices)]))
        return coreai.scatter_nd(x, indices, flat_update)

    def _lower_broadcast_arrays(self, node: Node) -> Value:
        spec = self.inferred.get(node.output)
        if spec is None or spec.shape is None:
            raise ValueError(f"broadcast_arrays node '{node.output}' requires inferred shape.")
        output_index = int(node.attrs.get("output_index", 0))
        if any(int(dim) < 0 for dim in spec.shape):
            shapes = [coreai.get_shape(self.env[name]) for name in node.inputs]
            target_shape = shapes[0]
            for shape in shapes[1:]:
                target_shape = coreai.broadcast_shapes(target_shape, shape)
            return coreai.broadcast_to(self.env[node.inputs[output_index]], target_shape)
        return coreai.broadcast_to(self.env[node.inputs[output_index]], _as_shape_value(spec.shape))

    def _lower_take(self, node: Node) -> Value:
        x = self.env[node.inputs[0]]
        indices = self.env[node.inputs[1]]
        rank = _rank(x)
        axis = _normalize_axis(int(node.attrs.get("axis", 0)), rank)
        index_shape = _static_or_dynamic_shape(indices)
        if axis == 0:
            source = x
        else:
            perm = [axis, *range(axis), *range(axis + 1, rank)]
            source = coreai.transpose(x, np.asarray(perm, dtype=np.uint32))

        coords = (
            coreai.reshape(indices, _as_shape_value([1]))
            if not index_shape
            else coreai.expand_dims(indices, [len(index_shape)])
        )
        gathered = coreai.gather_nd(source, coords)
        if not index_shape or axis == 0:
            return gathered

        index_rank = len(index_shape)
        before_rank = axis
        out_rank = index_rank + rank - 1
        perm = (
            list(range(index_rank, index_rank + before_rank))
            + list(range(index_rank))
            + list(range(index_rank + before_rank, out_rank))
        )
        return coreai.transpose(gathered, np.asarray(perm, dtype=np.uint32))

    def _lower_gather(self, node: Node) -> Value:
        if "slice_shape" not in node.attrs and "shape" not in node.attrs:
            return self._lower_take(node)

        x = self.env[node.inputs[0]]
        indices = self.env[node.inputs[1]]
        rank = _rank(x)
        axis = _normalize_axis(int(node.attrs.get("axis", 0)), rank)
        slice_shape = [int(v) for v in node.attrs.get("slice_shape", [])]
        if slice_shape:
            if len(slice_shape) != rank:
                raise ValueError(
                    f"gather node '{node.output}' slice_shape rank {len(slice_shape)} does not match input rank {rank}."
                )
            if int(slice_shape[axis]) != 1:
                raise ValueError(
                    f"gather node '{node.output}' only supports unit slice size on the gathered axis."
                )
            x_shape = _static_or_dynamic_shape(x)
            unsupported_slices = [
                (dim, slice_dim, x_dim)
                for dim, (slice_dim, x_dim) in enumerate(zip(slice_shape, x_shape, strict=True))
                if dim != axis and int(x_dim) >= 0 and int(slice_dim) != int(x_dim)
            ]
            if unsupported_slices:
                raise ValueError(
                    f"gather node '{node.output}' only supports full slices on non-gathered axes: {unsupported_slices}."
                )

        source = x
        if axis != 0:
            source = coreai.transpose(x, np.asarray([axis, *range(axis), *range(axis + 1, rank)], dtype=np.uint32))
        index_shape = _static_or_dynamic_shape(indices)
        coords = (
            coreai.reshape(indices, _as_shape_value([1]))
            if not index_shape
            else coreai.expand_dims(indices, [len(index_shape)])
        )
        gathered = coreai.gather_nd(source, coords)
        target_shape = node.attrs.get("shape")
        if target_shape is None:
            if not slice_shape:
                return gathered
            target_shape = [*index_shape, *slice_shape]
        return self._reshape(gathered, target_shape)

    def _lower_fill(self, node: Node, op: str) -> Value:
        shape = node.attrs.get("shape")
        if shape is None:
            spec = self.inferred.get(node.output)
            if spec is None or spec.shape is None:
                raise ValueError(f"{op} node '{node.output}' requires shape.")
            shape = spec.shape
        dtype = _normalize_dtype(str(node.attrs.get("dtype", "fp32")))
        value = 0 if op == "zeros" else 1
        if op == "full":
            value = node.attrs.get("value", value)
        if _shape_has_runtime_dims(shape):
            scalar = coreai.constant(value, dtype=_element_type(dtype))
            return coreai.broadcast_to(scalar, self._shape_operand(shape))
        if any(int(dim) < 0 for dim in shape):
            raise ValueError(f"{op} node '{node.output}' has dynamic shape without runtime dimension refs: {shape!r}.")
        return self._constant(node.output, np.full(shape, value, dtype=_np_dtype_for_ir(dtype)), dtype=dtype, source=op)

    def _lower_linspace(self, node: Node) -> Value:
        start = float(node.attrs.get("start", 0.0))
        stop = float(node.attrs.get("stop", 1.0))
        num = int(node.attrs.get("num", node.attrs.get("num_steps", 1)))
        return self._constant(node.output, np.linspace(start, stop, num, dtype=np.float32), source="linspace")

    def _lower_layernorm(self, node: Node) -> Value:
        x = self.env[node.inputs[0]]
        gamma = self.env[node.inputs[1]] if len(node.inputs) > 1 else coreai.constant(1.0, dtype=x.type.element_type)
        beta = self.env[node.inputs[2]] if len(node.inputs) > 2 else coreai.constant(0.0, dtype=x.type.element_type)
        axes = _axes(node.attrs.get("axes"), rank=_rank(x))
        if not axes:
            normalized_shape = node.attrs.get("normalized_shape")
            if normalized_shape is not None:
                axes = list(range(_rank(x) - len(normalized_shape), _rank(x)))
            else:
                axes = [_rank(x) - 1]
        eps = float(node.attrs.get("eps", 1e-5))
        mean = coreai.reduce_mean(x, axes)
        centered = coreai.broadcasting_sub(x, mean)
        var = coreai.reduce_mean(coreai.broadcasting_mul(centered, centered), axes)
        inv = coreai.rsqrt(coreai.broadcasting_add(var, eps))
        norm = coreai.broadcasting_mul(centered, inv)
        return coreai.broadcasting_add(coreai.broadcasting_mul(norm, gamma), beta)

    def _lower_rmsnorm(self, node: Node) -> Value:
        x = self.env[node.inputs[0]]
        scale = self.env[node.inputs[1]] if len(node.inputs) > 1 else coreai.constant(1.0, dtype=x.type.element_type)
        eps = float(node.attrs.get("eps", node.attrs.get("epsilon", 1e-5)))
        axes = node.attrs.get("axes", [-1])
        axes = _axes(axes, rank=_rank(x))

        def body(args: list[Value]) -> Value:
            bx, bscale = args
            square = coreai.broadcasting_mul(bx, bx)
            mean_square = coreai.reduce_mean(square, axes)
            inv = coreai.rsqrt(coreai.broadcasting_add(mean_square, eps))
            out = coreai.broadcasting_mul(coreai.broadcasting_mul(bx, inv), bscale)
            return _reshape_like(out, bx)

        return self._emit_private_composite(
            node=node,
            composite_name="rms_norm",
            input_values=[x, scale],
            input_names=["input", "scale"],
            attrs={"axes": axes, "eps": eps},
            body=body,
            result_type=x.type,
        )

    def _lower_sdpa(self, node: Node) -> Value:
        q = self.env[node.inputs[0]]
        k = self.env[node.inputs[1]]
        v = self.env[node.inputs[2]]
        inputs = [q, k, v]
        names = ["query", "key", "value"]
        mask = self.env[node.inputs[3]] if len(node.inputs) > 3 else None
        if mask is not None:
            inputs.append(mask)
            names.append("attn_mask")
        scale = node.attrs.get("scale")
        scale_f = float(scale) if scale is not None else None
        is_causal = bool(node.attrs.get("do_causal", node.attrs.get("is_causal", False)))
        window_size = int(node.attrs.get("window_size", 0))

        def body(args: list[Value]) -> Value:
            bq, bk, bv = args[:3]
            bm = args[3] if len(args) > 3 else None
            if _rank(bq) >= 4 and _rank(bk) >= 4:
                target_heads = int(bq.type.shape[1])
                bk = _repeat_attention_heads(bk, target_heads)
                bv = _repeat_attention_heads(bv, target_heads)
            head_dim = int(bq.type.shape[-1])
            effective_scale = scale_f if scale_f is not None else 1.0 / math.sqrt(float(head_dim))
            scaled_q = coreai.broadcasting_mul(bq, effective_scale)
            kt = coreai.transpose(bk, np.asarray([0, 1, 3, 2], dtype=np.uint32))
            scores = coreai.broadcasting_batch_matmul(scaled_q, kt)
            if bm is not None:
                scores = coreai.broadcasting_add(scores, coreai.cast(bm, scores.type.element_type))
            if is_causal:
                scores = coreai.broadcasting_add(scores, _causal_mask_like(scores))
            weights = coreai.softmax(scores, _rank(scores) - 1)
            return coreai.broadcasting_batch_matmul(weights, bv)

        attrs: dict[str, Any] = {"is_causal": is_causal, "window_size": window_size}
        if scale_f is not None:
            attrs["scale"] = scale_f
        return self._emit_private_composite(
            node=node,
            composite_name="scaled_dot_product_attention",
            input_values=inputs,
            input_names=names,
            attrs=attrs,
            body=body,
        )

    def _lower_rope(self, node: Node) -> Value:
        x = self.env[node.inputs[0]]
        dims = int(node.attrs.get("dims") or int(x.type.shape[-1]))
        interleaved = bool(node.attrs.get("interleaved", node.attrs.get("traditional", False)))
        scale = float(node.attrs.get("scale", 1.0))
        base = float(node.attrs.get("base", 10000.0))
        extra_values: list[Value] = []
        extra_names: list[str] = []
        if len(node.inputs) >= 2:
            extra_values.append(self.env[node.inputs[1]])
            extra_names.append("offset")
        if len(node.inputs) >= 3:
            extra_values.append(self.env[node.inputs[2]])
            extra_names.append("freqs")

        def body(args: list[Value]) -> Value:
            bx = args[0]
            offset = args[1] if len(args) > 1 and extra_names[0] == "offset" else None
            freqs = args[-1] if extra_names and extra_names[-1] == "freqs" else None
            return _reshape_like(
                _rope_body(bx, dims=dims, interleaved=interleaved, scale=scale, base=base, offset=offset, freqs=freqs),
                bx,
            )

        return self._emit_private_composite(
            node=node,
            composite_name="rope",
            input_values=[x, *extra_values],
            input_names=["input", *extra_names],
            attrs={"scale": scale, "base": base, "dims": dims, "interleaved": interleaved},
            body=body,
            result_type=x.type,
        )

    def _lower_tensordot(self, node: Node) -> Value:
        x = self.env[node.inputs[0]]
        y = self.env[node.inputs[1]]
        x_shape = _static_shape(x)
        y_shape = _static_shape(y)
        axes = node.attrs.get("axes", 2)
        if isinstance(axes, int):
            count = int(axes)
            x_axes = list(range(len(x_shape) - count, len(x_shape)))
            y_axes = list(range(count))
        else:
            x_raw, y_raw = axes
            x_axes = [int(v) % len(x_shape) for v in (x_raw if isinstance(x_raw, (list, tuple)) else [x_raw])]
            y_axes = [int(v) % len(y_shape) for v in (y_raw if isinstance(y_raw, (list, tuple)) else [y_raw])]
        x_keep = [i for i in range(len(x_shape)) if i not in x_axes]
        y_keep = [i for i in range(len(y_shape)) if i not in y_axes]
        x_perm = x_keep + x_axes
        y_perm = y_axes + y_keep
        x_t = coreai.transpose(x, np.asarray(x_perm, dtype=np.uint32)) if x_perm != list(range(len(x_shape))) else x
        y_t = coreai.transpose(y, np.asarray(y_perm, dtype=np.uint32)) if y_perm != list(range(len(y_shape))) else y
        x_keep_size = math.prod(x_shape[i] for i in x_keep) if x_keep else 1
        y_keep_size = math.prod(y_shape[i] for i in y_keep) if y_keep else 1
        contract = math.prod(x_shape[i] for i in x_axes) if x_axes else 1
        x_2d = coreai.reshape(x_t, _as_shape_value([x_keep_size, contract]))
        y_2d = coreai.reshape(y_t, _as_shape_value([contract, y_keep_size]))
        out_2d = coreai.broadcasting_batch_matmul(x_2d, y_2d)
        out_shape = [x_shape[i] for i in x_keep] + [y_shape[i] for i in y_keep]
        if not out_shape:
            out_shape = [1]
        out = coreai.reshape(out_2d, _as_shape_value(out_shape))
        if out_shape == [1] and not (x_keep or y_keep):
            return coreai.shrink_dims(out, [0])
        return out

    def _lower_var_std(self, node: Node, *, compute_std: bool) -> Value:
        x = self.env[node.inputs[0]]
        axes = _axes(node.attrs.get("axes"), rank=_rank(x))
        keep_dims = bool(node.attrs.get("keep_dims", False))
        correction = float(node.attrs.get("correction", node.attrs.get("ddof", 0.0)))
        mean = coreai.reduce_mean(x, axes)
        centered = coreai.broadcasting_sub(x, mean)
        square = coreai.broadcasting_mul(centered, centered)
        var = coreai.reduce_mean(square, axes)
        if correction:
            shape = _static_shape(x)
            count = float(math.prod(shape[axis] for axis in axes))
            if count > correction:
                var = coreai.broadcasting_mul(var, count / (count - correction))
        result = coreai.sqrt(var) if compute_std else var
        return _shrink_if_needed(result, axes, keep_dims)

    def _lower_bool_reduce(self, node: Node, *, any_mode: bool) -> Value:
        x = self.env[node.inputs[0]]
        mask = x if _is_bool_value(x) else coreai.broadcasting_not_equal(x, _zero_for_value(x))
        axes = _axes(node.attrs.get("axes"), rank=_rank(mask))
        reduced = coreai.any_(mask, axes) if any_mode else coreai.all_(mask, axes)
        return _shrink_if_needed(reduced, axes, bool(node.attrs.get("keep_dims", False)))

    def _lower_array_equal(self, node: Node) -> Value:
        x = self.env[node.inputs[0]]
        y = self.env[node.inputs[1]]
        if _static_shape(x) != _static_shape(y):
            return self._constant(node.output, np.asarray(False), dtype="bool", source="array_equal_shape")
        eq = coreai.broadcasting_equal(x, y)
        axes = list(range(_rank(eq)))
        if not axes:
            return eq
        return _shrink_if_needed(coreai.all_(eq, axes), axes, False)

    def _lower_isclose(self, node: Node) -> Value:
        x, y = self.env[node.inputs[0]], self.env[node.inputs[1]]
        rtol = float(node.attrs.get("rtol", 1e-5))
        atol = float(node.attrs.get("atol", 1e-8))
        diff = coreai.abs_(coreai.broadcasting_sub(x, y))
        tol = coreai.broadcasting_add(atol, coreai.broadcasting_mul(rtol, coreai.abs_(y)))
        close = _BINARY_OPS["less_equal"](diff, tol)
        equal = coreai.broadcasting_equal(x, y)
        close = coreai.broadcasting_or(close, equal)
        if bool(node.attrs.get("equal_nan", False)):
            x_nan = coreai.broadcasting_not_equal(x, x)
            y_nan = coreai.broadcasting_not_equal(y, y)
            close = coreai.broadcasting_or(close, coreai.broadcasting_and(x_nan, y_nan))
        return close

    def _lower_nan_to_num(self, node: Node) -> Value:
        x = self.env[node.inputs[0]]
        nan = float(node.attrs.get("nan", 0.0))
        posinf = float(node.attrs.get("posinf", np.finfo(np.float32).max))
        neginf = float(node.attrs.get("neginf", np.finfo(np.float32).min))
        out = coreai.broadcasting_where(coreai.broadcasting_not_equal(x, x), nan, x)
        out = coreai.broadcasting_where(coreai.broadcasting_equal(out, float("inf")), posinf, out)
        out = coreai.broadcasting_where(coreai.broadcasting_equal(out, float("-inf")), neginf, out)
        return out

    def _lower_diag(self, node: Node) -> Value:
        x = self.env[node.inputs[0]]
        shape = _static_shape(x)
        offset = int(node.attrs.get("k", node.attrs.get("offset", 0)))
        if len(shape) == 1:
            return self._lower_diag_vector(node, x, offset)
        if len(shape) == 2:
            return self._extract_diagonal(x, axis1=0, axis2=1, offset=offset, name=node.output)
        raise ValueError(f"diag node '{node.output}' supports rank-1 or rank-2 input, got rank {len(shape)}.")

    def _lower_diag_vector(self, node: Node, x: Value, offset: int) -> Value:
        n = int(x.type.shape[0])
        out_size = n + abs(int(offset))
        if offset >= 0:
            rows = np.arange(n, dtype=np.int32)
            cols = rows + np.int32(offset)
        else:
            cols = np.arange(n, dtype=np.int32)
            rows = cols + np.int32(-offset)
        indices = np.stack([rows, cols], axis=1).astype(np.int32)
        base = coreai.constant(np.zeros((out_size, out_size), dtype=np.float32), dtype=x.type.element_type)
        return coreai.scatter_nd(base, indices, x)

    def _lower_diagonal(self, node: Node) -> Value:
        x = self.env[node.inputs[0]]
        rank = _rank(x)
        axis1 = _normalize_axis(int(node.attrs.get("axis1", 0)), rank)
        axis2 = _normalize_axis(int(node.attrs.get("axis2", 1)), rank)
        offset = int(node.attrs.get("offset", node.attrs.get("k", 0)))
        return self._extract_diagonal(x, axis1=axis1, axis2=axis2, offset=offset, name=node.output)

    def _lower_trace(self, node: Node) -> Value:
        x = self.env[node.inputs[0]]
        rank = _rank(x)
        axis1 = _normalize_axis(int(node.attrs.get("axis1", 0)), rank)
        axis2 = _normalize_axis(int(node.attrs.get("axis2", 1)), rank)
        offset = int(node.attrs.get("offset", node.attrs.get("k", 0)))
        diagonal = self._extract_diagonal(x, axis1=axis1, axis2=axis2, offset=offset, name=f"{node.output}_diag")
        last_axis = _rank(diagonal) - 1
        return _shrink_if_needed(coreai.reduce_sum(diagonal, [last_axis]), [last_axis], False)

    def _extract_diagonal(self, x: Value, *, axis1: int, axis2: int, offset: int, name: str) -> Value:
        shape = _static_shape(x)
        if axis1 == axis2:
            raise ValueError(f"{name}: diagonal axes must differ.")
        indices = _diagonal_gather_indices(shape, axis1=axis1, axis2=axis2, offset=offset)
        return coreai.gather_nd(x, indices)

    def _lower_tri(self, node: Node) -> Value:
        n = int(node.attrs["n"])
        m = int(node.attrs.get("m", n))
        k = int(node.attrs.get("k", 0))
        dtype = _normalize_dtype(str(node.attrs.get("dtype", "fp32")))
        rows = np.arange(n, dtype=np.int32)[:, None]
        cols = np.arange(m, dtype=np.int32)[None, :]
        mask = cols <= (rows + k)
        value = mask if dtype == "bool" else mask.astype(_np_dtype_for_ir(dtype))
        return self._constant(node.output, value, dtype=dtype, source="tri")

    def _lower_triangular_band(self, node: Node, *, lower: bool) -> Value:
        x = self.env[node.inputs[0]]
        shape = _static_shape(x)
        if len(shape) < 2:
            raise ValueError(f"{node.op} node '{node.output}' requires rank >= 2 input.")
        rows, cols = int(shape[-2]), int(shape[-1])
        k = int(node.attrs.get("k", 0))
        row_idx = np.arange(rows, dtype=np.int32)[:, None]
        col_idx = np.arange(cols, dtype=np.int32)[None, :]
        mask = col_idx <= (row_idx + k) if lower else col_idx >= (row_idx + k)
        masked = coreai.broadcasting_where(coreai.constant(mask), x, _zero_for_value(x))
        return masked

    def _lower_eye(self, node: Node) -> Value:
        n = int(node.attrs["n"])
        m = int(node.attrs.get("m", n))
        k = int(node.attrs.get("k", 0))
        dtype = _normalize_dtype(str(node.attrs.get("dtype", "fp32")))
        value = np.eye(n, m, k=k, dtype=_np_dtype_for_ir(dtype))
        if dtype == "bool":
            value = value.astype(np.bool_)
        return self._constant(node.output, value, dtype=dtype, source="eye")

    def _lower_meshgrid(self, node: Node) -> Value:
        if bool(node.attrs.get("sparse", False)):
            raise ValueError(f"meshgrid node '{node.output}' does not support sparse=True yet.")
        input_index = int(node.attrs.get("input_index", 0))
        indexing = str(node.attrs.get("indexing", "xy")).strip().lower()
        vectors = [self.env[name] for name in node.inputs]
        dims = [int(_static_shape(vector)[0]) for vector in vectors]
        out_dims, varying_axis = _meshgrid_dims_and_axis(dims, input_index, indexing, node.output)
        src = vectors[input_index]
        base_shape = [1] * len(out_dims)
        base_shape[varying_axis] = dims[input_index]
        reshaped = coreai.reshape(src, _as_shape_value(base_shape))
        repeats = [1 if axis == varying_axis else int(dim) for axis, dim in enumerate(out_dims)]
        return reshaped if all(rep == 1 for rep in repeats) else coreai.tile(reshaped, _as_tile_value(repeats))

    def _lower_kron(self, node: Node) -> Value:
        x, y = self.env[node.inputs[0]], self.env[node.inputs[1]]
        x_shape = _static_shape(x)
        y_shape = _static_shape(y)
        rank = max(len(x_shape), len(y_shape))
        if rank == 0:
            return coreai.broadcasting_mul(x, y)
        x_pad = [1] * (rank - len(x_shape)) + x_shape
        y_pad = [1] * (rank - len(y_shape)) + y_shape
        x_reshape: list[int] = []
        y_reshape: list[int] = []
        out_shape: list[int] = []
        for x_dim, y_dim in zip(x_pad, y_pad, strict=True):
            x_reshape.extend([x_dim, 1])
            y_reshape.extend([1, y_dim])
            out_shape.append(x_dim * y_dim)
        x_work = coreai.reshape(x, _as_shape_value(x_reshape))
        y_work = coreai.reshape(y, _as_shape_value(y_reshape))
        return coreai.reshape(coreai.broadcasting_mul(x_work, y_work), _as_shape_value(out_shape))

    def _lower_conv(self, node: Node, *, transpose: bool) -> Value:
        if len(node.inputs) not in {2, 3}:
            raise ValueError(f"{node.op} node '{node.output}' requires x, weight[, bias] inputs.")
        x = self.env[node.inputs[0]]
        weight = self.env[node.inputs[1]]
        bias = self.env[node.inputs[2]] if len(node.inputs) == 3 else None
        spatial = _rank(x) - 2
        if spatial not in {1, 2, 3}:
            raise ValueError(f"{node.op} node '{node.output}' requires rank 3, 4, or 5 input.")
        strides = _as_int_list(node.attrs.get("strides", node.attrs.get("stride")), spatial, default=1)
        dilations = _as_int_list(node.attrs.get("dilations", node.attrs.get("dilation")), spatial, default=1)
        groups = int(node.attrs.get("groups", 1))
        padding = _conv_padding(node, x, weight, strides, dilations, transpose=transpose)
        if any(padding):
            if transpose:
                raise ValueError(f"{node.op} node '{node.output}' does not support nonzero transposed padding yet.")
            x = coreai.pad(
                x,
                np.asarray([0, 0, 0, 0, *padding], dtype=np.int32),
                coreai.constant(0, dtype=x.type.element_type),
                "constant",
            )

        if spatial == 1:
            x = coreai.expand_dims(x, [2])
            weight = coreai.expand_dims(weight, [2])
            strides = [1, *strides]
            dilations = [1, *dilations]
            padding = [0, 0, *padding]
            spatial = 2

        if transpose:
            output_pad = _as_int_list(node.attrs.get("output_pad", node.attrs.get("output_padding")), spatial, default=0)
            pointwise = self._lower_pointwise_conv_transpose(
                node,
                x=x,
                weight=weight,
                strides=strides,
                dilations=dilations,
                padding=padding,
                output_pad=output_pad,
                groups=groups,
            )
            if pointwise is not None:
                out = pointwise
            elif spatial == 2:
                out = self._lower_unsupported_conv_transpose_composite(node, x=x, weight=weight, bias=bias)
            elif spatial == 3:
                out = self._lower_unsupported_conv_transpose_composite(node, x=x, weight=weight, bias=bias)
            else:
                raise ValueError(f"{node.op} node '{node.output}' has unsupported transposed spatial rank {spatial}.")
        else:
            if spatial == 2:
                out = coreai.conv2d(
                    x,
                    weight,
                    np.asarray(strides, dtype=np.int32),
                    np.asarray(dilations, dtype=np.int32),
                    np.asarray(groups, dtype=np.int32),
                )
            else:
                out = coreai.conv3d(
                    x,
                    weight,
                    np.asarray(strides, dtype=np.int32),
                    np.asarray(dilations, dtype=np.int32),
                    np.asarray(groups, dtype=np.int32),
                )
        if bias is not None:
            bias_shape = [1, int(bias.type.shape[0])] + [1] * spatial
            out = coreai.broadcasting_add(out, coreai.reshape(bias, _as_shape_value(bias_shape)))
        if _rank(out) == 4 and len(_static_shape(self.env[node.inputs[0]])) == 3:
            out = coreai.shrink_dims(out, [2])
        return out

    def _lower_pointwise_conv_transpose(
        self,
        node: Node,
        *,
        x: Value,
        weight: Value,
        strides: Sequence[int],
        dilations: Sequence[int],
        padding: Sequence[int],
        output_pad: Sequence[int],
        groups: int,
    ) -> Value | None:
        if groups != 1 or list(strides) != [1, 1] or list(dilations) != [1, 1]:
            return None
        if any(int(v) != 0 for v in [*padding, *output_pad]):
            return None
        x_shape = _static_shape(x)
        w_shape = _static_shape(weight)
        if len(x_shape) != 4 or len(w_shape) != 4 or w_shape[2:] != [1, 1]:
            return None
        batch, channels_in, height, width = x_shape
        channels_out = w_shape[1]
        x_nhwc = coreai.transpose(x, np.asarray([0, 2, 3, 1], dtype=np.uint32))
        x_2d = coreai.reshape(x_nhwc, _as_shape_value([batch * height * width, channels_in]))
        w_2d = coreai.reshape(weight, _as_shape_value([channels_in, channels_out]))
        y_2d = coreai.broadcasting_batch_matmul(x_2d, w_2d)
        y_nhwc = coreai.reshape(y_2d, _as_shape_value([batch, height, width, channels_out]))
        return coreai.transpose(y_nhwc, np.asarray([0, 3, 1, 2], dtype=np.uint32))

    def _lower_unsupported_conv_transpose_composite(
        self,
        node: Node,
        *,
        x: Value,
        weight: Value,
        bias: Value | None,
    ) -> Value:
        spec = self.inferred.get(node.output)
        if spec is None or spec.shape is None:
            raise ValueError(f"{node.op} node '{node.output}' requires inferred shape for composite fallback.")
        result_type = RankedTensorType.get(list(spec.shape), x.type.element_type)
        input_values = [x, weight] + ([bias] if bias is not None else [])

        def body(args: list[Value]) -> Value:
            return coreai.constant(np.zeros(spec.shape, dtype=np.float32), dtype=args[0].type.element_type)

        return self._emit_private_composite(
            node=node,
            composite_name="mlx_conv_transpose",
            input_values=input_values,
            input_names=[f"input_{idx}" for idx in range(len(input_values))],
            attrs={"source_op": node.op, "fallback": "unsupported_coreai_beta_asset_writer"},
            body=body,
            result_type=result_type,
        )


def _pad_index(value: Any, rank: int, default: int | None, *, x: Value | None = None) -> list[int]:
    if value is None:
        if default is None:
            if x is None:
                raise ValueError("Cannot infer slice end without input shape.")
            return [int(v) for v in x.type.shape]
        return [int(default)] * rank
    out = [int(v) for v in value]
    if len(out) < rank:
        fill = int(default) if default is not None else 0
        if default is None and x is not None:
            return out + [int(v) for v in x.type.shape[len(out):]]
        out += [fill] * (rank - len(out))
    return out


def _causal_mask_like(scores: Value) -> Value:
    shape = _static_or_dynamic_shape(scores)
    q_len, k_len = shape[-2], shape[-1]
    dyn = RankedTensorType.get_dynamic_size()
    si32 = IntegerType.get_signed(32)
    zero = coreai.constant(0, dtype=np.int32)
    one = coreai.constant(1, dtype=np.int32)
    q_end = coreai.constant(q_len, dtype=np.int32) if q_len >= 0 else _dim_scalar_from_value(scores, -2)
    k_end = coreai.constant(k_len, dtype=np.int32) if k_len >= 0 else _dim_scalar_from_value(scores, -1)
    q = coreai.RangeOp(
        zero,
        q_end,
        one,
        results=[RankedTensorType.get([q_len if q_len >= 0 else dyn], si32)],
    ).result
    k = coreai.RangeOp(
        zero,
        k_end,
        one,
        results=[RankedTensorType.get([k_len if k_len >= 0 else dyn], si32)],
    ).result
    q_dim = q_len if q_len >= 0 else _dim_1d_from_value(scores, -2)
    k_dim = k_len if k_len >= 0 else _dim_1d_from_value(scores, -1)
    q = _reshape_with_mixed_shape(q, [q_dim, 1])
    k = _reshape_with_mixed_shape(k, [1, k_dim])
    mask_shape = [q_dim, k_dim]
    mask_shape_operand = _mixed_shape_operand(mask_shape)
    q = coreai.broadcast_to(q, mask_shape_operand)
    k = coreai.broadcast_to(k, mask_shape_operand)
    future = coreai.greater(k, q)
    return coreai.broadcasting_mul(coreai.cast(future, scores.type.element_type), coreai.constant(-1e4, dtype=scores.type.element_type))


def _repeat_attention_heads(value: Value, target_heads: int) -> Value:
    shape = _static_or_dynamic_shape(value)
    if len(shape) < 4:
        return value
    heads = int(shape[1])
    target_heads = int(target_heads)
    if heads == target_heads:
        return value
    if heads <= 0 or target_heads % heads != 0:
        raise ValueError(f"Cannot repeat attention heads from {heads} to {target_heads}.")
    expanded = coreai.expand_dims(value, [2])
    repeats = [1] * (len(shape) + 1)
    repeats[2] = target_heads // heads
    tiled = coreai.tile(expanded, _as_tile_value(repeats))
    out_shape = list(shape)
    out_shape[1] = target_heads
    target_parts: list[Any] = []
    for axis, dim in enumerate(out_shape):
        if axis == 1:
            target_parts.append(target_heads)
        elif int(dim) >= 0:
            target_parts.append(int(dim))
        else:
            target_parts.append(_dim_1d_from_value(value, axis))
    return _reshape_with_mixed_shape(tiled, target_parts)


def _diagonal_bounds(dim1: int, dim2: int, offset: int) -> tuple[int, int, int]:
    row_start = max(-int(offset), 0)
    col_start = max(int(offset), 0)
    diag_len = min(int(dim1) - row_start, int(dim2) - col_start)
    if diag_len <= 0:
        raise ValueError(f"Diagonal has non-positive length for shape ({dim1}, {dim2}) and offset={offset}.")
    return row_start, col_start, diag_len


def _diagonal_gather_indices(shape: Sequence[int], *, axis1: int, axis2: int, offset: int) -> np.ndarray:
    shape = [int(dim) for dim in shape]
    rank = len(shape)
    row_start, col_start, diag_len = _diagonal_bounds(shape[axis1], shape[axis2], offset)
    prefix_axes = [axis for axis in range(rank) if axis not in {axis1, axis2}]
    prefix_shape = [shape[axis] for axis in prefix_axes]
    out_shape = [*prefix_shape, diag_len, rank]
    indices = np.zeros(out_shape, dtype=np.int32)
    prefix_iter = np.ndindex(*prefix_shape) if prefix_shape else [()]
    for prefix in prefix_iter:
        for diag_idx in range(diag_len):
            full = [0] * rank
            for axis, value in zip(prefix_axes, prefix, strict=True):
                full[axis] = int(value)
            full[axis1] = row_start + diag_idx
            full[axis2] = col_start + diag_idx
            indices[(*prefix, diag_idx)] = np.asarray(full, dtype=np.int32)
    return indices


def _meshgrid_dims_and_axis(dims: Sequence[int], input_index: int, indexing: str, output_name: str) -> tuple[list[int], int]:
    dims = [int(dim) for dim in dims]
    rank = len(dims)
    input_index = int(input_index)
    if input_index < 0 or input_index >= rank:
        raise ValueError(f"meshgrid node '{output_name}' input_index={input_index} is out of range.")
    if indexing == "ij" or rank < 2:
        return dims, input_index
    if indexing != "xy":
        raise ValueError(f"meshgrid node '{output_name}' only supports indexing='xy' or indexing='ij'.")
    out_dims = [dims[1], dims[0], *dims[2:]]
    if input_index == 0:
        return out_dims, 1
    if input_index == 1:
        return out_dims, 0
    return out_dims, input_index


def _conv_padding(
    node: Node,
    x: Value,
    weight: Value,
    strides: Sequence[int],
    dilations: Sequence[int],
    *,
    transpose: bool,
) -> list[int]:
    spatial = _rank(x) - 2
    raw_pad = node.attrs.get("pad", node.attrs.get("padding", node.attrs.get("pads")))
    pad_type = str(node.attrs.get("pad_type", "custom" if raw_pad is not None else "valid")).strip().lower()
    if pad_type == "valid":
        return [0] * (2 * spatial)
    if raw_pad is not None:
        if isinstance(raw_pad, int):
            return [int(raw_pad), int(raw_pad)] * spatial
        parsed = [int(v) for v in raw_pad]
        if len(parsed) == spatial:
            return [v for item in parsed for v in (item, item)]
        if len(parsed) == 2 * spatial:
            return parsed
        raise ValueError(f"{node.op} node '{node.output}' has invalid padding attr {parsed}.")
    if pad_type not in {"same", "same_lower"}:
        raise ValueError(f"{node.op} node '{node.output}' has unsupported pad_type '{pad_type}'.")
    if transpose:
        raise ValueError(f"{node.op} node '{node.output}' does not support pad_type={pad_type!r} for transposed conv.")
    x_shape = _static_shape(x)
    w_shape = _static_shape(weight)
    padding: list[int] = []
    for i in range(spatial):
        input_size = int(x_shape[2 + i])
        kernel = int(w_shape[2 + i])
        stride = int(strides[i])
        dilation = int(dilations[i])
        output_size = math.ceil(input_size / stride)
        needed = max(0, (output_size - 1) * stride + dilation * (kernel - 1) + 1 - input_size)
        before = needed // 2
        after = needed - before
        if pad_type == "same_lower":
            before, after = after, before
        padding.extend([before, after])
    return padding


def _slice_last(x: Value, start: int, end: int) -> Value:
    rank = _rank(x)
    begin = np.zeros(rank, dtype=np.int32)
    finish = _value_shape_operand(x, overrides={rank - 1: int(end)})
    begin[-1] = int(start)
    return coreai.slice_(x, begin, finish, np.ones(rank, dtype=np.int32))


def _rope_body(
    x: Value,
    *,
    dims: int,
    interleaved: bool,
    scale: float,
    base: float,
    offset: Value | None,
    freqs: Value | None,
) -> Value:
    shape = _static_or_dynamic_shape(x)
    rank = _rank(x)
    seq_len = int(shape[-2])
    feature_dim = int(shape[-1])
    dims = min(int(dims), feature_dim)
    if dims % 2:
        raise ValueError(f"RoPE dims must be even, got {dims}.")
    half = dims // 2
    dyn = RankedTensorType.get_dynamic_size()
    seq_end = coreai.constant(seq_len, dtype=np.int32) if seq_len >= 0 else _dim_scalar_from_value(x, -2)
    pos = coreai.RangeOp(
        coreai.constant(0, dtype=np.int32),
        seq_end,
        coreai.constant(1, dtype=np.int32),
        results=[RankedTensorType.get([seq_len if seq_len >= 0 else dyn], IntegerType.get_signed(32))],
    ).result
    pos = coreai.cast(pos, F32Type.get())
    if scale != 1.0:
        pos = coreai.broadcasting_mul(pos, scale)
    if offset is not None:
        pos = coreai.broadcasting_add(pos, coreai.cast(offset, F32Type.get()))
    seq_dim = seq_len if seq_len >= 0 else _dim_1d_from_value(x, -2)
    pos = _reshape_with_mixed_shape(pos, [seq_dim, 1])
    if freqs is None:
        freq_values = np.power(np.float32(base), np.arange(0, dims, 2, dtype=np.float32) / np.float32(dims)).astype(np.float32)
        freqs = coreai.constant(freq_values)
    freqs = _reshape_with_mixed_shape(coreai.cast(freqs, F32Type.get()), [1, half])
    angles = coreai.broadcasting_divide(pos, freqs)
    cos = coreai.cos(angles)
    sin = coreai.sin(angles)
    trig_shape = [1] * (rank - 2) + [seq_dim, half]
    cos = _reshape_with_mixed_shape(cos, trig_shape)
    sin = _reshape_with_mixed_shape(sin, trig_shape)
    x_rot = _slice_last(x, 0, dims)
    half_shape = [
        dim if int(dim) >= 0 else _dim_1d_from_value(x, axis)
        for axis, dim in enumerate(shape[:-1])
    ] + [half]
    half_shape_operand = _mixed_shape_operand(half_shape)
    cos = coreai.broadcast_to(cos, half_shape_operand)
    sin = coreai.broadcast_to(sin, half_shape_operand)
    if interleaved:
        pairs_shape = [
            dim if int(dim) >= 0 else _dim_1d_from_value(x, axis)
            for axis, dim in enumerate(shape[:-1])
        ] + [half, 2]
        pairs = _reshape_with_mixed_shape(x_rot, pairs_shape)
        pair_rank = _rank(pairs)
        even_end = _value_shape_operand(pairs, overrides={pair_rank - 1: 1})
        even = coreai.shrink_dims(coreai.slice_(pairs, [0] * pair_rank, even_end, [1] * pair_rank), [pair_rank - 1])
        odd_begin = [0] * pair_rank
        odd_begin[-1] = 1
        odd = coreai.shrink_dims(coreai.slice_(pairs, odd_begin, _value_shape_operand(pairs), [1] * pair_rank), [pair_rank - 1])
        rot_even = coreai.broadcasting_sub(coreai.broadcasting_mul(even, cos), coreai.broadcasting_mul(odd, sin))
        rot_odd = coreai.broadcasting_add(coreai.broadcasting_mul(even, sin), coreai.broadcasting_mul(odd, cos))
        rotated_shape = [
            dim if int(dim) >= 0 else _dim_1d_from_value(x, axis)
            for axis, dim in enumerate(shape[:-1])
        ] + [dims]
        rotated = _reshape_with_mixed_shape(
            coreai.concat(pair_rank - 1, [coreai.expand_dims(rot_even, [pair_rank - 1]), coreai.expand_dims(rot_odd, [pair_rank - 1])]),
            rotated_shape,
        )
    else:
        first = _slice_last(x_rot, 0, half)
        second = _slice_last(x_rot, half, dims)
        rot_first = coreai.broadcasting_sub(coreai.broadcasting_mul(first, cos), coreai.broadcasting_mul(second, sin))
        rot_second = coreai.broadcasting_add(coreai.broadcasting_mul(first, sin), coreai.broadcasting_mul(second, cos))
        rotated = coreai.concat(_rank(x) - 1, [rot_first, rot_second])
    if dims < feature_dim:
        tail = _slice_last(x, dims, feature_dim)
        return coreai.concat(_rank(x) - 1, [rotated, tail])
    return rotated


_BINARY_OPS: dict[str, Callable[[Value, Value], Value]] = {
    "add": coreai.broadcasting_add,
    "maximum": coreai.broadcasting_maximum,
    "minimum": coreai.broadcasting_minimum,
    "sub": coreai.broadcasting_sub,
    "mul": coreai.broadcasting_mul,
    "real_div": coreai.broadcasting_divide,
    "divide": coreai.broadcasting_divide,
    "pow": coreai.broadcasting_pow,
    "mod": coreai.broadcasting_modulo,
    "greater": coreai.broadcasting_greater,
    "greater_equal": lambda x, y: coreai.broadcasting_or(
        coreai.broadcasting_greater(x, y), coreai.broadcasting_equal(x, y)
    ),
    "less": lambda x, y: coreai.broadcasting_greater(y, x),
    "less_equal": lambda x, y: coreai.broadcasting_or(
        coreai.broadcasting_greater(y, x), coreai.broadcasting_equal(x, y)
    ),
    "equal": coreai.broadcasting_equal,
    "not_equal": coreai.broadcasting_not_equal,
}


_UNARY_OPS: dict[str, Callable[[Value], Value]] = {
    "sigmoid": coreai.sigmoid,
    "silu": coreai.silu,
    "gelu": coreai.gelu,
    "tanh": coreai.tanh,
    "sin": coreai.sin,
    "cos": coreai.cos,
    "erf": coreai.erf,
    "acos": coreai.acos,
    "asin": coreai.asin,
    "atan": coreai.atan,
    "atanh": coreai.atanh,
    "exp": coreai.exp,
    "expm1": lambda x: coreai.broadcasting_sub(coreai.exp(x), 1.0),
    "log": coreai.log,
    "log1p": lambda x: coreai.log(coreai.broadcasting_add(x, 1.0)),
    "log2": lambda x: coreai.broadcasting_divide(coreai.log(x), math.log(2.0)),
    "log10": lambda x: coreai.broadcasting_divide(coreai.log(x), math.log(10.0)),
    "sqrt": coreai.sqrt,
    "rsqrt": coreai.rsqrt,
    "abs": coreai.abs_,
    "degrees": lambda x: coreai.broadcasting_mul(x, 180.0 / math.pi),
    "radians": lambda x: coreai.broadcasting_mul(x, math.pi / 180.0),
    "isnan": lambda x: coreai.broadcasting_not_equal(x, x),
    "isinf": lambda x: coreai.broadcasting_equal(coreai.abs_(x), float("inf")),
    "isfinite": lambda x: coreai.not_(coreai.broadcasting_equal(coreai.abs_(x), float("inf"))),
    "isneginf": lambda x: coreai.broadcasting_equal(x, float("-inf")),
    "isposinf": lambda x: coreai.broadcasting_equal(x, float("inf")),
}


_REDUCE_OPS: dict[str, Callable[[Value, list[int]], Value]] = {
    "reduce_sum": coreai.reduce_sum,
    "reduce_mean": coreai.reduce_mean,
    "reduce_min": coreai.reduce_min,
    "reduce_max": coreai.reduce_max,
    "reduce_prod": coreai.reduce_product,
    "reduce_log_sum_exp": lambda x, axes: coreai.log(coreai.reduce_sum(coreai.exp(x), axes)),
    "all": coreai.all_,
    "any": coreai.any_,
}


def build_coreai_program(
    graph: Graph,
    *,
    config: CoreAILoweringConfig | None = None,
) -> LoweredCoreAIProgram:
    return CoreAILowerer(config).lower(graph)


def _optimization_skip_reason(graph: Graph) -> str | None:
    has_dynamic_input = any(any(int(dim) < 0 for dim in spec.shape) for spec in graph.inputs)
    if not has_dynamic_input:
        return None
    for node in graph.nodes:
        if node.op == "scaled_dot_product_attention" and bool(node.attrs.get("do_causal", node.attrs.get("is_causal", False))):
            return "coreai_optimize_dynamic_causal_sdpa_reshape_bug"
    return None


def save_coreai_program(program: AIProgram, output_path: str | Path) -> Any:
    return program.save_asset(Path(output_path))
