from __future__ import annotations

from collections import OrderedDict

import numpy as np
from coreai._compiler.dialects import coreai
from coreai._compiler.ir import InsertionPoint, Module, Value

from ._composite_declaration import generate_composite_decl


def _transpose(value, axes):
    return coreai.transpose(value, np.asarray(axes, dtype=np.uint32))


def _step_at(value, index):
    # Put sequence first so a scalar gather removes just that dimension.
    axes = [2, 0, 1, *range(3, value.type.rank)]
    source = _transpose(value, axes)
    return coreai.gather_nd(source, coreai.reshape(index, [1]))


def _body(args):
    query, key, value, g, beta, state = args
    decay = coreai.exp(g)
    # Output initially uses B,H,T,Dv, with T resolved from the value tensor.
    output = coreai.broadcast_to(coreai.constant(0, dtype=np.float32), coreai.get_shape(value))
    length = coreai.shrink_dims(coreai.slice_(coreai.get_shape(query), [2], [3], [1]), [0])
    length = coreai.cast(length, dtype=np.int32)
    zero = coreai.constant(0, dtype=np.int32)
    results, (before, after) = coreai.while_(
        results=[zero.type, state.type, output.type], inits=[zero, state, output],
    )
    with before:
        t, s, out = before.arguments
        coreai.condition(coreai.broadcasting_greater(length, t), t, s, out)
    with after:
        t, s, out = after.arguments
        q = coreai.expand_dims(_step_at(query, t), [3])
        k = coreai.expand_dims(_step_at(key, t), [3])
        v = _step_at(value, t)
        d = coreai.expand_dims(_step_at(decay, t), [2, 3])
        b = coreai.expand_dims(_step_at(beta, t), [2])
        s = coreai.broadcasting_mul(s, d)
        memory = coreai.shrink_dims(coreai.reduce_sum(coreai.broadcasting_mul(s, k), [2]), [2])
        delta = coreai.broadcasting_mul(coreai.broadcasting_sub(v, memory), b)
        s = coreai.broadcasting_add(s, coreai.broadcasting_mul(k, coreai.expand_dims(delta, [2])))
        y = coreai.reduce_sum(coreai.broadcasting_mul(s, q), [2])
        indices = coreai.broadcast_to(t, coreai.get_shape(y))
        out = coreai.scatter_along_axis(out.type, out, indices, y, 2)
        coreai.yield_([coreai.broadcasting_add(t, np.int32(1)), s, out])
    return _transpose(results[2], [0, 2, 1, 3]), results[1]


def lower_gated_delta_update(
    inputs: list[Value], *, module: Module, graph: coreai.GraphOp, name: str,
    implementation: str = "native",
) -> tuple[Value, Value]:
    """Adapt MLX's normalized, scaled Q and decay to Apple's v1 composite."""
    if len(inputs) != 6:
        raise ValueError("gated_delta_update expects q, k, v, decay, beta, state.")
    q, k, v, decay, beta, state = inputs
    if any(x.type.rank != 4 for x in (q, k, v, state)) or any(x.type.rank != 3 for x in (decay, beta)):
        raise ValueError("Only scalar-gated, unmasked MLX gated delta is supported.")
    if int(q.type.shape[2]) != int(v.type.shape[2]):
        raise ValueError("gated_delta_update currently requires equal Q/K and value head counts.")
    dtype, state_dtype = q.type.element_type, state.type.element_type
    q, k, v, decay, beta, state = [coreai.cast(x, dtype=np.float32) for x in inputs]
    operands = [_transpose(x, [0, 2, 1, 3]) for x in (q, k, v)]
    operands += [_transpose(coreai.log(decay), [0, 2, 1]), _transpose(beta, [0, 2, 1]),
                 _transpose(state, [0, 1, 3, 2])]
    names = ["query", "key", "value", "g", "beta", "initial_state"]
    outputs = ["output_0", "output_1"]
    # macOS 27's native false-normalization path also skips query scaling.
    # MLX has already normalized and scaled Q, so the fallback does neither.
    composite_name = "gated_delta_update" if implementation == "native" else "mlx_gated_delta_update"
    declaration = generate_composite_decl(module.context, composite_name, names, outputs,
                                          {"use_qk_l2_norm": False})
    with InsertionPoint(module.body):
        private = coreai.GraphOp(
            name=name, input_types=[x.type for x in operands], result_types=[], input_names=names,
            private=True, no_inline=True, composite_decl=declaration,
        )
        with private.block:
            results = _body(list(private.arguments))
            private.set_outputs_spec_from_dict(OrderedDict(zip(outputs, results, strict=True)))
            result_types = [x.type for x in results]
    with graph.block:
        output, next_state = coreai.invoke(results=result_types, callee=name, operands=operands)
        return coreai.cast(output, dtype=dtype), coreai.cast(_transpose(next_state, [0, 1, 3, 2]), dtype=state_dtype)
