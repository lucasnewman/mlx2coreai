"""Structured control flow for explicit recipe IR, not Python/MLX tracing."""
from coreai._compiler.dialects import coreai
from coreai._compiler.ir import Value

from .ir import Graph, TensorType


def _analyze(node, key, inputs):
    from .passes import analyze_graph

    graph = node.attrs.get(key)
    if not isinstance(graph, Graph):
        raise ValueError(f"{node.op} requires a Graph attribute '{key}'.")
    analysis = analyze_graph(graph)
    declared = [TensorType(spec.shape, spec.dtype) for spec in analysis.graph.inputs]
    if declared != list(inputs):
        raise ValueError(f"{node.op} {key} input types must match its operands: {declared} != {inputs}.")
    return analysis


def _outputs(analysis):
    outputs = [analysis.specs[name] for name in analysis.graph.outputs]
    if any(spec.shape is None or spec.dtype is None for spec in outputs):
        raise ValueError("Control-flow outputs need complete tensor types.")
    return outputs


def _predicate(spec):
    if spec != TensorType((), 'bool'):
        raise ValueError("Control-flow predicates must be scalar bool tensors.")


def infer_cond(node, inputs):
    if not inputs:
        raise ValueError("cond requires a predicate.")
    _predicate(inputs[0])
    then = _outputs(_analyze(node, 'then', inputs[1:]))
    otherwise = _outputs(_analyze(node, 'else', inputs[1:]))
    if then != otherwise or len(then) != int(node.attrs.get('num_outputs', len(node.outputs))):
        raise ValueError("cond branches must have matching output types and arity.")
    return then[int(node.attrs.get('output_index', 0))]


def infer_while(node, inputs):
    count = int(node.attrs.get('carried_count', len(inputs)))
    if count <= 0 or count > len(inputs):
        raise ValueError("while_loop requires a valid positive carried_count.")
    condition = _outputs(_analyze(node, 'condition', inputs))
    if len(condition) != 1:
        raise ValueError("while_loop condition must return one predicate.")
    _predicate(condition[0])
    body = _outputs(_analyze(node, 'body', inputs))
    if body != list(inputs[:count]) or count != int(node.attrs.get('num_outputs', len(node.outputs))):
        raise ValueError("while_loop body must preserve carried tensor types and arity.")
    return body[int(node.attrs.get('output_index', 0))]


def _inline(context, graph, operands):
    from .passes import analyze_graph

    analysis = analyze_graph(graph)
    saved_env, saved_inferred = context.env, context.inferred
    saved_delta = context._legacy_gated_delta_results
    try:
        context.env = dict(zip((spec.name for spec in analysis.graph.inputs), operands, strict=True))
        context.inferred = analysis.specs
        context._legacy_gated_delta_results = {}
        for node in analysis.graph.nodes:
            result = context._lower_node(node)
            values = tuple(result) if isinstance(result, (tuple, list)) else (result,)
            context.env.update(zip(node.outputs, values, strict=True))
        return [context.env[name] for name in analysis.graph.outputs]
    finally:
        context.env, context.inferred = saved_env, saved_inferred
        context._legacy_gated_delta_results = saved_delta


def emit_cond(context, node):
    from .lower_to_coreai import _tensor_type

    operands = [context.env[name] for name in node.inputs]
    # Validate even when captured value_types bypass the ordinary inference rule.
    infer_cond(node, [context.inferred[name] for name in node.inputs])
    result_types = [_tensor_type(context.inferred[name]) for name in node.outputs]
    result, builders = coreai.if_(results=result_types, condition=operands[0])
    for key, builder in zip(('then', 'else'), builders, strict=True):
        with builder:
            coreai.yield_(_inline(context, node.attrs[key], operands[1:]))
    return result if isinstance(result, Value) else tuple(result)


def emit_while(context, node):
    operands = [context.env[name] for name in node.inputs]
    infer_while(node, [context.inferred[name] for name in node.inputs])
    count = int(node.attrs.get('carried_count', len(operands)))
    carried, additional = operands[:count], operands[count:]
    result, (before, after) = coreai.while_(results=[value.type for value in carried], inits=carried)
    with before:
        args = list(before.arguments)
        [predicate] = _inline(context, node.attrs['condition'], args + additional)
        coreai.condition(predicate, *args)
    with after:
        coreai.yield_(_inline(context, node.attrs['body'], list(after.arguments) + additional))
    return result if isinstance(result, Value) else tuple(result)
