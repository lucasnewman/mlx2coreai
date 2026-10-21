"""Capture adaptations for the mlx-audio Redux encoder and one TDT step."""


def prepare_model(model, *, preserve_quantization=False):
    """Materialize losslessly dequantized ternary weights before capture."""
    import mlx.core as mx
    from mlx2coreai import prepare_quantized_linears

    model.eval()
    model.set_dtype(mx.float32)
    packed = prepare_quantized_linears(model, preserve=preserve_quantization, require_all=True)
    mx.eval(model.parameters())
    return packed


def encode(model, mel, lengths, pos_emb, positions, *, floating_indices=False):
    """Supply relative positions explicitly so capture doesn't freeze PE slices."""
    import mlx.core as mx

    encoder = model.encoder
    # Host-supplied frame indices avoid freezing arange extents after strided
    # convolutions, whose ceil-divided dimensions capture cannot always infer.
    import mlx.nn as nn

    x = mel[..., None]
    stage_lengths, stage = lengths, 0
    for layer in encoder.pre_encode.conv:
        x = layer(x)
        if isinstance(layer, nn.Conv2d) and layer.stride != (1, 1):
            if floating_indices:
                # Host-computed ceil divisions avoid integer ops and a beta
                # ANE ceil bug that returns negative lengths.
                lengths = stage_lengths[:, stage]
                stage += 1
                positions = positions[:, ::2] * 0.5
            else:
                lengths = (lengths + 1) // 2
                positions = positions[:, ::2] // 2
            valid = positions < lengths[:, None]
            x = mx.where(valid[:, :, None, None], x, 0)
    x = x.transpose(0, 1, 3, 2).reshape(x.shape[0], x.shape[1], -1)
    x = encoder.pre_encode.out(x)
    x = x * encoder.pos_enc.scale
    valid = positions < lengths[:, None]
    mask = positions[:, None, None, :] >= lengths[:, None, None, None]
    for layer in encoder.layers:
        x = layer(x, pos_emb=pos_emb, mask=mask, valid=valid)
    return x, lengths


def fp16_encoder_graph(graph):
    """Author the fixed encoder in FP16; leave the source and decoder FP32."""
    from dataclasses import replace
    import numpy as np

    nodes = []
    for node in graph.nodes:
        attrs = dict(node.attrs)
        if attrs.get("dtype") == "fp32":
            attrs["dtype"] = "fp16"
        value = attrs.get("value")
        if isinstance(value, np.ndarray) and value.dtype == np.float32:
            attrs["value"] = value.astype(np.float16)
        nodes.append(replace(node, attrs=attrs))
    return replace(graph, nodes=nodes,
                   inputs=[replace(value, dtype="fp16") if value.dtype == "fp32" else value
                           for value in graph.inputs],
                   value_types={name: replace(value, dtype="fp16") if value.dtype == "fp32" else value
                                for name, value in graph.value_types.items()})


def decode_step(model, feature, current_token, hidden, cell):
    """The source's uncompiled TDT step, exposing logits for parity checks."""
    import mlx.core as mx

    prediction = model.decoder.prediction
    embedded = prediction["embed"](current_token)
    embedded = mx.where((current_token == model.blank_id)[..., None], 0, embedded)
    output, (hidden, cell) = prediction["dec_rnn"](embedded, (hidden, cell))
    logits = model.joint(feature, output)[0, 0, 0]
    return logits[:model.blank_id + 1], logits[model.blank_id + 1:], hidden, cell
