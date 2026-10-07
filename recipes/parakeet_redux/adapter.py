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


def encode(model, mel, lengths, pos_emb, positions):
    """Supply relative positions explicitly so capture doesn't freeze PE slices."""
    import mlx.core as mx

    encoder = model.encoder
    # Host-supplied frame indices avoid freezing arange extents after strided
    # convolutions, whose ceil-divided dimensions capture cannot always infer.
    import mlx.nn as nn

    x = mel[..., None]
    for layer in encoder.pre_encode.conv:
        x = layer(x)
        if isinstance(layer, nn.Conv2d) and layer.stride != (1, 1):
            lengths = (lengths + 1) // 2
            positions = positions[:, ::2] // 2
            valid = positions < lengths[:, None]
            x = mx.where(valid[:, :, None, None], x, 0)
    x = x.transpose(0, 1, 3, 2).reshape(x.shape[0], x.shape[1], -1)
    x = encoder.pre_encode.out(x)
    x = x * encoder.pos_enc.scale
    valid = positions < lengths[:, None]
    mask = ~valid[:, None, None, :]
    for layer in encoder.layers:
        x = layer(x, pos_emb=pos_emb, mask=mask, valid=valid)
    return x, lengths


def decode_step(model, feature, current_token, hidden, cell):
    """The source's uncompiled TDT step, exposing logits for parity checks."""
    import mlx.core as mx

    prediction = model.decoder.prediction
    embedded = prediction["embed"](current_token)
    embedded = mx.where((current_token == model.blank_id)[..., None], 0, embedded)
    output, (hidden, cell) = prediction["dec_rnn"](embedded, (hidden, cell))
    logits = model.joint(feature, output)[0, 0, 0]
    return logits[:model.blank_id + 1], logits[model.blank_id + 1:], hidden, cell
