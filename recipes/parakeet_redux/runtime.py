"""Offline Redux greedy TDT decoding using only CoreAI and NumPy."""
from dataclasses import dataclass
import json

import numpy as np


@dataclass(frozen=True)
class Request:
    mel: np.ndarray
    # Redux preprocessing appends one padded frame; its default valid length is T - 1.
    length: int | None = None


def encoder_inputs(mel, length, metadata):
    factor = metadata["subsampling_factor"]
    count = (mel.shape[1] + factor - 1) // factor
    # Match mlx-audio's FP32 sinusoidal PE, including the order of operations.
    positions = np.arange(count - 1, -count, -1, dtype=np.float32)[:, None]
    dim = metadata["encoder_dim"]
    rates = np.exp(np.arange(0, dim, 2, dtype=np.float32) * np.float32(-np.log(10000.0) / dim))
    pe = np.empty((1, 2 * count - 1, dim), np.float32)
    pe[0, :, 0::2] = np.sin(positions * rates)
    pe[0, :, 1::2] = np.cos(positions * rates)
    return {"mel": mel, "lengths": np.array([length], np.int32), "pos_emb": pe,
            "positions": np.arange(mel.shape[1], dtype=np.int32)[None]}


def _special(piece):
    return (piece.startswith("<|") and piece.endswith("|>")) or piece in ("<unk>", "<pad>")


async def run(session, request: Request, *, report=None):
    if session.bundle.manifest["recipe"] != "parakeet_redux":
        raise ValueError("Expected a Parakeet Redux recipe bundle.")
    metadata = session.bundle.metadata
    mel = np.asarray(request.mel)
    if mel.ndim == 2:
        mel = mel[None]
    if mel.ndim != 3 or mel.shape[0] != 1 or mel.shape[1] < 3 or mel.shape[2] != metadata["mel_bins"]:
        raise ValueError(f"Mel must have shape [time >= 3, {metadata['mel_bins']}] or [1, time, mel_bins].")
    if not np.issubdtype(mel.dtype, np.floating) or not np.isfinite(mel).all():
        raise ValueError("Mel must contain finite floating-point values.")
    length = mel.shape[1] - 1 if request.length is None else request.length
    if isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)) or not 2 <= length <= mel.shape[1]:
        raise ValueError("Length must be an integer between 2 and the mel frame count.")
    mel = mel.astype(np.float32)
    if not np.isfinite(mel).all():
        raise ValueError("Mel must be representable as finite FP32 values.")
    encoded = await session.run("encoder", encoder_inputs(mel, length, metadata), readback=True)
    features = encoded["features"]
    factor = metadata["subsampling_factor"]
    count = (mel.shape[1] + factor - 1) // factor
    valid = (length + factor - 1) // factor
    if (features.shape != (1, count, metadata["encoder_dim"]) or not np.isfinite(features).all()
            or encoded["lengths"].shape != (1,) or encoded["lengths"][0] != valid):
        raise RuntimeError("Redux encoder produced invalid features or lengths.")
    vocabulary = json.loads(session.bundle.resource("vocabulary.json").read_text())
    blank = metadata["blank_id"]
    if len(vocabulary) != blank:
        raise ValueError("Redux bundle vocabulary does not match its blank ID.")
    state_shape = (metadata["decoder_layers"], 1, metadata["decoder_hidden"])
    hidden = np.zeros(state_shape, np.float32)
    cell = np.zeros_like(hidden)
    last_token, frame, steps = blank, 0, 0
    tokens = []
    durations = metadata["durations"]
    for _ in range(metadata["max_symbols"] * valid):
        if frame >= valid:
            break
        output = await session.run("decoder_step", {
            "feature": features[:, frame:frame + 1], "current_token": np.array([[last_token]], np.int32),
            "hidden": hidden, "cell": cell,
        }, readback=True)
        shapes = {"token_logits": (blank + 1,), "duration_logits": (len(durations),),
                  "hidden": state_shape, "cell": state_shape}
        if any(output[k].shape != shape or not np.isfinite(output[k]).all() for k, shape in shapes.items()):
            raise RuntimeError("Redux decoder produced invalid logits or state.")
        token = int(np.argmax(output["token_logits"]))
        duration = durations[int(np.argmax(output["duration_logits"]))]
        steps += 1
        if token == blank:
            duration = max(duration, 1)
        else:
            last_token, hidden, cell = token, output["hidden"], output["cell"]
            piece = vocabulary[token]
            if not _special(piece):
                start = frame * metadata["frame_seconds"]
                seconds = duration * metadata["frame_seconds"]
                tokens.append({"id": token, "text": piece.replace("▁", " "),
                               "start": start, "duration": seconds, "end": start + seconds})
        frame += duration
    if report is not None:
        report.update(encoder_frames=valid, decoder_steps=steps, completed=frame >= valid,
                      budget_exhausted=frame < valid)
    return {"text": "".join(token["text"] for token in tokens).strip(), "tokens": tokens,
            "completed": frame >= valid}
