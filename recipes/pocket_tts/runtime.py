"""Pocket's generation policy; no MLX or source-model imports."""
from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class Request:
    text: str = "Hello! This is Pocket TTS running with Core AI."
    max_frames: int = 150
    prefill_chunk_size: int = 64
    state_capacity: int | None = None
    frames_after_eos: int = 3
    eos_threshold: float = -4.0
    temperature: float = 0.7
    seed: int = 42
    ignore_eos: bool = False


async def run(session, request: Request, *, report=None):
    """Yield mono FP32 chunks. Reset all model state for each serial request.

    The caller owns the session context, including when iteration stops early.
    Optional report timings exclude session opening and caller time between yields.
    """
    import sentencepiece

    if (request.max_frames <= 0 or request.prefill_chunk_size <= 0 or request.frames_after_eos < 0
            or not np.isfinite(request.temperature) or request.temperature < 0
            or not np.isfinite(request.eos_threshold)):
        raise ValueError("Invalid frame limit, chunk size, tail count, temperature, or EOS threshold.")
    bundle = session.bundle
    if bundle.manifest["recipe"] != "pocket_tts":
        raise ValueError("Expected a Pocket TTS recipe bundle.")
    metadata = bundle.metadata
    blocks = metadata["backbone_components"]
    if not blocks or set(("conditioner", "sampler", "decoder", *blocks)) - set(session.names):
        raise ValueError("Missing Pocket TTS components; finish conversion first.")
    tokenizer = sentencepiece.SentencePieceProcessor(str(bundle.resource("tokenizer.model")))
    tokens = np.array([tokenizer.encode(request.text, out_type=int)], np.int32)
    if tokens.shape[1] == 0:
        raise ValueError("Text must contain at least one token.")
    with np.load(bundle.resource("conditioning.npz")) as conditioning:
        voice, bos = conditioning["voice"], conditioning["bos"]
    required = voice.shape[1] + tokens.shape[1] + request.max_frames
    capacity = required if request.state_capacity is None else request.state_capacity
    if capacity < required:
        raise ValueError(f"State capacity must be at least {required} for this request.")
    stride = metadata["decoder_steps_per_frame"]
    session.reset_state({**{block: capacity for block in blocks}, "decoder": request.max_frames * stride})
    stats = {} if report is None else report
    offset = 0

    async def backbone(embeddings):
        nonlocal offset
        if offset + embeddings.shape[1] > capacity:
            raise ValueError("Backbone KV cache capacity exceeded.")
        hidden = embeddings
        for block in blocks:
            outputs = await session.run(block, {"embeddings": hidden, "position": np.array([offset], np.int32)},
                                        readback=block == blocks[-1])
            hidden = outputs["hidden"]
        offset += embeddings.shape[1]
        return hidden, outputs["eos"]

    start = time.perf_counter()
    text = (await session.run("conditioner", {"tokens": tokens}, readback=True))["embeddings"]
    for conditioning in (voice, text):
        for begin in range(0, conditioning.shape[1], request.prefill_chunk_size):
            await backbone(conditioning[:, begin:begin + request.prefill_chunk_size])
    prefill_seconds = time.perf_counter() - start
    embeddings = bos
    rng = np.random.default_rng(request.seed)
    eos_step = None
    first_audio_seconds = None
    decode_start = time.perf_counter()
    frames = samples = 0
    consumer_seconds = 0.0
    stopped_on_eos = False
    for step in range(request.max_frames):
        hidden, eos = await backbone(embeddings)
        stats["eos_logit"] = float(eos[0, -1, 0])
        if stats["eos_logit"] > request.eos_threshold and eos_step is None:
            eos_step = step
        if not request.ignore_eos and eos_step is not None and step >= eos_step + request.frames_after_eos:
            stopped_on_eos = True
            break
        noise = rng.normal(size=(1, metadata["latent_dim"])).astype(np.float32) * np.float32(request.temperature**0.5)
        sampled = await session.run("sampler", {"hidden": hidden[:, -1], "noise": noise}, readback=True)
        embeddings = sampled["embedding"]
        audio = (await session.run("decoder", {"latent": sampled["latent"][:, None],
            "position": np.array([step * stride], np.int32)}, readback=True))["audio"][0, 0]
        if not np.isfinite(audio).all():
            raise RuntimeError("Decoder produced nonfinite audio.")
        if first_audio_seconds is None:
            first_audio_seconds = time.perf_counter() - start
        frames += 1
        samples += len(audio)
        before_yield = time.perf_counter()
        yield audio
        consumer_seconds += time.perf_counter() - before_yield
    if not frames:
        raise RuntimeError("Generation produced no audio.")
    elapsed = time.perf_counter() - start - consumer_seconds
    decode_seconds = time.perf_counter() - decode_start - consumer_seconds
    stats.update({"bundle": str(bundle.path), "text": request.text, "seed": request.seed,
        "temperature": request.temperature, "frames": frames, "eos_step": eos_step,
        "stopped_on_eos": stopped_on_eos, "sample_rate": metadata["sample_rate"],
        "audio_seconds": samples / metadata["sample_rate"], "elapsed_seconds": elapsed,
        "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
        "first_audio_seconds": first_audio_seconds, "frames_per_second": frames / decode_seconds,
        "frames_per_second_after_first": (frames - 1) / (elapsed - first_audio_seconds) if frames > 1 else None,
        "realtime_factor": elapsed / (samples / metadata["sample_rate"]),
        "positions": {"backbone": offset, "decoder": frames * stride}})
