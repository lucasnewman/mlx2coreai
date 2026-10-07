"""Shared request-scoped token generation; no MLX-LM imports at execution time."""
from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class Request:
    prompt: str | None = None
    token_ids: tuple[int, ...] | None = None
    max_new_tokens: int = 32
    prefill_chunk_size: int = 128
    prefill_chunks: tuple[int, ...] = ()
    state_capacity: int | None = None
    temperature: float = 0.0
    top_k: int = 0
    seed: int = 42
    chat: bool = False
    ignore_eos: bool = False
    allow_experimental: bool = False


def load_tokenizer(bundle):
    from transformers import AutoTokenizer

    if not any(name.startswith("tokenizer/") for name in bundle.manifest["resources"]):
        raise ValueError("Bundle has no tokenizer; supply token IDs directly.")
    return AutoTokenizer.from_pretrained(bundle.path / "tokenizer", local_files_only=True, trust_remote_code=False)


def sample(logits, request, rng):
    logits = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise RuntimeError("Model produced nonfinite logits.")
    if request.temperature == 0:
        return int(logits.argmax())
    indices = np.arange(logits.size)
    if request.top_k and request.top_k < logits.size:
        indices = np.argpartition(logits, -request.top_k)[-request.top_k:]
        logits = logits[indices]
    logits = (logits - logits.max()) / request.temperature
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return int(rng.choice(indices, p=probabilities))


async def run(session, request: Request, *, report=None):
    """Yield token IDs; reset KV/conv/recurrent state and positions per request."""
    metadata = session.bundle.metadata
    if metadata.get("experimental") and not request.allow_experimental:
        raise ValueError("This export is experimental and may be incorrect or crash the beta runtime. "
                         "Set allow_experimental=True only for diagnostics.")
    if (request.max_new_tokens <= 0 or request.prefill_chunk_size <= 0
            or any(n <= 0 for n in request.prefill_chunks) or request.top_k < 0
            or not np.isfinite(request.temperature) or request.temperature < 0):
        raise ValueError("Invalid generation length, prefill chunks, temperature, or top_k.")
    if (request.prompt is None) == (request.token_ids is None):
        raise ValueError("Supply exactly one of prompt or token_ids.")
    if request.token_ids is not None:
        ids = np.asarray(request.token_ids)
    else:
        tokenizer = load_tokenizer(session.bundle)
        if request.chat:
            ids = np.asarray(tokenizer.apply_chat_template(
                [{"role": "user", "content": request.prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False))
        else:
            ids = np.asarray(tokenizer.encode(request.prompt))
    vocab = metadata.get("vocab_size")
    if (ids.ndim != 1 or ids.size == 0 or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids > np.iinfo(np.int32).max)
            or (vocab is not None and np.any(ids >= vocab))):
        raise ValueError("Expected a nonempty sequence of in-vocabulary integer token IDs.")
    required = len(ids) + request.max_new_tokens
    capacity = required if request.state_capacity is None else request.state_capacity
    if capacity < required:
        raise ValueError(f"State capacity must be at least {required}.")
    session.reset_state({"main": capacity})
    position = 0

    async def forward(tokens):
        nonlocal position
        tokens = np.asarray(tokens, dtype=np.int32)
        result = await session.run("main", {"input_ids": tokens[None],
            "position_ids": np.arange(position, position + len(tokens), dtype=np.int32)[None]}, readback=True)
        position += len(tokens)
        logits = result["logits"]
        if logits.ndim != 3 or logits.shape[:2] != (1, len(tokens)):
            raise RuntimeError(f"Unexpected logits shape {logits.shape}.")
        return logits[0, -1]

    start = time.perf_counter()
    for count in request.prefill_chunks:
        if position == len(ids):
            break
        logits = await forward(ids[position:position + count])
    while position < len(ids):
        logits = await forward(ids[position:position + request.prefill_chunk_size])
    prefill_seconds = time.perf_counter() - start
    rng = np.random.default_rng(request.seed)
    eos = set(metadata["eos_token_ids"])
    generated, consumer_seconds, finish = [], 0.0, "length"
    for index in range(request.max_new_tokens):
        token = sample(logits, request, rng)
        if token in eos and not request.ignore_eos:
            finish = "eos"
            break
        generated.append(token)
        before_yield = time.perf_counter()
        yield token
        consumer_seconds += time.perf_counter() - before_yield
        if index + 1 < request.max_new_tokens:
            logits = await forward([token])
    if report is not None:
        report.update({"tokens": generated, "finish_reason": finish, "prompt_tokens": len(ids),
            "position": position, "state_capacity": capacity, "prefill_seconds": prefill_seconds,
            "elapsed_seconds": time.perf_counter() - start - consumer_seconds})
