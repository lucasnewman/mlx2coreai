#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ml_dtypes

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx2coreai._convert_mlx_lm import build_mlx_lm_inputs, load_mlx_lm_model


@dataclass(slots=True)
class StatefulBenchmarkRow:
    context_length: int
    steps: int
    elapsed_sec: float
    tokens_per_sec: float
    output_name: str
    position_start: int
    position_end: int
    sampled_tokens: list[int]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one-token decode throughput for a coreai-models-style stateful .aimodel asset."
    )
    parser.add_argument(
        "asset",
        type=Path,
        help="Unified stateful .aimodel asset or coreai-models-style bundle directory.",
    )
    parser.add_argument("--contexts", default="16,32,64,128,256")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--function-name", default="main")
    parser.add_argument("--input-name", default="input_ids")
    parser.add_argument("--position-ids-name", default="position_ids")
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional tokenizer override. Defaults to the bundle tokenizer when available.",
    )
    parser.add_argument("--revision", default=None, help="Revision for --model-id tokenizer override.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--fill-token-id", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--grow-context",
        action="store_true",
        help="Increment position_ids after each sampled token. Default keeps each interval fixed.",
    )
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args(argv)


async def benchmark(args: argparse.Namespace) -> list[StatefulBenchmarkRow]:
    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}.")
    if args.warmup < 0:
        raise ValueError(f"--warmup must be non-negative, got {args.warmup}.")
    contexts = parse_contexts(args.contexts)
    asset_path = resolve_asset_path(args.asset)
    tokenizer = load_tokenizer(
        args.model_id,
        revision=args.revision,
        bundle_path=resolve_bundle_path(args.asset, asset_path=asset_path),
    )
    if args.prompt is not None and tokenizer is None:
        print(
            "warning: --prompt was provided but no tokenizer was available; using synthetic token ids",
            file=sys.stderr,
        )

    from coreai.authoring import AIModelAsset  # noqa: PLC0415
    from coreai.runtime import NDArray  # noqa: PLC0415

    asset = AIModelAsset.load(asset_path)
    rng = np.random.default_rng(args.seed)
    rows: list[StatefulBenchmarkRow] = []

    print(f"loading executable from {asset_path}", file=sys.stderr)
    async with asset.executable() as model:
        function = model.load_function(args.function_name)
        output_name = args.output_name or first_output_name(function)
        print_table_header()
        for context_length in contexts:
            state_capacity = context_length + (args.steps if args.grow_context else 1)
            state = allocate_state(function, NDArray, state_capacity=state_capacity)
            token_ids = context_token_ids(
                context_length,
                tokenizer=tokenizer,
                prompt=args.prompt,
                fill_token_id=args.fill_token_id,
            )
            outputs = await run_main(
                function,
                NDArray,
                token_ids,
                np.arange(context_length, dtype=np.int32),
                state,
                input_name=args.input_name,
                position_ids_name=args.position_ids_name,
            )
            token = sample_next_token(
                last_token_logits(outputs[output_name].numpy()),
                rng=rng,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            position = context_length
            for _ in range(args.warmup):
                outputs = await run_main(
                    function,
                    NDArray,
                    np.asarray([token], dtype=np.int32),
                    np.asarray([position], dtype=np.int32),
                    state,
                    input_name=args.input_name,
                    position_ids_name=args.position_ids_name,
                )
                token = greedy_token(outputs[output_name].numpy())

            sampled_tokens: list[int] = []
            start_position = position
            start = time.perf_counter()
            for _ in range(args.steps):
                outputs = await run_main(
                    function,
                    NDArray,
                    np.asarray([token], dtype=np.int32),
                    np.asarray([position], dtype=np.int32),
                    state,
                    input_name=args.input_name,
                    position_ids_name=args.position_ids_name,
                )
                token = sample_next_token(
                    last_token_logits(outputs[output_name].numpy()),
                    rng=rng,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                sampled_tokens.append(token)
                if args.grow_context:
                    position += 1
            elapsed = time.perf_counter() - start
            row = StatefulBenchmarkRow(
                context_length=context_length,
                steps=args.steps,
                elapsed_sec=elapsed,
                tokens_per_sec=args.steps / elapsed if elapsed > 0 else float("inf"),
                output_name=output_name,
                position_start=start_position,
                position_end=position,
                sampled_tokens=sampled_tokens,
            )
            rows.append(row)
            print_table_row(row)
            if args.decode and tokenizer is not None:
                print(f"decoded[{context_length}]: {decode_tokens(tokenizer, sampled_tokens)}")

    return rows


def resolve_asset_path(path: Path) -> Path:
    if path.suffix == ".aimodel":
        return path
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        asset_name = metadata.get("assets", {}).get("main")
        if not isinstance(asset_name, str):
            raise ValueError(f"{metadata_path} does not contain assets.main.")
        return path / asset_name
    candidates = sorted(path.glob("*.aimodel")) if path.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Could not resolve .aimodel asset from {path}.")


def resolve_bundle_path(path: Path, *, asset_path: Path) -> Path | None:
    if path.is_dir() and (path / "tokenizer").is_dir():
        return path
    if asset_path.parent.is_dir() and (asset_path.parent / "tokenizer").is_dir():
        return asset_path.parent
    return None


async def run_main(
    function: Any,
    NDArray: Any,
    token_ids: np.ndarray,
    position_ids: np.ndarray,
    state: dict[str, Any],
    *,
    input_name: str,
    position_ids_name: str,
) -> dict[str, Any]:
    return await function(
        inputs={
            input_name: NDArray(np.asarray(token_ids, dtype=np.int32)[None, :]),
            position_ids_name: NDArray(np.asarray(position_ids, dtype=np.int32)[None, :]),
        },
        state=state,
    )


def allocate_state(function: Any, NDArray: Any, *, state_capacity: int) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name in function.desc.state_names:
        descriptor = function.desc.state_descriptor(name=name)
        shape = tuple(int(state_capacity) if int(dim) < 0 else int(dim) for dim in descriptor.shape)
        state[name] = NDArray(np.zeros(shape, dtype=_runtime_dtype_to_numpy(descriptor.dtype)))
    return state


def _runtime_dtype_to_numpy(dtype: Any) -> Any:
    text = str(dtype).strip().lower()
    if "bfloat16" in text or "bf16" in text:
        return ml_dtypes.bfloat16
    if "float16" in text or "fp16" in text:
        return np.float16
    if "float32" in text or "fp32" in text:
        return np.float32
    if "int32" in text:
        return np.int32
    raise ValueError(f"Unsupported runtime state dtype: {dtype!r}.")


def first_output_name(function: Any) -> str:
    output_names = list(function.desc.output_names)
    if not output_names:
        raise ValueError("decode function has no runtime outputs.")
    return str(output_names[0])


def context_token_ids(
    context_length: int,
    *,
    tokenizer: Any | None,
    prompt: str | None,
    fill_token_id: int,
) -> np.ndarray:
    if tokenizer is None and prompt is None:
        return np.full((context_length,), int(fill_token_id), dtype=np.int32)
    inputs = build_mlx_lm_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        sequence_length=max(1, context_length),
        batch_size=1,
    )
    return np.asarray(inputs.input_ids[0], dtype=np.int32)


def parse_contexts(value: str) -> list[int]:
    contexts = [int(chunk.strip()) for chunk in value.split(",") if chunk.strip()]
    if not contexts or any(context <= 0 for context in contexts):
        raise ValueError("--contexts must contain positive integers.")
    return contexts


def load_tokenizer(
    model_id: str | None,
    *,
    revision: str | None,
    bundle_path: Path | None,
) -> Any | None:
    if model_id is None:
        tokenizer_dir = bundle_path / "tokenizer" if bundle_path is not None else None
        if tokenizer_dir is None or not tokenizer_dir.is_dir():
            return None
        print(f"loading tokenizer from {tokenizer_dir}", file=sys.stderr)
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
        except Exception as exc:
            print(
                f"warning: could not import transformers to load embedded tokenizer: {exc}",
                file=sys.stderr,
            )
            return None
        return AutoTokenizer.from_pretrained(str(tokenizer_dir))
    print(f"loading tokenizer from {model_id}", file=sys.stderr)
    model, tokenizer = load_mlx_lm_model(model_id, lazy_load=True, revision=revision)
    del model
    return tokenizer


def last_token_logits(logits: np.ndarray) -> np.ndarray:
    array = np.asarray(logits)
    if array.ndim == 3:
        return np.asarray(array[0, -1, :])
    if array.ndim == 2:
        return np.asarray(array[-1, :])
    if array.ndim == 1:
        return array
    raise ValueError(f"Expected logits with rank 1, 2, or 3, got shape {array.shape}.")


def greedy_token(logits: np.ndarray) -> int:
    return int(np.nanargmax(last_token_logits(logits)))


def sample_next_token(
    logits: np.ndarray,
    *,
    rng: np.random.Generator,
    temperature: float,
    top_k: int,
) -> int:
    scores = np.asarray(logits, dtype=np.float64)
    if temperature == 0:
        return int(np.nanargmax(scores))
    scores = np.nan_to_num(scores / float(temperature), nan=-np.inf)
    if 0 < top_k < scores.shape[-1]:
        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        candidate_scores = scores[candidate_indices]
    else:
        candidate_indices = np.arange(scores.shape[-1])
        candidate_scores = scores
    shifted = candidate_scores - np.max(candidate_scores)
    probabilities = np.exp(shifted)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0:
        return int(candidate_indices[np.nanargmax(candidate_scores)])
    probabilities = probabilities / total
    return int(rng.choice(candidate_indices, p=probabilities))


def decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return " ".join(str(token_id) for token_id in token_ids)
    return str(decode(token_ids))


def print_table_header() -> None:
    print(f"{'context':>8} {'steps':>6} {'elapsed_s':>10} {'tok/s':>10} {'output':>10} {'pos0':>8} {'pos1':>8}")
    print(f"{'-' * 8} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 8}", flush=True)


def print_table_row(row: StatefulBenchmarkRow) -> None:
    print(
        f"{row.context_length:8d} "
        f"{row.steps:6d} "
        f"{row.elapsed_sec:10.3f} "
        f"{row.tokens_per_sec:10.2f} "
        f"{row.output_name:>10} "
        f"{row.position_start:8d} "
        f"{row.position_end:8d}",
        flush=True,
    )


def write_json(path: Path, rows: list[StatefulBenchmarkRow], args: argparse.Namespace) -> None:
    payload = {
        "asset": str(args.asset),
        "function_name": str(args.function_name),
        "contexts": parse_contexts(args.contexts),
        "steps": int(args.steps),
        "warmup": int(args.warmup),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "grow_context": bool(args.grow_context),
        "results": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = asyncio.run(benchmark(args))
    if args.json_output is not None:
        write_json(args.json_output, rows, args)
        print(f"wrote {args.json_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
