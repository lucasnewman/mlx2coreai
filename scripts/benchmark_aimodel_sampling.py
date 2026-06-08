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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx2coreai._convert_mlx_lm import build_mlx_lm_inputs, load_mlx_lm_model
from mlx2coreai.runtime import (
    _load_coreai_runtime,
    _output_to_numpy,
    _resolve_storage_kind,
    _to_ndarray,
)


@dataclass(slots=True)
class BenchmarkRow:
    context_length: int
    steps: int
    elapsed_sec: float
    tokens_per_sec: float
    output_name: str
    final_context_length: int
    sampled_tokens: list[int]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a converted CoreAI language-model .aimodel and report "
            "tokens/sec at fixed context intervals."
        )
    )
    parser.add_argument("asset", type=Path, help="Path to the .aimodel directory.")
    parser.add_argument(
        "--contexts",
        default="16,32,64,128,256",
        help="Comma-separated context lengths to benchmark.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of generated tokens to time for each context length.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed forwards to run before each context length.",
    )
    parser.add_argument("--function-name", default="main")
    parser.add_argument("--input-name", default="input_ids")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Logits output name. Defaults to the first output returned by the runtime.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help=(
            "Optional mlx-lm model id used only to load a tokenizer for the "
            "prompt and --decode output. If omitted, synthetic token ids are used."
        ),
    )
    parser.add_argument("--revision", default=None, help="Optional mlx-lm revision.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt to tokenize when --model-id is provided.",
    )
    parser.add_argument(
        "--fill-token-id",
        type=int,
        default=0,
        help="Synthetic token id used when no tokenizer is loaded.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for deterministic greedy decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k candidate count for non-greedy sampling. Use 0 to sample all logits.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--grow-context",
        action="store_true",
        help=(
            "Append generated tokens without trimming. By default the script "
            "keeps a fixed sliding window for each context interval."
        ),
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Decode sampled token ids when --model-id provides a tokenizer.",
    )
    parser.add_argument(
        "--storage-kind",
        default=None,
        help="Optional coreai.runtime.StorageKind name for input NDArrays.",
    )
    parser.add_argument(
        "--compute-unit",
        default="auto",
        choices=("auto", "default", "cpu", "cpu-preferred", "gpu", "neural-engine"),
        help=(
            "CoreAI specialization target. 'auto' preserves asset.executable() "
            "behavior, 'default' passes SpecializationOptions.default(), 'cpu' "
            "uses CPU-only specialization, and CPU-preferred/GPU/Neural Engine "
            "are preferred compute-unit hints."
        ),
    )
    parser.add_argument(
        "--debug-specialization",
        action="store_true",
        help="Enable CoreAI specialization debug mode when supported by the runtime.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write benchmark results as JSON.",
    )
    return parser.parse_args(argv)


async def benchmark(args: argparse.Namespace) -> list[BenchmarkRow]:
    contexts = parse_contexts(args.contexts)
    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}.")
    if args.warmup < 0:
        raise ValueError(f"--warmup must be non-negative, got {args.warmup}.")
    if args.temperature < 0:
        raise ValueError(f"--temperature must be non-negative, got {args.temperature}.")
    if args.top_k < 0:
        raise ValueError(f"--top-k must be non-negative, got {args.top_k}.")

    tokenizer = load_tokenizer(args.model_id, revision=args.revision)
    if args.prompt is not None and tokenizer is None:
        print(
            "warning: --prompt was provided without --model-id; using synthetic token ids",
            file=sys.stderr,
        )

    bindings = _load_coreai_runtime()
    storage_kind = _resolve_storage_kind(args.storage_kind, bindings.StorageKind)
    specialization_options = resolve_specialization_options(args, bindings)
    asset = bindings.AIModelAsset.load(args.asset)
    rng = np.random.default_rng(args.seed)
    output_name = args.output_name
    rows: list[BenchmarkRow] = []

    print(
        f"loading executable from {args.asset} "
        f"(compute_unit={args.compute_unit})",
        file=sys.stderr,
    )
    async with asset.executable(specialization_options=specialization_options) as ai_model:
        function = ai_model.load_function(args.function_name)
        print_table_header()

        for context_length in contexts:
            context = make_initial_context(
                context_length,
                tokenizer=tokenizer,
                prompt=args.prompt,
                fill_token_id=args.fill_token_id,
            )
            for _ in range(args.warmup):
                _, output_name = await run_logits(
                    function,
                    context,
                    input_name=args.input_name,
                    output_name=output_name,
                    bindings=bindings,
                    storage_kind=storage_kind,
                )

            sampled_tokens: list[int] = []
            start = time.perf_counter()
            for _ in range(args.steps):
                logits, output_name = await run_logits(
                    function,
                    context,
                    input_name=args.input_name,
                    output_name=output_name,
                    bindings=bindings,
                    storage_kind=storage_kind,
                )
                token_id = sample_next_token(
                    last_token_logits(logits),
                    rng=rng,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                sampled_tokens.append(token_id)
                context = append_token(
                    context,
                    token_id,
                    max_length=None if args.grow_context else context_length,
                )
            elapsed_sec = time.perf_counter() - start
            row = BenchmarkRow(
                context_length=context_length,
                steps=args.steps,
                elapsed_sec=elapsed_sec,
                tokens_per_sec=args.steps / elapsed_sec if elapsed_sec > 0 else float("inf"),
                output_name=str(output_name),
                final_context_length=int(context.shape[1]),
                sampled_tokens=sampled_tokens,
            )
            rows.append(row)
            print_table_row(row)
            if args.decode and tokenizer is not None:
                print(f"decoded[{context_length}]: {decode_tokens(tokenizer, sampled_tokens)}")

    return rows


def parse_contexts(value: str) -> list[int]:
    contexts: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        context = int(chunk)
        if context <= 0:
            raise ValueError(f"context lengths must be positive, got {context}.")
        contexts.append(context)
    if not contexts:
        raise ValueError("--contexts must contain at least one positive integer.")
    return contexts


def load_tokenizer(model_id: str | None, *, revision: str | None) -> Any | None:
    if model_id is None:
        return None
    print(f"loading tokenizer from {model_id}", file=sys.stderr)
    model, tokenizer = load_mlx_lm_model(
        model_id,
        lazy_load=True,
        revision=revision,
    )
    del model
    return tokenizer


def resolve_specialization_options(args: argparse.Namespace, bindings: Any) -> Any | None:
    if args.compute_unit == "auto" and not args.debug_specialization:
        return None

    SpecializationOptions = getattr(bindings, "SpecializationOptions", None)
    ComputeUnitKind = getattr(bindings, "ComputeUnitKind", None)
    if SpecializationOptions is None or ComputeUnitKind is None:
        raise RuntimeError("Installed coreai.runtime does not expose specialization options.")

    if args.compute_unit == "auto" or args.compute_unit == "default":
        options = SpecializationOptions.default()
    elif args.compute_unit == "cpu":
        options = SpecializationOptions.cpu_only()
    elif args.compute_unit == "cpu-preferred":
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.cpu())
    elif args.compute_unit == "gpu":
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    elif args.compute_unit == "neural-engine":
        options = SpecializationOptions.from_preferred_compute_unit_kind(
            ComputeUnitKind.neural_engine()
        )
    else:  # pragma: no cover - argparse choices prevent this path.
        raise ValueError(f"Unknown compute unit: {args.compute_unit!r}")

    if args.debug_specialization:
        with_debug = getattr(options, "with_debug", None)
        if not callable(with_debug):
            raise RuntimeError("Installed coreai.runtime does not support debug specialization.")
        options = with_debug(enabled=True)
    return options


def make_initial_context(
    context_length: int,
    *,
    tokenizer: Any | None,
    prompt: str | None,
    fill_token_id: int,
) -> np.ndarray:
    if tokenizer is None and prompt is None:
        return np.full((1, context_length), int(fill_token_id), dtype=np.int32)
    inputs = build_mlx_lm_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        sequence_length=context_length,
        batch_size=1,
    )
    return np.asarray(inputs.input_ids, dtype=np.int32)


async def run_logits(
    function: Any,
    input_ids: np.ndarray,
    *,
    input_name: str,
    output_name: str | None,
    bindings: Any,
    storage_kind: Any | None,
) -> tuple[np.ndarray, str]:
    outputs = await function(
        inputs={
            input_name: _to_ndarray(input_ids, bindings.NDArray, storage_kind),
        }
    )
    resolved_name = output_name
    if resolved_name is None:
        try:
            resolved_name = next(iter(outputs))
        except StopIteration as exc:
            raise ValueError("CoreAI runtime returned no outputs.") from exc
    if resolved_name not in outputs:
        available = ", ".join(str(name) for name in outputs)
        raise KeyError(f"Output {resolved_name!r} not found. Available outputs: {available}")
    return _output_to_numpy(outputs[resolved_name]), str(resolved_name)


def last_token_logits(logits: np.ndarray) -> np.ndarray:
    array = np.asarray(logits)
    if array.ndim == 3:
        return np.asarray(array[0, -1, :])
    if array.ndim == 2:
        return np.asarray(array[-1, :])
    if array.ndim == 1:
        return array
    raise ValueError(f"Expected logits with rank 1, 2, or 3, got shape {array.shape}.")


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


def append_token(
    input_ids: np.ndarray,
    token_id: int,
    *,
    max_length: int | None,
) -> np.ndarray:
    token = np.asarray([[int(token_id)]], dtype=np.int32)
    appended = np.concatenate([input_ids, token], axis=1)
    if max_length is None:
        return appended
    return appended[:, -int(max_length) :]


def decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return " ".join(str(token_id) for token_id in token_ids)
    return str(decode(token_ids))


def print_table_header() -> None:
    print(f"{'context':>8} {'steps':>6} {'elapsed_s':>10} {'tok/s':>10} {'output':>10} {'final_seq':>10}")
    print(f"{'-' * 8} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}", flush=True)


def print_table_row(row: BenchmarkRow) -> None:
    print(
        f"{row.context_length:8d} "
        f"{row.steps:6d} "
        f"{row.elapsed_sec:10.3f} "
        f"{row.tokens_per_sec:10.2f} "
        f"{row.output_name:>10} "
        f"{row.final_context_length:10d}",
        flush=True,
    )


def write_json(path: Path, rows: list[BenchmarkRow], args: argparse.Namespace) -> None:
    payload = {
        "asset": str(args.asset),
        "contexts": parse_contexts(args.contexts),
        "steps": int(args.steps),
        "warmup": int(args.warmup),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "grow_context": bool(args.grow_context),
        "compute_unit": str(args.compute_unit),
        "debug_specialization": bool(args.debug_specialization),
        "results": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = asyncio.run(benchmark(args))
    except RuntimeError as exc:
        hint = runtime_error_hint(exc, args)
        if hint is not None:
            print(hint, file=sys.stderr)
            return 1
        raise
    if args.json_output is not None:
        write_json(args.json_output, rows, args)
        print(f"wrote {args.json_output}", file=sys.stderr)
    return 0


def runtime_error_hint(exc: RuntimeError, args: argparse.Namespace) -> str | None:
    message = str(exc)
    if "coreai.reshape" not in message or "The output shape must have the same number of elements" not in message:
        return None
    command = (
        "python -m mlx2coreai convert-mlx-lm mlx-community/Qwen3-0.6B-bf16 "
        "--output qwen3-fixed.aimodel"
    )
    if args.model_id is not None:
        command = f"python -m mlx2coreai convert-mlx-lm {args.model_id} --output qwen3-fixed.aimodel"
    return (
        "\nCoreAI runtime rejected a reshape while executing the asset. This "
        "matches the stale optimized dynamic causal-attention asset failure "
        "seen with earlier qwen3.aimodel builds.\n\n"
        "Regenerate the .aimodel with the current converter defaults, which "
        "skip the beta CoreAI optimizer for dynamic causal SDPA graphs, or "
        "pass --no-optimize explicitly. Example:\n\n"
        f"  {command}\n\n"
        "Then benchmark the regenerated asset. Original runtime error:\n"
        f"  {message}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
