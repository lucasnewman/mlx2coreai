#!/usr/bin/env python3
"""Compare stateful CoreAI prefill/decode against MLX-LM on identical tokens."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_aimodel_sampling import allocate_state, resolve_asset_path, run_main


async def validate(args):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from coreai.authoring import AIModelAsset
    from coreai.runtime import NDArray, SpecializationOptions, ComputeUnitKind

    print(f"Loading MLX reference: {args.model}", flush=True)
    model, tokenizer = load(args.model, lazy=False)
    model.eval()
    cache = make_prompt_cache(model)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True, add_generation_prompt=True,
        enable_thinking=False,
    )
    lengths = [int(n) for n in args.prefill_chunks.split(",") if n]
    if any(n <= 0 for n in lengths):
        raise ValueError("Prefill chunk sizes must be positive.")
    batches = []
    for count in lengths:
        if not prompt:
            break
        batches.append(prompt[:count])
        prompt = prompt[count:]
    if prompt:
        batches.append(prompt)
    capacity = sum(map(len, batches)) + args.steps + 1
    options = SpecializationOptions.cpu_only() if args.device == "cpu" else (
        SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu()))
    asset = AIModelAsset.load(resolve_asset_path(args.asset))
    rows, generated = [], []
    async with asset.executable(specialization_options=options) as executable:
        fn = executable.load_function("main")
        output_name = fn.desc.output_names[0]
        state = allocate_state(fn, NDArray, state_capacity=capacity)
        position = 0
        token = 0
        for index in range(len(batches) + args.steps):
            ids = np.asarray(batches[index] if index < len(batches) else [token], dtype=np.int32)
            expected = np.asarray(model(mx.array(ids[None]), cache=cache).astype(mx.float32))
            if not np.isfinite(expected).all():
                raise AssertionError("MLX reference produced nonfinite logits")
            actual = await run_main(fn, NDArray, ids, np.arange(position, position + len(ids), dtype=np.int32),
                                    state, input_name="input_ids", position_ids_name="position_ids")
            logits = np.asarray(actual[output_name].numpy(), dtype=np.float32)
            if logits.shape != expected.shape or not np.isfinite(logits).all():
                raise AssertionError(f"Invalid CoreAI logits: shape={logits.shape}, expected={expected.shape}")
            error = logits - expected
            relative_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12))
            token = int(logits[0, -1].argmax())
            row = {"position": position, "query_length": len(ids),
                   "max_abs_error": float(np.max(np.abs(error))),
                   "mean_abs_error": float(np.mean(np.abs(error))), "relative_l2": relative_l2,
                   "coreai_token": token, "mlx_token": int(expected[0, -1].argmax())}
            rows.append(row)
            print(json.dumps(row), flush=True)
            if relative_l2 > args.max_relative_l2:
                raise AssertionError(f"Relative L2 {relative_l2:.6f} exceeds {args.max_relative_l2}")
            if index >= len(batches) - 1:
                generated.append(token)
            position += len(ids)
    result = {"model": args.model, "asset": str(args.asset), "device": args.device,
              "prompt": args.prompt, "results": rows, "generated_text": tokenizer.decode(generated)}
    print(result["generated_text"], flush=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="What is the capital of France? Answer in one short sentence.")
    parser.add_argument("--prefill-chunks", default="3,5")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--max-relative-l2", type=float, default=0.05)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.steps < 0 or args.max_relative_l2 <= 0:
        parser.error("steps must be nonnegative and max-relative-l2 must be positive")
    asyncio.run(validate(args))


if __name__ == "__main__":
    main()
