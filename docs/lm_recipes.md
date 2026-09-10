# Language-Model Recipes

Use `recipes.qwen3`, `recipes.qwen35`, or `recipes.lfm2` to convert a supported
MLX-LM checkpoint and generate text with CoreAI. GPU-preferred execution
requires macOS 27 and a compatible CoreAI SDK. Keep model assets in `artifacts/`.

## Build and Run

```bash
python -m recipes.qwen3 convert --output artifacts/recipes/qwen3_fp32
python -m recipes.qwen3 run artifacts/recipes/qwen3_fp32 --chat \
  --prompt "What is the capital of France?" --max-new-tokens 32
```

| Recipe | Default checkpoint | Default precision | Status |
| --- | --- | --- | --- |
| `qwen3` | `mlx-community/Qwen3-0.6B-bf16` | FP32 | Validated generation |
| `qwen35` | `Qwen/Qwen3.5-0.8B` | Source precision (BF16) | Experimental; decoding parity unresolved |
| `lfm2` | `LiquidAI/LFM2.5-2.6B-MLX-bf16` | FP32 | Exports; full-model parity remains unresolved |

The LFM recipe also accepts `LiquidAI/LFM2-8B-A1B`. MoE execution is experimental:
it can produce incorrect cache contents and encounter Metal command-buffer
errors. Export success is not a guarantee of correct generation.

Supply a checkpoint directory or Hub ID as the positional conversion argument.
Use `--revision` to pin a Hub revision:

```bash
python -m recipes.lfm2 convert LiquidAI/LFM2-8B-A1B \
  --revision c1c44ff9fc00db3ebf4516970563f5f383d23670 \
  --output artifacts/recipes/lfm2_moe_fp32
```

Missing checkpoint files can be downloaded during conversion. No additional
weight quantization is applied. FP32 exports can be much larger than source
BF16 weights; the LFM MoE asset is approximately 32 GiB and conversion needs
additional working memory and disk space.

`--compute-precision` selects `auto`, `fp32`, `fp16`, or `bf16`.
`--cache-dtype` independently selects cache precision. Reduced-precision LFM
execution can fail validation or compilation; FP32 is the correctness baseline,
not a guarantee that every model passes. Experimental bundles require
`--allow-experimental` for diagnostic execution.

Each bundle contains `manifest.json`, a packaged tokenizer, and `main.aimodel`.
Use the recipe's `run` command, not the legacy sampling benchmark, with these
bundles. Conversion and optional MLX validation need MLX-LM; normal generation
does not load the source checkpoint.

## Generation Options

- `--chat` formats the prompt using the packaged tokenizer's chat template.
- `--max-new-tokens` sets the generation budget. Generation normally stops at EOS;
  `--ignore-eos` overrides that behavior for diagnostics.
- `--temperature 0` selects greedy generation. Positive temperatures support
  sampling with `--top-k` and `--seed`.
- `--prefill-chunk-size` sets the usual prefill chunk size. `--prefill-chunks 3,5`
  specifies initial chunk lengths before the remaining prompt is processed.
- `--state-capacity` allocates room for prompt and generated tokens. It must be
  at least the prompt length plus the requested generation budget.
- `--json-output results.json` saves generated text, token IDs, and timings.

Query length and cache capacity are dynamic. Conversion's `--max-context-length`
sets the capture example, not a fixed execution length. Each request resets its
state; do not run concurrent requests through the same session.

## Python API

```python
from coreai.runtime import ComputeUnitKind, SpecializationOptions
from mlx2coreai.recipe import Bundle
from recipes import qwen3

bundle = Bundle.open("artifacts/recipes/qwen3_fp32")
options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())

async def generate():
    async with bundle.session(specialization_options=options, storage_kind="metal") as session:
        request = qwen3.Request(prompt="Hello!", chat=True, max_new_tokens=32)
        async for token_id in qwen3.run(session, request):
            print(token_id)
```

Use `token_ids=(...)` instead of `prompt` to bypass tokenization. For experimental
models, inspect `bundle.metadata["experimental"]` before opening a session and
explicitly set `Request(allow_experimental=True, ...)` only for diagnostics.

## Validate a Conversion

```bash
python -m recipes.qwen3 run artifacts/recipes/qwen3_fp32 --chat \
  --prompt "What is the capital of France?" --prefill-chunks 3,5 \
  --max-new-tokens 32 --state-capacity 2048 --validate-mlx \
  --json-output artifacts/recipes/qwen3_fp32/validation.json
```

Validation compares logits and cache state against MLX. `--source` overrides
the source checkpoint path if it moved. The default maximum absolute and
relative L2 error bounds are both 0.01; set `--max-abs-error` and
`--max-relative-l2` deliberately if using a different precision policy.

Validation loads both implementations and uses additional memory. Do not use
its timings as inference benchmarks. Omit `--validate-mlx` for normal execution.
The largest FP32 models may not fit alongside their MLX reference on the same
machine. See [model recipes](../recipes/README.md) for bundle and session usage.
