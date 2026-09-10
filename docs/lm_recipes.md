# Language-Model Recipes

The language-model recipes share a CLI and Python runtime. For setup,
checkpoint selection, build/run commands, and compatibility warnings, start
with the model's README:

| Recipe | Guide |
| --- | --- |
| [Qwen3](../recipes/qwen3/README.md) | Validated default FP32 generation |
| [Qwen3.5](../recipes/qwen35/README.md) | Experimental text-only hybrid decoder |
| [LFM2 / LFM2.5](../recipes/lfm2/README.md) | Dense and MoE conversion; parity unresolved |

## Shared Conversion Options

Pass a local checkpoint directory or Hub ID after `convert` to override the
recipe's default source. Missing files are downloaded as needed. Use
`--revision` to pin a Hub revision and `--output` to choose the bundle path.

`--compute-precision` selects `auto`, `fp32`, `fp16`, or `bf16`.
`--cache-dtype` independently selects cache precision. No additional weight
quantization is applied. Default precision and known failures are documented
per recipe; changing precision is not a general fix for conversion correctness.
FP32 exports need more disk and memory than BF16 sources.

Each bundle contains `manifest.json`, a packaged tokenizer, and `main.aimodel`.
Use the recipe's `run` command to generate text or collect timings. Conversion and optional MLX validation need MLX-LM; normal generation
does not load the source checkpoint. Keep bundles in `artifacts/`.

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
import asyncio
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

asyncio.run(generate())
```

Use `token_ids=(...)` instead of `prompt` to bypass tokenization. For experimental
models, inspect `bundle.metadata["experimental"]` before opening a session and
explicitly set `Request(allow_experimental=True, ...)` only for diagnostics.

## Validate a Conversion

Add `--validate-mlx` to your recipe's run command. See the
[Qwen3 validation example](../recipes/qwen3/README.md#validate) for a complete
command with chunked prefill and a saved JSON report.

Validation compares logits and cache state against MLX. `--source` overrides
the source checkpoint path if it moved. The default maximum absolute and
relative L2 error bounds are both 0.01; set `--max-abs-error` and
`--max-relative-l2` deliberately if using a different precision policy.

Validation loads both implementations and uses additional memory. Do not use
its timings as inference benchmarks. Omit `--validate-mlx` for normal execution.
The largest FP32 models may not fit alongside their MLX reference on the same
machine. See the [recipe API](recipe_api.md) for bundle and session usage.
