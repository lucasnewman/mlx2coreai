# Qwen and SmartTurn Post-Refactor Validation

Tested on macOS 27 build 26A428, CoreAI 1.0.0b2, MLX 0.32.2,
mlx-lm 0.31.3, and mlx-audio 0.5.3, using GPU-preferred specialization.
This specifies a preference, not proof of exclusive GPU placement.

## Qwen

The previously working model is **Qwen3-0.6B**, not Qwen3-0.8B. The separately
investigated Qwen3.5-0.8B hybrid model is not covered by these results; its
[known limitations](qwen35_conversion.md) remain separate.

Reused cached `mlx-community/Qwen3-0.6B-bf16` revision
`42096995f6402fde107068cf530136fe64b604f8`, without downloading or quantizing.
Both fresh exports have dynamic query and KV-cache dimensions and optimization
enabled. The reference uses the same weights and matches export precision.

| Export | Calls | Next-token matches | Max logit absolute error | Max relative L2 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 35 | 33/35 | 0.73828125 | 0.04055925 |
| FP32 | 35 | 35/35 | 0.000124097 | 0.00000547715 |

Each check uses a 3/5/remainder chunked prefill, then 32 advancing one-token
calls, with cache capacity 2048. This is not a 2048-token prompt test. Every call
compares all logits against MLX on identical input tokens; subsequent tokens are
chosen by CoreAI and fed to both runtimes. The fixed-step diagnostic continues
past EOS. These checks do not directly compare every KV-cache element.

BF16 executes and answers the prompt, but is not numerically interchangeable
with the original BF16 MLX forward pass. FP32's much closer agreement is evidence
against a large conversion error on this workload, not proof of bitwise parity
or correctness on all prompts. The two precisions use their own generated token
streams, so the table is not a controlled same-stream precision ablation.

Assets: `artifacts/refactor_qwen3_06b_{bf16,fp32}`.
Reports: `artifacts/refactor_validation/qwen3_{bf16,fp32}.json`.

```bash
MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-bf16/snapshots/42096995f6402fde107068cf530136fe64b604f8"
.build/coreai-b2-env/bin/python -m mlx2coreai._convert_mlx_lm_stateful "$MODEL" \
  --output artifacts/refactor_qwen3_06b_fp32 --max-context-length 256 \
  --revision 42096995f6402fde107068cf530136fe64b604f8 --compute-precision fp32
.build/coreai-b2-env/bin/python scripts/validate_aimodel_mlx.py \
  artifacts/refactor_qwen3_06b_fp32 --model "$MODEL" --compute-precision fp32 \
  --state-capacity 2048 --steps 32 --prefill-chunks 3,5 \
  --max-relative-l2 0.001 --max-abs-error 0.01 \
  --json-output artifacts/refactor_validation/qwen3_fp32.json
```

## SmartTurn

Reused `mlx-community/smart-turn-v3` revision
`74af1538ebace37b0fcc667a6a8a3d902d73eee0`, loaded strictly by mlx-audio.
Although the repository is named v3, its config identifies the source as
`smart-turn-v3.2-gpu.onnx`. This check validates that cached MLX snapshot, not a
separate v3.0 or ONNX implementation.

The unquantized FP32 asset accepts `[batch, 80, 800]` mel features and returns
logits and probabilities. Audio resampling, normalization, left-padding/cropping,
and mel extraction remain unchanged in mlx-audio, outside the CoreAI asset.
The 800-frame window is the source model's native contract, not a workaround.
Batch is dynamic: captured at 1, probed at 2, and tested at 1, 2, and unseen 3.

Eight cases, covering 11 predictions, all agree with MLX endpoint decisions.
Maximum logit absolute error: **0.0000228733**.
Maximum probability absolute error: **0.00000566244**.
Cases include silence, noise, a tone, resampling, cropping, synthetic speech,
and mixed batches. This checks conversion parity, not real-world turn-detection
accuracy on a labeled speech dataset.

Asset: `artifacts/refactor_smart_turn_v3/smart_turn.aimodel`.
Report: `artifacts/refactor_smart_turn_v3/validation.json`.

```bash
.build/coreai-b2-env/bin/python scripts/validate_smart_turn.py \
  --model "$HOME/.cache/huggingface/hub/models--mlx-community--smart-turn-v3/snapshots/74af1538ebace37b0fcc667a6a8a3d902d73eee0" \
  --audio .build/smart_turn_complete.wav
```

The optional speech fixture was generated locally with macOS `say`, then
converted to a 16 kHz WAV with `afconvert`. Omit `--audio` for deterministic
signal-only checks, or supply other audio files. mlx-audio is an optional
requirement for this script, not a new mandatory converter dependency.

## Issues Found

- Captured `AddMM` uses `(A, B, C)` and stores `alpha`/`beta`; the old lowerer
  assumed the public API's `(C, A, B)` and unit coefficients. The codec now marks
  captured ordering and preserves coefficients; manual IR retains bias-first
  defaults. See [MLX's AddMM implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/ops.cpp#L5439).
- Captured `LayerNorm` lacked its last-axis and epsilon attributes. The lowerer
  consequently used all axes. Capture now preserves both, and manual IR's
  missing-axis fallback correctly uses the normalized shape or last axis.
- Derived shape operands such as `batch * frames` were frozen to trace values.
  Reshape now infers a sole unmatched extent using `-1`; later shape operands
  can reference observed intermediate runtime dimensions. No arithmetic formula
  is guessed from two samples.
- CoreAI b2's authoring optimizer incorrectly folds reshape/addmm/reshape while
  leaving the bias flattened. A minimal batched linear repro fails with
  optimization and passes without it. SmartTurn therefore uses `optimize=False`;
  dynamic execution stays enabled. Qwen's optimization remains enabled.

Regression tests cover live AddMM order/scaling, LayerNorm axis/epsilon, derived
dynamic shapes, and optional tiny SmartTurn integration without external weights.
The final suite passes **257 tests**, with the same five existing expected
failures. `git diff --check` passes. The FP32 Qwen conversion was rebuilt and
revalidated after all fixes, reproducing the table above.
