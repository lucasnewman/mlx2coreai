# Conversion Simplification

This refactor preserves generic MLX, manual IR, stateless and stateful MLX-LM,
dynamic query/cache dimensions, multiple CoreAI entrypoints, precision policy,
and Python/Swift execution. Existing public entrypoints remain compatible.

## Stages

1. Shared conversion pipeline: bind state outputs before graph preparation;
   normalize, validate, and infer once; share lowering/save/metadata orchestration.
2. Runtime session: own executable/function/state lifetime in the library;
   make benchmark and validation scripts clients, retaining compatibility helpers.
3. Operation definitions: consolidate decoding, lowering dispatch, optional type
   inference, support reporting, and dtype policy without generating numerical
   reference implementations from those definitions.
4. Typed graph: preserve callback value metadata and multi-output operations;
   retain explicit runtime dimensions/probing and manual-IR inference fallback.
5. Compatibility isolation: separate legacy DOT and reverse-MLX evaluation from
   callback capture, with existing imports preserved.

## Validation Gates

Every stage runs the complete regression suite and a newly converted, unquantized
LFM2.5-2.6B FP32 asset against MLX FP32. Full-model comparisons replay the same
BF16-reference-chosen tokens: 3/5/21-token prefill followed by 32 decode calls,
with KV capacity 2048. Both logits and all three states must stay below 0.01
maximum absolute error and relative L2. This is 61 occupied positions, not a
2048-token prompt. Tiny tests additionally cover nonzero caches, untouched
regions, multiple attention layers, and several dynamic capacities.

The baseline suite passed 209 tests with five expected failures (the existing
Qwen3.5, FP16 LFM, and deeper BF16 LFM runtime/precision limitations). The full
baseline reproduced maximum logit absolute error 0.0001468658447265625 and
maximum state absolute error 0.00013494491577148438, with 35/35 next-token matches.
Raw baseline metrics: `artifacts/refactor_validation/baseline.json`.

Stage reports and logs live in Git-ignored `artifacts/refactor_validation/` and
`.build/refactor_*`. The original validated assets are retained; refactored
conversions use `artifacts/refactor_lfm_fp32`.

## Architecture

The main path is now:

```text
MLX callback capture -> dynamic-shape probing -> signature/state binding
    -> analyze_graph -> CoreAI lowering -> asset/bundle writing
```

- `from_mlx.py` captures primitives, constants, and tensor types. DOT parsing and
  reverse-MLX evaluation live in `_legacy_capture.py`, loaded only on request;
  old capture imports remain available. `reporting.write_graph_dot` renders IR
  directly without executing MLX.
- `conversion.py` owns preparation, lowering, saving, and common metadata.
  `CaptureSignature` binds functional outputs to mutable states before analysis.
  Model adapters describe cache layout and capture the model forward pass rather
  than duplicating the conversion pipeline. `bundle.py` owns bundle packaging.
- `passes.py` returns an `AnalyzedGraph` reused by lowering. Captured
  `TensorType` metadata is authoritative; `_type_inference.py` remains a fallback
  for manual IR and incompletely observed dynamic values. Multi-output
  primitives remain one `Node`, rather than duplicate nodes per result.
- `op_rules.py` connects each supported operation to source aliases, an optional
  attribute decoder, its lowerer, and optional fallback type inference. Support
  reporting derives from these definitions. `dtypes.py` centralizes precision
  policy; it does not change the existing runtime workarounds.
- `runtime.CoreAISession` owns executable/function lifetime and reusable state.
  Python benchmark and validation scripts are clients. Existing one-shot and
  low-level helpers remain compatible, and the Swift runner is unchanged.

To add an operation, register its aliases and handlers in `op_rules.py`, add a
decoder in `_op_attrs.py` only if needed, implement the backend emitter in
`lower_to_coreai.py`, and add independent numerical tests. Add fallback inference
when needed for manual IR or dynamic shapes; do not reconstruct types already
provided by capture. To add a stateful model, extend the model/cache adapter and
its `StateSpec`/`StateBinding` declarations, not the shared pipeline or runtime.

This is a reduction in duplicated responsibilities, not a promise of fewer total
lines. Compatibility implementations and regression tests are retained. Dynamic
shape probing and precision/compiler guards remain deliberate boundaries; no
fixed-length execution or relaxed correctness threshold was introduced.

## Results

- Stage 1 passes 212 tests with the same five expected failures. Fresh full-model
  conversion passes all 35 calls, with metrics identical to baseline (including
  each state's errors). Report: `artifacts/refactor_validation/stage1.json`.
- Stage 2 passes 214 tests with the same five expected failures. Fresh full-model
  conversion through `CoreAISession` matches baseline metrics for all 35 calls.
  Report: `artifacts/refactor_validation/stage2.json`.
- Stage 3 passes 230 tests with the same five expected failures. Fresh full-model
  conversion again matches baseline metrics for all 35 calls. Definitions now
  live in `op_rules.py`, with optional codecs/type rules and backend bindings;
  all boundary precision policies live in `dtypes.py`.
  Report: `artifacts/refactor_validation/stage3.json`.
- Stage 4 passes 240 tests with the same five expected failures. Fresh full-model
  conversion matches baseline metrics for all 35 calls. Captured type metadata
  raises shape coverage from 1787/2210 to 2210/2210 tensors without freezing
  dynamic dimensions. Report: `artifacts/refactor_validation/stage4.json`.
- Stage 5 passes 249 tests with the same five expected failures, including
  compatibility imports, non-executing IR visualization, multi-output replay,
  and result-specific dtype tests. Fresh full-model conversion matches baseline
  metrics for all 35 calls. Report: `artifacts/refactor_validation/stage5.json`.

All five stages replay identical token batches and reproduce the baseline error
summaries exactly. Final maximum logit absolute error is 0.000146866, maximum
logit relative L2 is 0.00000787409, and maximum state absolute error is
0.000134945. Next-token agreement is 35/35. These are numerical parity checks,
not a claim that CoreAI and MLX are bitwise identical.

The full-model gate used the existing local `.build/probe_lfm2_precision.py`
precision probe with `--asset artifacts/refactor_lfm_fp32 --precisions fp32
--state-capacity 2048 --steps 32 --prefill-chunks 3,5 --max-error 0.01` and the
prompt "Explain why the sky appears blue and why sunsets often look red. Use two
short sentences." The reports retain every input batch and per-call metric.
The checked-in `tests/test_lfm2_stateful.py` independently exercises dynamic
prefill/decode, state correctness, and untouched cache regions on tiny models.
