"""Evaluate a compressed CoreAI bundle directly against an FP32 bundle."""
import json
from pathlib import Path

import numpy as np

from mlx2coreai.recipe import Bundle
from .runtime import Request, run


def tensor_error(actual, expected, *, atol, rtol):
    actual, expected = np.asarray(actual), np.asarray(expected)
    valid = actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    if not valid:
        return {"valid": False, "allclose": False}
    error = actual.astype(np.float64) - expected.astype(np.float64)
    return {"valid": True, "allclose": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
            "max_abs_error": float(np.max(np.abs(error))) if error.size else 0.0,
            "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12))}


def _size(path):
    return sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())


def _asset_size(bundle):
    return sum(_size(bundle.path / asset) for asset in
               {entry["asset"] for entry in bundle.manifest["components"].values()})


def _compatible(reference, candidate):
    if any(b.manifest["recipe"] != "parakeet_redux" for b in (reference, candidate)):
        raise ValueError("Comparison requires two Parakeet Redux recipe bundles.")
    if reference.metadata.get("weight_format", "fp32") != "fp32":
        raise ValueError("The reference bundle must use FP32 weights.")
    for key in ("mel_bins", "encoder_dim", "subsampling_factor", "decoder_layers", "decoder_hidden",
                "blank_id", "durations", "max_symbols", "processor", "frame_seconds"):
        if reference.metadata[key] != candidate.metadata[key]:
            raise ValueError(f"Bundle contracts differ: {key}")
    if reference.resource("vocabulary.json").read_bytes() != candidate.resource("vocabulary.json").read_bytes():
        raise ValueError("Bundle vocabularies differ.")


async def compare(args):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    reference, candidate = Bundle.open(args.reference), Bundle.open(args.candidate)
    _compatible(reference, candidate)
    cases = [(str(path), np.load(path, allow_pickle=False)) for path in args.mel]
    if args.audio or not cases:
        import mlx.core as mx
        from mlx_audio.stt.models.parakeet.audio import PreprocessArgs, log_mel_spectrogram
        from .cli import prepare_audio

        processor = PreprocessArgs(**reference.metadata["processor"])
        rng = np.random.default_rng(56)
        cases += [("silence_1s", np.asarray(log_mel_spectrogram(mx.zeros(16000), processor))),
                  ("noise_0.37s", np.asarray(log_mel_spectrogram(mx.array(
                      rng.normal(0, 0.01, 5920).astype(np.float32)), processor)))]
        cases += [(str(path), prepare_audio(path, reference.metadata["processor"])) for path in args.audio]
    report = {"reference": str(reference.path.resolve()), "candidate": str(candidate.path.resolve()),
              "reference_asset_bytes": _asset_size(reference), "candidate_asset_bytes": _asset_size(candidate),
              "atol": args.atol, "rtol": args.rtol, "results": [],
              "scope": "Agreement with FP32 on supplied clips; not a labeled ASR benchmark."}
    report["asset_size_reduction"] = 1 - report["candidate_asset_bytes"] / report["reference_asset_bytes"]
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    traces = {}

    def observer(session, component, inputs, outputs):
        traces.setdefault(id(session), {}).setdefault(component, []).append(
            {name: value.numpy().copy() for name, value in outputs.items()})

    async with reference.session(specialization_options=options, observer=observer) as ref_session:
        async with candidate.session(specialization_options=options, observer=observer) as test_session:
            for name, mel in cases:
                traces.clear()
                ref_result = await run(ref_session, Request(mel))
                result = await run(test_session, Request(mel))
                ref_trace, test_trace = traces[id(ref_session)], traces[id(test_session)]
                row = {"case": name, "fp32_text": ref_result["text"], "candidate_text": result["text"],
                       "matching_text": result["text"] == ref_result["text"],
                       "matching_token_ids": [t["id"] for t in result["tokens"]] == [t["id"] for t in ref_result["tokens"]],
                       "matching_timestamps": result["tokens"] == ref_result["tokens"],
                       "completed": result["completed"] and ref_result["completed"],
                       "encoder": tensor_error(test_trace["encoder"][0]["features"],
                                               ref_trace["encoder"][0]["features"], atol=args.atol, rtol=args.rtol)}
                ref_steps, test_steps = ref_trace.get("decoder_step", []), test_trace.get("decoder_step", [])
                row["fp32_decoder_steps"], row["candidate_decoder_steps"] = len(ref_steps), len(test_steps)
                row["matching_step_decisions"] = len(ref_steps) == len(test_steps) and all(
                    np.argmax(a[k]) == np.argmax(b[k]) for a, b in zip(test_steps, ref_steps, strict=True)
                    for k in ("token_logits", "duration_logits"))
                row["decoder"] = {}
                for key in ("token_logits", "duration_logits", "hidden", "cell"):
                    checks = [tensor_error(a[key], b[key], atol=args.atol, rtol=args.rtol)
                              for a, b in zip(test_steps, ref_steps)]
                    row["decoder"][key] = {
                        "allclose": len(ref_steps) == len(test_steps) and all(c["allclose"] for c in checks),
                        "max_abs_error": max((c.get("max_abs_error", float("inf")) for c in checks), default=0.0),
                        "max_relative_l2": max((c.get("relative_l2", float("inf")) for c in checks), default=0.0),
                    }
                row["passed"] = all(row[k] for k in ("matching_text", "matching_token_ids", "matching_timestamps",
                                                     "matching_step_decisions", "completed")) and row["encoder"]["allclose"] and all(
                    c["allclose"] for c in row["decoder"].values())
                report["results"].append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    report["passed"] = all(row["passed"] for row in report["results"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(cases)} cases; "
          f"asset reduction {report['asset_size_reduction']:.1%}; saved {args.output}", flush=True)
    if not report["passed"]:
        raise AssertionError("Compressed bundle differs from FP32; inspect the comparison report.")
    return report
