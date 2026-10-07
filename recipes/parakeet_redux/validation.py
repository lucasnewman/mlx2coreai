"""Native parity against the packed Redux source, including greedy decoding."""
from dataclasses import asdict
import json

import numpy as np

from mlx2coreai.recipe import export
from recipes._validation import require_full_precision_mlx
from .adapter import decode_step
from .build import load_source, from_model
from .cli import prepare_audio
from .runtime import Request, run


async def validate(args):
    require_full_precision_mlx("Parakeet Redux")
    import mlx.core as mx
    from coreai.runtime import ComputeUnitKind, SpecializationOptions
    from mlx_audio.stt.models.parakeet.audio import log_mel_spectrogram

    model = load_source(args.model, revision=args.revision)
    model.set_dtype(mx.float32)
    rng = np.random.default_rng(56)
    cases = [("silence_1s", np.zeros(16000, np.float32)),
             ("noise_0.37s", rng.normal(0, 0.01, 5920).astype(np.float32))]
    features = [(name, np.asarray(log_mel_spectrogram(mx.array(audio), model.preprocessor_config)))
                for name, audio in cases]
    features += [(path.name, prepare_audio(path, asdict(model.preprocessor_config))) for path in args.audio]
    references = []
    # Capture real packed-source references before recipe preparation expands weights.
    for name, mel in features:
        encoded, lengths = model.encoder(mx.array(mel))
        result = model.decode(mx.array(mel))[0]
        tokens = [asdict(token) for sentence in result.sentences for token in sentence.tokens]
        references.append((np.asarray(encoded), np.asarray(lengths), result.text, tokens))
        print(f"MLX reference {name}: {result.text!r}", flush=True)
    bundle = export(from_model(model, args.model, revision=args.revision, frames=tuple(args.frames),
                               weight_format=args.weight_format), args.output)
    report = {"model": args.model, "mlx_enable_tf32": "0", "atol": args.atol, "rtol": args.rtol,
              "weight_format": args.weight_format,
              "conversion": bundle.manifest["components"], "results": [], "decoder": {}}

    def compare(name, actual, expected, stats):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError(f"{name}: invalid shape or nonfinite values")
        np.testing.assert_allclose(actual, expected, atol=args.atol, rtol=args.rtol, err_msg=name)
        row = stats.setdefault(name, {"calls": 0, "max_abs_error": 0.0})
        row["calls"] += 1
        row["max_abs_error"] = max(row["max_abs_error"], float(np.max(np.abs(actual - expected))))

    def observer(session, name, inputs, outputs):
        if name != "decoder_step":
            return
        reference = decode_step(model, **{k: mx.array(v) for k, v in inputs.items()})
        for key, expected in zip(("token_logits", "duration_logits", "hidden", "cell"), reference, strict=True):
            compare(key, outputs[key].numpy(), np.asarray(expected), report["decoder"])

    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    async with bundle.session(specialization_options=options, observer=observer) as session:
        for (name, mel), (encoded, lengths, text, tokens) in zip(features, references, strict=True):
            row = {"case": name, "mel_frames": mel.shape[1]}
            from .runtime import encoder_inputs
            actual = await session.run("encoder", encoder_inputs(mel, mel.shape[1] - 1, bundle.metadata), readback=True)
            compare("features", actual["features"], encoded, row)
            np.testing.assert_array_equal(actual["lengths"], lengths)
            stats = {}
            result = await run(session, Request(mel), report=stats)
            assert result["completed"], f"{name}: decode budget exhausted"
            assert result["text"] == text, (name, result["text"], text)
            assert result["tokens"] == tokens, (name, result["tokens"], tokens)
            row.update(text=text, matching_tokens=len(tokens), **stats)
            report["results"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (args.output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS: {len(features)} cases; saved {args.output / 'validation.json'}", flush=True)
    return report
