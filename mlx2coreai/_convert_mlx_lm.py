from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .conversion import ConversionConfig, ConvertedCoreAIModel, convert_mlx_to_coreai


@dataclass(slots=True)
class MLXLMConversionInputs:
    input_ids: np.ndarray
    prompt: str | None
    token_count: int
    padded_token_count: int
    synthetic: bool = False

    def as_dict(self, *, input_name: str = "input_ids") -> dict[str, np.ndarray]:
        return {input_name: self.input_ids}


def convert_mlx_lm(
    model_id: str,
    output_path: str | Path,
    *,
    prompt: str | None = None,
    sequence_length: int | None = None,
    batch_size: int = 1,
    input_ids: Any | None = None,
    input_name: str = "input_ids",
    lazy_load: bool = False,
    revision: str | None = None,
    dynamic_sequence: bool = True,
    config: ConversionConfig | None = None,
    dot_output_path: str | Path | None = None,
    load_fn: Callable[..., tuple[Any, Any]] | None = None,
    capture_function: Callable[..., Any] | None = None,
) -> ConvertedCoreAIModel:
    """Load an ``mlx-lm`` model, capture its logits path, and save an ``.aimodel``.

    ``capture_function`` can be supplied for models that need a non-standard
    signature, masks, or cache/state arguments. By default the helper captures
    ``model(input_ids)`` and selects the first output when the model returns a
    tuple, list, or mapping.
    """

    model, tokenizer = load_mlx_lm_model(
        model_id,
        lazy_load=lazy_load,
        revision=revision,
        load_fn=load_fn,
    )
    if hasattr(model, "eval"):
        model.eval()

    mlx_lm_inputs = build_mlx_lm_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        sequence_length=sequence_length,
        batch_size=batch_size,
        input_ids=input_ids,
    )
    resolved_capture = capture_function or _default_mlx_lm_capture(model)
    resolved_config = _resolve_lm_conversion_config(
        config or ConversionConfig(),
        input_name=input_name,
        dynamic_sequence=dynamic_sequence,
        inputs=mlx_lm_inputs,
    )
    converted = convert_mlx_to_coreai(
        model,
        mlx_lm_inputs.as_dict(input_name=input_name),
        config=resolved_config,
        output_path=Path(output_path),
        dot_output_path=Path(dot_output_path) if dot_output_path is not None else None,
        capture_function=resolved_capture,
    )
    converted.metadata["mlx_lm"] = {
        "model_id": model_id,
        "revision": revision,
        "lazy_load": bool(lazy_load),
        "prompt": prompt,
        "sequence_length": None if sequence_length is None else int(sequence_length),
        "capture_sequence_length": int(mlx_lm_inputs.input_ids.shape[1]),
        "batch_size": int(batch_size),
        "input_name": input_name,
        "dynamic_sequence": bool(dynamic_sequence),
        "token_count": int(mlx_lm_inputs.token_count),
        "padded_token_count": int(mlx_lm_inputs.padded_token_count),
        "synthetic_input_ids": bool(mlx_lm_inputs.synthetic),
    }
    return converted


def load_mlx_lm_model(
    model_id: str,
    *,
    lazy_load: bool = False,
    revision: str | None = None,
    load_fn: Callable[..., tuple[Any, Any]] | None = None,
) -> tuple[Any, Any]:
    if load_fn is None:
        try:
            from mlx_lm import load as load_fn
        except ImportError as exc:  # pragma: no cover - depends on optional package install
            raise ImportError(
                "convert_mlx_lm requires mlx-lm. Install it with "
                "`pip install mlx-lm` or pass a custom load_fn."
            ) from exc

    kwargs: dict[str, Any] = {"lazy": bool(lazy_load)}
    if revision is not None:
        kwargs["revision"] = revision
    return load_fn(model_id, **kwargs)


def build_mlx_lm_inputs(
    *,
    tokenizer: Any | None,
    prompt: str | None = None,
    sequence_length: int | None = None,
    batch_size: int = 1,
    input_ids: Any | None = None,
) -> MLXLMConversionInputs:
    if sequence_length is not None and sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    if input_ids is not None:
        array = np.asarray(input_ids, dtype=np.int32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError(f"input_ids must be rank 1 or 2, got shape {array.shape}.")
        return MLXLMConversionInputs(
            input_ids=array,
            prompt=prompt,
            token_count=int(array.shape[-1]),
            padded_token_count=0,
            synthetic=False,
        )

    synthetic = prompt is None
    token_ids = _synthesize_token_ids(tokenizer, sequence_length) if synthetic else _tokenize_prompt(tokenizer, prompt)
    if not token_ids:
        token_ids = [_fallback_token_id(tokenizer)]
    if sequence_length is None:
        token_count = len(token_ids)
        padded_token_count = 0
    else:
        token_count = min(len(token_ids), int(sequence_length))
        if len(token_ids) > sequence_length:
            token_ids = token_ids[:sequence_length]
        padded_token_count = max(0, int(sequence_length) - len(token_ids))
        if padded_token_count:
            token_ids = token_ids + [_pad_token_id(tokenizer, token_ids)] * padded_token_count
    array = np.asarray([token_ids], dtype=np.int32)
    if batch_size > 1:
        array = np.repeat(array, int(batch_size), axis=0)
    return MLXLMConversionInputs(
        input_ids=array,
        prompt=prompt,
        token_count=int(token_count),
        padded_token_count=int(padded_token_count),
        synthetic=synthetic,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an mlx-lm text model to a CoreAI .aimodel asset."
    )
    parser.add_argument("model_id", help="Hugging Face model id loadable with mlx_lm.load().")
    parser.add_argument("--output", type=Path, required=True, help="Output .aimodel directory.")
    parser.add_argument("--prompt", default=None, help="Optional prompt used to build input_ids.")
    parser.add_argument(
        "--sequence-length",
        "--seq-len",
        type=int,
        default=None,
        help="Optional capture sequence length. Defaults to the prompt token length, or 1 for synthesized inputs.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--lazy-load", action="store_true", help="Pass lazy=True to mlx_lm.load().")
    parser.add_argument("--dot-output", type=Path, default=None, help="Optional capture DOT path.")
    parser.add_argument("--no-optimize", action="store_true", help="Skip CoreAI rewrite optimization.")
    parser.add_argument(
        "--dynamic-sequence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit input_ids with a dynamic sequence axis using a probe capture.",
    )
    parser.add_argument(
        "--externalize-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit constants at or above the threshold as CoreAI dense resources.",
    )
    parser.add_argument(
        "--external-weight-threshold",
        type=int,
        default=10,
        help="Minimum element count for resource-backed constants. Use -1 to keep all constants inline.",
    )
    parser.add_argument(
        "--capture-is-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Temporarily set the model to training mode during MLX graph capture.",
    )
    parser.add_argument(
        "--allow-unknown-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the MLX graph parser to retain unknown source nodes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    converted = convert_mlx_lm(
        args.model_id,
        args.output,
        prompt=args.prompt,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        lazy_load=args.lazy_load,
        revision=args.revision,
        dynamic_sequence=bool(args.dynamic_sequence),
        dot_output_path=args.dot_output,
        config=ConversionConfig(
            allow_unknown_sources=bool(args.allow_unknown_sources),
            capture_is_training=bool(args.capture_is_training),
            optimize=not bool(args.no_optimize),
            externalize_weights=bool(args.externalize_weights),
            external_weight_threshold=int(args.external_weight_threshold),
        ),
    )
    print(f"Wrote {converted.asset_path}")
    print(f"Nodes: {len(converted.prepared.normalized_graph.nodes)}")
    resource_count = sum(1 for entry in converted.weight_manifest if entry.get("storage") == "resource")
    inline_count = sum(1 for entry in converted.weight_manifest if entry.get("storage") == "inline")
    print(
        f"Weights: {len(converted.weight_manifest)} constants "
        f"({resource_count} resource, {inline_count} inline)"
    )
    return 0


def _default_mlx_lm_capture(model: Any) -> Callable[[Any], Any]:
    def capture(input_ids: Any) -> Any:
        return _select_primary_output(model(input_ids))

    return capture


def _resolve_lm_conversion_config(
    config: ConversionConfig,
    *,
    input_name: str,
    dynamic_sequence: bool,
    inputs: MLXLMConversionInputs,
) -> ConversionConfig:
    if not dynamic_sequence:
        return config
    dynamic_axes = config.dynamic_axes or {input_name: [1]}
    dynamic_probe_inputs = config.dynamic_probe_inputs or _dynamic_sequence_probe_inputs(inputs, input_name=input_name)
    return replace(
        config,
        capture_shapeless=True,
        dynamic_axes=dynamic_axes,
        dynamic_probe_inputs=dynamic_probe_inputs,
    )


def _dynamic_sequence_probe_inputs(
    inputs: MLXLMConversionInputs,
    *,
    input_name: str,
) -> dict[str, np.ndarray]:
    base = np.asarray(inputs.input_ids, dtype=np.int32)
    if base.ndim != 2:
        raise ValueError(f"LM dynamic sequence probe expects rank-2 input_ids, got {base.shape}.")
    extension = base[:, -1:] if base.shape[1] else np.zeros((base.shape[0], 1), dtype=np.int32)
    return {input_name: np.concatenate([base, extension], axis=1)}


def _select_primary_output(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not value:
            raise ValueError("Model returned an empty mapping output.")
        return next(iter(value.values()))
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Model returned an empty sequence output.")
        return value[0]
    return value


def _tokenize_prompt(tokenizer: Any | None, prompt: str | None) -> list[int]:
    if prompt is None:
        return []
    if tokenizer is None:
        return [_fallback_token_id(None)]
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return [_fallback_token_id(tokenizer)]
    encoded = encode(prompt)
    ids = getattr(encoded, "ids", encoded)
    return [int(token_id) for token_id in ids]


def _synthesize_token_ids(tokenizer: Any | None, sequence_length: int | None) -> list[int]:
    length = int(sequence_length) if sequence_length is not None else 1
    return [_fallback_token_id(tokenizer)] * length


def _pad_token_id(tokenizer: Any | None, token_ids: list[int]) -> int:
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return int(token_ids[-1]) if token_ids else _fallback_token_id(tokenizer)


def _fallback_token_id(tokenizer: Any | None) -> int:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
