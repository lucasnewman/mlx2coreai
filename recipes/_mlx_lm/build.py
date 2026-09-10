"""Shared component scaffolding; family adapters own model-specific policy."""
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx2coreai import ConversionConfig, build_mlx_lm_inputs
from mlx2coreai._convert_mlx_lm import load_mlx_lm_model
from mlx2coreai.dtypes import cast_model_precision, normalize_compute_precision
from mlx2coreai.recipe import Build
from .stateful import (
    TRACE_QUERY_LENGTH, TRACE_POSITION_OFFSET, _resolve_compute_precision,
    _make_state_specs, _stateful_main_capture_function, _stateful_component,
)


def load_source(source, adapter, *, revision=None, precision="fp32", load_fn=None):
    model, tokenizer = load_mlx_lm_model(str(source), lazy_load=False, revision=revision, load_fn=load_fn)
    model = adapter.adapt(model)
    model.eval()
    resolved = _resolve_compute_precision(model, precision)
    adapter.warn_precision(resolved)
    if precision != "auto":
        cast_model_precision(model, resolved)
    return model, tokenizer, resolved


def build_model(source, *, recipe, adapter, revision=None, compute_precision="fp32", cache_dtype=None,
                max_context_length=256, config=None, load_fn=None):
    if max_context_length <= 0:
        raise ValueError("Capture context capacity must be positive.")
    model, tokenizer, precision = load_source(source, adapter, revision=revision,
                                             precision=compute_precision, load_fn=load_fn)
    layout = adapter.cache_layout(model)
    cache_dtype = normalize_compute_precision(cache_dtype or precision)
    if cache_dtype == "auto":
        cache_dtype = precision
    specs = _make_state_specs(layout, batch_size=1, max_context_length=max_context_length,
                             cache_dtype=cache_dtype, key_cache_name="keyCache", value_cache_name="valueCache")
    length = min(TRACE_QUERY_LENGTH, max_context_length)
    offset = min(TRACE_POSITION_OFFSET, max_context_length - length)
    inputs = build_mlx_lm_inputs(tokenizer=tokenizer, sequence_length=length)
    forward = _stateful_main_capture_function(model, layout=layout, input_name="input_ids",
        position_ids_name="position_ids", key_cache_name="keyCache", value_cache_name="valueCache",
        cast_bf16_logits_to_fp16=True, array_cache_factory=getattr(adapter, "ArrayCache", None))
    component = _stateful_component(model, lm_inputs=inputs, layout=layout,
        max_context_length=max_context_length, cache_dtype=cache_dtype, input_name="input_ids",
        position_ids_name="position_ids", position_length=offset + length, key_cache_name="keyCache",
        value_cache_name="valueCache", config=config or ConversionConfig(), state_specs=specs,
        capture_function=forward, dynamic_sequence=True, dynamic_state=True)
    resources = {}
    if tokenizer is not None:
        with TemporaryDirectory(prefix="recipe-tokenizer-") as directory:
            tokenizer.save_pretrained(directory)
            root = Path(directory)
            resources = {f"tokenizer/{path.relative_to(root).as_posix()}": path.read_bytes()
                         for path in sorted(root.rglob("*")) if path.is_file()}
    eos = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)
    eos = [] if eos is None else sorted(int(value) for value in (eos if isinstance(eos, (set, list, tuple)) else [eos]))
    vocab = getattr(getattr(model, "args", None), "vocab_size", getattr(tokenizer, "vocab_size", None))
    return Build(recipe, {"main": component}, resources, {
        "source": str(Path(source).resolve()) if Path(source).is_dir() else str(source), "revision": revision,
        "compute_precision": precision, "source_precision_policy": compute_precision, "cache_dtype": cache_dtype,
        "max_context_length": max_context_length, "dynamic_sequence": True, "dynamic_state": True,
        "batch_size": 1, "vocab_size": vocab, "eos_token_ids": eos,
        "attention_layers": [i for i in range(layout.num_layers)
            if not (layout.linear_layers and layout.linear_layers[i])
            and not (layout.short_conv_layers and layout.short_conv_layers[i])],
        "conv_layers": [i for i in range(layout.num_layers)
            if (layout.linear_layers and layout.linear_layers[i]) or (layout.short_conv_layers and layout.short_conv_layers[i])],
        "recurrent": bool(layout.num_linear_layers), "experimental": adapter.experimental(precision, cache_dtype),
        "gated_delta_implementation": component.config.gated_delta_implementation,
        "cast_bf16_logits_to_fp16": True,
        "workarounds": ["per-layer gather/select KV updates with final packing", "BF16 logits cast to FP16",
                        *adapter.WORKAROUNDS],
    })
