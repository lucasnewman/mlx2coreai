from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from mlx2coreai.bundle import (
    _resolve_bundle_paths,
    _write_coreai_models_bundle,
    _write_tokenizer,
    _build_bundle_metadata,
    _vocab_size,
)
from mlx2coreai.dtypes import (
    normalize_compute_precision as _normalize_cache_dtype,
    floating_precision as _dtype_to_precision,
    cast_model_precision as _apply_model_compute_precision,
    cache_numpy_dtype as _cache_np_dtype,
)
from mlx2coreai._convert_mlx_lm import MLXLMConversionInputs, build_mlx_lm_inputs, load_mlx_lm_model
from mlx2coreai.conversion import (
    ConversionConfig,
    PreparedMLXGraph,
    convert_prepared_mlx_to_coreai,
    prepare_mlx_conversion,
)
from mlx2coreai.dynamic_shapes import DynamicAxes
from mlx2coreai.ir import Graph, StateSpec
from mlx2coreai.lower_to_coreai import LoweredCoreAIProgram
from mlx2coreai.signature import CaptureSignature, StateBinding
from mlx2coreai.kv_cache import LayeredKVCacheState as _LayeredKVCacheState, LayeredKVCache as _ExportableLayeredKVCache
from mlx2coreai.recipe import Component


TRACE_QUERY_LENGTH = 16
TRACE_POSITION_OFFSET = 8


@dataclass(slots=True)
class MLXLMStatefulConversion:
    main: PreparedMLXGraph
    lowered: LoweredCoreAIProgram
    asset: Any | None
    bundle_path: Path | None
    asset_path: Path | None
    bundle_metadata: dict[str, Any] | None
    max_context_length: int
    inputs: MLXLMConversionInputs
    state_specs: list[StateSpec]
    metadata: dict[str, Any]

    @property
    def program(self) -> Any:
        return self.lowered.program

    @property
    def weight_manifest(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.lowered.weight_manifest]


@dataclass(frozen=True, slots=True)
class _CacheLayout:
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    linear_layers: tuple[bool, ...] = ()
    conv_shape: tuple[int, int] = (0, 0)
    recurrent_shape: tuple[int, int, int] = (0, 0, 0)
    short_conv_layers: tuple[bool, ...] = ()

    @property
    def num_linear_layers(self) -> int:
        return sum(self.linear_layers)

    @property
    def num_short_conv_layers(self) -> int:
        return sum(self.short_conv_layers)

    @property
    def num_conv_layers(self) -> int:
        return self.num_linear_layers + self.num_short_conv_layers


class _ExportableRecurrentCache:
    """The unpadded ArraysCache interface used by MLX-LM's hybrid layers."""

    lengths = None

    def __init__(self, conv: Any, recurrent: Any = None):
        self.state = [conv] if recurrent is None else [conv, recurrent]

    def __getitem__(self, index: int) -> Any:
        return self.state[index]

    def __setitem__(self, index: int, value: Any) -> None:
        self.state[index] = value

    def advance(self, count: int) -> None:
        pass

    def make_mask(self, count: int) -> None:
        return None




def convert_mlx_lm_stateful(
    model_id: str,
    output_path: str | Path,
    *,
    max_context_length: int = 256,
    batch_size: int = 1,
    revision: str | None = None,
    input_name: str = "input_ids",
    position_ids_name: str = "position_ids",
    key_cache_name: str = "keyCache",
    value_cache_name: str = "valueCache",
    compute_precision: str = "auto",
    cache_dtype: str | None = None,
    entrypoint_name: str = "main",
    dynamic_sequence: bool = True,
    dynamic_state: bool = True,
    cast_bf16_logits_to_fp16: bool = True,
    config: ConversionConfig | None = None,
    load_fn: Callable[..., tuple[Any, Any]] | None = None,
) -> MLXLMStatefulConversion:
    """Convert an mlx-lm model into one stateful CoreAI asset.

    The generated ``.aimodel`` follows the macOS LLM contract used by
    ``coreai-models``: a single dynamic ``main`` entrypoint with ``input_ids``,
    ``position_ids``, and mutable KV-cache state tensors named ``keyCache``
    and ``valueCache`` by default. Short-convolution models also carry
    ``convState``; hybrid gated-delta models additionally carry an FP32
    ``recurrentState`` tensor.
    """

    if max_context_length <= 0:
        raise ValueError(f"max_context_length must be positive, got {max_context_length}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if batch_size != 1:
        raise ValueError("stateful mlx-lm conversion currently supports batch_size=1 only.")

    model, tokenizer = load_mlx_lm_model(
        model_id,
        lazy_load=False,
        revision=revision,
        load_fn=load_fn,
    )
    from .policy import unwrap, warn_precision
    model = unwrap(model)
    if hasattr(model, "eval"):
        model.eval()

    layout = _infer_cache_layout(model)
    resolved_compute_precision = _resolve_compute_precision(model, compute_precision)
    warn_precision(layout, resolved_compute_precision)
    if compute_precision != "auto":
        _apply_model_compute_precision(model, resolved_compute_precision)
    resolved_cache_dtype = _normalize_cache_dtype(cache_dtype or resolved_compute_precision)
    state_specs = _make_state_specs(
        layout,
        batch_size=batch_size,
        max_context_length=max_context_length,
        cache_dtype=resolved_cache_dtype,
        key_cache_name=key_cache_name,
        value_cache_name=value_cache_name,
    )

    trace_sequence_length = min(TRACE_QUERY_LENGTH, int(max_context_length))
    trace_offset = TRACE_POSITION_OFFSET
    if trace_offset + trace_sequence_length > max_context_length:
        trace_offset = max(0, int(max_context_length) - int(trace_sequence_length))

    lm_inputs = build_mlx_lm_inputs(
        tokenizer=tokenizer,
        sequence_length=trace_sequence_length,
        batch_size=batch_size,
    )

    base_config = config or ConversionConfig()
    capture_function = _stateful_main_capture_function(
        model,
        layout=layout,
        input_name=input_name,
        position_ids_name=position_ids_name,
        key_cache_name=key_cache_name,
        value_cache_name=value_cache_name,
        cast_bf16_logits_to_fp16=bool(cast_bf16_logits_to_fp16),
    )

    main = _prepare_stateful_entry(
        model,
        lm_inputs=lm_inputs,
        layout=layout,
        max_context_length=max_context_length,
        cache_dtype=resolved_cache_dtype,
        input_name=input_name,
        position_ids_name=position_ids_name,
        position_length=trace_offset + trace_sequence_length,
        key_cache_name=key_cache_name,
        value_cache_name=value_cache_name,
        config=base_config,
        state_specs=state_specs,
        capture_function=capture_function,
        dynamic_sequence=bool(dynamic_sequence),
        dynamic_state=bool(dynamic_state),
    )

    conversion_config = replace(
        main.config or base_config,
        entrypoint_name=entrypoint_name,
        state_specs=state_specs,
    )

    bundle_path, asset_path, bundle_name = _resolve_bundle_paths(output_path)
    bundle_path.mkdir(parents=True, exist_ok=True)
    converted = convert_prepared_mlx_to_coreai(main, config=conversion_config, output_path=asset_path)
    lowered, asset = converted.lowered, converted.asset
    bundle_metadata = _write_coreai_models_bundle(
        bundle_path,
        tokenizer=tokenizer,
        model=model,
        model_id=model_id,
        revision=revision,
        name=bundle_name,
        asset_path=asset_path,
        max_context_length=max_context_length,
        entrypoint_name=entrypoint_name,
    )
    metadata = {
        **converted.metadata,
        "mlx_lm_stateful": {
            "model_id": model_id,
            "revision": revision,
            "max_context_length": int(max_context_length),
            "trace_sequence_length": int(lm_inputs.input_ids.shape[1]),
            "trace_offset": int(trace_offset),
            "dynamic_sequence": bool(dynamic_sequence),
            "dynamic_state": bool(dynamic_state),
            "batch_size": int(batch_size),
            "input_name": input_name,
            "position_ids_name": position_ids_name,
            "key_cache_name": key_cache_name,
            "value_cache_name": value_cache_name,
            "compute_precision": resolved_compute_precision,
            "cache_dtype": resolved_cache_dtype,
            "entrypoints": {"main": entrypoint_name},
            "state_count": len(state_specs),
            "num_layers": layout.num_layers,
            "num_linear_layers": layout.num_linear_layers,
            "num_short_conv_layers": layout.num_short_conv_layers,
            "num_attention_layers": layout.num_layers - layout.num_conv_layers,
            "gated_delta_implementation": base_config.gated_delta_implementation,
            "num_key_value_heads": layout.num_key_value_heads,
            "head_dim": layout.head_dim,
            "cast_bf16_logits_to_fp16": bool(cast_bf16_logits_to_fp16),
        },
        "coreai_models_bundle": bundle_metadata,
        "entrypoint_names": list(lowered.entrypoint_names),
        "state_specs": [spec.to_dict() for spec in state_specs],
    }

    (bundle_path / "conversion.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return MLXLMStatefulConversion(
        main=main,
        lowered=lowered,
        asset=asset,
        bundle_path=bundle_path,
        asset_path=asset_path,
        bundle_metadata=bundle_metadata,
        max_context_length=int(max_context_length),
        inputs=lm_inputs,
        state_specs=state_specs,
        metadata=metadata,
    )







def _stateful_component(
    model: Any,
    *,
    lm_inputs: MLXLMConversionInputs,
    layout: _CacheLayout,
    max_context_length: int,
    cache_dtype: str,
    input_name: str,
    position_ids_name: str,
    position_length: int,
    key_cache_name: str,
    value_cache_name: str,
    config: ConversionConfig,
    state_specs: list[StateSpec],
    capture_function: Callable[..., tuple[Any, ...]],
    dynamic_sequence: bool,
    dynamic_state: bool,
) -> Component:
    inputs = _stateful_inputs(
        lm_inputs,
        state_specs=state_specs,
        input_name=input_name,
        position_ids_name=position_ids_name,
        position_length=position_length,
        cache_dtype=cache_dtype,
    )
    dynamic_axes: DynamicAxes | None = None
    dynamic_probe_inputs: Mapping[str, Any] | None = None
    if dynamic_sequence or dynamic_state:
        dynamic_axes_dict: dict[str, list[int]] = {}
        if dynamic_sequence:
            dynamic_axes_dict[input_name] = [1]
            dynamic_axes_dict[position_ids_name] = [1]
        if dynamic_state:
            dynamic_axes_dict.update({
                spec.name: [spec.capacity_axis] for spec in state_specs if spec.capacity_axis is not None
            })
        dynamic_axes = dynamic_axes_dict
        probe_length = _probe_sequence_length(
            int(lm_inputs.input_ids.shape[1]),
            max_context_length=max_context_length,
        )
        probe_position_length = max(int(position_length) + (probe_length - int(lm_inputs.input_ids.shape[1])), 1)
        probe_state_context_length = int(max_context_length) + 1 if dynamic_state else int(max_context_length)
        if (
            probe_length != int(lm_inputs.input_ids.shape[1])
            or probe_position_length != int(position_length)
            or probe_state_context_length != int(max_context_length)
        ):
            probe_inputs = _resize_lm_inputs(lm_inputs, sequence_length=probe_length)
            dynamic_probe_inputs = _stateful_inputs(
                probe_inputs,
                state_specs=state_specs,
                input_name=input_name,
                position_ids_name=position_ids_name,
                position_length=probe_position_length,
                cache_dtype=cache_dtype,
                state_context_length=probe_state_context_length,
                context_state_names=(key_cache_name, value_cache_name),
            )
        else:
            dynamic_axes = None

    stateful_config = replace(
        config,
        capture_shapeless=bool(dynamic_axes),
        dynamic_axes=dynamic_axes,
        dynamic_probe_inputs=dynamic_probe_inputs,
        state_specs=None,
        signature=CaptureSignature(
            input_order=(input_name, position_ids_name, *(spec.name for spec in state_specs)),
            states=tuple(StateBinding(spec, index + 1) for index, spec in enumerate(state_specs)),
            output_count=1 + len(state_specs),
        ),
    )
    return Component(capture_function, inputs, ("logits",), stateful_config)


def _prepare_stateful_entry(model, **kwargs):
    component = _stateful_component(model, **kwargs)
    return prepare_mlx_conversion(component.forward, component.inputs, config=component.config)


def _stateful_main_capture_function(
    model: Any,
    *,
    layout: _CacheLayout,
    input_name: str,
    position_ids_name: str,
    key_cache_name: str,
    value_cache_name: str,
    cast_bf16_logits_to_fp16: bool,
    array_cache_factory=None,
) -> Callable[..., tuple[Any, ...]]:
    if array_cache_factory is None:
        from .policy import array_cache_factory as select_array_cache
        array_cache_factory = select_array_cache(layout)

    def capture(**kwargs: Any) -> tuple[Any, ...]:
        import mlx.core as mx  # noqa: PLC0415

        input_ids = kwargs[input_name]
        position_ids = kwargs[position_ids_name]
        offset = _offset_from_position_ids(input_ids, position_ids)
        state = _LayeredKVCacheState(
            keys=kwargs[key_cache_name],
            values=kwargs[value_cache_name],
        )
        caches = []
        array_caches = []
        attention_caches = []
        attention_index = 0
        linear_layers = layout.linear_layers or (False,) * layout.num_layers
        short_conv_layers = layout.short_conv_layers or (False,) * layout.num_layers
        for is_linear, is_short_conv in zip(linear_layers, short_conv_layers, strict=True):
            if is_linear or is_short_conv:
                index = len(array_caches)
                recurrent = kwargs["recurrentState"][index] if is_linear else None
                cache = array_cache_factory(kwargs["convState"][index], recurrent)
                array_caches.append(cache)
            else:
                cache = _ExportableLayeredKVCache(state, layer_idx=attention_index, offset=offset)
                attention_caches.append(cache)
                attention_index += 1
            caches.append(cache)
        logits = _select_primary_output(model(input_ids, cache=caches))
        if cast_bf16_logits_to_fp16 and "bfloat16" in str(getattr(logits, "dtype", "")).lower():
            logits = logits.astype(mx.float16)
        outputs = (
            logits,
            mx.stack([cache.keys for cache in attention_caches]),
            mx.stack([cache.values for cache in attention_caches]),
        )
        if array_caches:
            outputs += (mx.stack([cache[0] for cache in array_caches]),)
            if layout.num_linear_layers:
                outputs += (mx.stack([cache[1] for cache in array_caches]),)
        return outputs

    return capture


def _offset_from_position_ids(input_ids: Any, position_ids: Any) -> Any:
    import mlx.core as mx  # noqa: PLC0415

    query_indices = mx.arange(input_ids.shape[1], dtype=mx.int32)
    query_len = mx.max(query_indices) + mx.array(1, dtype=mx.int32)
    last_position = mx.max(position_ids)
    return last_position - query_len + mx.array(1, dtype=mx.int32)


def _stateful_inputs(
    lm_inputs: MLXLMConversionInputs,
    *,
    state_specs: list[StateSpec],
    input_name: str,
    position_ids_name: str,
    position_length: int,
    cache_dtype: str,
    state_context_length: int | None = None,
    context_state_names: tuple[str, str] = ("keyCache", "valueCache"),
) -> dict[str, np.ndarray]:
    inputs = lm_inputs.as_dict(input_name=input_name)
    inputs[position_ids_name] = np.arange(int(position_length), dtype=np.int32)[None, :]
    for spec in state_specs:
        shape = list(spec.resolved_shape(state_context_length))
        if state_context_length is not None and spec.capacity_axis is None and spec.name in context_state_names:
            shape[3] = int(state_context_length)
        inputs[spec.name] = np.zeros(tuple(shape), dtype=_cache_np_dtype(spec.dtype))
    return inputs


def _add_state_writes(
    graph: Graph,
    expected_outputs: Mapping[str, np.ndarray],
    *,
    state_specs: list[StateSpec],
    non_state_output_count: int,
) -> tuple[Graph, dict[str, np.ndarray]]:
    if len(graph.outputs) - non_state_output_count != len(state_specs):
        raise ValueError(
            "stateful capture returned "
            f"{len(graph.outputs) - non_state_output_count} state outputs for {len(state_specs)} state specs."
        )
    return CaptureSignature(states=tuple(
        StateBinding(spec, index + non_state_output_count) for index, spec in enumerate(state_specs)
    )).bind(graph, expected_outputs)


def _reorder_graph_inputs(graph: Graph, preferred_order: list[str]) -> Graph:
    present = {spec.name for spec in graph.inputs}
    order = tuple(name for name in preferred_order if name in present)
    reordered, _ = CaptureSignature(input_order=order).bind(graph, dict.fromkeys(graph.outputs))
    return reordered


def _infer_cache_layout(model: Any) -> _CacheLayout:
    from .policy import layout
    return layout(model)


def _layers(model):
    layers = getattr(model, "layers", None)
    if layers is None and hasattr(model, "model"):
        layers = getattr(model.model, "layers", None)
    if layers is None:
        raise ValueError("Could not infer mlx-lm transformer layers for stateful cache conversion.")
    return layers


def attention_layout(model, *, linear_layers=(), short_conv_layers=(), conv_shape=(0, 0),
                     recurrent_shape=(0, 0, 0)):
    layers = _layers(model)
    num_layers = len(layers)
    if sum(linear_layers) + sum(short_conv_layers) == num_layers:
        raise ValueError("Stateful conversion requires at least one full-attention layer.")
    args = getattr(model, "args", None)
    n_kv_heads = getattr(args, "num_key_value_heads", None)
    head_dim = getattr(args, "head_dim", None)
    if n_kv_heads is None or head_dim is None:
        attn = next((layer.self_attn for layer in layers if hasattr(layer, "self_attn")), None)
        n_kv_heads = n_kv_heads or getattr(attn, "n_kv_heads", None)
        head_dim = head_dim or getattr(args, "hidden_size", None)
        n_heads = getattr(args, "num_attention_heads", None)
        if head_dim is not None and n_heads:
            head_dim = int(head_dim) // int(n_heads)
    if n_kv_heads is None or head_dim is None:
        raise ValueError("Could not infer KV-cache head layout from mlx-lm model args.")
    return _CacheLayout(
        num_layers=int(num_layers),
        num_key_value_heads=int(n_kv_heads),
        head_dim=int(head_dim),
        linear_layers=linear_layers,
        conv_shape=conv_shape,
        recurrent_shape=recurrent_shape,
        short_conv_layers=short_conv_layers,
    )


def _resolve_compute_precision(model: Any, compute_precision: str) -> str:
    normalized = _normalize_cache_dtype(compute_precision)
    if normalized != "auto":
        return normalized
    for value in _iter_model_values(model):
        dtype = _dtype_to_precision(getattr(value, "dtype", None))
        if dtype in {"bf16", "fp16", "fp32"}:
            return dtype
    return "fp32"


def _iter_model_values(model: Any):
    params_fn = getattr(model, "parameters", None)
    if callable(params_fn):
        yield from _flatten_tree(params_fn())


def _flatten_tree(value: Any):
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _flatten_tree(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_tree(item)
        return
    yield value


def _make_state_specs(
    layout: _CacheLayout,
    *,
    batch_size: int,
    max_context_length: int,
    cache_dtype: str,
    key_cache_name: str,
    value_cache_name: str,
) -> list[StateSpec]:
    shape = (
        layout.num_layers - layout.num_conv_layers,
        int(batch_size),
        layout.num_key_value_heads,
        int(max_context_length),
        layout.head_dim,
    )
    specs = [
        StateSpec(key_cache_name, shape, cache_dtype, capacity_axis=3),
        StateSpec(value_cache_name, shape, cache_dtype, capacity_axis=3),
    ]
    if layout.num_conv_layers:
        specs.append(StateSpec("convState", (layout.num_conv_layers, int(batch_size), *layout.conv_shape), cache_dtype))
    if layout.num_linear_layers:
        specs.append(StateSpec("recurrentState", (layout.num_linear_layers, int(batch_size), *layout.recurrent_shape), "fp32"))
    return specs


def _probe_sequence_length(base_length: int, *, max_context_length: int) -> int:
    if base_length < max_context_length:
        return base_length + 1
    if base_length > 1:
        return base_length - 1
    return base_length


def _resize_lm_inputs(
    inputs: MLXLMConversionInputs,
    *,
    sequence_length: int,
) -> MLXLMConversionInputs:
    base = np.asarray(inputs.input_ids, dtype=np.int32)
    if base.ndim != 2:
        raise ValueError(f"LM stateful probe expects rank-2 input_ids, got {base.shape}.")
    target = int(sequence_length)
    if target <= 0:
        raise ValueError(f"sequence_length must be positive, got {target}.")
    if base.shape[1] == target:
        resized = base.copy()
    elif base.shape[1] > target:
        resized = base[:, :target]
    else:
        extension = base[:, -1:] if base.shape[1] else np.zeros((base.shape[0], 1), dtype=np.int32)
        resized = np.concatenate(
            [base, np.repeat(extension, target - base.shape[1], axis=1)],
            axis=1,
        )
    return MLXLMConversionInputs(
        input_ids=resized,
        prompt=inputs.prompt,
        token_count=min(inputs.token_count, target),
        padded_token_count=max(0, target - inputs.token_count),
        synthetic=inputs.synthetic,
    )


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an mlx-lm model into a coreai-models-style stateful CoreAI asset."
    )
    parser.add_argument("model_id")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output coreai-models-style bundle directory. A .aimodel suffix is treated as the nested asset name.",
    )
    parser.add_argument("--max-context-length", type=int, default=256)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--input-name", default="input_ids")
    parser.add_argument("--position-ids-name", default="position_ids")
    parser.add_argument("--key-cache-name", default="keyCache")
    parser.add_argument("--value-cache-name", default="valueCache")
    parser.add_argument("--compute-precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--cache-dtype", default=None, choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--entrypoint", default="main")
    parser.add_argument("--dynamic-sequence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cast-bf16-logits-to-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--externalize-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--external-weight-threshold", type=int, default=10)
    parser.add_argument("--allow-unknown-sources", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture-is-training", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--gated-delta-implementation", choices=["native", "decomposed"], default="native",
                        help="Experimental gated-delta lowering. Neither path is validated for full Qwen3.5 on build 26A428.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    converted = convert_mlx_lm_stateful(
        args.model_id,
        args.output,
        max_context_length=args.max_context_length,
        revision=args.revision,
        input_name=args.input_name,
        position_ids_name=args.position_ids_name,
        key_cache_name=args.key_cache_name,
        value_cache_name=args.value_cache_name,
        compute_precision=args.compute_precision,
        cache_dtype=args.cache_dtype,
        entrypoint_name=args.entrypoint,
        dynamic_sequence=bool(args.dynamic_sequence),
        dynamic_state=bool(args.dynamic_state),
        cast_bf16_logits_to_fp16=bool(args.cast_bf16_logits_to_fp16),
        config=ConversionConfig(
            allow_unknown_sources=bool(args.allow_unknown_sources),
            capture_is_training=bool(args.capture_is_training),
            externalize_weights=bool(args.externalize_weights),
            external_weight_threshold=int(args.external_weight_threshold),
            optimize=not bool(args.no_optimize),
            gated_delta_implementation=args.gated_delta_implementation,
        ),
    )
    print(f"Wrote bundle {converted.bundle_path}")
    print(f"Asset: {converted.asset_path}")
    print(f"Entrypoints: {', '.join(converted.lowered.entrypoint_names)}")
    print(f"States: {len(converted.state_specs)}")
    print(f"Compute precision: {converted.metadata['mlx_lm_stateful']['compute_precision']}")
    print(f"Cache dtype: {converted.metadata['mlx_lm_stateful']['cache_dtype']}")
    print(f"Max context: {converted.max_context_length}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
