from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import ml_dtypes
import numpy as np

from ._convert_mlx_lm import MLXLMConversionInputs, build_mlx_lm_inputs, load_mlx_lm_model
from .conversion import (
    CapturedMLXGraph,
    ConversionConfig,
    PreparedMLXGraph,
    find_extra_input_names,
    prepare_mlx_conversion,
)
from .dynamic_shapes import DynamicAxes
from .ir import Graph, Node, StateSpec
from .lower_to_coreai import (
    CoreAILoweringConfig,
    LoweredCoreAIProgram,
    build_coreai_program,
    save_coreai_program,
)
from .op_registry import ensure_supported, unsupported_op_details
from .passes import infer_graph_specs, normalize_graph, summarize_inference


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


@dataclass(slots=True)
class _LayeredKVCacheState:
    keys: Any
    values: Any


class _ExportableLayeredKVCache:
    def __init__(
        self,
        state: _LayeredKVCacheState,
        *,
        layer_idx: int,
        offset: Any,
    ):
        self.state = state
        self.layer_idx = int(layer_idx)
        self.keys = state.keys[self.layer_idx]
        self.values = state.values[self.layer_idx]
        self.offset = offset

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        import mlx.core as mx  # noqa: PLC0415

        offset = mx.reshape(self.offset, (1,))
        layer = mx.array([self.layer_idx], dtype=mx.int32)
        start = mx.concatenate(
            [
                layer,
                mx.array([0, 0], dtype=mx.int32),
                offset,
                mx.array([0], dtype=mx.int32),
            ]
        )
        expanded_keys = mx.expand_dims(keys, 0)
        expanded_values = mx.expand_dims(values, 0)
        self.state.keys = mx.slice_update(self.state.keys, expanded_keys, start, [0, 1, 2, 3, 4])
        self.state.values = mx.slice_update(self.state.values, expanded_values, start, [0, 1, 2, 3, 4])
        self.keys = self.state.keys[self.layer_idx]
        self.values = self.state.values[self.layer_idx]
        return self.keys, self.values

    def make_mask(
        self,
        N: int,
        window_size: int | None = None,
        return_array: bool = False,
    ) -> Any:
        import mlx.core as mx  # noqa: PLC0415

        if window_size is not None:
            raise NotImplementedError(
                "stateful KV-cache export does not support sliding-window masks yet."
            )
        query_positions = mx.arange(N) + self.offset
        key_positions = mx.arange(self.state.keys.shape[3])
        return (query_positions[:, None] >= key_positions[None, :])[None, None, :, :]

    def size(self) -> int:
        return self.state.keys.shape[3]

    def empty(self) -> bool:
        return False


def convert_mlx_lm_stateful(
    model_id: str,
    output_path: str | Path,
    *,
    prompt: str | None = None,
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
    ``position_ids``, and two mutable KV-cache state tensors named
    ``keyCache`` and ``valueCache`` by default.
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
    if hasattr(model, "eval"):
        model.eval()

    layout = _infer_cache_layout(model)
    resolved_compute_precision = _resolve_compute_precision(model, compute_precision)
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
        prompt=prompt,
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

    lowering_config = CoreAILoweringConfig(
        entrypoint_name=entrypoint_name,
        optimize=base_config.optimize,
        state_specs=state_specs,
        constant_inputs=base_config.constant_inputs,
        externalize_weights=base_config.externalize_weights,
        external_weight_threshold=base_config.external_weight_threshold,
    )
    lowered = build_coreai_program(
        main.normalized_graph,
        config=lowering_config,
    )

    bundle_path, asset_path, bundle_name = _resolve_bundle_paths(output_path)
    bundle_path.mkdir(parents=True, exist_ok=True)
    asset = save_coreai_program(lowered.program, asset_path)
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
            "num_key_value_heads": layout.num_key_value_heads,
            "head_dim": layout.head_dim,
            "cast_bf16_logits_to_fp16": bool(cast_bf16_logits_to_fp16),
        },
        "coreai_models_bundle": bundle_metadata,
        "entrypoint_names": list(lowered.entrypoint_names),
        "optimized": bool(lowered.optimized),
        "optimization_skip_reason": lowered.optimization_skip_reason,
        "state_specs": [spec.to_dict() for spec in state_specs],
        "weight_manifest": [entry.to_dict() for entry in lowered.weight_manifest],
        "inference_summary": main.inference_summary,
    }

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


def _resolve_bundle_paths(output_path: str | Path) -> tuple[Path, Path, str]:
    path = Path(output_path)
    if path.suffix == ".aimodel":
        name = path.stem
        bundle_path = path.with_suffix("")
        asset_path = bundle_path / path.name
    else:
        name = path.name
        bundle_path = path
        asset_path = bundle_path / f"{name}.aimodel"
    return bundle_path, asset_path, name


def _write_coreai_models_bundle(
    bundle_path: Path,
    *,
    tokenizer: Any | None,
    model: Any,
    model_id: str,
    revision: str | None,
    name: str,
    asset_path: Path,
    max_context_length: int,
    entrypoint_name: str,
) -> dict[str, Any]:
    tokenizer_dir = bundle_path / "tokenizer"
    _write_tokenizer(tokenizer_dir, tokenizer=tokenizer, model_id=model_id, revision=revision)
    metadata = _build_bundle_metadata(
        tokenizer=tokenizer,
        model=model,
        model_id=model_id,
        name=name,
        asset_name=asset_path.name,
        max_context_length=max_context_length,
        entrypoint_name=entrypoint_name,
    )
    (bundle_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def _write_tokenizer(
    dest: Path,
    *,
    tokenizer: Any | None,
    model_id: str,
    revision: str | None,
) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    save_pretrained = getattr(tokenizer, "save_pretrained", None)
    if callable(save_pretrained):
        save_pretrained(str(dest))
        return
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except Exception as exc:
        raise RuntimeError(
            "Could not save tokenizer: mlx-lm did not return a tokenizer with "
            "save_pretrained(), and transformers is not importable."
        ) from exc
    kwargs: dict[str, Any] = {}
    if revision is not None:
        kwargs["revision"] = revision
    AutoTokenizer.from_pretrained(model_id, **kwargs).save_pretrained(str(dest))


def _build_bundle_metadata(
    *,
    tokenizer: Any | None,
    model: Any,
    model_id: str,
    name: str,
    asset_name: str,
    max_context_length: int,
    entrypoint_name: str,
) -> dict[str, Any]:
    return {
        "metadata_version": "0.2",
        "kind": "llm",
        "name": name,
        "assets": {"main": asset_name},
        "language": {
            "tokenizer": model_id,
            "vocab_size": _vocab_size(tokenizer, model),
            "max_context_length": int(max_context_length),
            "embedded_tokenizer": True,
            "function_map": {"main": [entrypoint_name]},
        },
        "source": {
            "model_definition": "mlx",
            "hf_model_id": model_id,
        },
        "compression": None,
        "compilation": {
            "date": datetime.now().astimezone().isoformat(),
            "targets": [],
        },
    }


def _vocab_size(tokenizer: Any | None, model: Any) -> int | None:
    for obj in (getattr(model, "args", None), getattr(model, "config", None), model, tokenizer):
        if obj is None:
            continue
        value = getattr(obj, "vocab_size", None)
        if value is not None:
            return int(value)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        return len(get_vocab())
    if tokenizer is not None:
        try:
            return len(tokenizer)
        except TypeError:
            pass
    return None


def _prepare_stateful_entry(
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
) -> PreparedMLXGraph:
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
            dynamic_axes_dict[key_cache_name] = [3]
            dynamic_axes_dict[value_cache_name] = [3]
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
            )
        else:
            dynamic_axes = None

    stateful_config = replace(
        config,
        capture_shapeless=bool(dynamic_axes),
        dynamic_axes=dynamic_axes,
        dynamic_probe_inputs=dynamic_probe_inputs,
        state_specs=state_specs,
    )
    prepared = prepare_mlx_conversion(
        model,
        inputs,
        config=stateful_config,
        capture_function=capture_function,
    )
    graph, expected_outputs = _add_state_writes(
        prepared.normalized_graph,
        prepared.expected_outputs,
        state_specs=state_specs,
        non_state_output_count=1,
    )
    graph = _reorder_graph_inputs(
        graph,
        [input_name, position_ids_name, key_cache_name, value_cache_name],
    )
    graph = normalize_graph(graph)
    ensure_supported(graph)
    return PreparedMLXGraph(
        captured=CapturedMLXGraph(
            graph=graph,
            normalized_inputs=prepared.normalized_inputs,
            expected_outputs=expected_outputs,
        ),
        normalized_graph=graph,
        expected_outputs=expected_outputs,
        inference_summary=summarize_inference(infer_graph_specs(graph)),
        unsupported_details=unsupported_op_details(graph),
        extra_input_names=find_extra_input_names(graph, prepared.normalized_inputs),
    )


def _stateful_main_capture_function(
    model: Any,
    *,
    layout: _CacheLayout,
    input_name: str,
    position_ids_name: str,
    key_cache_name: str,
    value_cache_name: str,
    cast_bf16_logits_to_fp16: bool,
) -> Callable[..., tuple[Any, ...]]:
    def capture(**kwargs: Any) -> tuple[Any, ...]:
        import mlx.core as mx  # noqa: PLC0415

        input_ids = kwargs[input_name]
        position_ids = kwargs[position_ids_name]
        offset = _offset_from_position_ids(input_ids, position_ids)
        state = _LayeredKVCacheState(
            keys=kwargs[key_cache_name],
            values=kwargs[value_cache_name],
        )
        caches = [
            _ExportableLayeredKVCache(
                state,
                layer_idx=layer_idx,
                offset=offset,
            )
            for layer_idx in range(layout.num_layers)
        ]
        logits = _select_primary_output(model(input_ids, cache=caches))
        if cast_bf16_logits_to_fp16 and "bfloat16" in str(getattr(logits, "dtype", "")).lower():
            logits = logits.astype(mx.float16)
        return logits, state.keys, state.values

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
) -> dict[str, np.ndarray]:
    inputs = lm_inputs.as_dict(input_name=input_name)
    inputs[position_ids_name] = np.arange(int(position_length), dtype=np.int32)[None, :]
    dtype = _cache_np_dtype(cache_dtype)
    for spec in state_specs:
        shape = list(spec.shape)
        if state_context_length is not None:
            shape[3] = int(state_context_length)
        inputs[spec.name] = np.zeros(tuple(shape), dtype=dtype)
    return inputs


def _add_state_writes(
    graph: Graph,
    expected_outputs: Mapping[str, np.ndarray],
    *,
    state_specs: list[StateSpec],
    non_state_output_count: int,
) -> tuple[Graph, dict[str, np.ndarray]]:
    original_outputs = list(graph.outputs)
    non_state_outputs = original_outputs[:non_state_output_count]
    state_value_outputs = original_outputs[non_state_output_count:]
    if len(state_value_outputs) != len(state_specs):
        raise ValueError(
            "stateful capture returned "
            f"{len(state_value_outputs)} state outputs for {len(state_specs)} state specs."
        )
    nodes = list(graph.nodes)
    rewritten_outputs = list(non_state_outputs)
    rewritten_expected = {name: np.asarray(expected_outputs[name]) for name in non_state_outputs}
    for spec, value_name in zip(state_specs, state_value_outputs, strict=True):
        output_name = f"{spec.name}__updated"
        nodes.append(
            Node(
                "write_state",
                (spec.name, value_name),
                output_name,
                attrs={"coreai_output_name": spec.name},
                source="mlx2coreai:stateful_kv_cache",
            )
        )
        rewritten_outputs.append(output_name)
        rewritten_expected[output_name] = np.asarray(expected_outputs[value_name])
    rewritten = Graph(inputs=list(graph.inputs), nodes=nodes, outputs=rewritten_outputs)
    rewritten.validate()
    return rewritten, rewritten_expected


def _reorder_graph_inputs(graph: Graph, preferred_order: list[str]) -> Graph:
    by_name = {spec.name: spec for spec in graph.inputs}
    ordered = [by_name[name] for name in preferred_order if name in by_name]
    ordered_names = {spec.name for spec in ordered}
    ordered.extend(spec for spec in graph.inputs if spec.name not in ordered_names)
    out = Graph(inputs=ordered, nodes=list(graph.nodes), outputs=list(graph.outputs))
    out.validate()
    return out


def _infer_cache_layout(model: Any) -> _CacheLayout:
    layers = getattr(model, "layers", None)
    if layers is None and hasattr(model, "model"):
        layers = getattr(model.model, "layers", None)
    if layers is None:
        raise ValueError("Could not infer mlx-lm transformer layers for stateful cache conversion.")
    num_layers = len(layers)
    args = getattr(model, "args", None)
    n_kv_heads = getattr(args, "num_key_value_heads", None)
    head_dim = getattr(args, "head_dim", None)
    if n_kv_heads is None or head_dim is None:
        attn = getattr(layers[0], "self_attn", None) if num_layers else None
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


def _apply_model_compute_precision(model: Any, compute_precision: str) -> None:
    set_dtype = getattr(model, "set_dtype", None)
    if not callable(set_dtype):
        return
    try:
        import mlx.core as mx  # noqa: PLC0415
    except ImportError:
        return
    dtype = {
        "bf16": mx.bfloat16,
        "fp16": mx.float16,
        "fp32": mx.float32,
    }[compute_precision]
    set_dtype(dtype)


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


def _dtype_to_precision(dtype: Any) -> str | None:
    if dtype is None:
        return None
    text = str(dtype).strip().lower()
    if "bfloat16" in text or "bf16" in text:
        return "bf16"
    if "float16" in text or "fp16" in text:
        return "fp16"
    if "float32" in text or "fp32" in text:
        return "fp32"
    return None


def _normalize_cache_dtype(dtype: str) -> str:
    normalized = dtype.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"fp32", "float32"}:
        return "fp32"
    if normalized in {"fp16", "float16"}:
        return "fp16"
    if normalized in {"bf16", "bfloat16"}:
        return "bf16"
    raise ValueError(f"Unsupported compute/cache dtype: {dtype!r}.")


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
        layout.num_layers,
        int(batch_size),
        layout.num_key_value_heads,
        int(max_context_length),
        layout.head_dim,
    )
    return [
        StateSpec(key_cache_name, shape, cache_dtype),
        StateSpec(value_cache_name, shape, cache_dtype),
    ]


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


def _cache_np_dtype(dtype: str) -> Any:
    normalized = dtype.strip().lower()
    if normalized in {"fp32", "float32"}:
        return np.float32
    if normalized in {"fp16", "float16"}:
        return np.float16
    if normalized in {"bf16", "bfloat16"}:
        return ml_dtypes.bfloat16
    raise ValueError(f"Unsupported cache dtype: {dtype!r}.")


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
    parser.add_argument("--prompt", default=None)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    converted = convert_mlx_lm_stateful(
        args.model_id,
        args.output,
        prompt=args.prompt,
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
