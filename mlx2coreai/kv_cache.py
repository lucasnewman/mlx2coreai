"""Functional bounded KV cache views shared by capture adapters."""
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LayeredKVCacheState:
    keys: Any
    values: Any


class LayeredKVCache:
    def __init__(
        self,
        state: LayeredKVCacheState,
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

        # The beta runtime can read stale data after packed slice updates;
        # updating a layer view with slice_update also crashes MPSGraph. Gather
        # the new tokens and select only the written interval, then pack once.
        positions = mx.arange(self.keys.shape[2], dtype=mx.int32) - self.offset
        last = mx.max(mx.arange(keys.shape[2], dtype=mx.int32))
        indices = mx.clip(positions, 0, last)
        mask = ((positions >= 0) & (positions <= last))[None, None, :, None]
        self.keys = mx.where(mask, mx.take(keys.astype(self.keys.dtype), indices, axis=2), self.keys)
        self.values = mx.where(mask, mx.take(values.astype(self.values.dtype), indices, axis=2), self.values)
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
