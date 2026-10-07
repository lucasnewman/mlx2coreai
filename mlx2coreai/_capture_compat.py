"""Temporary decompositions for primitives the MLX exporter cannot serialize."""
from contextlib import contextmanager
from threading import RLock


_export_lock = RLock()


@contextmanager
def export_compatibility(mx):
    from mlx.nn.layers import pooling
    # Serialize our process-global capture patches, and always restore the
    # native operations before evaluating reference outputs.
    with _export_lock:
        contiguous = mx.contiguous
        invert = mx.bitwise_invert
        array_invert = mx.array.__invert__
        sliding_windows = pooling._sliding_windows

        def serializable_invert(a, stream=None):
            a = mx.array(a) if not isinstance(a, mx.array) else a
            mask = mx.array(True if a.dtype == mx.bool_ else -1, dtype=a.dtype)
            return mx.bitwise_xor(a, mask, stream=stream)

        def serializable_windows(x, window_shape, window_strides):
            if x.ndim < 3 or len(window_shape) != x.ndim - 2 or len(window_strides) != x.ndim - 2:
                return sliding_windows(x, window_shape, window_strides)
            # Preserve relative slice stops instead of exporting products of
            # concrete spatial extents as AsStrided sizes and memory strides.
            for axis, (window, stride) in enumerate(zip(window_shape, window_strides, strict=True), 1):
                windows = []
                for offset in range(window):
                    index = [slice(None)] * x.ndim
                    tail = window - 1 - offset
                    index[axis] = slice(offset, -tail if tail else None, stride)
                    windows.append(x[tuple(index)])
                x = mx.stack(windows, axis=-2)
            return x

        try:
            mx.contiguous = lambda value, *args, **kwargs: value
            mx.bitwise_invert = serializable_invert
            mx.array.__invert__ = serializable_invert
            pooling._sliding_windows = serializable_windows
            yield
        finally:
            mx.contiguous = contiguous
            mx.bitwise_invert = invert
            mx.array.__invert__ = array_invert
            pooling._sliding_windows = sliding_windows
