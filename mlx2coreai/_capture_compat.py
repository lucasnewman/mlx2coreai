"""Temporary decompositions for primitives the MLX exporter cannot serialize."""
from contextlib import contextmanager
from threading import RLock


_export_lock = RLock()


@contextmanager
def export_compatibility(mx):
    # Serialize our process-global capture patches, and always restore the
    # native operations before evaluating reference outputs.
    with _export_lock:
        contiguous = mx.contiguous
        invert = mx.bitwise_invert
        array_invert = mx.array.__invert__

        def serializable_invert(a, stream=None):
            a = mx.array(a) if not isinstance(a, mx.array) else a
            mask = mx.array(True if a.dtype == mx.bool_ else -1, dtype=a.dtype)
            return mx.bitwise_xor(a, mask, stream=stream)

        try:
            mx.contiguous = lambda value, *args, **kwargs: value
            mx.bitwise_invert = serializable_invert
            mx.array.__invert__ = serializable_invert
            yield
        finally:
            mx.contiguous = contiguous
            mx.bitwise_invert = invert
            mx.array.__invert__ = array_invert
