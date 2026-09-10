"""Optional numerical checks shared by recipe validation, not normal execution."""
import numpy as np


def compare(report, name, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f"{name}: invalid shape or nonfinite values: {actual.shape}, {expected.shape}")
    error = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(error))) if error.size else 0.0
    relative = float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12))
    entry = report.setdefault(name, {"calls": 0, "max_abs_error": 0.0, "max_relative_l2": 0.0})
    entry["calls"] += 1
    entry["max_abs_error"] = max(entry["max_abs_error"], maximum)
    entry["max_relative_l2"] = max(entry["max_relative_l2"], relative)
    if maximum > 1e-3 or relative > 1e-3:
        raise AssertionError(f"{name}: max_abs={maximum}, relative_l2={relative}")


def compare_cache(report, prefix, state, caches, offset):
    for name, attr in (("keyCache", "keys"), ("valueCache", "values")):
        expected = np.stack([np.asarray(getattr(cache, attr)[:, :, :offset]) for cache in caches])
        compare(report, prefix + "." + name, state[name][:, :, :, :offset], expected)
        np.testing.assert_array_equal(state[name][:, :, :, offset:], 0)
