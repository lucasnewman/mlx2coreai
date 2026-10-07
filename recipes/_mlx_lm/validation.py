"""Native MLX-LM reference observer; not imported by normal generation."""
import numpy as np

from recipes._validation import require_full_precision_mlx
from .build import load_source


class Reference:
    def __init__(self, bundle, adapter, *, source=None, model=None, max_abs_error=0.01, max_relative_l2=0.01):
        require_full_precision_mlx("MLX-LM")
        if any(not np.isfinite(value) or value <= 0 for value in (max_abs_error, max_relative_l2)):
            raise ValueError("Validation tolerances must be positive and finite.")
        self.metadata = bundle.metadata
        self.model = model
        if model is None:
            self.model, _, _ = load_source(source or self.metadata["source"], adapter,
                revision=self.metadata.get("revision"), precision=self.metadata["source_precision_policy"])
            import mlx.core as mx
            # Do not retain freed loading/casting buffers alongside CoreAI's
            # second copy of a full checkpoint.
            mx.clear_cache()
        self.max_abs_error = max_abs_error
        self.max_relative_l2 = max_relative_l2
        self.checks, self.calls = {}, []
        self.position = 0

    def compare(self, name, actual, expected):
        actual, expected = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError(f"{name}: shape mismatch or nonfinite values")
        # Corrupted FP32 states can still be finite; accumulate their norms in
        # FP64 so the diagnostic remains meaningful instead of overflowing.
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        maximum = float(np.max(np.abs(delta))) if delta.size else 0.0
        relative = float(np.linalg.norm(delta) / max(np.linalg.norm(expected.astype(np.float64)), 1e-12))
        row = self.checks.setdefault(name, {"calls": 0, "max_abs_error": 0.0, "max_relative_l2": 0.0})
        row["calls"] += 1
        row["max_abs_error"] = max(row["max_abs_error"], maximum)
        row["max_relative_l2"] = max(row["max_relative_l2"], relative)
        if maximum > self.max_abs_error or relative > self.max_relative_l2:
            raise AssertionError(f"{name} at position {self.position}: max_abs={maximum}, relative_l2={relative}")

    def __call__(self, session, component, inputs, outputs):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        ids = np.asarray(inputs["input_ids"])
        position = int(inputs["position_ids"][0, 0])
        if position == 0:
            self.cache = make_prompt_cache(self.model)
            self.position = 0
        if position != self.position:
            raise AssertionError(f"Non-advancing reference position: {position}, expected {self.position}")
        expected = np.asarray(self.model(mx.array(ids), cache=self.cache).astype(mx.float32))
        actual = outputs["logits"].numpy().astype(np.float32)
        self.compare("logits", actual, expected)
        self.calls.append({"position": position, "query_length": ids.shape[1],
            "coreai_token": int(actual[0, -1].argmax()), "mlx_token": int(expected[0, -1].argmax())})
        self.position += ids.shape[1]
        state = session.snapshot_state(component)
        for name, attr in (("keyCache", "keys"), ("valueCache", "values")):
            reference = np.stack([np.asarray(getattr(self.cache[i], attr)[:, :, :self.position].astype(mx.float32))
                                  for i in self.metadata["attention_layers"]])
            self.compare(name, state[name][:, :, :, :self.position], reference)
            np.testing.assert_array_equal(state[name][:, :, :, self.position:], 0)
        if self.metadata["conv_layers"]:
            self.compare("convState", state["convState"], np.stack([
                np.asarray(self.cache[i][0].astype(mx.float32)) for i in self.metadata["conv_layers"]]))
        if self.metadata["recurrent"]:
            self.compare("recurrentState", state["recurrentState"], np.stack([
                np.asarray(self.cache[i][1].astype(mx.float32)) for i in self.metadata["conv_layers"]]))
