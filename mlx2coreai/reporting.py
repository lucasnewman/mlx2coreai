from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = "mlx2coreai.run_report.v1"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unavailable"


def collect_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "coreai-core": _package_version("coreai-core"),
        "numpy": _package_version("numpy"),
        "mlx": _package_version("mlx"),
    }


def build_run_context(
    *,
    run_kind: str,
    runtime_target: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_kind": run_kind,
        "runtime_target": runtime_target,
        "seed": int(seed),
        "versions": collect_versions(),
    }


def init_stage_timings(stage_names: Iterable[str]) -> dict[str, float | None]:
    return {str(name): None for name in stage_names}


@contextmanager
def timed_stage(stage_timings: dict[str, float | None], stage_name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        stage_timings[stage_name] = max(0.0, time.perf_counter() - start)


def summarize_stage_timings(
    stage_timings: dict[str, float | None],
    *,
    ndigits: int = 6,
) -> tuple[dict[str, float | None], float]:
    summarized: dict[str, float | None] = {}
    total = 0.0
    for stage_name, seconds in stage_timings.items():
        if seconds is None:
            summarized[stage_name] = None
            continue
        value = round(float(seconds), ndigits)
        summarized[stage_name] = value
        total += value
    return summarized, round(total, ndigits)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
