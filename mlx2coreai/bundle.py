"""CoreAI model bundle paths, metadata, and tokenizer packaging."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import shutil


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


def resolve_asset_path(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix == ".aimodel":
        return path
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        asset_name = metadata.get("assets", {}).get("main")
        if not isinstance(asset_name, str):
            raise ValueError(f"{metadata_path} does not contain assets.main.")
        return path / asset_name
    candidates = sorted(path.glob("*.aimodel")) if path.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Could not resolve .aimodel asset from {path}.")


def resolve_bundle_path(path: Path, *, asset_path: Path) -> Path | None:
    if path.is_dir() and (path / "tokenizer").is_dir():
        return path
    if asset_path.parent.is_dir() and (asset_path.parent / "tokenizer").is_dir():
        return asset_path.parent
    return None
