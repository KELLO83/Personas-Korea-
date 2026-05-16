from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "persona_similarity"


def default_cpu_workers() -> int:
    return min(max((os.cpu_count() or 1) - 4, 1), 18)


def resolve_worker_count(requested: int | None = None) -> int:
    if requested is None or requested <= 0:
        return default_cpu_workers()
    return max(1, min(int(requested), os.cpu_count() or 1))


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def ensure_parent(path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = ensure_parent(path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str | None:
    resolved = resolve_path(path)
    if not resolved.exists():
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    resolved = resolve_path(path)
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def cache_metadata_matches(path: str | Path, expected: dict[str, Any]) -> tuple[bool, str]:
    metadata = load_json_if_exists(path)
    if metadata is None:
        return False, "metadata_missing"
    for key, value in expected.items():
        if metadata.get(key) != value:
            return False, f"metadata_mismatch:{key}"
    return True, "metadata_match"


def should_use_cache(
    artifact_path: str | Path,
    metadata_path: str | Path,
    expected_metadata: dict[str, Any],
    force: bool = False,
) -> tuple[bool, str]:
    if force:
        return False, "force_rebuild"
    if not resolve_path(artifact_path).exists():
        return False, "artifact_missing"
    return cache_metadata_matches(metadata_path, expected_metadata)


def mark_cache_hit(metadata_path: str | Path, expected_metadata: dict[str, Any], artifact_path: str | Path) -> None:
    metadata = load_json_if_exists(metadata_path) or {}
    resolved_artifact = resolve_path(artifact_path)
    try:
        artifact_label = str(resolved_artifact.relative_to(PROJECT_ROOT))
    except ValueError:
        artifact_label = str(resolved_artifact)
    metadata.update(
        {
            **expected_metadata,
            "cache_hit": True,
            "artifact_path": artifact_label,
            "runtime_seconds": 0.0,
        }
    )
    write_json(metadata_path, metadata)
