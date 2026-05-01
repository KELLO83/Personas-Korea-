from __future__ import annotations

import hashlib
import time
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.linalg import norm

from .text_embedding import KURE_MODEL_NAME, _load_kure_model


HOBBY_MATRIX_CACHE_SUBDIR = "hobby_matrix"


class PersonEmbeddingCache:
    """Cache persona text embeddings to avoid repeated KURE encoding."""

    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory: dict[str, np.ndarray] = {}

    def _cache_path(self, text: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"person_emb_{key}.npy"

    def get(self, text: str) -> np.ndarray | None:
        if text in self._memory:
            return self._memory[text]
        cache_path = self._cache_path(text)
        if cache_path and cache_path.exists():
            arr = np.load(cache_path)
            self._memory[text] = arr
            return arr
        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        self._memory[text] = embedding
        cache_path = self._cache_path(text)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embedding)

    def encode(self, text: str) -> np.ndarray:
        cached = self.get(text)
        if cached is not None:
            return cached
        model = _load_kure_model()
        emb = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        self.set(text, emb)
        return emb


class HobbyEmbeddingCache:
    """Cache hobby name embeddings to avoid repeated KURE encoding."""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        model_name: str = KURE_MODEL_NAME,
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.device = device if device else self._default_device()
        self._memory: dict[str, np.ndarray] = {}

    def _default_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            return "cpu"
        return "cpu"

    def _hobby_cache_key(self, hobby_names: list[str]) -> str:
        payload = {
            "model_name": self.model_name,
            "hobby_names": sorted(hobby_names),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _matrix_cache_paths(self, hobby_names: list[str]) -> tuple[Path, Path]:
        key = self._hobby_cache_key(hobby_names)
        if self.cache_dir is None:
            raise ValueError("cache_dir required for matrix cache")
        base = self.cache_dir / HOBBY_MATRIX_CACHE_SUBDIR
        return base / f"hobby_matrix_{key}.npy", base / f"hobby_matrix_{key}.json"

    def _cache_path(self, hobby_name: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(hobby_name.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"hobby_emb_{key}.npy"

    def get(self, hobby_name: str) -> np.ndarray | None:
        if hobby_name in self._memory:
            return self._memory[hobby_name]
        cache_path = self._cache_path(hobby_name)
        if cache_path and cache_path.exists():
            arr = np.load(cache_path)
            self._memory[hobby_name] = arr
            return arr
        return None

    def set(self, hobby_name: str, embedding: np.ndarray) -> None:
        self._memory[hobby_name] = embedding
        cache_path = self._cache_path(hobby_name)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embedding)

    def encode(self, hobby_name: str) -> np.ndarray:
        cached = self.get(hobby_name)
        if cached is not None:
            return cached
        model = _load_kure_model()
        emb = model.encode(hobby_name, convert_to_numpy=True, show_progress_bar=False)
        self.set(hobby_name, emb)
        return emb

    def encode_batch(self, hobby_names: list[str]) -> dict[str, np.ndarray]:
        missing = [name for name in hobby_names if self.get(name) is None]
        if missing:
            model = _load_kure_model()
            embeddings = model.encode(
                missing,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
                device=self.device,
            )
            for name, emb in zip(missing, embeddings, strict=False):
                self.set(name, emb)
        return {name: self.get(name) for name in hobby_names if self.get(name) is not None}

    def load_matrix(self, hobby_names: list[str]) -> tuple[np.ndarray | None, dict[str, Any] | None]:
        if self.cache_dir is None:
            return None, None
        cache_path, meta_path = self._matrix_cache_paths(hobby_names)
        if not cache_path.exists() or not meta_path.exists():
            return None, None

        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None, None

        if not isinstance(metadata, dict):
            return None, None

        if metadata.get("model_name") != self.model_name:
            return None, None

        if metadata.get("hobby_names_hash") != self._hobby_names_hash(hobby_names):
            return None, None

        if metadata.get("num_hobbies", 0) != len(hobby_names):
            return None, None

        try:
            matrix = np.load(cache_path)
        except OSError:
            return None, None

        if matrix.ndim != 2:
            return None, None

        return matrix.astype(np.float32), metadata

    def save_matrix(self, hobby_names: list[str], matrix: np.ndarray, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.cache_dir is None:
            return {
                "cache_enabled": False,
                "model_name": self.model_name,
            }

        cache_path, meta_path = self._matrix_cache_paths(hobby_names)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        meta = self._build_matrix_metadata(hobby_names, matrix)
        if metadata:
            meta.update(metadata)
        np.save(cache_path, matrix.astype(np.float32))
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    def load_or_build_matrix(self, hobby_names: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
        matrix, metadata = self.load_matrix(hobby_names)
        if matrix is not None:
            return matrix, {
                "cache_enabled": True,
                "cache_key": self._hobby_cache_key(hobby_names),
                **metadata,
            }

        embeddings = self.encode_batch(hobby_names)
        ordered_vectors = []
        for name in hobby_names:
            vector = embeddings.get(name)
            if vector is None:
                if self._memory:
                    first_vec = next(iter(self._memory.values()))
                    vector = np.zeros_like(first_vec, dtype=np.float32)
                else:
                    vector = np.zeros(1, dtype=np.float32)
            else:
                vector = np.asarray(vector)
            ordered_vectors.append(vector)

        if ordered_vectors:
            matrix = np.vstack([vec.reshape(1, -1) for vec in ordered_vectors]).astype(np.float32)
            matrix = _l2_normalize_rows(matrix)
        else:
            matrix = np.empty((0, 0), dtype=np.float32)

        matrix_metadata = self.save_matrix(hobby_names, matrix)
        matrix_metadata["cache_enabled"] = True
        return matrix, matrix_metadata

    @staticmethod
    def _hobby_names_hash(hobby_names: list[str]) -> str:
        names = sorted(set(hobby_names))
        payload = {"hobby_names": names}
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _build_matrix_metadata(self, hobby_names: list[str], matrix: np.ndarray) -> dict[str, Any]:
        return {
            "cache_version": 1,
            "cache_enabled": True,
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "device": self.device,
            "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 and matrix.size else 0,
            "num_hobbies": int(len(hobby_names)),
            "hobby_names_hash": self._hobby_names_hash(hobby_names),
            "cache_key": self._hobby_cache_key(hobby_names),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return matrix / norms
