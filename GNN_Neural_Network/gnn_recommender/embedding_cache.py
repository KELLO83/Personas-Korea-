from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .text_embedding import KURE_MODEL_NAME, _load_kure_model


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

    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory: dict[str, np.ndarray] = {}

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
            embeddings = model.encode(missing, convert_to_numpy=True, show_progress_bar=False)
            for name, emb in zip(missing, embeddings, strict=False):
                self.set(name, emb)
        return {name: self.get(name) for name in hobby_names if self.get(name) is not None}
