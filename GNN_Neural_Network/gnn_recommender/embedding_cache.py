from __future__ import annotations

import hashlib
import time
import json
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from .text_embedding import KURE_MODEL_NAME, _load_kure_model


HOBBY_MATRIX_CACHE_SUBDIR = "hobby_matrix"
ENCODE_CHUNK_BATCHES = 1
CACHE_VERSION = 2
DEFAULT_PREPROCESSING_VERSION = "raw_v1"


class PersonEmbeddingCache:
    """Cache persona text embeddings to avoid repeated KURE encoding."""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        model_name: str = KURE_MODEL_NAME,
        model_revision: str = "",
        preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.base_cache_dir = Path(cache_dir) if cache_dir else None
        self.model_name = model_name
        self.model_revision = model_revision
        self.preprocessing_version = preprocessing_version
        self.cache_dir = _model_cache_dir(
            self.base_cache_dir, self.model_name, self.model_revision, self.preprocessing_version,
        ) if self.base_cache_dir else None
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

    def _cache_path(self, text: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = _embedding_cache_key(text, self.model_name, self.model_revision, self.preprocessing_version)
        return self.cache_dir / f"person_emb_{key}.npy"

    def get(self, text: str) -> np.ndarray | None:
        if text in self._memory:
            return self._memory[text]
        cache_path = self._cache_path(text)
        if cache_path and cache_path.exists():
            arr = _safe_load_embedding(cache_path)
            if arr is None:
                return None
            if _metadata_matches(
                _metadata_path(cache_path),
                model_name=self.model_name,
                model_revision=self.model_revision,
                preprocessing_version=self.preprocessing_version,
                embedding_dim=_embedding_dim(arr),
            ):
                self._memory[text] = arr
                return arr
        legacy_path = self._legacy_cache_path(text, "person_emb")
        if legacy_path and legacy_path.exists():
            arr = _safe_load_embedding(legacy_path)
            if arr is None:
                return None
            self.set(text, arr)
            return arr
        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        self._memory[text] = embedding
        cache_path = self._cache_path(text)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embedding)
            _metadata_path(cache_path).write_text(
                json.dumps(
                    _embedding_metadata(
                        self.model_name,
                        self.model_revision,
                        self.preprocessing_version,
                        embedding,
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def _legacy_cache_path(self, text: str, prefix: str) -> Path | None:
        if not _allow_legacy_cache_lookup(self.model_name, self.model_revision, self.preprocessing_version):
            return None
        if self.base_cache_dir is None:
            return None
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return self.base_cache_dir / f"{prefix}_{key}.npy"

    def encode(self, text: str) -> np.ndarray:
        cached = self.get(text)
        if cached is not None:
            return cached
        model = _load_kure_model(
            self.device,
            model_name=self.model_name,
            model_revision=self.model_revision,
        )
        emb = model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        self.set(text, emb)
        return emb

    def encode_batch(
        self,
        texts: list[str],
        *,
        show_progress_bar: bool = False,
        progress_desc: str = "KURE persona embeddings",
    ) -> dict[str, np.ndarray]:
        unique_texts = list(dict.fromkeys(text for text in texts if text))
        missing = [text for text in unique_texts if self.get(text) is None]
        if missing:
            model = _load_kure_model(
                self.device,
                model_name=self.model_name,
                model_revision=self.model_revision,
            )
            chunks = list(_iter_encode_chunks(missing, self.batch_size))
            iterator = tqdm(
                chunks,
                desc=progress_desc,
                unit="batch",
                dynamic_ncols=False,
                leave=True,
                mininterval=1.0,
                maxinterval=10.0,
                file=sys.stderr,
                disable=not show_progress_bar,
            )
            for chunk in iterator:
                embeddings = model.encode(
                    chunk,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                )
                for text, emb in zip(chunk, embeddings, strict=False):
                    self.set(text, emb)
        result: dict[str, np.ndarray] = {}
        for text in unique_texts:
            embedding = self.get(text)
            if embedding is not None:
                result[text] = embedding
        return result


class HobbyEmbeddingCache:
    """Cache hobby name embeddings to avoid repeated KURE encoding."""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        model_name: str = KURE_MODEL_NAME,
        model_revision: str = "",
        preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.preprocessing_version = preprocessing_version
        self.base_cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_dir = _model_cache_dir(
            self.base_cache_dir, self.model_name, self.model_revision, self.preprocessing_version,
        ) if self.base_cache_dir else None
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
            "model_revision": self.model_revision,
            "preprocessing_version": self.preprocessing_version,
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
        key = _embedding_cache_key(hobby_name, self.model_name, self.model_revision, self.preprocessing_version)
        return self.cache_dir / f"hobby_emb_{key}.npy"

    def get(self, hobby_name: str) -> np.ndarray | None:
        if hobby_name in self._memory:
            return self._memory[hobby_name]
        cache_path = self._cache_path(hobby_name)
        if cache_path and cache_path.exists():
            arr = _safe_load_embedding(cache_path)
            if arr is None:
                return None
            if _metadata_matches(
                _metadata_path(cache_path),
                model_name=self.model_name,
                model_revision=self.model_revision,
                preprocessing_version=self.preprocessing_version,
                embedding_dim=_embedding_dim(arr),
            ):
                self._memory[hobby_name] = arr
                return arr
        legacy_path = self._legacy_cache_path(hobby_name, "hobby_emb")
        if legacy_path and legacy_path.exists():
            arr = _safe_load_embedding(legacy_path)
            if arr is None:
                return None
            self.set(hobby_name, arr)
            return arr
        return None

    def set(self, hobby_name: str, embedding: np.ndarray) -> None:
        self._memory[hobby_name] = embedding
        cache_path = self._cache_path(hobby_name)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embedding)
            _metadata_path(cache_path).write_text(
                json.dumps(
                    _embedding_metadata(
                        self.model_name,
                        self.model_revision,
                        self.preprocessing_version,
                        embedding,
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def _legacy_cache_path(self, hobby_name: str, prefix: str) -> Path | None:
        if not _allow_legacy_cache_lookup(self.model_name, self.model_revision, self.preprocessing_version):
            return None
        if self.base_cache_dir is None:
            return None
        key = hashlib.sha256(hobby_name.encode("utf-8")).hexdigest()[:16]
        return self.base_cache_dir / f"{prefix}_{key}.npy"

    def encode(self, hobby_name: str) -> np.ndarray:
        cached = self.get(hobby_name)
        if cached is not None:
            return cached
        model = _load_kure_model(
            self.device,
            model_name=self.model_name,
            model_revision=self.model_revision,
        )
        emb = model.encode(hobby_name, convert_to_numpy=True, show_progress_bar=False)
        self.set(hobby_name, emb)
        return emb

    def encode_batch(
        self,
        hobby_names: list[str],
        *,
        show_progress_bar: bool = False,
        progress_desc: str = "KURE hobby embeddings",
    ) -> dict[str, np.ndarray]:
        missing = [name for name in hobby_names if self.get(name) is None]
        if missing:
            model = _load_kure_model(
                self.device,
                model_name=self.model_name,
                model_revision=self.model_revision,
            )
            chunks = list(_iter_encode_chunks(missing, self.batch_size))
            iterator = tqdm(
                chunks,
                desc=progress_desc,
                unit="batch",
                dynamic_ncols=False,
                leave=True,
                mininterval=1.0,
                maxinterval=10.0,
                file=sys.stderr,
                disable=not show_progress_bar,
            )
            for chunk in iterator:
                embeddings = model.encode(
                    chunk,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                    device=self.device,
                )
                for name, emb in zip(chunk, embeddings, strict=False):
                    self.set(name, emb)
        result: dict[str, np.ndarray] = {}
        for name in hobby_names:
            embedding = self.get(name)
            if embedding is not None:
                result[name] = embedding
        return result

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

        if metadata.get("model_revision", "") != self.model_revision:
            return None, None

        if metadata.get("preprocessing_version") != self.preprocessing_version:
            return None, None

        if metadata.get("cache_version") != CACHE_VERSION:
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

        if metadata.get("embedding_dim") != _embedding_dim(matrix):
            return None, None

        return matrix.astype(np.float32), metadata

    def save_matrix(self, hobby_names: list[str], matrix: np.ndarray, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.cache_dir is None:
            return {
                "cache_enabled": False,
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "preprocessing_version": self.preprocessing_version,
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
        if matrix is not None and metadata is not None:
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
            "cache_version": CACHE_VERSION,
            "cache_enabled": True,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "preprocessing_version": self.preprocessing_version,
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


def _allow_legacy_cache_lookup(
    model_name: str,
    model_revision: str,
    preprocessing_version: str,
) -> bool:
    return (
        model_name == KURE_MODEL_NAME
        and model_revision == ""
        and preprocessing_version == DEFAULT_PREPROCESSING_VERSION
    )


def _model_cache_dir(
    cache_dir: Path,
    model_name: str,
    model_revision: str = "",
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
) -> Path:
    identity = "|".join((model_name, model_revision, preprocessing_version))
    safe_name = identity.replace("\\", "__").replace("/", "__").replace(":", "__").replace("|", "__")
    return cache_dir / safe_name


def _embedding_cache_key(
    text: str,
    model_name: str,
    model_revision: str,
    preprocessing_version: str,
) -> str:
    payload = {
        "text": text,
        "model_name": model_name,
        "model_revision": model_revision,
        "preprocessing_version": preprocessing_version,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".json")


def _safe_load_embedding(cache_path: Path) -> np.ndarray | None:
    try:
        return np.load(cache_path)
    except (OSError, ValueError):
        return None


def _embedding_dim(embedding: np.ndarray) -> int:
    arr = np.asarray(embedding)
    if arr.ndim == 1:
        return int(arr.shape[0])
    if arr.ndim == 2:
        return int(arr.shape[1])
    return 0


def _embedding_metadata(
    model_name: str,
    model_revision: str,
    preprocessing_version: str,
    embedding: np.ndarray,
) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "model_name": model_name,
        "model_revision": model_revision,
        "preprocessing_version": preprocessing_version,
        "embedding_dim": _embedding_dim(embedding),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _metadata_matches(
    meta_path: Path,
    *,
    model_name: str,
    model_revision: str,
    preprocessing_version: str,
    embedding_dim: int,
) -> bool:
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(metadata, dict):
        return False
    return (
        metadata.get("cache_version") == CACHE_VERSION
        and metadata.get("model_name") == model_name
        and metadata.get("model_revision", "") == model_revision
        and metadata.get("preprocessing_version") == preprocessing_version
        and metadata.get("embedding_dim") == embedding_dim
    )


def _iter_encode_chunks(values: list[str], batch_size: int) -> Iterator[list[str]]:
    chunk_size = max(1, int(batch_size) * ENCODE_CHUNK_BATCHES)
    for index in range(0, len(values), chunk_size):
        yield values[index : index + chunk_size]
