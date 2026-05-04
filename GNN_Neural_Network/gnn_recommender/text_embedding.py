from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

KURE_MODEL_NAME = "nlpai-lab/KURE-v1"
CACHE_DIR = "GNN_Neural_Network/artifacts/embeddings_cache"


class HobbyEmbeddingCache:
    def __init__(self, cache_file: str | Path) -> None:
        self.cache_file = str(cache_file)
        self._embeddings: dict[str, np.ndarray] = {}
        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._load()

    def _load(self) -> None:
        npy_path = self.cache_file.replace(".txt", ".npy")
        if os.path.exists(npy_path):
            try:
                loaded = np.load(npy_path, allow_pickle=True).item()
                if isinstance(loaded, dict):
                    self._embeddings = {
                        str(key): np.asarray(value, dtype=np.float32)
                        for key, value in loaded.items()
                    }
                    return
            except (OSError, ValueError, TypeError):
                pass
        try:
            with open(self.cache_file, "r", encoding="utf-8") as file:
                for line in file:
                    hobby, vector = line.strip().split("\t", 1)
                    self._embeddings[hobby] = np.fromstring(vector, sep=" ", dtype=np.float32)
        except FileNotFoundError:
            return

    def get(self, hobby: str) -> np.ndarray | None:
        return self._embeddings.get(hobby)

    def set(self, hobby: str, embedding: np.ndarray) -> None:
        self._embeddings[hobby] = np.asarray(embedding, dtype=np.float32)

    def save(self) -> None:
        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.save(self.cache_file.replace(".txt", ".npy"), self._embeddings)

    def load_cache_np(self) -> None:
        self._load()


def _compile_alias_patterns(alias_map: dict[str, list[str]]) -> dict[str, list[re.Pattern[str]]]:
    patterns: dict[str, list[re.Pattern[str]]] = {}
    for canonical, aliases in alias_map.items():
        names = [canonical, *aliases]
        patterns[canonical] = [_compile_hobby_pattern(name) for name in names if name]
    return patterns


def _compile_hobby_pattern(hobby: str) -> re.Pattern[str]:
    escaped = re.escape(hobby)
    return re.compile(rf"(?<![\w가-힣]){escaped}(?![\w가-힣])", flags=re.IGNORECASE)


def mask_holdout_hobbies(
    text: str,
    holdout_hobbies: set[str] | list[str] | tuple[str, ...],
    alias_map: dict[str, list[str]] | None = None,
    mask_token: str = "[MASK]",
) -> str:
    if not text or not holdout_hobbies:
        return text

    alias_patterns = _compile_alias_patterns(alias_map) if alias_map else {}
    masked = text
    for hobby in sorted(set(holdout_hobbies), key=len, reverse=True):
        masked = _compile_hobby_pattern(hobby).sub(mask_token, masked)
        for pattern in alias_patterns.get(hobby, []):
            masked = pattern.sub(mask_token, masked)
    return masked


def post_mask_leakage_audit(
    masked_text: str,
    holdout_hobbies: set[str] | list[str] | tuple[str, ...],
    alias_map: dict[str, list[str]] | None = None,
) -> bool:
    if not masked_text or not holdout_hobbies:
        return True

    normalized = _normalize_for_audit(masked_text)
    for hobby in holdout_hobbies:
        if _normalize_for_audit(hobby) in normalized:
            return False
        if alias_map:
            for alias in alias_map.get(hobby, []):
                if _normalize_for_audit(alias) in normalized:
                    return False
    return True


def _normalize_for_audit(text: str) -> str:
    return " ".join(text.lower().split())


def _load_kure_model(device: str | None = None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers is required for KURE text embeddings") from exc

    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    model = SentenceTransformer(KURE_MODEL_NAME, **kwargs)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = 512
    return model


def compute_text_embedding_similarity(*args: Any, **kwargs: Any) -> float | np.ndarray:
    if args and not isinstance(args[0], str) and hasattr(args[0], "encode"):
        return _compute_similarity_matrix(*args, **kwargs)
    return _compute_similarity_scalar(*args, **kwargs)


def _compute_similarity_scalar(persona_text: str, hobby_name: str) -> float:
    if not persona_text or not hobby_name:
        return 0.0
    return _lexical_similarity(persona_text, hobby_name)


def _compute_similarity_matrix(
    model: Any,
    persona_texts: list[str] | tuple[str, ...],
    hobby_names: list[str] | tuple[str, ...],
    cache: HobbyEmbeddingCache | None = None,
) -> np.ndarray:
    if not persona_texts or not hobby_names:
        return np.zeros((len(persona_texts), len(hobby_names)), dtype=np.float32)

    persona_embeddings = _encode_texts(model, list(persona_texts), batch_size=32)
    hobby_embeddings: list[np.ndarray] = []
    missing_hobbies: list[str] = []
    missing_indices: list[int] = []

    for index, hobby in enumerate(hobby_names):
        cached = cache.get(hobby) if cache else None
        if cached is None:
            missing_hobbies.append(hobby)
            missing_indices.append(index)
            hobby_embeddings.append(np.empty((0,), dtype=np.float32))
        else:
            hobby_embeddings.append(_normalize_vector(cached))

    if missing_hobbies:
        encoded = _encode_texts(model, missing_hobbies, batch_size=32)
        for hobby, index, embedding in zip(missing_hobbies, missing_indices, encoded, strict=False):
            normalized = _normalize_vector(embedding)
            hobby_embeddings[index] = normalized
            if cache:
                cache.set(hobby, normalized)

    if cache:
        cache.save()

    hobby_matrix = np.vstack(hobby_embeddings)
    return np.matmul(persona_embeddings, hobby_matrix.T).clip(0.0, 1.0)


def _encode_texts(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return array
    return array / norm


def _lexical_similarity(persona_text: str, hobby_name: str) -> float:
    if hobby_name in persona_text:
        return 1.0
    persona_chars = {char for char in persona_text.lower() if not char.isspace()}
    hobby_chars = {char for char in hobby_name.lower() if not char.isspace()}
    if not persona_chars or not hobby_chars:
        return 0.0
    return float(len(persona_chars & hobby_chars) / len(hobby_chars))


def batch_compute_embedding_similarity(
    persona_texts: list[str],
    hobby_names: list[str],
) -> list[float]:
    if len(persona_texts) != len(hobby_names):
        return []
    return [
        float(_compute_similarity_scalar(persona_text, hobby_name))
        for persona_text, hobby_name in zip(persona_texts, hobby_names, strict=False)
    ]
