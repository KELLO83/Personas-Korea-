from __future__ import annotations

import re
from typing import Any

import numpy as np


# Default KURE model used by the main project
KURE_MODEL_NAME = "nlpai-lab/KURE-v1"


def _compile_alias_patterns(alias_map: dict[str, list[str]]) -> dict[str, list[re.Pattern[str]]]:
    """Compile word-boundary regex patterns for each canonical hobby and its aliases."""
    patterns: dict[str, list[re.Pattern[str]]] = {}
    for canonical, aliases in alias_map.items():
        all_names = [canonical] + aliases
        compiled: list[re.Pattern[str]] = []
        for name in all_names:
            escaped = re.escape(name)
            compiled.append(re.compile(rf"\b{escaped}\b"))
        patterns[canonical] = compiled
    return patterns


def mask_holdout_hobbies(
    text: str,
    holdout_hobbies: set[str],
    alias_map: dict[str, list[str]] | None = None,
    mask_token: str = "[MASK]",
) -> str:
    """Mask hold-out hobby names and their aliases from text using word-boundary regex.

    Args:
        text: Input text (e.g., persona_text, hobbies_text).
        holdout_hobbies: Set of canonical hobby names to mask.
        alias_map: Optional map from canonical hobby to list of alias names.
        mask_token: Token to replace matched hobby names with.

    Returns:
        Text with hold-out hobbies masked.
    """
    if not text or not holdout_hobbies:
        return text

    alias_patterns = _compile_alias_patterns(alias_map) if alias_map else {}

    masked = text
    for hobby in sorted(holdout_hobbies, key=len, reverse=True):
        escaped = re.escape(hobby)
        masked = re.sub(rf"\b{escaped}\b", mask_token, masked)
        for pattern in alias_patterns.get(hobby, []):
            masked = pattern.sub(mask_token, masked)

    return masked


def post_mask_leakage_audit(
    masked_text: str,
    holdout_hobbies: set[str],
    alias_map: dict[str, list[str]] | None = None,
) -> bool:
    """Check if any hold-out hobby or alias still remains in the masked text.

    Args:
        masked_text: Text after masking.
        holdout_hobbies: Set of canonical hobby names that should be masked.
        alias_map: Optional alias map.

    Returns:
        True if no leakage detected, False otherwise.
    """
    normalized = _normalize_for_audit(masked_text)
    for hobby in holdout_hobbies:
        if hobby in normalized:
            return False
        if alias_map:
            for alias in alias_map.get(hobby, []):
                if alias in normalized:
                    return False
    return True


def _normalize_for_audit(text: str) -> str:
    """Normalize text for leakage audit: lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


def _load_kure_model() -> Any:
    """Lazy-load the KURE sentence embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for text embedding. "
            "Install: pip install sentence-transformers>=2.3.0"
        ) from exc
    return SentenceTransformer(KURE_MODEL_NAME)


def compute_text_embedding_similarity(
    persona_text: str,
    hobby_name: str,
) -> float:
    """Compute cosine similarity between persona text and hobby name using KURE embeddings.

    Args:
        persona_text: Masked persona text.
        hobby_name: Canonical hobby name.

    Returns:
        Cosine similarity in [0, 1] range. Returns 0.0 on error or empty input.
    """
    if not persona_text or not hobby_name:
        return 0.0

    try:
        model = _load_kure_model()
        embeddings = model.encode([persona_text, hobby_name], convert_to_numpy=True, show_progress_bar=False)
    except Exception:
        return 0.0

    if embeddings.shape[0] < 2:
        return 0.0

    vec_a = embeddings[0]
    vec_b = embeddings[1]

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    cosine_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    # Clamp to [0, 1] for use as a feature
    return max(0.0, min(1.0, cosine_sim))


def batch_compute_embedding_similarity(
    persona_texts: list[str],
    hobby_names: list[str],
) -> list[float]:
    """Compute similarities for a batch of (persona_text, hobby_name) pairs.

    Args:
        persona_texts: List of persona texts.
        hobby_names: List of hobby names (same length as persona_texts).

    Returns:
        List of cosine similarities.
    """
    if not persona_texts or not hobby_names or len(persona_texts) != len(hobby_names):
        return []

    try:
        model = _load_kure_model()
        texts = persona_texts + hobby_names
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
    except Exception:
        return [0.0] * len(persona_texts)

    n = len(persona_texts)
    results: list[float] = []
    for i in range(n):
        vec_a = embeddings[i]
        vec_b = embeddings[n + i]
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0.0 or norm_b == 0.0:
            results.append(0.0)
            continue
        cosine_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        results.append(max(0.0, min(1.0, cosine_sim)))
    return results
