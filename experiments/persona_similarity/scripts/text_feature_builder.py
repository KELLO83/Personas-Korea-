from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np


TEXT_DOMAINS: dict[str, tuple[str, ...]] = {
    "all": (
        "persona",
        "professional_persona",
        "sports_persona",
        "arts_persona",
        "travel_persona",
        "culinary_persona",
        "family_persona",
        "cultural_background",
        "career_goals_and_ambitions",
        "skills_and_expertise",
        "hobbies_and_interests",
    ),
    "persona": ("persona",),
    "professional": ("professional_persona",),
    "hobbies": ("hobbies_and_interests", "sports_persona", "arts_persona", "travel_persona", "culinary_persona"),
    "skills": ("skills_and_expertise",),
    "career": ("career_goals_and_ambitions", "professional_persona"),
    "family": ("family_persona",),
    "lifestyle": ("hobbies_and_interests", "sports_persona", "arts_persona", "travel_persona", "culinary_persona", "family_persona"),
}


TEXT_FEATURE_BY_DOMAIN = {
    "all": "all_text_cosine",
    "persona": "persona_text_cosine",
    "professional": "professional_text_cosine",
    "hobbies": "hobbies_text_cosine",
    "skills": "skills_text_cosine",
    "career": "career_text_cosine",
    "family": "family_text_cosine",
    "lifestyle": "lifestyle_text_cosine",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def build_domain_text(row: dict[str, Any], domain: str) -> str:
    parts: list[str] = []
    for column in TEXT_DOMAINS[domain]:
        text = clean_text(row.get(column))
        if text:
            parts.append(f"{column}: {text}")
    return "\n".join(parts)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def embedding_key(uuid: str, domain: str) -> str:
    return f"{uuid}::{domain}"
