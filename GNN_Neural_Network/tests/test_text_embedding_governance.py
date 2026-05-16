from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from GNN_Neural_Network.gnn_recommender.data import (
    PersonContext,
    build_domain_tagged_persona_text,
)
from GNN_Neural_Network.gnn_recommender.embedding_cache import HobbyEmbeddingCache
from GNN_Neural_Network.scripts.evaluate_ranker import (
    TEXT_EMBEDDING_PREPROCESSING_VERSION,
    _feature_cache_key,
    _prepare_text_leakage_context,
)


def _context(**overrides: str) -> PersonContext:
    values = {
        "person_uuid": "p1",
        "age": "",
        "age_group": "",
        "sex": "",
        "occupation": "",
        "district": "",
        "province": "",
        "family_type": "",
        "housing_type": "",
        "education_level": "",
        "persona_text": "quiet culture note",
        "professional_text": "office text",
        "sports_text": "sports text",
        "arts_text": "arts text",
        "travel_text": "travel text",
        "culinary_text": "food text",
        "family_text": "family text",
        "hobbies_text": "hobby text",
        "skills_text": "",
        "career_goals": "",
        "embedding_text": "embedding text",
    }
    values.update(overrides)
    return PersonContext(**values)


def test_domain_tagged_builder_accepts_masked_field_overrides() -> None:
    text = build_domain_tagged_persona_text(
        _context(professional_text="raw hobby"),
        {"professional_text": "[ACT]"},
    )
    assert "[PROF] [ACT]" in text
    assert "raw hobby" not in text
    assert "[SPORT] sports text" in text


def test_prepare_text_context_counts_empty_domain_text_as_coverage_miss() -> None:
    payload = _prepare_text_leakage_context(
        person_ids=[1],
        target_edges=[(1, 10)],
        id_to_person={1: "p1"},
        contexts={"p1": _context(persona_text="", professional_text="", sports_text="", arts_text="", travel_text="", culinary_text="", family_text="")},
        id_to_hobby={10: "hobby"},
        alias_map={},
    )
    summary = payload["summary"]
    assert summary["failed_person_count"] == 0
    assert summary["missing_context_person_count"] == 1
    assert summary["audit_eligible_person_count"] == 0
    assert payload["person_text_by_id"] == {}


def test_feature_cache_key_includes_text_embedding_preprocessing_policy(tmp_path: Path) -> None:
    args = argparse.Namespace(split="validation")
    key = _feature_cache_key(
        args,
        [1],
        {},
        ["text_embedding_similarity"],
        tmp_path / "person_context.csv",
        tmp_path / "hobby_profile.json",
        tmp_path / "hobby_taxonomy.json",
        tmp_path / "hobby_aliases.json",
    )
    assert isinstance(key, str)
    assert TEXT_EMBEDDING_PREPROCESSING_VERSION == "domain_tagged_masked_v1"


def test_embedding_caches_do_not_collide_across_backbones(tmp_path: Path) -> None:
    hobby_names = ["hobby-a", "hobby-b"]
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    kure_cache = HobbyEmbeddingCache(tmp_path, model_name="nlpai-lab/KURE-v1")
    snowflake_cache = HobbyEmbeddingCache(tmp_path, model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko")
    e5_cache = HobbyEmbeddingCache(tmp_path, model_name="dragonkue/multilingual-e5-small-ko-v2")

    kure_cache.save_matrix(hobby_names, matrix)

    assert kure_cache.load_matrix(hobby_names)[0] is not None
    assert snowflake_cache.load_matrix(hobby_names)[0] is None
    assert e5_cache.load_matrix(hobby_names)[0] is None
