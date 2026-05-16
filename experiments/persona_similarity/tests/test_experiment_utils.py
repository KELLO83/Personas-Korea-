import pandas as pd

from experiments.persona_similarity.scripts.common import cache_metadata_matches, mark_cache_hit, should_use_cache, write_json
from experiments.persona_similarity.scripts.evaluation_utils import add_diversity_rerank_score, evaluate_score_column, topk_overlap_at_k
from experiments.persona_similarity.scripts.experiment_specs import FEATURE_EXCLUSION_SETS, structured_text_feature_columns, text_feature_columns, feature_columns
from experiments.persona_similarity.scripts.text_feature_builder import build_domain_text, cosine_similarity


def test_feature_columns_excludes_requested_features() -> None:
    columns = feature_columns(FEATURE_EXCLUSION_SETS["without_low_info"])

    assert "same_sex" not in columns
    assert "same_marital" not in columns
    assert "same_community" not in columns
    assert "fastrp_score" in columns


def test_text_feature_column_sets_are_disjoint_from_identifiers() -> None:
    text_columns = text_feature_columns()
    combined = structured_text_feature_columns()

    assert "all_text_cosine" in text_columns
    assert "source_uuid" not in combined
    assert "target_uuid" not in combined
    assert "fastrp_score" in combined
    assert "hobbies_text_cosine" in combined


def test_build_domain_text_adds_domain_tags_and_skips_empty_values() -> None:
    row = {
        "persona": "quiet reader",
        "professional_persona": "",
        "hobbies_and_interests": "walks after work",
    }

    assert build_domain_text(row, "persona") == "persona: quiet reader"
    assert "hobbies_and_interests: walks after work" in build_domain_text(row, "hobbies")


def test_cosine_similarity_handles_zero_vectors() -> None:
    import numpy as np

    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_evaluate_score_column_returns_ranking_and_explanation_metrics() -> None:
    frame = pd.DataFrame(
        [
            {
                "source_uuid": "s1",
                "target_uuid": "a",
                "label": 1.0,
                "score": 0.9,
                "explanation_feature_count": 2,
                "same_occupation": 1,
                "same_district": 0,
                "same_education": 0,
                "same_field": 0,
                "shared_hobby_count": 0,
                "shared_skill_count": 0,
                "same_sex": 0,
                "same_marital": 0,
                "same_province": 0,
                "same_community": 0,
            },
            {
                "source_uuid": "s1",
                "target_uuid": "b",
                "label": 0.2,
                "score": 0.1,
                "explanation_feature_count": 1,
                "same_occupation": 0,
                "same_district": 0,
                "same_education": 0,
                "same_field": 0,
                "shared_hobby_count": 0,
                "shared_skill_count": 0,
                "same_sex": 1,
                "same_marital": 0,
                "same_province": 0,
                "same_community": 0,
            },
        ]
    )

    metrics = evaluate_score_column(frame, "score", [1, 2], progress=False)

    assert metrics["ndcg@1"] == 1.0
    assert metrics["explanation_coverage@2"] == 1.0
    assert metrics["strong_reason_coverage@2"] == 0.5
    assert metrics["low_information_dominance@2"] == 0.5


def test_topk_overlap_at_k_compares_target_sets_by_source() -> None:
    frame = pd.DataFrame(
        [
            {"source_uuid": "s1", "target_uuid": "a", "left": 2.0, "right": 1.0},
            {"source_uuid": "s1", "target_uuid": "b", "left": 1.0, "right": 2.0},
            {"source_uuid": "s1", "target_uuid": "c", "left": 0.0, "right": 0.0},
        ]
    )

    assert topk_overlap_at_k(frame, "left", "right", 1, progress=False) == 0.0
    assert topk_overlap_at_k(frame, "left", "right", 2, progress=False) == 1.0


def test_diversity_rerank_penalizes_repeated_attributes() -> None:
    frame = pd.DataFrame(
        [
            {"source_uuid": "s1", "target_uuid": "a", "score": 1.0, "target_occupation": "dev", "target_province": "seoul"},
            {"source_uuid": "s1", "target_uuid": "b", "score": 0.9, "target_occupation": "dev", "target_province": "seoul"},
            {"source_uuid": "s1", "target_uuid": "c", "score": 0.8, "target_occupation": "artist", "target_province": "busan"},
        ]
    )

    reranked = add_diversity_rerank_score(frame, "score", "diverse_score", diversity_lambda=0.2, penalty_columns=["target_occupation", "target_province"])
    ordered = reranked.sort_values("diverse_score", ascending=False)["target_uuid"].tolist()

    assert ordered[:2] == ["a", "c"]


def test_cache_helpers_detect_and_mark_cache_hits(tmp_path) -> None:
    artifact = tmp_path / "artifact.parquet"
    metadata = tmp_path / "artifact.status.json"
    artifact.write_text("data", encoding="utf-8")
    write_json(metadata, {"stage": "unit", "config_hash": "abc", "old_metric": 1.0})

    expected = {"stage": "unit", "config_hash": "abc"}

    assert cache_metadata_matches(metadata, expected) == (True, "metadata_match")
    assert should_use_cache(artifact, metadata, expected) == (True, "metadata_match")

    mark_cache_hit(metadata, expected, artifact)
    assert cache_metadata_matches(metadata, {**expected, "cache_hit": True}) == (True, "metadata_match")
