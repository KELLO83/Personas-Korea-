import pytest

from GNN_Neural_Network.gnn_recommender.data import PersonContext
from GNN_Neural_Network.gnn_recommender.recommend import Candidate, merge_candidates_by_hobby
from GNN_Neural_Network.gnn_recommender.rerank import (
    RerankerConfig,
    build_reranker_config,
    build_rerank_features,
    merge_stage1_candidates,
    rerank_candidates,
)


def _context() -> PersonContext:
    return PersonContext(
        person_uuid="p1",
        age="30",
        age_group="30대",
        sex="여자",
        occupation="개발자",
        district="강남구",
        province="서울특별시",
        family_type="1인 가구",
        housing_type="아파트",
        education_level="학사",
        persona_text="요가를 즐김",
        professional_text="",
        sports_text="",
        arts_text="",
        travel_text="",
        culinary_text="",
        family_text="",
        hobbies_text="요가",
        skills_text="",
        career_goals="",
        embedding_text="요가",
    )


def _profile() -> dict[str, object]:
    return {
        "source": "train_split_only",
        "hobbies": {
            "요가": {
                "train_popularity": 10,
                "distributions": {"age_group": {"30대": 8}, "occupation": {"개발자": 5}, "province": {"서울특별시": 7}},
                "cooccurring_hobbies": [{"hobby_name": "등산", "count": 3}],
            },
            "독서": {"train_popularity": 5, "distributions": {}, "cooccurring_hobbies": []},
        },
    }


def test_rerank_requires_train_only_profile() -> None:
    candidate = merge_stage1_candidates([Candidate(1, "lightgcn", 1.0, 1.0)], {1: "요가"})

    with pytest.raises(ValueError, match="train split only"):
        rerank_candidates(_context(), candidate, {"source": "full_graph", "hobbies": {}}, set())


def test_rerank_features_are_bounded_and_no_text_by_default() -> None:
    merged = merge_candidates_by_hobby(
        {
            "lightgcn": [Candidate(1, "lightgcn", 1.0, 1.0, source_scores={"lightgcn": 1.0})],
            "segment_popularity": [Candidate(1, "segment_popularity", 0.9, 0.8, source_scores={"segment_popularity": 0.8})],
        },
        1,
    )
    candidate = merge_stage1_candidates(merged, {1: "요가"})[0]

    features = build_rerank_features(_context(), candidate, _profile(), {"등산"}, RerankerConfig())

    assert features["persona_text_fit"] == 0.0
    assert features["age_group_fit"] == 1.0
    assert features["known_hobby_compatibility"] == 1.0
    assert features["segment_popularity_score"] == 0.8


def test_rerank_penalizes_clear_demographic_mismatch() -> None:
    merged = merge_candidates_by_hobby({"lightgcn": [Candidate(1, "lightgcn", 1.0, 1.0)]}, 1)
    candidate = merge_stage1_candidates(merged, {1: "요가"})[0]
    profile: dict[str, object] = {
        "source": "train_split_only",
        "hobbies": {
            "요가": {
                "train_popularity": 10,
                "distributions": {"age_group": {"50대": 8}, "occupation": {"교사": 5}, "sex": {"남자": 7}},
                "cooccurring_hobbies": [],
            }
        },
    }

    features = build_rerank_features(_context(), candidate, profile, set(), RerankerConfig())

    assert features["mismatch_penalty"] > 0.0


def test_rerank_uses_stage1_fallback_without_context() -> None:
    candidate = merge_stage1_candidates([Candidate(1, "lightgcn", 1.0, 0.7)], {1: "요가"})

    reranked = rerank_candidates(None, candidate, _profile(), set())

    assert reranked[0].final_score == 0.7
    assert reranked[0].reason_features["fallback"] == "stage1_score_only"


def test_build_reranker_config_applies_weight_overrides() -> None:
    config = build_reranker_config(False, {"lightgcn_score": 0.9, "segment_popularity_score": 0.2, "age_group_fit": 0.2})

    assert config.weights.lightgcn_score == 0.9
    assert config.weights.segment_popularity_score == 0.2
    assert config.weights.age_group_fit == 0.2
    assert config.weights.cooccurrence_score == 0.10


def test_segment_popularity_weight_can_change_rerank_order() -> None:
    candidates = merge_stage1_candidates(
        merge_candidates_by_hobby(
            {
                "lightgcn": [
                    Candidate(1, "lightgcn", 1.0, 1.0, source_scores={"lightgcn": 1.0}),
                    Candidate(2, "lightgcn", 0.8, 0.8, source_scores={"lightgcn": 0.8}),
                ],
                "segment_popularity": [
                    Candidate(1, "segment_popularity", 0.2, 0.2, source_scores={"segment_popularity": 0.2}),
                    Candidate(2, "segment_popularity", 1.0, 1.0, source_scores={"segment_popularity": 1.0}),
                ],
            },
            2,
        ),
        {1: "요가", 2: "독서"},
    )

    profile: dict[str, object] = _profile()

    reranked = rerank_candidates(
        _context(),
        candidates,
        profile,
        set(),
        build_reranker_config(False, {"lightgcn_score": 0.0, "segment_popularity_score": 1.0, "mismatch_penalty": 0.0}),
    )

    assert reranked[0].hobby_id == 2


def test_build_reranker_config_rejects_unknown_weight() -> None:
    with pytest.raises(ValueError, match="Unknown reranker weight"):
        build_reranker_config(False, {"unknown": 1.0})
