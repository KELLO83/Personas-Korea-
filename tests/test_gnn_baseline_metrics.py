from GNN_Neural_Network.gnn_recommender.baseline import (
    baseline_ranking_metrics,
    cooccurrence_candidate_provider,
    cooccurrence_recommendations,
    popularity_candidate_provider,
    popularity_recommendations,
    segment_popularity_candidate_provider,
)
from GNN_Neural_Network.gnn_recommender.data import PersonContext


def test_popularity_recommendations_exclude_known_items() -> None:
    train_edges = [(1, 10), (2, 10), (3, 11), (4, 12)]

    recommendations = popularity_recommendations(train_edges, [1], {1: {10}}, top_k=2)

    assert recommendations[1] == [11, 12]


def test_cooccurrence_recommendations_fall_back_to_popularity() -> None:
    train_edges = [(1, 10), (1, 11), (2, 10), (3, 12)]

    recommendations = cooccurrence_recommendations(train_edges, [3], {3: {12}}, top_k=2)

    assert recommendations[3] == [10, 11]


def test_baseline_ranking_metrics_reports_popularity_and_cooccurrence() -> None:
    train_edges = [(1, 10), (1, 11), (2, 10), (2, 12), (3, 10)]
    target_edges = [(1, 12)]
    known_by_person = {1: {10, 11}}

    metrics = baseline_ranking_metrics(train_edges, target_edges, known_by_person, (1, 2))

    assert set(metrics) == {"popularity", "cooccurrence"}
    assert metrics["popularity"]["recall@2"] == 1.0
    assert metrics["cooccurrence"]["hit_rate@2"] == 1.0


def test_popularity_candidate_provider_returns_scores_and_provider() -> None:
    candidates = popularity_candidate_provider(
        train_edges=[(1, 10), (2, 10), (3, 11)],
        person_id=1,
        known_hobbies={11},
        top_k=2,
    )

    assert len(candidates) == 1
    assert candidates[0].hobby_id == 10
    assert candidates[0].provider == "popularity"
    assert candidates[0].raw_score == 2.0


def test_cooccurrence_candidate_provider_does_not_hide_popularity_fallback() -> None:
    candidates = cooccurrence_candidate_provider(
        train_edges=[(1, 10), (1, 11), (2, 12)],
        person_id=2,
        known_hobbies={12},
        top_k=2,
    )

    assert candidates == []


def test_segment_popularity_provider_uses_train_only_profile_context() -> None:
    context = PersonContext(
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
        persona_text="",
        professional_text="",
        sports_text="",
        arts_text="",
        travel_text="",
        culinary_text="",
        family_text="",
        hobbies_text="",
        skills_text="",
        career_goals="",
        embedding_text="",
    )
    profile = {
        "source": "train_split_only",
        "hobbies": {
            "요가": {
                "hobby_id": 10,
                "train_popularity": 4,
                "distributions": {"age_group": {"30대": 3, "40대": 1}, "occupation": {"개발자": 2}},
            },
            "낚시": {"hobby_id": 11, "train_popularity": 10, "distributions": {"age_group": {"60대": 10}}},
        },
    }

    candidates = segment_popularity_candidate_provider(profile, context, known_hobbies={11}, top_k=2)

    assert [candidate.hobby_id for candidate in candidates] == [10]
    assert candidates[0].provider == "segment_popularity"
    assert candidates[0].reason_features["segment_scores"] == {"age_group": 0.75, "occupation": 1.0}


def test_segment_popularity_provider_requires_train_only_profile() -> None:
    candidates = segment_popularity_candidate_provider({"source": "full_graph", "hobbies": {}}, None, set(), 2)

    assert candidates == []
