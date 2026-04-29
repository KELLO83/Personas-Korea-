from GNN_Neural_Network.gnn_recommender.recommend import (
    Candidate,
    build_provider_contribution_artifact,
    merge_candidates_by_hobby,
    normalize_candidate_scores,
)


def test_normalize_candidate_scores_rank_percentile_preserves_raw_scores() -> None:
    candidates = [Candidate(10, "popularity", 3.0), Candidate(11, "popularity", 1.0)]

    normalized = normalize_candidate_scores(candidates, "rank_percentile")

    assert [candidate.hobby_id for candidate in normalized] == [10, 11]
    assert [candidate.raw_score for candidate in normalized] == [3.0, 1.0]
    assert [candidate.normalized_score for candidate in normalized] == [1.0, 0.0]


def test_normalize_candidate_scores_min_max() -> None:
    candidates = [Candidate(10, "cooccurrence", 2.0), Candidate(11, "cooccurrence", 6.0)]

    normalized = normalize_candidate_scores(candidates, "min_max")

    assert [candidate.hobby_id for candidate in normalized] == [11, 10]
    assert [candidate.normalized_score for candidate in normalized] == [1.0, 0.0]


def test_merge_candidates_by_hobby_deduplicates_by_best_score() -> None:
    selected = merge_candidates_by_hobby(
        {
            "lightgcn": [Candidate(10, "lightgcn", 0.2, 0.2)],
            "popularity": [Candidate(10, "popularity", 5.0, 1.0), Candidate(11, "popularity", 4.0, 0.5)],
        },
        top_k=2,
    )

    assert [(candidate.hobby_id, candidate.provider) for candidate in selected] == [(10, "popularity"), (11, "popularity")]
    assert selected[0].source_scores == {"lightgcn": 0.2, "popularity": 1.0}


def test_build_provider_contribution_artifact_counts_selected_sources() -> None:
    provider_candidates = {
        "lightgcn": [Candidate(10, "lightgcn", 0.5, 1.0)],
        "popularity": [Candidate(11, "popularity", 3.0, 1.0)],
    }

    artifact = build_provider_contribution_artifact(
        provider_candidates,
        [provider_candidates["lightgcn"][0]],
        requested_top_k=1,
        fallback_type="none",
    )

    assert artifact["providers"]["lightgcn"]["selected_count"] == 1
    assert artifact["providers"]["popularity"]["selected_count"] == 0


def test_candidate_to_dict_includes_source_scores() -> None:
    selected = merge_candidates_by_hobby(
        {
            "lightgcn": [Candidate(10, "lightgcn", 0.2, 0.2)],
            "cooccurrence": [Candidate(10, "cooccurrence", 3.0, 1.0)],
        },
        top_k=1,
    )

    artifact = build_provider_contribution_artifact({"merged": selected}, selected, requested_top_k=1, fallback_type="none")

    assert artifact["selected_candidates"][0]["source_scores"] == {"lightgcn": 0.2, "cooccurrence": 1.0}
