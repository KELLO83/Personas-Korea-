from __future__ import annotations

import pytest

from GNN_Neural_Network.gnn_recommender.data import empty_person_context
from GNN_Neural_Network.gnn_recommender.phase6 import (
    Phase6ExperimentSpec,
    build_cross_feature_values,
    build_positive_blacklist,
    build_smoothed_fit_table,
    topic_calibrated_scores,
    validate_negative_samples,
    validate_phase6_spec,
)


def test_phase6_spec_allows_one_changed_variable() -> None:
    spec = Phase6ExperimentSpec(
        experiment_id="phase6_cf_provider_smoke",
        changed_variable="stage1_provider",
        stage1_provider="similar_person_cf_quota",
    )

    validate_phase6_spec(spec)


def test_phase6_spec_rejects_multiple_changed_variables() -> None:
    spec = Phase6ExperimentSpec(
        experiment_id="bad",
        changed_variable="too_many",
        stage1_provider="similar_person_cf_quota",
        candidate_text_builder="name_plus_aliases",
    )

    with pytest.raises(ValueError, match="only one"):
        validate_phase6_spec(spec)


def test_positive_blacklist_blocks_validation_and_test_edges() -> None:
    blacklist = build_positive_blacklist(
        train_edges=[(1, 10)],
        validation_edges=[(1, 11)],
        test_edges=[(2, 20)],
    )

    assert blacklist == {1: {10, 11}, 2: {20}}
    with pytest.raises(ValueError, match="held-out positives"):
        validate_negative_samples({1: [11]}, blacklist)


def test_smoothed_fit_table_and_cross_feature_values() -> None:
    table = build_smoothed_fit_table(
        [
            (("20대", "여성", "클라이밍"), True),
            (("20대", "여성", "클라이밍"), False),
        ],
        alpha=1.0,
        prior=0.5,
    )
    context = empty_person_context("p1")
    context = context.__class__(
        **{**context.__dict__, "age_group": "20대", "sex": "여성", "occupation": "개발자", "province": "서울"}
    )

    features = build_cross_feature_values(context, "클라이밍", {"age_group_sex_hobby": table})

    assert 0.0 < features["age_group_sex_fit"] < 1.0
    assert features["occupation_region_fit"] == 0.0


def test_topic_calibration_rewards_underrepresented_topics() -> None:
    scores = topic_calibrated_scores(
        [
            ("a", 1.0, "sports"),
            ("b", 1.0, "arts"),
        ],
        {"arts": 1.0},
        calibration_lambda=0.2,
    )

    assert scores[0][1] == 1.0
    assert scores[1][1] > 1.0
