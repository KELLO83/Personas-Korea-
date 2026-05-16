from experiments.persona_similarity.scripts.feature_builder import (
    DeterministicScoreWeights,
    WeakLabelWeights,
    build_pair_features,
    deterministic_score,
    parse_json_list,
    weak_label,
)


def test_parse_json_list_handles_json_and_invalid_values() -> None:
    assert parse_json_list('["등산", "독서"]') == ["등산", "독서"]
    assert parse_json_list("not-json") == []
    assert parse_json_list(None) == []


def test_build_pair_features_uses_pair_matches_and_shared_counts() -> None:
    row = {
        "fastrp_score": 0.7,
        "source_age": 31,
        "target_age": 35,
        "source_age_group": "30대",
        "target_age_group": "30대",
        "source_sex": "여자",
        "target_sex": "남자",
        "source_province": "서울",
        "target_province": "서울",
        "source_district": "강남구",
        "target_district": "서초구",
        "source_occupation": "개발자",
        "target_occupation": "개발자",
        "source_education": "대학교",
        "target_education": "대학교",
        "source_field": "공학",
        "target_field": "공학",
        "source_marital": "미혼",
        "target_marital": "기혼",
        "source_family": "1인 가구",
        "target_family": "1인 가구",
        "source_housing": "아파트",
        "target_housing": "아파트",
        "source_community_id": 7,
        "target_community_id": 7,
        "shared_hobbies": '["등산", "독서"]',
        "shared_skills": '["Python"]',
    }

    features = build_pair_features(row)

    assert features["same_age_group"] == 1
    assert features["same_sex"] == 0
    assert features["same_province"] == 1
    assert features["same_district"] == 0
    assert features["same_occupation"] == 1
    assert features["shared_hobby_count"] == 2
    assert features["shared_skill_count"] == 1
    assert features["age_diff"] == 4


def test_weak_label_is_positive_for_matching_pair() -> None:
    features = {
        "fastrp_score": 0.5,
        "same_occupation": 1,
        "same_province": 1,
        "same_district": 0,
        "same_age_group": 1,
        "same_education": 1,
        "same_community": 1,
        "shared_hobby_count": 2,
        "shared_skill_count": 0,
    }

    assert weak_label(features, WeakLabelWeights()) > 0.5


def test_deterministic_score_rewards_strong_visible_matches() -> None:
    strong_pair = {
        "fastrp_score": 0.2,
        "same_occupation": 1,
        "same_province": 0,
        "same_district": 1,
        "same_age_group": 1,
        "same_education": 1,
        "same_field": 1,
        "same_family": 0,
        "same_housing": 0,
        "same_community": 0,
        "shared_hobby_count": 2,
        "shared_skill_count": 1,
    }
    weak_pair = {
        **strong_pair,
        "same_occupation": 0,
        "same_district": 0,
        "same_education": 0,
        "same_field": 0,
        "shared_hobby_count": 0,
        "shared_skill_count": 0,
        "same_province": 1,
        "same_community": 1,
    }

    weights = DeterministicScoreWeights()
    assert deterministic_score(strong_pair, weights) > deterministic_score(weak_pair, weights)
