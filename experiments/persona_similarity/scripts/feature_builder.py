from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


FEATURE_COLUMNS = [
    "fastrp_score",
    "age_diff",
    "same_age_group",
    "same_sex",
    "same_province",
    "same_district",
    "same_occupation",
    "same_education",
    "same_field",
    "same_marital",
    "same_family",
    "same_housing",
    "same_community",
    "shared_hobby_count",
    "shared_skill_count",
    "explanation_feature_count",
]


@dataclass(frozen=True)
class WeakLabelWeights:
    same_occupation: float = 0.25
    same_province: float = 0.15
    same_district: float = 0.10
    same_age_group: float = 0.10
    same_education: float = 0.10
    same_community: float = 0.10
    shared_hobby: float = 0.12
    shared_skill: float = 0.08
    fastrp_score: float = 0.10
    max_shared_hobbies: int = 5
    max_shared_skills: int = 5

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WeakLabelWeights":
        return cls(**{key: value for key, value in config.items() if key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class DeterministicScoreWeights:
    same_occupation: float = 0.24
    same_district: float = 0.14
    same_province: float = 0.08
    same_education: float = 0.10
    same_field: float = 0.10
    same_age_group: float = 0.08
    same_family: float = 0.06
    same_housing: float = 0.04
    same_community: float = 0.04
    shared_hobby: float = 0.08
    shared_skill: float = 0.04
    fastrp_score: float = 0.10
    max_shared_hobbies: int = 5
    max_shared_skills: int = 5

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DeterministicScoreWeights":
        return cls(**{key: value for key, value in config.items() if key in cls.__dataclass_fields__})


def same_value(left: Any, right: Any) -> int:
    if left is None or right is None:
        return 0
    if left == "" or right == "":
        return 0
    return int(left == right)


def parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, str) and item]
    return []


def build_pair_features(row: dict[str, Any]) -> dict[str, float]:
    source_age = row.get("source_age")
    target_age = row.get("target_age")
    age_diff = abs(float(source_age) - float(target_age)) if source_age is not None and target_age is not None else 0.0

    shared_hobbies = parse_json_list(row.get("shared_hobbies"))
    shared_skills = parse_json_list(row.get("shared_skills"))

    features = {
        "fastrp_score": float(row.get("fastrp_score") or 0.0),
        "age_diff": age_diff,
        "same_age_group": same_value(row.get("source_age_group"), row.get("target_age_group")),
        "same_sex": same_value(row.get("source_sex"), row.get("target_sex")),
        "same_province": same_value(row.get("source_province"), row.get("target_province")),
        "same_district": same_value(row.get("source_district"), row.get("target_district")),
        "same_occupation": same_value(row.get("source_occupation"), row.get("target_occupation")),
        "same_education": same_value(row.get("source_education"), row.get("target_education")),
        "same_field": same_value(row.get("source_field"), row.get("target_field")),
        "same_marital": same_value(row.get("source_marital"), row.get("target_marital")),
        "same_family": same_value(row.get("source_family"), row.get("target_family")),
        "same_housing": same_value(row.get("source_housing"), row.get("target_housing")),
        "same_community": same_value(row.get("source_community_id"), row.get("target_community_id")),
        "shared_hobby_count": len(shared_hobbies),
        "shared_skill_count": len(shared_skills),
    }
    features["explanation_feature_count"] = sum(
        int(features[column] > 0)
        for column in [
            "same_age_group",
            "same_sex",
            "same_province",
            "same_district",
            "same_occupation",
            "same_education",
            "same_field",
            "same_marital",
            "same_family",
            "same_housing",
            "same_community",
            "shared_hobby_count",
            "shared_skill_count",
        ]
    )
    return features


def weak_label(features: dict[str, float], weights: WeakLabelWeights) -> float:
    hobby_signal = min(features["shared_hobby_count"], weights.max_shared_hobbies) / max(1, weights.max_shared_hobbies)
    skill_signal = min(features["shared_skill_count"], weights.max_shared_skills) / max(1, weights.max_shared_skills)
    return (
        weights.same_occupation * features["same_occupation"]
        + weights.same_province * features["same_province"]
        + weights.same_district * features["same_district"]
        + weights.same_age_group * features["same_age_group"]
        + weights.same_education * features["same_education"]
        + weights.same_community * features["same_community"]
        + weights.shared_hobby * hobby_signal
        + weights.shared_skill * skill_signal
        + weights.fastrp_score * features["fastrp_score"]
    )


def deterministic_score(features: dict[str, float], weights: DeterministicScoreWeights) -> float:
    hobby_signal = min(features["shared_hobby_count"], weights.max_shared_hobbies) / max(1, weights.max_shared_hobbies)
    skill_signal = min(features["shared_skill_count"], weights.max_shared_skills) / max(1, weights.max_shared_skills)
    return (
        weights.same_occupation * features["same_occupation"]
        + weights.same_district * features["same_district"]
        + weights.same_province * features["same_province"]
        + weights.same_education * features["same_education"]
        + weights.same_field * features["same_field"]
        + weights.same_age_group * features["same_age_group"]
        + weights.same_family * features["same_family"]
        + weights.same_housing * features["same_housing"]
        + weights.same_community * features["same_community"]
        + weights.shared_hobby * hobby_signal
        + weights.shared_skill * skill_signal
        + weights.fastrp_score * features["fastrp_score"]
    )
