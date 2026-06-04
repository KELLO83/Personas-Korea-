from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .data import PersonContext


PHASE6_BASELINE_STAGE1 = "popularity+cooccurrence"
PHASE6_BASELINE_STAGE2 = "lightgbm_num_leaves31_e5_domain"


@dataclass(frozen=True)
class Phase6ExperimentSpec:
    experiment_id: str
    changed_variable: str
    stage1_provider: str = PHASE6_BASELINE_STAGE1
    stage2_recipe: str = PHASE6_BASELINE_STAGE2
    candidate_text_builder: str = "name_only"
    embedding_model: str = "dragonkue/multilingual-e5-small-ko-v2"
    validation_first: bool = True
    winner_only_test: bool = True


def validate_phase6_spec(spec: Phase6ExperimentSpec) -> None:
    if not spec.experiment_id.strip():
        raise ValueError("experiment_id must not be empty")
    if not spec.changed_variable.strip():
        raise ValueError("changed_variable must describe the single changed variable")
    changed = [
        spec.stage1_provider != PHASE6_BASELINE_STAGE1,
        spec.stage2_recipe != PHASE6_BASELINE_STAGE2,
        spec.candidate_text_builder != "name_only",
        spec.embedding_model != "dragonkue/multilingual-e5-small-ko-v2",
    ]
    if sum(changed) > 1:
        raise ValueError("Phase 6 experiments must change only one controlled variable at a time")
    if not spec.validation_first:
        raise ValueError("Phase 6 experiments must run validation before test")
    if not spec.winner_only_test:
        raise ValueError("Phase 6 experiments must use winner-only test")


def build_positive_blacklist(
    train_edges: Iterable[tuple[int, int]],
    validation_edges: Iterable[tuple[int, int]],
    test_edges: Iterable[tuple[int, int]],
) -> dict[int, set[int]]:
    blacklist: dict[int, set[int]] = defaultdict(set)
    for person_id, hobby_id in [*train_edges, *validation_edges, *test_edges]:
        blacklist[int(person_id)].add(int(hobby_id))
    return dict(blacklist)


def validate_negative_samples(
    samples: Mapping[int, Iterable[int]],
    positive_blacklist: Mapping[int, set[int]],
) -> None:
    violations: list[str] = []
    for person_id, hobby_ids in samples.items():
        blocked = positive_blacklist.get(int(person_id), set())
        overlap = sorted(blocked.intersection(int(hobby_id) for hobby_id in hobby_ids))
        if overlap:
            violations.append(f"{person_id}:{overlap[:5]}")
    if violations:
        joined = ", ".join(violations[:10])
        raise ValueError(f"negative samples include held-out positives: {joined}")


def build_cross_feature_values(
    context: PersonContext,
    hobby_name: str,
    fit_tables: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    age_sex_key = _join_key(context.age_group, context.sex, hobby_name)
    occupation_region_key = _join_key(context.occupation, context.province or context.district, hobby_name)
    occupation_district_key = _join_key(context.occupation, context.district, hobby_name)
    return {
        "age_group_sex_fit": float(fit_tables.get("age_group_sex_hobby", {}).get(age_sex_key, 0.0)),
        "occupation_region_fit": max(
            float(fit_tables.get("occupation_region_hobby", {}).get(occupation_region_key, 0.0)),
            float(fit_tables.get("occupation_region_hobby", {}).get(occupation_district_key, 0.0)),
        ),
    }


def build_stage2_cross_features(base_features: Mapping[str, float]) -> dict[str, float]:
    age_group_fit = float(base_features.get("age_group_fit", 0.0))
    occupation_fit = float(base_features.get("occupation_fit", 0.0))
    region_fit = float(base_features.get("region_fit", 0.0))
    text_fit = max(
        float(base_features.get("text_embedding_similarity", 0.0)),
        float(base_features.get("e5_professional_similarity", 0.0)),
        float(base_features.get("e5_sports_similarity", 0.0)),
        float(base_features.get("e5_arts_similarity", 0.0)),
        float(base_features.get("e5_travel_similarity", 0.0)),
        float(base_features.get("e5_food_similarity", 0.0)),
        float(base_features.get("e5_family_similarity", 0.0)),
    )
    demographic_fit = (age_group_fit + occupation_fit + region_fit) / 3.0
    return {
        "age_group_region_cross_fit": age_group_fit * region_fit,
        "occupation_region_cross_fit": occupation_fit * region_fit,
        "demographic_text_cross_fit": demographic_fit * text_fit,
    }


def build_smoothed_fit_table(
    observations: Iterable[tuple[Sequence[str], bool]],
    *,
    alpha: float = 1.0,
    prior: float | None = None,
) -> dict[str, float]:
    totals: Counter[str] = Counter()
    positives: Counter[str] = Counter()
    global_total = 0
    global_positive = 0
    for key_parts, is_positive in observations:
        key = _join_key(*key_parts)
        totals[key] += 1
        global_total += 1
        if is_positive:
            positives[key] += 1
            global_positive += 1
    resolved_prior = prior if prior is not None else (global_positive / global_total if global_total else 0.0)
    return {
        key: (positives[key] + alpha * resolved_prior) / (totals[key] + alpha)
        for key in totals
    }


def topic_calibrated_scores(
    candidates: Iterable[tuple[str, float, str]],
    target_distribution: Mapping[str, float],
    *,
    calibration_lambda: float = 0.1,
) -> list[tuple[str, float]]:
    if calibration_lambda < 0:
        raise ValueError("calibration_lambda must be non-negative")
    normalized_target = _normalize_distribution(target_distribution)
    seen: Counter[str] = Counter()
    output: list[tuple[str, float]] = []
    for index, (candidate_id, base_score, topic) in enumerate(candidates):
        prefix_count = max(index, 1)
        current_share = seen[str(topic)] / prefix_count
        desired_share = normalized_target.get(str(topic), 0.0)
        calibration = calibration_lambda * (desired_share - current_share)
        output.append((str(candidate_id), float(base_score) + calibration))
        seen[str(topic)] += 1
    return output


def _normalize_distribution(values: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(float(value), 0.0) for value in values.values())
    if total <= 0:
        return {}
    return {str(key): max(float(value), 0.0) / total for key, value in values.items()}


def _join_key(*parts: object) -> str:
    return "|".join(str(part or "").strip().lower() for part in parts)
