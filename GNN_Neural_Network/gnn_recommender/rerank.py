from __future__ import annotations

from dataclasses import dataclass, field
import math

from .data import PersonContext
from .recommend import Candidate


@dataclass(frozen=True)
class HobbyCandidate:
    hobby_id: int
    hobby_name: str
    source_scores: dict[str, float]
    raw_source_scores: dict[str, float]
    reason_features: dict[str, object]


@dataclass(frozen=True)
class RerankerWeights:
    lightgcn_score: float = 0.25
    cooccurrence_score: float = 0.10
    segment_popularity_score: float = 0.20
    similar_person_score: float = 0.0
    persona_text_fit: float = 0.0
    known_hobby_compatibility: float = 0.15
    age_group_fit: float = 0.10
    occupation_fit: float = 0.10
    region_fit: float = 0.05
    popularity_prior: float = 0.03
    mismatch_penalty: float = 0.25
    popularity_penalty: float = 0.0
    novelty_bonus: float = 0.0
    category_diversity_reward: float = 0.0
    is_cold_start: float = 0.0


@dataclass(frozen=True)
class RerankerConfig:
    use_text_fit: bool = False
    weights: RerankerWeights = field(default_factory=RerankerWeights)


@dataclass(frozen=True)
class RerankedCandidate:
    hobby_id: int
    hobby_name: str
    final_score: float
    stage1_score: float
    features: dict[str, float]
    source_scores: dict[str, float]
    reason_features: dict[str, object]


def build_reranker_config(use_text_fit: bool, weights: dict[str, float] | None = None) -> RerankerConfig:
    if not weights:
        return RerankerConfig(use_text_fit=use_text_fit)
    defaults = RerankerWeights()
    allowed = set(defaults.__dataclass_fields__)
    unknown = sorted(set(weights) - allowed)
    if unknown:
        raise ValueError(f"Unknown reranker weight keys: {', '.join(unknown)}")
    return RerankerConfig(
        use_text_fit=use_text_fit,
        weights=RerankerWeights(**{key: float(value) for key, value in weights.items()}),
    )


def merge_stage1_candidates(candidates: list[Candidate], id_to_hobby: dict[int, str]) -> list[HobbyCandidate]:
    output: list[HobbyCandidate] = []
    for candidate in candidates:
        raw_scores = candidate.reason_features.get("raw_source_scores", {}) if candidate.reason_features else {}
        raw_source_scores = {str(key): _safe_float(value) for key, value in raw_scores.items()} if isinstance(raw_scores, dict) else {}
        output.append(
            HobbyCandidate(
                hobby_id=candidate.hobby_id,
                hobby_name=id_to_hobby[candidate.hobby_id],
                source_scores=candidate.source_scores or {candidate.provider: candidate.score},
                raw_source_scores=raw_source_scores,
                reason_features=candidate.reason_features or {},
            )
        )
    return output


def rerank_candidates(
    context: PersonContext | None,
    candidates: list[HobbyCandidate],
    hobby_profile: dict[str, object] | None,
    known_hobby_names: set[str],
    config: RerankerConfig | None = None,
    hobby_taxonomy: dict[str, object] | None = None,
) -> list[RerankedCandidate]:
    if context is None or hobby_profile is None:
        return [_stage1_only(candidate) for candidate in candidates]
    _require_train_only_profile(hobby_profile)
    active_config = config or RerankerConfig()
    reranked = [
        _score_candidate(context, candidate, hobby_profile, known_hobby_names, active_config)
        for candidate in candidates
    ]
    initial_ranked = sorted(reranked, key=lambda item: (-item.final_score, item.hobby_id))
    return _diversity_aware_sort(initial_ranked, hobby_taxonomy, active_config.weights)


def _score_candidate(
    context: PersonContext,
    candidate: HobbyCandidate,
    hobby_profile: dict[str, object],
    known_hobby_names: set[str],
    config: RerankerConfig,
) -> RerankedCandidate:
    features = build_rerank_features(context, candidate, hobby_profile, known_hobby_names, config)
    final_score = score_rerank_features(features, config.weights)
    return RerankedCandidate(
        hobby_id=candidate.hobby_id,
        hobby_name=candidate.hobby_name,
        final_score=final_score,
        stage1_score=max(candidate.source_scores.values()) if candidate.source_scores else 0.0,
        features=features,
        source_scores=candidate.source_scores,
        reason_features={"stage1": candidate.reason_features, "features": features},
    )


def score_rerank_features(features: dict[str, float], weights: RerankerWeights) -> float:
    return (
        weights.lightgcn_score * features["lightgcn_score"]
        + weights.cooccurrence_score * features["cooccurrence_score"]
        + weights.segment_popularity_score * features["segment_popularity_score"]
        + weights.similar_person_score * features["similar_person_score"]
        + weights.persona_text_fit * features["persona_text_fit"]
        + weights.known_hobby_compatibility * features["known_hobby_compatibility"]
        + weights.age_group_fit * features["age_group_fit"]
        + weights.occupation_fit * features["occupation_fit"]
        + weights.region_fit * features["region_fit"]
        + weights.popularity_prior * features["popularity_prior"]
        - weights.mismatch_penalty * features["mismatch_penalty"]
        - weights.popularity_penalty * features["popularity_penalty"]
        + weights.novelty_bonus * features["novelty_bonus"]
        + weights.category_diversity_reward * features["category_diversity_reward"]
        + weights.is_cold_start * features.get("is_cold_start", 0.0)
    )


def build_rerank_features(
    context: PersonContext,
    candidate: HobbyCandidate,
    hobby_profile: dict[str, object] | None,
    known_hobby_names: set[str],
    config: RerankerConfig,
    text_embedding_similarity: float = 0.0,
) -> dict[str, float]:
    resolved_hobby_profile = hobby_profile or {}
    profile = _hobby_entry(resolved_hobby_profile, candidate.hobby_name)
    source_keys = set(candidate.source_scores)
    source_is_popularity = 1.0 if "popularity" in source_keys else 0.0
    source_is_cooccurrence = 1.0 if "cooccurrence" in source_keys else 0.0
    return {
        "lightgcn_score": candidate.source_scores.get("lightgcn", 0.0),
        "cooccurrence_score": candidate.source_scores.get("cooccurrence", 0.0),
        "segment_popularity_score": candidate.source_scores.get("segment_popularity", 0.0),
        "similar_person_score": 0.0,
        "persona_text_fit": _persona_text_fit(context, candidate) if config.use_text_fit else 0.0,
        "known_hobby_compatibility": _known_hobby_compatibility(profile, known_hobby_names),
        "age_group_fit": _distribution_fit(profile, "age_group", context.age_group),
        "occupation_fit": _distribution_fit(profile, "occupation", context.occupation),
        "region_fit": max(_distribution_fit(profile, "province", context.province), _distribution_fit(profile, "district", context.district)),
        "popularity_prior": _popularity_prior(profile, resolved_hobby_profile),
        "mismatch_penalty": _mismatch_penalty(profile, context),
        "popularity_penalty": _popularity_penalty(profile, resolved_hobby_profile),
        "novelty_bonus": _novelty_bonus(profile, resolved_hobby_profile),
        "category_diversity_reward": 0.0,
        "is_cold_start": 1.0 if len(known_hobby_names) <= 1 else 0.0,
        "source_is_popularity": source_is_popularity,
        "source_is_cooccurrence": 1.0 if "cooccurrence" in source_keys else 0.0,
        "source_count": source_is_popularity + source_is_cooccurrence,
        "text_embedding_similarity": text_embedding_similarity,
    }


def _require_train_only_profile(hobby_profile: dict[str, object]) -> None:
    if hobby_profile.get("source") != "train_split_only":
        raise ValueError("hobby_profile must be generated from train split only")


def _hobby_entry(hobby_profile: dict[str, object], hobby_name: str) -> dict[str, object]:
    hobbies = hobby_profile.get("hobbies", {})
    if not isinstance(hobbies, dict):
        return {}
    entry = hobbies.get(hobby_name, {})
    return dict(entry) if isinstance(entry, dict) else {}


def _distribution_fit(profile: dict[str, object], field: str, value: str) -> float:
    distributions = profile.get("distributions", {})
    if not isinstance(distributions, dict) or not value:
        return 0.0
    distribution = distributions.get(field, {})
    if not isinstance(distribution, dict) or not distribution:
        return 0.0
    total = sum(_safe_float(count) for count in distribution.values())
    return _safe_float(distribution.get(value, 0.0)) / total if total else 0.0


def _known_hobby_compatibility(profile: dict[str, object], known_hobby_names: set[str]) -> float:
    cooccurring = profile.get("cooccurring_hobbies", [])
    if not isinstance(cooccurring, list) or not cooccurring or not known_hobby_names:
        return 0.0
    total = 0.0
    matched = 0.0
    for item in cooccurring:
        if not isinstance(item, dict):
            continue
        count = _safe_float(item.get("count", 0.0))
        total += count
        if str(item.get("hobby_name", "")) in known_hobby_names:
            matched += count
    return matched / total if total else 0.0


def _popularity_prior(profile: dict[str, object], hobby_profile: dict[str, object]) -> float:
    count = _safe_float(profile.get("train_popularity", 0.0))
    hobbies = hobby_profile.get("hobbies", {})
    if not isinstance(hobbies, dict) or not hobbies:
        return 0.0
    max_count = max((_train_popularity(entry) for entry in hobbies.values() if isinstance(entry, dict)), default=0.0)
    return count / max_count if max_count else 0.0


def _popularity_penalty(profile: dict[str, object], hobby_profile: dict[str, object]) -> float:
    """Penalize very popular items. Uses log-normalized popularity."""
    count = _safe_float(profile.get("train_popularity", 0.0))
    hobbies = hobby_profile.get("hobbies", {})
    if not isinstance(hobbies, dict) or not hobbies:
        return 0.0
    max_count = max((_train_popularity(entry) for entry in hobbies.values() if isinstance(entry, dict)), default=0.0)
    if max_count <= 0.0:
        return 0.0
    return math.log1p(count) / math.log1p(max_count)


def _novelty_bonus(profile: dict[str, object], hobby_profile: dict[str, object]) -> float:
    """Reward rare/novel items. Inverse of popularity penalty."""
    return 1.0 - _popularity_penalty(profile, hobby_profile)


def _mismatch_penalty(profile: dict[str, object], context: PersonContext) -> float:
    fields = {
        "age_group": context.age_group,
        "occupation": context.occupation,
        "sex": context.sex,
    }
    penalties = [
        1.0 - fit
        for field, value in fields.items()
        if value and _has_distribution(profile, field) and (fit := _distribution_fit(profile, field, value)) < 0.05
    ]
    if not penalties:
        return 0.0
    return sum(penalties) / len(fields)


def _has_distribution(profile: dict[str, object], field: str) -> bool:
    distributions = profile.get("distributions", {})
    if not isinstance(distributions, dict):
        return False
    distribution = distributions.get(field, {})
    return isinstance(distribution, dict) and bool(distribution)


def _train_popularity(entry: dict[object, object]) -> float:
    value = entry.get("train_popularity", 0.0)
    return _safe_float(value)


def _safe_float(value: object) -> float:
    return float(value) if isinstance(value, int | float | str) else 0.0


def _persona_text_fit(context: PersonContext, candidate: HobbyCandidate) -> float:
    text = " ".join([context.persona_text, context.professional_text, context.sports_text, context.arts_text, context.travel_text, context.culinary_text, context.family_text])
    return 1.0 if candidate.hobby_name and candidate.hobby_name in text else 0.0


def _hobby_category(hobby_name: str, hobby_taxonomy: dict[str, object] | None) -> str:
    if hobby_taxonomy is None:
        return ""
    taxonomy_map = hobby_taxonomy.get("taxonomy", {})
    if isinstance(taxonomy_map, dict):
        entry = taxonomy_map.get(hobby_name, {})
        if isinstance(entry, dict):
            return str(entry.get("category", ""))
    rules = hobby_taxonomy.get("rules", [])
    if isinstance(rules, list):
        for rule in rules:
            if isinstance(rule, dict) and rule.get("canonical_hobby") == hobby_name:
                tax = rule.get("taxonomy", {})
                if isinstance(tax, dict):
                    return str(tax.get("category", ""))
    return ""


def _diversity_aware_sort(
    scored: list[RerankedCandidate],
    hobby_taxonomy: dict[str, object] | None,
    weights: RerankerWeights,
) -> list[RerankedCandidate]:
    if weights.category_diversity_reward <= 0.0 or hobby_taxonomy is None:
        return scored
    selected: list[RerankedCandidate] = []
    remaining = list(scored)
    seen_categories: set[str] = set()
    while remaining:
        best_idx = -1
        best_adjusted = float("-inf")
        best_is_new_category = False
        best_hobby_id = -1
        for index, candidate in enumerate(remaining):
            category = _hobby_category(candidate.hobby_name, hobby_taxonomy)
            is_new_category = bool(category) and category not in seen_categories
            diversity_bonus = 1.0 if is_new_category else 0.0
            adjusted_score = candidate.final_score + weights.category_diversity_reward * diversity_bonus
            if (
                adjusted_score > best_adjusted
                or (adjusted_score == best_adjusted and is_new_category and not best_is_new_category)
                or (
                    adjusted_score == best_adjusted
                    and is_new_category == best_is_new_category
                    and candidate.hobby_id < best_hobby_id
                )
            ):
                best_idx = index
                best_adjusted = adjusted_score
                best_is_new_category = is_new_category
                best_hobby_id = candidate.hobby_id
        if best_idx < 0:
            break
        winner = remaining.pop(best_idx)
        winner_category = _hobby_category(winner.hobby_name, hobby_taxonomy)
        if winner_category:
            seen_categories.add(winner_category)
        selected.append(winner)
    return selected


def _stage1_only(candidate: HobbyCandidate) -> RerankedCandidate:
    stage1_score = max(candidate.source_scores.values()) if candidate.source_scores else 0.0
    return RerankedCandidate(
        hobby_id=candidate.hobby_id,
        hobby_name=candidate.hobby_name,
        final_score=stage1_score,
        stage1_score=stage1_score,
        features={"stage1_score_only": 1.0},
        source_scores=candidate.source_scores,
        reason_features={"fallback": "stage1_score_only"},
    )
