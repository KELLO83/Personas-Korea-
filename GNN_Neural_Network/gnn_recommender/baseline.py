from __future__ import annotations

from collections import Counter

from .data import PersonContext
from .metrics import summarize_ranking_metrics
from .recommend import Candidate


def build_popularity_counts(train_edges: list[tuple[int, int]]) -> Counter[int]:
    return Counter(hobby_id for _, hobby_id in train_edges)


def build_cooccurrence_counts(train_edges: list[tuple[int, int]]) -> dict[int, Counter[int]]:
    return _build_cooccurrence_counts(train_edges)


def popularity_recommendations(
    train_edges: list[tuple[int, int]],
    person_ids: list[int],
    known_by_person: dict[int, set[int]],
    top_k: int,
) -> dict[int, list[int]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    counts = Counter(hobby_id for _, hobby_id in train_edges)
    ranked_hobbies = [hobby_id for hobby_id, _ in counts.most_common()]
    output: dict[int, list[int]] = {}
    for person_id in person_ids:
        known = known_by_person.get(person_id, set())
        output[person_id] = [hobby_id for hobby_id in ranked_hobbies if hobby_id not in known][:top_k]
    return output


def popularity_candidate_provider(
    train_edges: list[tuple[int, int]],
    person_id: int,
    known_hobbies: set[int],
    top_k: int,
    popularity_counts: Counter[int] | None = None,
) -> list[Candidate]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    counts = popularity_counts or build_popularity_counts(train_edges)
    candidates: list[Candidate] = []
    for rank, (hobby_id, count) in enumerate(counts.most_common(), start=1):
        if hobby_id in known_hobbies:
            continue
        candidates.append(
            Candidate(
                hobby_id=hobby_id,
                provider="popularity",
                raw_score=float(count),
                rank=rank,
                reason_features={"train_popularity": count, "person_id": person_id},
                source_scores={"popularity": float(count)},
            )
        )
        if len(candidates) >= top_k:
            break
    return candidates


def segment_popularity_candidate_provider(
    hobby_profile: dict[str, object] | None,
    context: PersonContext | None,
    known_hobbies: set[int],
    top_k: int,
) -> list[Candidate]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if context is None or hobby_profile is None or hobby_profile.get("source") != "train_split_only":
        return []
    hobbies = hobby_profile.get("hobbies", {})
    if not isinstance(hobbies, dict):
        return []
    scored: list[tuple[float, int, dict[str, float], float]] = []
    for entry in hobbies.values():
        if not isinstance(entry, dict):
            continue
        hobby_id_value = entry.get("hobby_id")
        if not isinstance(hobby_id_value, int) or hobby_id_value in known_hobbies:
            continue
        field_scores = _segment_field_scores(entry, context)
        popularity = _safe_float(entry.get("train_popularity", 0.0))
        score = sum(field_scores.values()) + (popularity * 1e-6)
        if score <= 0.0:
            continue
        scored.append((score, hobby_id_value, field_scores, popularity))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        Candidate(
            hobby_id=hobby_id,
            provider="segment_popularity",
            raw_score=score,
            rank=rank,
            reason_features={"segment_scores": field_scores, "train_popularity": popularity},
            source_scores={"segment_popularity": score},
        )
        for rank, (score, hobby_id, field_scores, popularity) in enumerate(scored[:top_k], start=1)
    ]


def _segment_field_scores(entry: dict[object, object], context: PersonContext) -> dict[str, float]:
    distributions = entry.get("distributions", {})
    if not isinstance(distributions, dict):
        return {}
    fields = {
        "age_group": context.age_group,
        "occupation": context.occupation,
        "province": context.province,
        "district": context.district,
        "family_type": context.family_type,
        "housing_type": context.housing_type,
        "education_level": context.education_level,
    }
    return {
        field: _distribution_ratio(distributions.get(field, {}), value)
        for field, value in fields.items()
        if value and _distribution_ratio(distributions.get(field, {}), value) > 0.0
    }


def _distribution_ratio(distribution: object, value: str) -> float:
    if not isinstance(distribution, dict) or not distribution:
        return 0.0
    total = sum(_safe_float(count) for count in distribution.values())
    return _safe_float(distribution.get(value, 0.0)) / total if total else 0.0


def _safe_float(value: object) -> float:
    return float(value) if isinstance(value, int | float | str) else 0.0


def cooccurrence_recommendations(
    train_edges: list[tuple[int, int]],
    person_ids: list[int],
    known_by_person: dict[int, set[int]],
    top_k: int,
) -> dict[int, list[int]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    hobbies_by_person: dict[int, set[int]] = {}
    for person_id, hobby_id in train_edges:
        hobbies_by_person.setdefault(person_id, set()).add(hobby_id)

    cooccurrence: dict[int, Counter[int]] = {}
    for hobbies in hobbies_by_person.values():
        for source in hobbies:
            target_counter = cooccurrence.setdefault(source, Counter())
            for target in hobbies:
                if source != target:
                    target_counter[target] += 1

    output: dict[int, list[int]] = {}
    fallback = popularity_recommendations(train_edges, person_ids, known_by_person, top_k)
    for person_id in person_ids:
        known = known_by_person.get(person_id, set())
        scores: Counter[int] = Counter()
        for hobby_id in known:
            scores.update(cooccurrence.get(hobby_id, Counter()))
        ranked = [hobby_id for hobby_id, _ in scores.most_common() if hobby_id not in known]
        if len(ranked) < top_k:
            ranked.extend(hobby_id for hobby_id in fallback[person_id] if hobby_id not in set(ranked))
        output[person_id] = ranked[:top_k]
    return output


def cooccurrence_candidate_provider(
    train_edges: list[tuple[int, int]],
    person_id: int,
    known_hobbies: set[int],
    top_k: int,
    cooccurrence_counts: dict[int, Counter[int]] | None = None,
) -> list[Candidate]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    cooccurrence = cooccurrence_counts or _build_cooccurrence_counts(train_edges)
    scores: Counter[int] = Counter()
    for hobby_id in known_hobbies:
        scores.update(cooccurrence.get(hobby_id, Counter()))
    candidates: list[Candidate] = []
    for rank, (hobby_id, count) in enumerate(scores.most_common(), start=1):
        if hobby_id in known_hobbies:
            continue
        candidates.append(
            Candidate(
                hobby_id=hobby_id,
                provider="cooccurrence",
                raw_score=float(count),
                rank=rank,
                reason_features={"cooccurrence_count": count, "known_hobby_count": len(known_hobbies), "person_id": person_id},
                source_scores={"cooccurrence": float(count)},
            )
        )
        if len(candidates) >= top_k:
            break
    return candidates


def _build_cooccurrence_counts(train_edges: list[tuple[int, int]]) -> dict[int, Counter[int]]:
    hobbies_by_person: dict[int, set[int]] = {}
    for person_id, hobby_id in train_edges:
        hobbies_by_person.setdefault(person_id, set()).add(hobby_id)

    cooccurrence: dict[int, Counter[int]] = {}
    for hobbies in hobbies_by_person.values():
        for source in hobbies:
            target_counter = cooccurrence.setdefault(source, Counter())
            for target in hobbies:
                if source != target:
                    target_counter[target] += 1
    return cooccurrence


def baseline_ranking_metrics(
    train_edges: list[tuple[int, int]],
    target_edges: list[tuple[int, int]],
    known_by_person: dict[int, set[int]],
    top_k_values: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    truth: dict[int, set[int]] = {}
    for person_id, hobby_id in target_edges:
        truth.setdefault(person_id, set()).add(hobby_id)
    if not truth:
        empty = {f"{name}@{k}": 0.0 for k in top_k_values for name in ("recall", "ndcg", "hit_rate")}
        return {"popularity": dict(empty), "cooccurrence": dict(empty)}
    person_ids = list(truth)
    max_k = max(top_k_values)
    popularity = popularity_recommendations(train_edges, person_ids, known_by_person, max_k)
    cooccurrence = cooccurrence_recommendations(train_edges, person_ids, known_by_person, max_k)
    return {
        "popularity": summarize_ranking_metrics(truth, popularity, top_k_values),
        "cooccurrence": summarize_ranking_metrics(truth, cooccurrence, top_k_values),
    }


def build_bm25_itemknn_counts(
    train_edges: list[tuple[int, int]],
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[int, dict[int, float]]:
    hobbies_by_person: dict[int, set[int]] = {}
    for person_id, hobby_id in train_edges:
        hobbies_by_person.setdefault(person_id, set()).add(hobby_id)
        
    num_persons = len(hobbies_by_person)
    hobby_doc_freq = Counter(hobby_id for _, hobby_id in train_edges)
    
    import math
    idf: dict[int, float] = {}
    for hobby_id, df in hobby_doc_freq.items():
        idf[hobby_id] = math.log(((num_persons - df + 0.5) / (df + 0.5)) + 1.0)
        
    bm25_scores: dict[int, dict[int, float]] = {}
    avg_item_deg = sum(hobby_doc_freq.values()) / len(hobby_doc_freq) if hobby_doc_freq else 1.0
    
    cooc = _build_cooccurrence_counts(train_edges)
    
    for source_id, targets in cooc.items():
        source_degree = hobby_doc_freq[source_id]
        scores: dict[int, float] = {}
        for target_id, count in targets.items():
            target_idf = idf.get(target_id, 0.0)
            numerator = count * (k1 + 1)
            denominator = count + k1 * (1 - b + b * (source_degree / avg_item_deg))
            scores[target_id] = target_idf * (numerator / denominator)
        bm25_scores[source_id] = scores
        
    return bm25_scores


def bm25_itemknn_candidate_provider(
    train_edges: list[tuple[int, int]],
    person_id: int,
    known_hobbies: set[int],
    top_k: int,
    bm25_counts: dict[int, dict[int, float]] | None = None,
) -> list[Candidate]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    bm25 = bm25_counts or build_bm25_itemknn_counts(train_edges)
    scores: Counter[int] = Counter()
    for hobby_id in known_hobbies:
        if hobby_id in bm25:
            for target_id, score in bm25[hobby_id].items():
                scores[target_id] += score
                
    candidates: list[Candidate] = []
    for rank, (hobby_id, score) in enumerate(scores.most_common(), start=1):
        if hobby_id in known_hobbies:
            continue
        candidates.append(
            Candidate(
                hobby_id=hobby_id,
                provider="bm25_itemknn",
                raw_score=score,
                rank=rank,
                reason_features={"bm25_score": score, "known_hobby_count": len(known_hobbies), "person_id": person_id},
                source_scores={"bm25_itemknn": score},
            )
        )
        if len(candidates) >= top_k:
            break
    return candidates
