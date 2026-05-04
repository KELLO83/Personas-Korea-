from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.data import PersonContext
from GNN_Neural_Network.gnn_recommender.metrics import summarize_ranking_metrics
from GNN_Neural_Network.gnn_recommender.recommend import Candidate


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
        source_idf = idf.get(source_id, 0.0)
        scores: dict[int, float] = {}
        for target_id, count in targets.items():
            target_degree = hobby_doc_freq[target_id]
            numerator = count * (k1 + 1)
            denominator = count + k1 * (1 - b + b * (target_degree / avg_item_deg))
            scores[target_id] = source_idf * (numerator / denominator)
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
    scores: dict[int, float] = {}
    for hobby_id in known_hobbies:
        if hobby_id in bm25:
            for target_id, score in bm25[hobby_id].items():
                scores[target_id] = scores.get(target_id, 0.0) + score

    candidates: list[Candidate] = []
    ranked_scores = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    for rank, (hobby_id, score) in enumerate(ranked_scores, start=1):
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


# ---------------------------------------------------------------------------
# IDF-weighted cooccurrence provider
#   cooccurrence(i,j) * IDF(j)
#   IDF(j) = log((N + 1) / (1 + df(j)))
#   Down-weights hobbies that appear in many user profiles.
# ---------------------------------------------------------------------------

def build_idf_weighted_cooccurrence_counts(
    train_edges: list[tuple[int, int]],
) -> dict[int, dict[int, float]]:
    """Compute IDF-weighted cooccurrence:  cooc(source, target) * IDF(target)."""
    import math

    hobbies_by_person: dict[int, set[int]] = {}
    for person_id, hobby_id in train_edges:
        hobbies_by_person.setdefault(person_id, set()).add(hobby_id)

    num_persons = len(hobbies_by_person)
    hobby_df: Counter[int] = Counter()
    for hobbies in hobbies_by_person.values():
        for hobby_id in hobbies:
            hobby_df[hobby_id] += 1

    idf: dict[int, float] = {}
    for hobby_id, df in hobby_df.items():
        idf[hobby_id] = math.log((num_persons + 1) / (1 + df))

    cooc = _build_cooccurrence_counts(train_edges)
    weighted: dict[int, dict[int, float]] = {}
    for source_id, targets in cooc.items():
        weighted[source_id] = {
            target_id: count * idf.get(target_id, 0.0)
            for target_id, count in targets.items()
        }
    return weighted


def idf_weighted_cooccurrence_provider(
    train_edges: list[tuple[int, int]],
    person_id: int,
    known_hobbies: set[int],
    top_k: int,
    idf_cooc_counts: dict[int, dict[int, float]] | None = None,
) -> list[Candidate]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    counts = idf_cooc_counts or build_idf_weighted_cooccurrence_counts(train_edges)
    scores: dict[int, float] = {}
    for hobby_id in known_hobbies:
        if hobby_id in counts:
            for target_id, score in counts[hobby_id].items():
                scores[target_id] = scores.get(target_id, 0.0) + score

    candidates: list[Candidate] = []
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    for rank, (hobby_id, score) in enumerate(ranked, start=1):
        if hobby_id in known_hobbies:
            continue
        candidates.append(
            Candidate(
                hobby_id=hobby_id,
                provider="idf_cooccurrence",
                raw_score=score,
                rank=rank,
                reason_features={"idf_cooccurrence_score": score, "known_hobby_count": len(known_hobbies), "person_id": person_id},
                source_scores={"idf_cooccurrence": score},
            )
        )
        if len(candidates) >= top_k:
            break
    return candidates


# ---------------------------------------------------------------------------
# Popularity-capped cooccurrence provider
#   cooccurrence(i,j) / log(1 + popularity(j))
#   Simple downweighting of popular items without zeroing out signal.
# ---------------------------------------------------------------------------

def build_pop_capped_cooccurrence_counts(
    train_edges: list[tuple[int, int]],
) -> dict[int, dict[int, float]]:
    """Compute cooccurrence divided by log(1 + item popularity)."""
    import math

    popularity = build_popularity_counts(train_edges)
    cooc = _build_cooccurrence_counts(train_edges)
    weighted: dict[int, dict[int, float]] = {}
    for source_id, targets in cooc.items():
        weighted[source_id] = {
            target_id: count / math.log(1 + popularity.get(target_id, 1))
            for target_id, count in targets.items()
        }
    return weighted


def pop_capped_cooccurrence_provider(
    train_edges: list[tuple[int, int]],
    person_id: int,
    known_hobbies: set[int],
    top_k: int,
    pop_capped_counts: dict[int, dict[int, float]] | None = None,
) -> list[Candidate]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    counts = pop_capped_counts or build_pop_capped_cooccurrence_counts(train_edges)
    scores: dict[int, float] = {}
    for hobby_id in known_hobbies:
        if hobby_id in counts:
            for target_id, score in counts[hobby_id].items():
                scores[target_id] = scores.get(target_id, 0.0) + score

    candidates: list[Candidate] = []
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    for rank, (hobby_id, score) in enumerate(ranked, start=1):
        if hobby_id in known_hobbies:
            continue
        candidates.append(
            Candidate(
                hobby_id=hobby_id,
                provider="pop_capped_cooccurrence",
                raw_score=score,
                rank=rank,
                reason_features={"pop_capped_score": score, "known_hobby_count": len(known_hobbies), "person_id": person_id},
                source_scores={"pop_capped_cooccurrence": score},
            )
        )
        if len(candidates) >= top_k:
            break
    return candidates


# ---------------------------------------------------------------------------
# Jaccard item-item similarity provider
#   J(A, B) = |A ∩ B| / |A ∪ B| where A, B are sets of persons with each hobby.
#   Aggregated per user: sum of Jaccard(known_hobby, candidate) over known hobbies.
# ---------------------------------------------------------------------------

def build_jaccard_itemknn_counts(
    train_edges: list[tuple[int, int]],
) -> dict[int, dict[int, float]]:
    """Compute Jaccard similarity between every pair of hobbies based on person overlap."""
    hobbies_by_person: dict[int, set[int]] = {}
    for person_id, hobby_id in train_edges:
        hobbies_by_person.setdefault(person_id, set()).add(hobby_id)

    person_sets: dict[int, set[int]] = {}
    for person_id, hobbies in hobbies_by_person.items():
        for hobby_id in hobbies:
            person_sets.setdefault(hobby_id, set()).add(person_id)

    all_hobby_ids = list(person_sets.keys())
    jaccard: dict[int, dict[int, float]] = {}
    for i, source_id in enumerate(all_hobby_ids):
        source_set = person_sets[source_id]
        jaccard[source_id] = {}
        for target_id in all_hobby_ids:
            if source_id == target_id:
                continue
            target_set = person_sets[target_id]
            intersection = len(source_set & target_set)
            union = len(source_set | target_set)
            if union > 0 and intersection > 0:
                jaccard[source_id][target_id] = intersection / union

    return jaccard


def jaccard_itemknn_candidate_provider(
    train_edges: list[tuple[int, int]],
    person_id: int,
    known_hobbies: set[int],
    top_k: int,
    jaccard_counts: dict[int, dict[int, float]] | None = None,
) -> list[Candidate]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    counts = jaccard_counts or build_jaccard_itemknn_counts(train_edges)
    scores: dict[int, float] = {}
    for hobby_id in known_hobbies:
        if hobby_id in counts:
            for target_id, score in counts[hobby_id].items():
                scores[target_id] = scores.get(target_id, 0.0) + score

    candidates: list[Candidate] = []
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    for rank, (hobby_id, score) in enumerate(ranked, start=1):
        if hobby_id in known_hobbies:
            continue
        candidates.append(
            Candidate(
                hobby_id=hobby_id,
                provider="jaccard_itemknn",
                raw_score=score,
                rank=rank,
                reason_features={"jaccard_score": score, "known_hobby_count": len(known_hobbies), "person_id": person_id},
                source_scores={"jaccard_itemknn": score},
            )
        )
        if len(candidates) >= top_k:
            break
    return candidates


# ---------------------------------------------------------------------------
# PMI (Pointwise Mutual Information) item-item provider
#   PMI(i, j) = log(P(i,j) / (P(i) * P(j)))
#   P(i,j) = cooc(i,j) / total_persons
#   P(i) = df(i) / total_persons
#   Aggregated per user: sum of PMI(known_hobby, candidate) over known hobbies.
# ---------------------------------------------------------------------------

def build_pmi_itemknn_counts(
    train_edges: list[tuple[int, int]],
    positive_pmi: bool = True,
) -> dict[int, dict[int, float]]:
    """Compute PMI (or PPMI) between every pair of hobbies.

    Args:
        positive_pmi: If True, clamp negative PMI values to 0 (PPMI).
    """
    import math

    hobbies_by_person: dict[int, set[int]] = {}
    for person_id, hobby_id in train_edges:
        hobbies_by_person.setdefault(person_id, set()).add(hobby_id)

    num_persons = len(hobbies_by_person)
    hobby_df: Counter[int] = Counter()
    for hobbies in hobbies_by_person.values():
        for hobby_id in hobbies:
            hobby_df[hobby_id] += 1

    cooc = _build_cooccurrence_counts(train_edges)
    pmi: dict[int, dict[int, float]] = {}
    for source_id, targets in cooc.items():
        pmi[source_id] = {}
        p_source = hobby_df[source_id] / num_persons
        for target_id, count in targets.items():
            p_joint = count / num_persons
            p_target = hobby_df[target_id] / num_persons
            denominator = p_source * p_target
            if denominator > 0.0 and p_joint > 0.0:
                val = math.log(p_joint / denominator)
                if positive_pmi and val < 0.0:
                    val = 0.0
                pmi[source_id][target_id] = val
    return pmi


def pmi_itemknn_candidate_provider(
    train_edges: list[tuple[int, int]],
    person_id: int,
    known_hobbies: set[int],
    top_k: int,
    pmi_counts: dict[int, dict[int, float]] | None = None,
) -> list[Candidate]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    counts = pmi_counts or build_pmi_itemknn_counts(train_edges)
    scores: dict[int, float] = {}
    for hobby_id in known_hobbies:
        if hobby_id in counts:
            for target_id, score in counts[hobby_id].items():
                scores[target_id] = scores.get(target_id, 0.0) + score

    candidates: list[Candidate] = []
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    for rank, (hobby_id, score) in enumerate(ranked, start=1):
        if hobby_id in known_hobbies:
            continue
        candidates.append(
            Candidate(
                hobby_id=hobby_id,
                provider="pmi_itemknn",
                raw_score=score,
                rank=rank,
                reason_features={"pmi_score": score, "known_hobby_count": len(known_hobbies), "person_id": person_id},
                source_scores={"pmi_itemknn": score},
            )
        )
        if len(candidates) >= top_k:
            break
    return candidates
