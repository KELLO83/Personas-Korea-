from __future__ import annotations

import math
from collections import defaultdict


def recall_at_k(relevant: set[int], recommended: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def hit_rate_at_k(relevant: set[int], recommended: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if set(recommended[:k]) & relevant else 0.0


def ndcg_at_k(relevant: set[int], recommended: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for index, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def intra_list_diversity_at_k(
    recommendations_by_person: dict[int, list[int]],
    hobby_categories: dict[int, str],
    k: int,
) -> float:
    person_diversities: list[float] = []
    for recs in recommendations_by_person.values():
        top_k = recs[:k]
        if len(top_k) < 2:
            continue
        categories = [hobby_categories.get(item, "") for item in top_k]
        pair_count = 0
        diff_count = 0
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                pair_count += 1
                if categories[i] != categories[j]:
                    diff_count += 1
        person_diversities.append(diff_count / pair_count if pair_count else 0.0)
    return sum(person_diversities) / len(person_diversities) if person_diversities else 0.0


def oracle_recall_at_k(
    truth_by_person: dict[int, set[int]],
    candidate_pool_by_person: dict[int, list[int]],
    k: int,
) -> float:
    recalls: list[float] = []
    for person_id, relevant in truth_by_person.items():
        if not relevant:
            continue
        pool = candidate_pool_by_person.get(person_id, [])
        pool_set = set(pool)
        hits = min(len(relevant & pool_set), k)
        recalls.append(hits / len(relevant))
    return sum(recalls) / len(recalls) if recalls else 0.0


def per_segment_metrics(
    truth_by_person: dict[int, set[int]],
    recommendations_by_person: dict[int, list[int]],
    person_segments: dict[int, dict[str, str]],
    k: int,
) -> dict[str, object]:
    segment_fields = ("age_group", "sex")
    result: dict[str, object] = {}
    recall_gap: dict[str, float] = {}

    for field in segment_fields:
        groups: dict[str, list[float]] = defaultdict(list)
        for person_id, relevant in truth_by_person.items():
            if not relevant:
                continue
            seg = person_segments.get(person_id, {})
            group_value = seg.get(field, "unknown")
            r = recall_at_k(relevant, recommendations_by_person.get(person_id, []), k)
            groups[group_value].append(r)

        field_result: dict[str, dict[str, float]] = {}
        group_recalls: list[float] = []
        for group_name, recall_values in groups.items():
            avg = sum(recall_values) / len(recall_values)
            field_result[group_name] = {"recall": avg, "count": len(recall_values)}
            group_recalls.append(avg)

        result[field] = field_result
        recall_gap[field] = (max(group_recalls) - min(group_recalls)) if len(group_recalls) >= 2 else 0.0

    result["recall_gap"] = recall_gap
    return result


def summarize_ranking_metrics(
    truth_by_person: dict[int, set[int]],
    recommendations_by_person: dict[int, list[int]],
    top_k_values: tuple[int, ...],
    num_total_items: int | None = None,
    item_popularity: dict[int, int] | None = None,
    hobby_categories: dict[int, str] | None = None,
    candidate_pool_by_person: dict[int, list[int]] | None = None,
    person_segments: dict[int, dict[str, str]] | None = None,
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    persons = [person_id for person_id, relevant in truth_by_person.items() if relevant]
    if not persons:
        default_names = ("recall", "ndcg", "hit_rate", "catalog_coverage", "novelty")
        return {f"{name}@{k}": 0.0 for k in top_k_values for name in default_names}

    total_edges = sum(item_popularity.values()) if item_popularity else 1

    for k in top_k_values:
        recalls = [recall_at_k(truth_by_person[p], recommendations_by_person.get(p, []), k) for p in persons]
        ndcgs = [ndcg_at_k(truth_by_person[p], recommendations_by_person.get(p, []), k) for p in persons]
        hits = [hit_rate_at_k(truth_by_person[p], recommendations_by_person.get(p, []), k) for p in persons]

        metrics[f"recall@{k}"] = sum(recalls) / len(recalls)
        metrics[f"ndcg@{k}"] = sum(ndcgs) / len(ndcgs)
        metrics[f"hit_rate@{k}"] = sum(hits) / len(hits)

        if num_total_items is not None and num_total_items > 0:
            recommended_items = {
                item for p in persons for item in recommendations_by_person.get(p, [])[:k]
            }
            metrics[f"catalog_coverage@{k}"] = len(recommended_items) / num_total_items

        if item_popularity is not None:
            novelties = []
            for p in persons:
                recs = recommendations_by_person.get(p, [])[:k]
                if not recs:
                    continue
                user_novelty = sum(
                    -math.log2((item_popularity.get(item, 0) + 1e-9) / total_edges)
                    for item in recs
                ) / len(recs)
                novelties.append(user_novelty)
            metrics[f"novelty@{k}"] = sum(novelties) / len(novelties) if novelties else 0.0

        if hobby_categories is not None:
            metrics[f"intra_list_diversity@{k}"] = intra_list_diversity_at_k(
                recommendations_by_person, hobby_categories, k,
            )

        if candidate_pool_by_person is not None:
            metrics[f"oracle_recall@{k}"] = oracle_recall_at_k(
                truth_by_person, candidate_pool_by_person, k,
            )

    if person_segments is not None:
        max_k = max(top_k_values)
        metrics["per_segment"] = per_segment_metrics(
            truth_by_person, recommendations_by_person, person_segments, max_k,
        )

    return metrics
