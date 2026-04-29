from __future__ import annotations

import math


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


def summarize_ranking_metrics(
    truth_by_person: dict[int, set[int]],
    recommendations_by_person: dict[int, list[int]],
    top_k_values: tuple[int, ...],
    num_total_items: int | None = None,
    item_popularity: dict[int, int] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
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
                # Inverse Popularity: -log2(p_i)
                user_novelty = sum(
                    -math.log2((item_popularity.get(item, 0) + 1e-9) / total_edges)
                    for item in recs
                ) / len(recs)
                novelties.append(user_novelty)
            metrics[f"novelty@{k}"] = sum(novelties) / len(novelties) if novelties else 0.0

    return metrics
