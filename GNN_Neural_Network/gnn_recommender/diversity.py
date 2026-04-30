"""MMR (Maximal Marginal Relevance) diversity reordering module.

Phase 3 implementation: category one-hot embedding fallback.
KURE embedding replacement planned for Phase 5.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.linalg import norm


def compute_hobby_embeddings(
    hobby_names: list[str],
    hobby_taxonomy: dict[str, object] | None = None,
) -> np.ndarray:
    """Compute hobby embeddings from taxonomy category one-hot encoding.

    Phase 3 fallback: uses hobby_taxonomy.json category field to build
    one-hot embeddings. Each unique category becomes one dimension.

    Phase 5 will replace this with KURE-v1 dense embeddings.

    Args:
        hobby_names: List of canonical hobby names to embed.
        hobby_taxonomy: Loaded hobby_taxonomy.json dict with
            "rules" list containing canonical_hobby -> taxonomy.category,
            or "taxonomy" dict mapping hobby_name -> {"category": ...}.

    Returns:
        ndarray of shape (n_hobbies, n_categories) with L2-normalized rows.
        Hobbies without a category get a zero vector.
    """
    if not hobby_names:
        return np.empty((0, 0), dtype=np.float32)

    
    category_map: dict[str, int] = {}
    hobby_categories: list[str | None] = []

    for name in hobby_names:
        cat = _get_category(name, hobby_taxonomy)
        hobby_categories.append(cat)
        if cat is not None and cat not in category_map:
            category_map[cat] = len(category_map)

    if not category_map:
        
        return np.zeros((len(hobby_names), 1), dtype=np.float32)

    n_cats = len(category_map)
    embeddings = np.zeros((len(hobby_names), n_cats), dtype=np.float32)

    for idx, cat in enumerate(hobby_categories):
        if cat is not None and cat in category_map:
            embeddings[idx, category_map[cat]] = 1.0

    
    norms = norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embeddings = embeddings / norms

    return embeddings


def mmr_rerank(
    hobby_ids: list[int],
    relevance_scores: np.ndarray,
    embeddings: np.ndarray,
    lambda_param: float = 0.7,
    top_k: int = 10,
) -> list[int]:
    """Greedy MMR (Maximal Marginal Relevance) selection.

    MMR(i) = λ * relevance(i) - (1 - λ) * max_similarity_to_selected(i)

    Args:
        hobby_ids: Candidate hobby IDs in ranker score order.
        relevance_scores: Ranker scores for each candidate.
        embeddings: Pre-computed hobby embeddings (n_candidates, dim).
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).
            Higher λ = more relevance, lower λ = more diversity.
        top_k: Number of items to select.

    Returns:
        List of selected hobby IDs in MMR order.
    """
    if len(hobby_ids) == 0:
        return []

    n = len(hobby_ids)
    k = min(top_k, n)

    sim_matrix = embeddings @ embeddings.T

    selected_indices: list[int] = []
    remaining = set(range(n))

    for _ in range(k):
        best_idx = -1
        best_score = float("-inf")

        for i in remaining:
            relevance = float(relevance_scores[i])

            if selected_indices:
                max_sim = float(max(sim_matrix[i, j] for j in selected_indices))
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim

            
            if mmr_score > best_score or (
                mmr_score == best_score
                and (best_idx == -1 or relevance > float(relevance_scores[best_idx]))
            ):
                best_score = mmr_score
                best_idx = i

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    return [hobby_ids[i] for i in selected_indices]


def compute_intra_list_diversity(
    recommended_hobby_names: list[str],
    hobby_taxonomy: dict[str, object] | None = None,
    embeddings: np.ndarray | None = None,
) -> float:
    """Compute intra-list diversity as 1 - mean_pairwise_similarity.

    Uses category one-hot embeddings (fallback) or pre-computed embeddings.

    Args:
        recommended_hobby_names: List of recommended hobby names.
        hobby_taxonomy: For fallback category one-hot embedding.
        embeddings: Pre-computed embeddings (n_items, dim). If provided,
            hobby_taxonomy is ignored.

    Returns:
        Float in [0, 1]. Higher = more diverse.
        Returns 1.0 for empty/single-item lists.
    """
    if len(recommended_hobby_names) <= 1:
        return 1.0

    if embeddings is None:
        embeddings = compute_hobby_embeddings(recommended_hobby_names, hobby_taxonomy)

    if embeddings.shape[0] != len(recommended_hobby_names):
        
        names = recommended_hobby_names
        categories = [_get_category(n, hobby_taxonomy) or n for n in names]
        unique_cats = len(set(categories))
        return unique_cats / len(categories) if categories else 1.0

    
    sim_matrix = embeddings @ embeddings.T
    n = len(recommended_hobby_names)

    
    total_sim = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += float(sim_matrix[i, j])
            count += 1

    if count == 0:
        return 1.0

    mean_sim = total_sim / count
    return 1.0 - mean_sim


def mmr_rerank_with_hobbies(
    hobby_ids: list[int],
    hobby_names: list[str],
    relevance_scores: np.ndarray,
    hobby_taxonomy: dict[str, object] | None = None,
    lambda_param: float = 0.7,
    top_k: int = 10,
) -> list[int]:
    """Convenience wrapper: compute embeddings from hobby names and run MMR.

    Args:
        hobby_ids: Candidate hobby IDs.
        hobby_names: Corresponding hobby names for embedding lookup.
        relevance_scores: Ranker scores for each candidate.
        hobby_taxonomy: For category one-hot embedding.
        lambda_param: MMR trade-off parameter.
        top_k: Number of items to select.

    Returns:
        List of selected hobby IDs in MMR order.
    """
    embeddings = compute_hobby_embeddings(hobby_names, hobby_taxonomy)
    return mmr_rerank(hobby_ids, relevance_scores, embeddings, lambda_param, top_k)


def _get_category(hobby_name: str, hobby_taxonomy: dict[str, object] | None) -> str | None:
    """Extract category from hobby taxonomy for a given hobby name."""
    if hobby_taxonomy is None:
        return None

    
    taxonomy_map = hobby_taxonomy.get("taxonomy", {})
    if isinstance(taxonomy_map, dict):
        entry = taxonomy_map.get(hobby_name, {})
        if isinstance(entry, dict):
            cat = entry.get("category")
            if cat:
                return str(cat)

    
    rules = hobby_taxonomy.get("rules", [])
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if rule.get("canonical_hobby") == hobby_name:
                tax = rule.get("taxonomy", {})
                if isinstance(tax, dict):
                    cat = tax.get("category")
                    if cat:
                        return str(cat)
            
            include_kws = rule.get("include_keywords", [])
            if isinstance(include_kws, list) and hobby_name in include_kws:
                tax = rule.get("taxonomy", {})
                if isinstance(tax, dict):
                    cat = tax.get("category")
                    if cat:
                        return str(cat)

    return None