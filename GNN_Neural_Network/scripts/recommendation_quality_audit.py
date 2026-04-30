from __future__ import annotations

import argparse
import math
import random
import sys
from collections.abc import Mapping
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.baseline import (
    build_cooccurrence_counts,
    build_popularity_counts,
    build_bm25_itemknn_counts,
)
from GNN_Neural_Network.gnn_recommender.config import load_config
from GNN_Neural_Network.gnn_recommender.data import load_json, load_person_contexts, save_json
from GNN_Neural_Network.gnn_recommender.model import LightGCN, build_normalized_adjacency, choose_device
from GNN_Neural_Network.gnn_recommender.recommend import (
    compute_lightgcn_embeddings,
    merge_candidates_by_hobby,
)
from GNN_Neural_Network.gnn_recommender.rerank import build_reranker_config, merge_stage1_candidates, rerank_candidates
from GNN_Neural_Network.scripts.evaluate_reranker import (
    _expect_mapping,
    _known_from_edges,
    _load_hobby_taxonomy,
    _normalization_method,
    _provider_candidates,
    _read_indexed_edges,
    _safe_torch_load,
    _selected_stage1_provider_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage 2 Recommendation Quality (Coverage, Bias, Samples).")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def calculate_entropy(freq_dict: Mapping[Any, int]) -> float:
    total = sum(freq_dict.values())
    if total == 0:
        return 0.0
    return sum(-(value / total) * math.log2(value / total) for value in freq_dict.values() if value > 0)


def avg_popularity_rank(freq_dict: Mapping[int, int], pop_rank_map: Mapping[int, int]) -> float:
    total = sum(freq_dict.values())
    if total == 0:
        return 0.0
    fallback_rank = len(pop_rank_map) + 1
    return sum(pop_rank_map.get(hobby_id, fallback_rank) * count for hobby_id, count in freq_dict.items()) / total


def _load_hobby_category_map(configured_path: Path) -> dict[str, str]:
    payload: object = {}
    fallback_path = Path("GNN_Neural_Network/artifacts/hobby_taxonomy.json")
    paths_to_try: list[Path] = []
    for path in (configured_path, fallback_path):
        if path not in paths_to_try:
            paths_to_try.append(path)
    for path in paths_to_try:
        if path.exists():
            payload = load_json(path)
            break

    if not isinstance(payload, dict):
        return {}

    category_map: dict[str, str] = {}

    taxonomy_entries = payload.get("taxonomy", {})
    if isinstance(taxonomy_entries, dict):
        for hobby_name, entry in taxonomy_entries.items():
            if not isinstance(hobby_name, str) or not isinstance(entry, dict):
                continue
            category = entry.get("category")
            if isinstance(category, str) and category:
                category_map[hobby_name] = category
            nested_taxonomy = entry.get("taxonomy")
            if isinstance(nested_taxonomy, dict):
                nested_category = nested_taxonomy.get("category")
                if isinstance(nested_category, str) and nested_category:
                    category_map[hobby_name] = nested_category

    rules = payload.get("rules", [])
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            hobby_name = rule.get("canonical_hobby")
            taxonomy = rule.get("taxonomy")
            if not isinstance(hobby_name, str) or not isinstance(taxonomy, dict):
                continue
            category = taxonomy.get("category")
            if isinstance(category, str) and category:
                category_map[hobby_name] = category

    return category_map


def _train_popularity_map(hobby_profile: dict[str, object] | None) -> tuple[dict[str, float], float, float]:
    if not isinstance(hobby_profile, dict):
        return {}, 0.0, 0.0
    hobbies = hobby_profile.get("hobbies", {})
    if not isinstance(hobbies, dict):
        return {}, 0.0, 0.0
    popularity_map = {
        hobby_name: float(entry.get("train_popularity", 0.0))
        for hobby_name, entry in hobbies.items()
        if isinstance(hobby_name, str) and isinstance(entry, dict)
    }
    max_train_popularity = max(popularity_map.values(), default=0.0)
    total_train_popularity = sum(popularity_map.values())
    return popularity_map, max_train_popularity, total_train_popularity


def _category_for_hobby(hobby_name: str, hobby_category_map: Mapping[str, str]) -> str:
    category = hobby_category_map.get(hobby_name)
    return category if isinstance(category, str) and category else "unknown"


def _popularity_penalty_candidate(train_popularity: float, max_train_popularity: float) -> float:
    if max_train_popularity <= 0:
        return 0.0
    return math.log1p(max(train_popularity, 0.0)) / math.log1p(max_train_popularity)


def _novelty_score(train_popularity: float, total_train_popularity: float) -> float:
    if train_popularity <= 0 or total_train_popularity <= 0:
        return 0.0
    probability = train_popularity / total_train_popularity
    if probability <= 0:
        return 0.0
    return -math.log2(probability)


def _average_categories_per_person(
    recs_by_person: Mapping[int, list[int]],
    id_to_hobby: Mapping[int, str],
    hobby_category_map: Mapping[str, str],
) -> float:
    if not recs_by_person:
        return 0.0
    unique_counts = []
    for hobby_ids in recs_by_person.values():
        categories = {
            _category_for_hobby(id_to_hobby[hobby_id], hobby_category_map)
            for hobby_id in hobby_ids
            if hobby_id in id_to_hobby
        }
        unique_counts.append(len(categories))
    return sum(unique_counts) / len(unique_counts) if unique_counts else 0.0


def _long_tail_share(freq_dict: Mapping[int, int], pop_rank_map: Mapping[int, int], threshold: int) -> float:
    total = sum(freq_dict.values())
    if total == 0:
        return 0.0
    fallback_rank = len(pop_rank_map) + 1
    long_tail_count = sum(
        count
        for hobby_id, count in freq_dict.items()
        if pop_rank_map.get(hobby_id, fallback_rank) > threshold
    )
    return long_tail_count / total


def _average_novelty(
    freq_dict: Mapping[int, int],
    id_to_hobby: Mapping[int, str],
    train_popularity_by_hobby: Mapping[str, float],
    total_train_popularity: float,
) -> float:
    total = sum(freq_dict.values())
    if total == 0:
        return 0.0
    novelty_sum = 0.0
    for hobby_id, count in freq_dict.items():
        hobby_name = id_to_hobby.get(hobby_id)
        if not isinstance(hobby_name, str):
            continue
        novelty_sum += _novelty_score(train_popularity_by_hobby.get(hobby_name, 0.0), total_train_popularity) * count
    return novelty_sum / total


def _category_counts(
    freq_dict: Mapping[int, int],
    id_to_hobby: Mapping[int, str],
    hobby_category_map: Mapping[str, str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for hobby_id, count in freq_dict.items():
        hobby_name = id_to_hobby.get(hobby_id)
        if not isinstance(hobby_name, str):
            continue
        counts[_category_for_hobby(hobby_name, hobby_category_map)] += count
    return dict(counts)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    config = load_config(args.config)
    candidate_k = args.candidate_k or config.rerank.candidate_pool_size
    top_k = args.top_k
    
    checkpoint = _safe_torch_load(config.paths.checkpoint)
    person_to_id = _expect_mapping(checkpoint.get("person_to_id"), "person_to_id")
    hobby_to_id = _expect_mapping(checkpoint.get("hobby_to_id"), "hobby_to_id")
    id_to_hobby = {value: key for key, value in hobby_to_id.items()}
    id_to_person = {value: key for key, value in person_to_id.items()}
    
    num_total_hobbies = len(hobby_to_id)
    
    train_edges = _read_indexed_edges(config.paths.train_edges)
    target_edges = _read_indexed_edges(config.paths.test_edges if args.split == "test" else config.paths.validation_edges)
    train_known = _known_from_edges(train_edges)
    truth = _known_from_edges(target_edges)
    
    contexts = load_person_contexts(config.paths.person_context_csv) if config.paths.person_context_csv.exists() else {}
    hobby_profile = load_json(config.paths.hobby_profile) if config.paths.hobby_profile.exists() else None
    hobby_category_map = _load_hobby_category_map(config.paths.hobby_taxonomy)
    hobby_taxonomy = _load_hobby_taxonomy(config.paths.hobby_taxonomy, config.paths.artifact_dir)
    train_popularity_by_hobby, max_train_popularity, total_train_popularity = _train_popularity_map(
        hobby_profile if isinstance(hobby_profile, dict) else None
    )
    normalization_method = _normalization_method(config.paths.score_normalization)
    reranker_config = build_reranker_config(config.rerank.use_text_fit, config.rerank.weights)

    device = choose_device(config.train.device)
    model = LightGCN(
        num_persons=int(checkpoint["num_persons"]),
        num_hobbies=int(checkpoint["num_hobbies"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        num_layers=int(checkpoint["num_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    adjacency = build_normalized_adjacency(model.num_persons, model.num_hobbies, train_edges, device)
    person_embeddings, hobby_embeddings = compute_lightgcn_embeddings(model, adjacency)
    
    popularity_counts = build_popularity_counts(train_edges)
    cooccurrence_counts = build_cooccurrence_counts(train_edges)
    bm25_counts = build_bm25_itemknn_counts(train_edges)
    
    pop_rank_map = {
        h_id: rank 
        for rank, (h_id, _) in enumerate(popularity_counts.most_common(), start=1)
    }

    # Tracking Structures
    stage1_recommended_set = set()
    stage2_recommended_set = set()
    stage2_recs_by_person: dict[int, list[int]] = {}
    stage1_recs_by_person: dict[int, list[int]] = {}
    
    provider_flags_counter = Counter()
    
    # Segment tracking (Age Group, Occupation)
    top_hobbies_by_age: dict[str, Counter[int]] = defaultdict(Counter)
    top_hobbies_by_occupation: dict[str, Counter[int]] = defaultdict(Counter)
    popularity_ranks_by_age: dict[str, list[int]] = defaultdict(list)
    popularity_ranks_by_occupation: dict[str, list[int]] = defaultdict(list)
    
    # Select random sample for qualitative review
    eval_person_ids = list(truth.keys())
    sample_ids = set(random.sample(eval_person_ids, min(args.sample_size, len(eval_person_ids))))
    sample_reviews = []

    for person_id in tqdm(eval_person_ids, desc=f"quality audit ({args.split})"):
        context = contexts.get(id_to_person.get(person_id, ""))
        
        provider_candidates = _provider_candidates(
            model=model,
            adjacency=adjacency,
            train_edges=train_edges,
            person_id=person_id,
            known=train_known.get(person_id, set()),
            candidate_k=candidate_k,
            chunk_size=config.eval.score_chunk_size,
            device=device,
            normalization_method=normalization_method,
            hobby_profile=hobby_profile if isinstance(hobby_profile, dict) else None,
            context=context,
            person_embeddings=person_embeddings,
            hobby_embeddings=hobby_embeddings,
            popularity_counts=popularity_counts,
            cooccurrence_counts=cooccurrence_counts,
            bm25_counts=bm25_counts,
        )
        
        selected_stage1_candidates = _selected_stage1_provider_candidates(provider_candidates)
        merged_stage1 = merge_candidates_by_hobby(selected_stage1_candidates, candidate_k)
        
        stage1_top_k = [c.hobby_id for c in merged_stage1[:top_k]]
        stage1_recs_by_person[person_id] = stage1_top_k
        stage1_recommended_set.update(stage1_top_k)
        
        hobby_candidates = merge_stage1_candidates(merged_stage1, id_to_hobby)
        known_names = {id_to_hobby[hobby_id] for hobby_id in train_known.get(person_id, set()) if hobby_id in id_to_hobby}
        
        reranked = rerank_candidates(
            context,
            hobby_candidates,
            hobby_profile if isinstance(hobby_profile, dict) else None,
            known_names,
            reranker_config,
            hobby_taxonomy=hobby_taxonomy,
        )
        
        stage2_top = reranked[:top_k]
        stage2_top_ids = [c.hobby_id for c in stage2_top]
        stage2_recs_by_person[person_id] = stage2_top_ids
        stage2_recommended_set.update(stage2_top_ids)
        
        # Segment tracking
        if context:
            if context.age_group:
                top_hobbies_by_age[context.age_group].update(stage2_top_ids)
                popularity_ranks_by_age[context.age_group].extend(
                    pop_rank_map.get(hobby_id, len(pop_rank_map) + 1) for hobby_id in stage2_top_ids
                )
            if context.occupation:
                top_hobbies_by_occupation[context.occupation].update(stage2_top_ids)
                popularity_ranks_by_occupation[context.occupation].extend(
                    pop_rank_map.get(hobby_id, len(pop_rank_map) + 1) for hobby_id in stage2_top_ids
                )
                
        # Provider tracking
        for c in stage2_top:
            providers = [p for p, score in c.source_scores.items() if score > 0]
            for p in providers:
                provider_flags_counter[p] += 1
                
        if person_id in sample_ids:
            review_entry = {
                "person_id": person_id,
                "person_uuid": id_to_person.get(person_id, ""),
                "context": {
                    "age_group": getattr(context, "age_group", None),
                    "sex": getattr(context, "sex", None),
                    "occupation": getattr(context, "occupation", None),
                    "family_type": getattr(context, "family_type", None),
                } if context else None,
                "known_hobbies": list(known_names),
                "target_hobbies": [id_to_hobby[hid] for hid in truth[person_id] if hid in id_to_hobby],
                "stage1_top": [id_to_hobby[hid] for hid in stage1_top_k],
                "stage2_top": [
                    {
                        "hobby_name": c.hobby_name,
                        "stage1_score": round(c.stage1_score, 4),
                        "final_score": round(c.final_score, 4),
                        "reason_features": c.reason_features,
                        "popularity_rank": pop_rank_map.get(c.hobby_id, -1),
                        "category": _category_for_hobby(c.hobby_name, hobby_category_map),
                        "popularity_penalty_candidate": round(
                            _popularity_penalty_candidate(
                                train_popularity_by_hobby.get(c.hobby_name, 0.0),
                                max_train_popularity,
                            ),
                            4,
                        ),
                        "novelty_bonus_candidate": round(
                            1.0
                            - _popularity_penalty_candidate(
                                train_popularity_by_hobby.get(c.hobby_name, 0.0),
                                max_train_popularity,
                            ),
                            4,
                        ),
                    }
                    for c in stage2_top
                ]
            }
            sample_reviews.append(review_entry)

    stage1_freq = Counter(hid for recs in stage1_recs_by_person.values() for hid in recs)
    stage2_freq = Counter(hid for recs in stage2_recs_by_person.values() for hid in recs)

    stage1_category_counts = _category_counts(stage1_freq, id_to_hobby, hobby_category_map)
    stage2_category_counts = _category_counts(stage2_freq, id_to_hobby, hobby_category_map)

    audit_report = {
        "split": args.split,
        "top_k": top_k,
        "total_canonical_hobbies": num_total_hobbies,
        "eval_persons": len(eval_person_ids),
        
        "coverage": {
            "stage1_unique_recommended": len(stage1_recommended_set),
            "stage1_coverage_ratio": len(stage1_recommended_set) / num_total_hobbies if num_total_hobbies else 0,
            "stage2_unique_recommended": len(stage2_recommended_set),
            "stage2_coverage_ratio": len(stage2_recommended_set) / num_total_hobbies if num_total_hobbies else 0,
        },
        "diversity_entropy": {
            "stage1_entropy": calculate_entropy(stage1_freq),
            "stage2_entropy": calculate_entropy(stage2_freq),
            "max_possible_entropy": calculate_entropy({i: 1 for i in range(num_total_hobbies)})
        },
        "popularity_bias": {
            "stage1_avg_popularity_rank": avg_popularity_rank(stage1_freq, pop_rank_map),
            "stage2_avg_popularity_rank": avg_popularity_rank(stage2_freq, pop_rank_map),
        },
        "provider_contribution": dict(provider_flags_counter),
        "segments": {
            "age_group": {
                age: [id_to_hobby[hid] for hid, _ in counts.most_common(5) if hid in id_to_hobby]
                for age, counts in top_hobbies_by_age.items()
            },
            "occupation": {
                occ: [id_to_hobby[hid] for hid, _ in counts.most_common(5) if hid in id_to_hobby]
                for occ, counts in list(top_hobbies_by_occupation.items())[:10]  # top 10 occupations
            }
        },
        "category_coverage": {
            "stage1": stage1_category_counts,
            "stage2": stage2_category_counts,
        },
        "category_entropy": {
            "stage1": calculate_entropy(stage1_category_counts),
            "stage2": calculate_entropy(stage2_category_counts),
        },
        "avg_categories_per_person": {
            "stage1": _average_categories_per_person(stage1_recs_by_person, id_to_hobby, hobby_category_map),
            "stage2": _average_categories_per_person(stage2_recs_by_person, id_to_hobby, hobby_category_map),
        },
        "long_tail_share": {
            "stage1_pop_rank_gt_50": _long_tail_share(stage1_freq, pop_rank_map, 50),
            "stage1_pop_rank_gt_100": _long_tail_share(stage1_freq, pop_rank_map, 100),
            "stage2_pop_rank_gt_50": _long_tail_share(stage2_freq, pop_rank_map, 50),
            "stage2_pop_rank_gt_100": _long_tail_share(stage2_freq, pop_rank_map, 100),
        },
        "novelty_distribution": {
            "stage1_avg_novelty": _average_novelty(
                stage1_freq,
                id_to_hobby,
                train_popularity_by_hobby,
                total_train_popularity,
            ),
            "stage2_avg_novelty": _average_novelty(
                stage2_freq,
                id_to_hobby,
                train_popularity_by_hobby,
                total_train_popularity,
            ),
        },
        "per_segment_popularity": {
            "age_group": {
                age: (sum(ranks) / len(ranks) if ranks else 0.0)
                for age, ranks in popularity_ranks_by_age.items()
            },
            "occupation": {
                occupation: (sum(ranks) / len(ranks) if ranks else 0.0)
                for occupation, ranks in popularity_ranks_by_occupation.items()
            },
        },
    }
    
    audit_out = config.paths.artifact_dir / "recommendation_quality_audit.json"
    sample_out = config.paths.artifact_dir / "sample_recommendations_review.json"
    
    save_json(audit_out, audit_report)
    save_json(sample_out, {"samples": sample_reviews})
    
    coverage = cast(dict[str, float | int], audit_report["coverage"])
    popularity_bias = cast(dict[str, float], audit_report["popularity_bias"])
    category_entropy = cast(dict[str, float], audit_report["category_entropy"])
    avg_categories_per_person = cast(dict[str, float], audit_report["avg_categories_per_person"])
    long_tail_share = cast(dict[str, float], audit_report["long_tail_share"])
    novelty_distribution = cast(dict[str, float], audit_report["novelty_distribution"])

    print("Audit Complete.")
    print(f"Stage 1 Coverage: {coverage['stage1_unique_recommended']} items")
    print(f"Stage 2 Coverage: {coverage['stage2_unique_recommended']} items")
    print(f"Stage 1 Avg Pop Rank: {popularity_bias['stage1_avg_popularity_rank']:.1f}")
    print(f"Stage 2 Avg Pop Rank: {popularity_bias['stage2_avg_popularity_rank']:.1f}")
    print(f"Stage 1/2 Category Entropy: {category_entropy['stage1']:.3f} / {category_entropy['stage2']:.3f}")
    print(
        f"Stage 1/2 Avg Categories per Person: "
        f"{avg_categories_per_person['stage1']:.2f} / {avg_categories_per_person['stage2']:.2f}"
    )
    print(
        f"Stage 1 Long-tail Share (>50, >100): "
        f"{long_tail_share['stage1_pop_rank_gt_50']:.3f}, {long_tail_share['stage1_pop_rank_gt_100']:.3f}"
    )
    print(
        f"Stage 2 Long-tail Share (>50, >100): "
        f"{long_tail_share['stage2_pop_rank_gt_50']:.3f}, {long_tail_share['stage2_pop_rank_gt_100']:.3f}"
    )
    print(
        f"Stage 1/2 Avg Novelty: "
        f"{novelty_distribution['stage1_avg_novelty']:.3f} / {novelty_distribution['stage2_avg_novelty']:.3f}"
    )
    print(f"Saved artifacts to {audit_out} and {sample_out}")

if __name__ == "__main__":
    main()
