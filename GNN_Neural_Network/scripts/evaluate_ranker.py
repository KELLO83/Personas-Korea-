from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.baseline import (  # noqa: E402
    build_cooccurrence_counts,
    build_popularity_counts,
    cooccurrence_candidate_provider,
    popularity_candidate_provider,
)
from GNN_Neural_Network.gnn_recommender.config import load_config  # noqa: E402
from GNN_Neural_Network.gnn_recommender.data import PersonContext, load_json, load_person_contexts, save_json  # noqa: E402
from GNN_Neural_Network.gnn_recommender.metrics import summarize_ranking_metrics  # noqa: E402
from GNN_Neural_Network.gnn_recommender.diversity import (
    compute_hobby_embeddings,
    mmr_rerank,
)
from GNN_Neural_Network.gnn_recommender.ranker import LightGBMRanker  # noqa: E402
from GNN_Neural_Network.gnn_recommender.recommend import merge_candidates_by_hobby, normalize_candidate_scores  # noqa: E402
from GNN_Neural_Network.gnn_recommender.rerank import (  # noqa: E402
    build_rerank_features,
    build_reranker_config,
    merge_stage1_candidates,
    rerank_candidates,
)

# PRD §4.4 promotion gate thresholds
RECALL_GATE = -0.002   # delta_recall@10 >= -0.002 (allow tiny regression)
NDCG_GATE = 0.005      # delta_ndcg@10 >= +0.005  (must improve)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LightGBM ranker vs deterministic reranker.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--use-mmr", action="store_true", help="Apply MMR diversity reordering after ranker scoring")
    parser.add_argument("--mmr-lambda", type=float, default=0.7, help="MMR lambda parameter (0=all diversity, 1=all relevance)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    candidate_k = config.rerank.candidate_pool_size
    if candidate_k <= 0:
        raise ValueError("candidate_pool_size must be positive")

    checkpoint = _safe_torch_load(config.paths.checkpoint)
    person_to_id = _expect_mapping(checkpoint.get("person_to_id"), "person_to_id")
    hobby_to_id = _expect_mapping(checkpoint.get("hobby_to_id"), "hobby_to_id")
    id_to_hobby = {v: k for k, v in hobby_to_id.items()}
    id_to_person = {v: k for k, v in person_to_id.items()}

    train_edges = _read_indexed_edges(config.paths.train_edges)
    target_edges = _read_indexed_edges(
        config.paths.validation_edges if args.split == "validation" else config.paths.test_edges,
    )
    train_known = _known_from_edges(train_edges)
    truth = _known_from_edges(target_edges)

    contexts = load_person_contexts(config.paths.person_context_csv) if config.paths.person_context_csv.exists() else {}
    hobby_profile = load_json(config.paths.hobby_profile) if config.paths.hobby_profile.exists() else None
    if not isinstance(hobby_profile, dict):
        raise ValueError("hobby_profile.json required")
    hobby_taxonomy = _load_hobby_taxonomy(config.paths.hobby_taxonomy, config.paths.artifact_dir)
    normalization_method = _normalization_method(config.paths.score_normalization)
    reranker_config = build_reranker_config(config.rerank.use_text_fit, config.rerank.weights)

    model_path = args.model_path or Path("GNN_Neural_Network/artifacts/ranker_model.txt")
    if not model_path.exists():
        raise FileNotFoundError(f"Ranker model not found: {model_path}. Run train_ranker.py first.")
    ranker = LightGBMRanker.load(model_path)
    model_feature_columns = ranker.feature_columns()
    print(f"Loaded ranker model: {model_path} (best_iteration={ranker.best_iteration})")

    popularity_counts = build_popularity_counts(train_edges)
    cooccurrence_counts = build_cooccurrence_counts(train_edges)
    max_k = max(config.eval.top_k)

    stage1_rankings: dict[int, list[int]] = {}
    v1_rankings: dict[int, list[int]] = {}
    v2_rankings: dict[int, list[int]] = {}
    candidate_rankings: dict[int, list[int]] = {}
    v2_fallback_count = 0

    # MMR precompute: hobby embeddings for diversity reordering
    all_hobby_names = list(hobby_to_id.keys())
    hobby_emb = compute_hobby_embeddings(all_hobby_names, hobby_taxonomy) if args.use_mmr else None
    hobby_id_to_emb_idx = {hid: idx for idx, name in enumerate(all_hobby_names) for hid in [hobby_to_id[name]]} if args.use_mmr else {}

    for person_id in tqdm(truth, desc=f"ranker eval ({args.split})"):
        known = train_known.get(person_id, set())

        pop = normalize_candidate_scores(
            popularity_candidate_provider(train_edges, person_id, known, candidate_k, popularity_counts=popularity_counts),
            normalization_method,
        )
        cooc = normalize_candidate_scores(
            cooccurrence_candidate_provider(train_edges, person_id, known, candidate_k, cooccurrence_counts=cooccurrence_counts),
            normalization_method,
        )
        merged = merge_candidates_by_hobby({"popularity": pop, "cooccurrence": cooc}, candidate_k)
        candidate_rankings[person_id] = [c.hobby_id for c in merged]
        stage1_rankings[person_id] = [c.hobby_id for c in merged[:max_k]]

        hobby_candidates = merge_stage1_candidates(merged, id_to_hobby)
        known_names = {id_to_hobby[hid] for hid in known if hid in id_to_hobby}

        reranked = rerank_candidates(
            contexts.get(id_to_person.get(person_id, "")),
            hobby_candidates,
            hobby_profile,
            known_names,
            reranker_config,
            hobby_taxonomy=hobby_taxonomy,
        )
        v1_rankings[person_id] = [c.hobby_id for c in reranked[:max_k]]

        person_uuid = id_to_person.get(person_id, "")
        person_context = contexts.get(person_uuid)

        if person_context and hobby_candidates:
            features_list: list[list[float]] = []
            hobby_ids_list: list[int] = []
            for candidate in hobby_candidates:
                features = build_rerank_features(
                    person_context, candidate, hobby_profile, known_names, reranker_config,
                )
                features.pop("similar_person_score", None)
                features.pop("persona_text_fit", None)
                features_list.append([features.get(col, 0.0) for col in model_feature_columns])
                hobby_ids_list.append(candidate.hobby_id)

            X = np.array(features_list, dtype=np.float32)
            scores = ranker.predict(X)
            sorted_indices = np.argsort(-scores)
            v2_rankings[person_id] = [hobby_ids_list[int(i)] for i in sorted_indices[:max_k]]

            if args.use_mmr and hobby_emb is not None:
                emb_indices = [hobby_id_to_emb_idx[hid] for hid in hobby_ids_list if hid in hobby_id_to_emb_idx]
                if emb_indices:
                    emb_subset = hobby_emb[emb_indices]
                    mmr_result = mmr_rerank(hobby_ids_list, scores, emb_subset, lambda_param=args.mmr_lambda, top_k=max_k)
                    v2_rankings[person_id] = mmr_result
        else:
            v2_fallback_count += 1
            v2_rankings[person_id] = stage1_rankings[person_id]

    hobby_categories = _build_hobby_categories(id_to_hobby, hobby_taxonomy)
    person_segments = _build_person_segments(truth.keys(), id_to_person, contexts)
    num_hobbies = len(hobby_to_id)

    stage1_metrics = summarize_ranking_metrics(
        truth, stage1_rankings, config.eval.top_k,
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, person_segments=person_segments,
    )
    v1_metrics = summarize_ranking_metrics(
        truth, v1_rankings, config.eval.top_k,
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, person_segments=person_segments,
    )
    v2_metrics = summarize_ranking_metrics(
        truth, v2_rankings, config.eval.top_k,
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, candidate_pool_by_person=candidate_rankings,
        person_segments=person_segments,
    )
    candidate_recall_metrics = summarize_ranking_metrics(
        truth, candidate_rankings, (candidate_k,),
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories,
    )

    delta_v2_vs_v1 = {
        "recall@10": _metric_value(v2_metrics, "recall@10") - _metric_value(v1_metrics, "recall@10"),
        "ndcg@10": _metric_value(v2_metrics, "ndcg@10") - _metric_value(v1_metrics, "ndcg@10"),
        "hit_rate@10": _metric_value(v2_metrics, "hit_rate@10") - _metric_value(v1_metrics, "hit_rate@10"),
    }
    delta_v2_vs_stage1 = {
        "recall@10": _metric_value(v2_metrics, "recall@10") - _metric_value(stage1_metrics, "recall@10"),
        "ndcg@10": _metric_value(v2_metrics, "ndcg@10") - _metric_value(stage1_metrics, "ndcg@10"),
        "hit_rate@10": _metric_value(v2_metrics, "hit_rate@10") - _metric_value(stage1_metrics, "hit_rate@10"),
    }

    promotion = _promotion_decision(args.split, delta_v2_vs_v1)

    payload: dict[str, object] = {
        "split": args.split,
        "candidate_k": candidate_k,
        "model_path": str(model_path),
        "feature_columns": model_feature_columns,
        "v2_fallback_count": v2_fallback_count,
        "stage1_baseline": {
            "providers": ["popularity", "cooccurrence"],
            "metrics": stage1_metrics,
        },
        "v1_deterministic_reranker": {
            "metrics": v1_metrics,
        },
        "v2_lightgbm_ranker": {
            "metrics": v2_metrics,
            "delta_vs_v1_reranker": delta_v2_vs_v1,
            "delta_vs_stage1": delta_v2_vs_stage1,
            "use_mmr": args.use_mmr,
            "mmr_lambda": args.mmr_lambda if args.use_mmr else None,
        },
        "candidate_recall": candidate_recall_metrics,
        "promotion_decision": promotion,
    }
    output_path = args.output or Path("GNN_Neural_Network/artifacts/ranker_eval_metrics.json")
    save_json(output_path, payload)
    print(f"\nResults saved: {output_path}")

    print(f"\n{'='*60}")
    mode_label = f"v2 LightGBM + MMR (λ={args.mmr_lambda})" if args.use_mmr else "v2 LightGBM Ranker"
    print(f"  LightGBM Ranker Evaluation ({args.split})")
    print(f"  Mode: {mode_label}")
    print(f"{'='*60}")
    _print_section("Stage1 (pop+cooc)", stage1_metrics)
    _print_section("v1 Deterministic Reranker", v1_metrics)
    _print_section("v2 LightGBM Ranker", v2_metrics)
    print(f"\n--- Delta v2 vs v1 ---")
    for key, val in delta_v2_vs_v1.items():
        sign = "+" if val >= 0 else ""
        print(f"  {key}: {sign}{val:.6f}")
    print(f"\n--- Delta v2 vs Stage1 ---")
    for key, val in delta_v2_vs_stage1.items():
        sign = "+" if val >= 0 else ""
        print(f"  {key}: {sign}{val:.6f}")
    print(f"\n--- Promotion Gate ---")
    print(f"  recall@10 gate (>= {RECALL_GATE}): {delta_v2_vs_v1['recall@10']:.6f} → {'PASS' if delta_v2_vs_v1['recall@10'] >= RECALL_GATE else 'FAIL'}")
    print(f"  ndcg@10 gate (>= {NDCG_GATE}): {delta_v2_vs_v1['ndcg@10']:.6f} → {'PASS' if delta_v2_vs_v1['ndcg@10'] >= NDCG_GATE else 'FAIL'}")
    print(f"  Decision: {promotion['status']}")
    if v2_fallback_count > 0:
        print(f"  v2 fallback (missing context): {v2_fallback_count}")


def _print_section(title: str, metrics: dict[str, object]) -> None:
    print(f"\n  {title}:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, int | float):
            print(f"    {key}: {value:.6f}")


def _build_hobby_categories(
    id_to_hobby: dict[int, str],
    hobby_taxonomy: dict[str, object] | None,
) -> dict[int, str]:
    if hobby_taxonomy is None:
        return {}
    taxonomy_map = hobby_taxonomy.get("taxonomy", {})
    rules = hobby_taxonomy.get("rules", [])
    result: dict[int, str] = {}
    for hobby_id, hobby_name in id_to_hobby.items():
        category = ""
        if isinstance(taxonomy_map, dict):
            entry = taxonomy_map.get(hobby_name, {})
            if isinstance(entry, dict):
                category = str(entry.get("category", ""))
        if not category and isinstance(rules, list):
            for rule in rules:
                if isinstance(rule, dict) and rule.get("canonical_hobby") == hobby_name:
                    tax = rule.get("taxonomy", {})
                    if isinstance(tax, dict):
                        category = str(tax.get("category", ""))
                    break
        if category:
            result[hobby_id] = category
    return result


def _build_person_segments(
    person_ids: Iterable[int],
    id_to_person: dict[int, str],
    contexts: dict[str, PersonContext],
) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for person_id in person_ids:
        person_uuid = id_to_person.get(person_id, "")
        ctx = contexts.get(person_uuid)
        if ctx is not None:
            result[person_id] = {
                "age_group": ctx.age_group,
                "sex": ctx.sex,
            }
    return result


def _promotion_decision(split: str, delta_v2_vs_v1: dict[str, float]) -> dict[str, object]:
    recall_delta = float(delta_v2_vs_v1.get("recall@10", 0.0))
    ndcg_delta = float(delta_v2_vs_v1.get("ndcg@10", 0.0))
    recall_pass = recall_delta >= RECALL_GATE
    ndcg_pass = ndcg_delta >= NDCG_GATE
    gate_pass = recall_pass and ndcg_pass

    if split == "validation":
        if gate_pass:
            status = "eligible_for_test"
            reason = f"v2 passes both gates on validation (recall@10 delta={recall_delta:+.6f} >= {RECALL_GATE}, ndcg@10 delta={ndcg_delta:+.6f} >= {NDCG_GATE})"
        else:
            failed_gates = []
            if not recall_pass:
                failed_gates.append(f"recall@10 delta={recall_delta:+.6f} < {RECALL_GATE}")
            if not ndcg_pass:
                failed_gates.append(f"ndcg@10 delta={ndcg_delta:+.6f} < {NDCG_GATE}")
            status = "blocked"
            reason = f"v2 fails gate(s) on validation: {'; '.join(failed_gates)}"
    elif split == "test":
        if gate_pass:
            status = "promoted"
            reason = f"v2 passes both gates on test (recall@10 delta={recall_delta:+.6f} >= {RECALL_GATE}, ndcg@10 delta={ndcg_delta:+.6f} >= {NDCG_GATE})"
        else:
            failed_gates = []
            if not recall_pass:
                failed_gates.append(f"recall@10 delta={recall_delta:+.6f} < {RECALL_GATE}")
            if not ndcg_pass:
                failed_gates.append(f"ndcg@10 delta={ndcg_delta:+.6f} < {NDCG_GATE}")
            status = "blocked"
            reason = f"v2 fails gate(s) on test: {'; '.join(failed_gates)}"
    else:
        status = "blocked"
        reason = "Unknown split"

    return {
        "status": status,
        "gates": {
            "recall@10": {"threshold": RECALL_GATE, "actual": recall_delta, "pass": recall_pass},
            "ndcg@10": {"threshold": NDCG_GATE, "actual": ndcg_delta, "pass": ndcg_pass},
        },
        "reason": reason,
    }


def _metric_value(metrics: Mapping[str, object], key: str) -> float:
    value = metrics.get(key, 0.0)
    return float(value) if isinstance(value, int | float | str) else 0.0


def _load_hobby_taxonomy(configured_path: Path, artifact_dir: Path) -> dict[str, object] | None:
    for path in (configured_path, artifact_dir / "hobby_taxonomy.json"):
        if path.exists():
            value = load_json(path)
            if isinstance(value, dict):
                return value
    return None


def _safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint {path} must contain a dictionary")
    return value


def _expect_mapping(value: object, name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint missing mapping: {name}")
    return {str(k): int(v) for k, v in value.items()}


def _read_indexed_edges(path: Path) -> list[tuple[int, int]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [(int(row["person_id"]), int(row["hobby_id"])) for row in reader]


def _known_from_edges(edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    known: dict[int, set[int]] = defaultdict(set)
    for pid, hid in edges:
        known[pid].add(hid)
    return dict(known)


def _normalization_method(path: Path) -> str:
    if not path.exists():
        return "rank_percentile"
    value = load_json(path)
    if not isinstance(value, dict):
        return "rank_percentile"
    return str(value.get("method", "rank_percentile"))


if __name__ == "__main__":
    main()
