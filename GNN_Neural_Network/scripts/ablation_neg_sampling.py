"""Negative Sampling Ablation: Compare neg_ratio and hard_ratio configurations.

⚠️ LEGACY / ANALYSIS-ONLY ⚠️

Per PRD §2.5 execution policy, hyperparameter sweep scripts are no longer part
of the default experiment path. Use `train_ranker.py` with single config
(--neg-ratio, --hard-ratio) + `evaluate_ranker.py --split validation` instead.

This script is retained for ad-hoc historical comparison only.
Do NOT use it for promotion decisions.

Phase 1: Test neg_ratio in [1, 2, 4, 8] with default hard_ratio=0.8
Phase 2: Test hard_ratio in [0.5, 0.8, 1.0] with best neg_ratio from Phase 1

Each configuration is trained and evaluated on both validation and test splits,
with results compared against the v1 deterministic baseline.

Usage (legacy only):
    python GNN_Neural_Network/scripts/ablation_neg_sampling.py
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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
from GNN_Neural_Network.gnn_recommender.data import load_json, load_person_contexts, save_json  # noqa: E402
from GNN_Neural_Network.gnn_recommender.metrics import summarize_ranking_metrics  # noqa: E402
from GNN_Neural_Network.gnn_recommender.ranker import LightGBMRanker, build_ranker_dataset  # noqa: E402
from GNN_Neural_Network.gnn_recommender.recommend import merge_candidates_by_hobby, normalize_candidate_scores  # noqa: E402
from GNN_Neural_Network.gnn_recommender.rerank import (  # noqa: E402
    HobbyCandidate, RerankerConfig, build_rerank_features,
    build_reranker_config, merge_stage1_candidates, rerank_candidates,
)

RECALL_GATE = -0.002
NDCG_GATE = 0.005

BASELINE_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 15,
    "min_data_in_leaf": 50,
    "learning_rate": 0.05,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "seed": 42,
    "num_threads": -1,
}

NEG_RATIOS = [1, 2, 4, 8]
HARD_RATIOS = [0.5, 0.8, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Negative sampling ablation.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("GNN_Neural_Network/artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--ranker-val-ratio", type=float, default=0.2)
    parser.add_argument("--include-source-features", action="store_true")
    return parser.parse_args()


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


def _generate_candidate_pools(
    person_ids: list[int],
    train_edges: list[tuple[int, int]],
    train_known: dict[int, set[int]],
    candidate_k: int,
    id_to_hobby: dict[int, str],
    popularity_counts: Counter[int],
    cooccurrence_counts: dict[int, Counter[int]],
    normalization_method: str,
) -> dict[int, list[HobbyCandidate]]:
    pools: dict[int, list[HobbyCandidate]] = {}
    for person_id in tqdm(person_ids, desc="candidate pools"):
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
        pools[person_id] = merge_stage1_candidates(merged, id_to_hobby)
    return pools


def _build_hobby_categories(id_to_hobby: dict[int, str], hobby_taxonomy: dict[str, object] | None) -> dict[int, str]:
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


def _build_person_segments(person_ids, id_to_person: dict[int, str], contexts: dict) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for person_id in person_ids:
        person_uuid = id_to_person.get(person_id, "")
        ctx = contexts.get(person_uuid)
        if ctx is not None:
            result[person_id] = {"age_group": ctx.age_group, "sex": ctx.sex}
    return result


def _metric_value(metrics: dict[str, object], key: str) -> float:
    value = metrics.get(key, 0.0)
    return float(value) if isinstance(value, (int, float, str)) else 0.0


def train_and_evaluate_config(
    neg_ratio: int,
    hard_ratio: float,
    params: dict[str, Any],
    ranker_train_edges: list[tuple[int, int]],
    ranker_val_edges: list[tuple[int, int]],
    pools: dict[int, list[HobbyCandidate]],
    all_hobby_ids: list[int],
    train_known: dict[int, set[int]],
    id_to_hobby: dict[int, str],
    id_to_person: dict[int, str],
    contexts: dict,
    hobby_profile: dict,
    reranker_config: RerankerConfig,
    split_truth: dict[int, set[int]],
    split_pools: dict[int, list[HobbyCandidate]],
    hobby_categories: dict[int, str],
    person_segments: dict[int, dict[str, str]],
    num_hobbies: int,
    popularity_counts: Counter[int],
    config,
    candidate_k: int,
    max_k: int,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    seed: int = 42,
    include_source_features: bool = False,
) -> dict[str, Any]:
    """Build dataset, train ranker, evaluate on the given split."""
    train_ds = build_ranker_dataset(
        ranker_train_edges, pools, all_hobby_ids, train_known,
        id_to_hobby, contexts, id_to_person, hobby_profile, reranker_config,
        neg_ratio=neg_ratio, hard_ratio=hard_ratio, seed=seed,
        include_source_features=include_source_features,
    )
    val_ds = build_ranker_dataset(
        ranker_val_edges, pools, all_hobby_ids, train_known,
        id_to_hobby, contexts, id_to_person, hobby_profile, reranker_config,
        neg_ratio=neg_ratio, hard_ratio=hard_ratio, seed=seed + 1,
        include_source_features=include_source_features,
    )
    train_lgb = train_ds.to_lgb_dataset()
    val_lgb = val_ds.to_lgb_dataset(reference=train_lgb)

    ranker = LightGBMRanker(params=params)
    metadata = ranker.fit(train_lgb, val_lgb, num_boost_round, early_stopping_rounds)
    model_feature_columns = ranker.feature_columns()

    train_pos = sum(1 for r in train_ds.rows if r.label == 1)
    val_pos = sum(1 for r in val_ds.rows if r.label == 1)

    v2_rankings: dict[int, list[int]] = {}
    v1_rankings: dict[int, list[int]] = {}

    for person_id in tqdm(split_truth, desc=f"eval (neg={neg_ratio}, hard={hard_ratio})", leave=False):
        known = train_known.get(person_id, set())
        pool_candidates = split_pools.get(person_id, [])
        if not pool_candidates:
            continue

        known_names = {id_to_hobby[hid] for hid in known if hid in id_to_hobby}
        reranked = rerank_candidates(
            contexts.get(id_to_person.get(person_id, "")),
            pool_candidates, hobby_profile, known_names, reranker_config,
        )
        v1_rankings[person_id] = [c.hobby_id for c in reranked[:max_k]]

        person_uuid = id_to_person.get(person_id, "")
        person_context = contexts.get(person_uuid)
        if person_context and pool_candidates and hobby_profile:
            features_list: list[list[float]] = []
            hobby_ids_list: list[int] = []
            for candidate in pool_candidates:
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
        else:
            v2_rankings[person_id] = [c.hobby_id for c in pool_candidates[:max_k]]

    v1_metrics = summarize_ranking_metrics(
        split_truth, v1_rankings, config.eval.top_k,
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, person_segments=person_segments,
    )
    v2_metrics = summarize_ranking_metrics(
        split_truth, v2_rankings, config.eval.top_k,
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, person_segments=person_segments,
    )

    return {
        "neg_ratio": neg_ratio,
        "hard_ratio": hard_ratio,
        "train_rows": len(train_ds.rows),
        "train_pos": train_pos,
        "train_neg": len(train_ds.rows) - train_pos,
        "val_rows": len(val_ds.rows),
        "best_iteration": metadata["best_iteration"],
        "best_auc": metadata["best_score"],
        "feature_importance": metadata["feature_importance"],
        "v1_recall@10": _metric_value(v1_metrics, "recall@10"),
        "v1_ndcg@10": _metric_value(v1_metrics, "ndcg@10"),
        "v2_recall@10": _metric_value(v2_metrics, "recall@10"),
        "v2_ndcg@10": _metric_value(v2_metrics, "ndcg@10"),
        "v2_coverage@10": _metric_value(v2_metrics, "catalog_coverage@10"),
        "v2_novelty@10": _metric_value(v2_metrics, "novelty@10"),
        "delta_recall@10": _metric_value(v2_metrics, "recall@10") - _metric_value(v1_metrics, "recall@10"),
        "delta_ndcg@10": _metric_value(v2_metrics, "ndcg@10") - _metric_value(v1_metrics, "ndcg@10"),
    }


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
    val_edges = _read_indexed_edges(config.paths.validation_edges)
    test_edges = _read_indexed_edges(config.paths.test_edges)
    train_known = _known_from_edges(train_edges)

    contexts = load_person_contexts(config.paths.person_context_csv) if config.paths.person_context_csv.exists() else {}
    hobby_profile = load_json(config.paths.hobby_profile) if config.paths.hobby_profile.exists() else None
    if not isinstance(hobby_profile, dict):
        raise ValueError("hobby_profile.json required")

    hobby_taxonomy = None
    for p in (config.paths.hobby_taxonomy, config.paths.artifact_dir / "hobby_taxonomy.json"):
        if p.exists():
            val = load_json(p)
            if isinstance(val, dict):
                hobby_taxonomy = val
                break

    normalization_method = _normalization_method(config.paths.score_normalization)
    reranker_config = build_reranker_config(config.rerank.use_text_fit, config.rerank.weights)

    popularity_counts = build_popularity_counts(train_edges)
    cooccurrence_counts = build_cooccurrence_counts(train_edges)
    all_hobby_ids = list(hobby_to_id.values())

    val_person_ids = sorted({pid for pid, _ in val_edges})
    rng = random.Random(args.seed)
    shuffled = list(val_person_ids)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1.0 - args.ranker_val_ratio)))
    ranker_train_persons = set(shuffled[:split_idx])
    ranker_val_persons = set(shuffled[split_idx:])

    ranker_train_edges = [(pid, hid) for pid, hid in val_edges if pid in ranker_train_persons]
    ranker_val_edges = [(pid, hid) for pid, hid in val_edges if pid in ranker_val_persons]

    val_pools = _generate_candidate_pools(
        val_person_ids, train_edges, train_known, candidate_k,
        id_to_hobby, popularity_counts, cooccurrence_counts, normalization_method,
    )

    val_truth = _known_from_edges(val_edges)
    test_truth = _known_from_edges(test_edges)
    test_person_ids = sorted(test_truth.keys())

    test_pools = _generate_candidate_pools(
        test_person_ids, train_edges, train_known, candidate_k,
        id_to_hobby, popularity_counts, cooccurrence_counts, normalization_method,
    )

    hobby_categories = _build_hobby_categories(id_to_hobby, hobby_taxonomy)
    val_segments = _build_person_segments(val_truth.keys(), id_to_person, contexts)
    test_segments = _build_person_segments(test_truth.keys(), id_to_person, contexts)
    num_hobbies = len(hobby_to_id)
    max_k = max(config.eval.top_k)

    # --- Phase 1: neg_ratio ablation ---
    print("=" * 60)
    print("  Phase 1: Negative Ratio Ablation")
    print("=" * 60)

    phase1_results: dict[str, dict[str, Any]] = {}

    for neg_ratio in NEG_RATIOS:
        print(f"\n--- neg_ratio={neg_ratio} (hard_ratio=0.8) ---")
        val_result = train_and_evaluate_config(
            neg_ratio=neg_ratio, hard_ratio=0.8, params=BASELINE_PARAMS,
            ranker_train_edges=ranker_train_edges, ranker_val_edges=ranker_val_edges,
            pools=val_pools, all_hobby_ids=all_hobby_ids, train_known=train_known,
            id_to_hobby=id_to_hobby, id_to_person=id_to_person, contexts=contexts,
            hobby_profile=hobby_profile, reranker_config=reranker_config,
            split_truth=val_truth, split_pools=val_pools,
            hobby_categories=hobby_categories, person_segments=val_segments,
            num_hobbies=num_hobbies, popularity_counts=popularity_counts,
            config=config, candidate_k=candidate_k, max_k=max_k,
            num_boost_round=args.num_boost_round, early_stopping_rounds=args.early_stopping,
            seed=args.seed,
            include_source_features=args.include_source_features,
        )
        phase1_results[str(neg_ratio)] = val_result
        print(f"  val: recall@10={val_result['v2_recall@10']:.4f}, "
              f"ndcg@10={val_result['v2_ndcg@10']:.4f}, "
              f"AUC={val_result['best_auc']:.4f}, "
              f"iter={val_result['best_iteration']}, "
              f"rows={val_result['train_rows']}")

    best_neg_ratio = max(
        NEG_RATIOS, key=lambda nr: phase1_results[str(nr)]["v2_recall@10"]
    )
    print(f"\nBest neg_ratio: {best_neg_ratio} "
          f"(recall@10={phase1_results[str(best_neg_ratio)]['v2_recall@10']:.4f})")

    # --- Phase 2: hard_ratio ablation ---
    print("\n" + "=" * 60)
    print(f"  Phase 2: Hard Ratio Ablation (neg_ratio={best_neg_ratio})")
    print("=" * 60)

    phase2_results: dict[str, dict[str, Any]] = {}

    for hard_ratio in HARD_RATIOS:
        print(f"\n--- hard_ratio={hard_ratio} (neg_ratio={best_neg_ratio}) ---")
        val_result = train_and_evaluate_config(
            neg_ratio=best_neg_ratio, hard_ratio=hard_ratio, params=BASELINE_PARAMS,
            ranker_train_edges=ranker_train_edges, ranker_val_edges=ranker_val_edges,
            pools=val_pools, all_hobby_ids=all_hobby_ids, train_known=train_known,
            id_to_hobby=id_to_hobby, id_to_person=id_to_person, contexts=contexts,
            hobby_profile=hobby_profile, reranker_config=reranker_config,
            split_truth=val_truth, split_pools=val_pools,
            hobby_categories=hobby_categories, person_segments=val_segments,
            num_hobbies=num_hobbies, popularity_counts=popularity_counts,
            config=config, candidate_k=candidate_k, max_k=max_k,
            num_boost_round=args.num_boost_round, early_stopping_rounds=args.early_stopping,
            seed=args.seed,
            include_source_features=args.include_source_features,
        )
        phase2_results[str(hard_ratio)] = val_result
        print(f"  val: recall@10={val_result['v2_recall@10']:.4f}, "
              f"ndcg@10={val_result['v2_ndcg@10']:.4f}, "
              f"AUC={val_result['best_auc']:.4f}")

    best_hard_ratio = max(
        HARD_RATIOS, key=lambda hr: phase2_results[str(hr)]["v2_recall@10"]
    )
    print(f"\nBest hard_ratio: {best_hard_ratio} "
          f"(recall@10={phase2_results[str(best_hard_ratio)]['v2_recall@10']:.4f})")

    # --- Final: test evaluation of best config ---
    print("\n" + "=" * 60)
    print(f"  Final Evaluation: neg_ratio={best_neg_ratio}, hard_ratio={best_hard_ratio}")
    print("=" * 60)

    best_val_result = phase2_results[str(best_hard_ratio)]

    test_result = train_and_evaluate_config(
        neg_ratio=best_neg_ratio, hard_ratio=best_hard_ratio, params=BASELINE_PARAMS,
        ranker_train_edges=ranker_train_edges, ranker_val_edges=ranker_val_edges,
        pools=val_pools, all_hobby_ids=all_hobby_ids, train_known=train_known,
        id_to_hobby=id_to_hobby, id_to_person=id_to_person, contexts=contexts,
        hobby_profile=hobby_profile, reranker_config=reranker_config,
        split_truth=test_truth, split_pools=test_pools,
        hobby_categories=hobby_categories, person_segments=test_segments,
        num_hobbies=num_hobbies, popularity_counts=popularity_counts,
        config=config, candidate_k=candidate_k, max_k=max_k,
        num_boost_round=args.num_boost_round, early_stopping_rounds=args.early_stopping,
        seed=args.seed,
        include_source_features=args.include_source_features,
    )

    recall_delta = test_result["delta_recall@10"]
    ndcg_delta = test_result["delta_ndcg@10"]
    recall_pass = recall_delta >= RECALL_GATE
    ndcg_pass = ndcg_delta >= NDCG_GATE
    status = "promoted" if (recall_pass and ndcg_pass) else "blocked"

    promotion_decision = {
        "best_neg_ratio": best_neg_ratio,
        "best_hard_ratio": best_hard_ratio,
        "status": status,
        "gates": {
            "recall@10": {"threshold": RECALL_GATE, "actual": recall_delta, "pass": recall_pass},
            "ndcg@10": {"threshold": NDCG_GATE, "actual": ndcg_delta, "pass": ndcg_pass},
        },
    }

    # --- Save results ---
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "baseline_params": BASELINE_PARAMS,
        "best_neg_ratio": best_neg_ratio,
        "best_hard_ratio": best_hard_ratio,
        "phase1_neg_ratios": {k: {kk: str(vv) if isinstance(vv, float) else vv for kk, vv in r.items()} for k, r in phase1_results.items()},
        "phase2_hard_ratios": {k: {kk: str(vv) if isinstance(vv, float) else vv for kk, vv in r.items()} for k, r in phase2_results.items()},
        "best_config": {"neg_ratio": best_neg_ratio, "hard_ratio": best_hard_ratio},
        "best_val_result": {k: str(v) if isinstance(v, float) else v for k, v in best_val_result.items()},
        "best_test_result": {k: str(v) if isinstance(v, float) else v for k, v in test_result.items()},
        "promotion_decision": promotion_decision,
    }
    save_json(output_dir / "neg_sampling_ablation.json", payload)

    summary_lines = [
        "# Negative Sampling Ablation Summary",
        "",
        "## Phase 1: neg_ratio (hard_ratio=0.8)",
        "",
        "| neg_ratio | train_rows | recall@10 | ndcg@10 | AUC | best_iter |",
        "|:---|:---|:---|:---|:---|:---|",
    ]
    for nr in NEG_RATIOS:
        r = phase1_results[str(nr)]
        marker = " **best**" if nr == best_neg_ratio else ""
        summary_lines.append(
            f"| {nr} | {r['train_rows']} | {r['v2_recall@10']:.4f} | {r['v2_ndcg@10']:.4f} | "
            f"{r['best_auc']:.4f} | {r['best_iteration']} |{marker}"
        )

    summary_lines.extend([
        "",
        "## Phase 2: hard_ratio (neg_ratio={best_neg_ratio})",
        "",
        "| hard_ratio | train_rows | recall@10 | ndcg@10 | AUC | best_iter |",
        "|:---|:---|:---|:---|:---|:---|",
    ])
    for hr in HARD_RATIOS:
        r = phase2_results[str(hr)]
        marker = " **best**" if hr == best_hard_ratio else ""
        summary_lines.append(
            f"| {hr} | {r['train_rows']} | {r['v2_recall@10']:.4f} | {r['v2_ndcg@10']:.4f} | "
            f"{r['best_auc']:.4f} | {r['best_iteration']} |{marker}"
        )

    summary_lines.extend([
        "",
        "## Best Configuration",
        "",
        f"- **neg_ratio**: {best_neg_ratio}",
        f"- **hard_ratio**: {best_hard_ratio}",
        f"- **Validation**: recall@10={best_val_result['v2_recall@10']:.4f}, "
        f"ndcg@10={best_val_result['v2_ndcg@10']:.4f}, AUC={best_val_result['best_auc']:.4f}",
        f"- **Test**: recall@10={test_result['v2_recall@10']:.4f}, "
        f"ndcg@10={test_result['v2_ndcg@10']:.4f}",
        f"- **Delta (test vs v1)**: recall@10={recall_delta:+.4f}, ndcg@10={ndcg_delta:+.4f}",
        "",
        "## Promotion Decision",
        "",
        f"- **Status**: {status}",
        f"- recall@10 gate: delta={recall_delta:+.6f} (>= {RECALL_GATE}) "
        f"→ {'PASS' if recall_pass else 'FAIL'}",
        f"- ndcg@10 gate: delta={ndcg_delta:+.6f} (>= {NDCG_GATE}) "
        f"→ {'PASS' if ndcg_pass else 'FAIL'}",
    ])

    (output_dir / "neg_sampling_ablation_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    print(f"\nResults saved: {output_dir / 'neg_sampling_ablation.json'}")
    print(f"Summary saved: {output_dir / 'neg_sampling_ablation_summary.md'}")
    print(f"\nBest config: neg_ratio={best_neg_ratio}, hard_ratio={best_hard_ratio}")
    print(f"Test: recall@10={test_result['v2_recall@10']:.4f}, ndcg@10={test_result['v2_ndcg@10']:.4f}")
    print(f"Promotion: {status}")


if __name__ == "__main__":
    main()
