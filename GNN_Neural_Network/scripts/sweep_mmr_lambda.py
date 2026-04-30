"""MMR lambda grid search sweep script.

Runs v2 LightGBM ranker + MMR for each lambda value and evaluates
accuracy (recall, ndcg) vs diversity (coverage, novelty, ILD).
Produces mmr_sweep.json and mmr_pareto.png.
"""
from __future__ import annotations

import argparse
import csv
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

from GNN_Neural_Network.gnn_recommender.baseline import (
    build_cooccurrence_counts,
    build_popularity_counts,
    cooccurrence_candidate_provider,
    popularity_candidate_provider,
)
from GNN_Neural_Network.gnn_recommender.config import load_config
from GNN_Neural_Network.gnn_recommender.data import PersonContext, load_json, load_person_contexts, save_json
from GNN_Neural_Network.gnn_recommender.diversity import (
    compute_hobby_embeddings,
    compute_intra_list_diversity,
    mmr_rerank,
)
from GNN_Neural_Network.gnn_recommender.metrics import summarize_ranking_metrics
from GNN_Neural_Network.gnn_recommender.ranker import LightGBMRanker
from GNN_Neural_Network.gnn_recommender.recommend import merge_candidates_by_hobby, normalize_candidate_scores
from GNN_Neural_Network.gnn_recommender.rerank import (
    build_rerank_features,
    build_reranker_config,
    merge_stage1_candidates,
)

LAMBDA_VALUES = [round(x * 0.1, 1) for x in range(1, 10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMR lambda grid sweep for diversity-accuracy trade-off.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("GNN_Neural_Network/artifacts/mmr_sweep.json"))
    parser.add_argument("--output-plot", type=Path, default=Path("GNN_Neural_Network/artifacts/mmr_pareto.png"))
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

    all_hobby_names = list(hobby_to_id.keys())
    hobby_embeddings = compute_hobby_embeddings(all_hobby_names, hobby_taxonomy)
    hobby_id_to_embedding_idx = {hid: idx for idx, name in enumerate(all_hobby_names) for hid in [hobby_to_id[name]]}

    hobby_categories = _build_hobby_categories(id_to_hobby, hobby_taxonomy)
    person_segments = _build_person_segments(truth.keys(), id_to_person, contexts)
    num_hobbies = len(hobby_to_id)

    person_data: dict[int, dict[str, Any]] = {}
    v2_fallback_count = 0

    for person_id in tqdm(truth, desc=f"precomputing v2 scores ({args.split})"):
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
        hobby_candidates = merge_stage1_candidates(merged, id_to_hobby)
        known_names = {id_to_hobby[hid] for hid in known if hid in id_to_hobby}
        person_uuid = id_to_person.get(person_id, "")
        person_context = contexts.get(person_uuid)

        if person_context and hobby_candidates:
            features_list: list[list[float]] = []
            for candidate in hobby_candidates:
                features = build_rerank_features(person_context, candidate, hobby_profile, known_names, reranker_config)
                features.pop("similar_person_score", None)
                features.pop("persona_text_fit", None)
                features_list.append([features.get(col, 0.0) for col in model_feature_columns])
            X = np.array(features_list, dtype=np.float32)
            scores = ranker.predict(X)
            sorted_indices = np.argsort(-scores)
            ranked_hobby_ids = [hobby_candidates[int(i)].hobby_id for i in sorted_indices[:max_k]]
            ranked_scores = [float(scores[int(i)]) for i in sorted_indices[:max_k]]
            ranked_names = [id_to_hobby[hid] for hid in ranked_hobby_ids if hid in id_to_hobby]
            emb_indices = [hobby_id_to_embedding_idx[hid] for hid in ranked_hobby_ids if hid in hobby_id_to_embedding_idx]
            emb_subset = hobby_embeddings[emb_indices] if emb_indices else np.empty((0, hobby_embeddings.shape[1]))

            person_data[person_id] = {
                "hobby_ids": ranked_hobby_ids,
                "hobby_names": ranked_names,
                "scores": np.array(ranked_scores, dtype=np.float32),
                "embeddings": emb_subset,
                "fallback": False,
            }
        else:
            v2_fallback_count += 1
            stage1_ids = [c.hobby_id for c in merged[:max_k]]
            person_data[person_id] = {
                "hobby_ids": stage1_ids,
                "hobby_names": [id_to_hobby.get(hid, "") for hid in stage1_ids],
                "scores": np.zeros(len(stage1_ids), dtype=np.float32),
                "embeddings": np.empty((0, hobby_embeddings.shape[1])),
                "fallback": True,
            }

    v2_rankings = {pid: data["hobby_ids"][:max_k] for pid, data in person_data.items()}
    v2_metrics = summarize_ranking_metrics(
        truth, v2_rankings, config.eval.top_k,
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, person_segments=person_segments,
    )

    sweep_results: list[dict[str, Any]] = []

    for lam in LAMBDA_VALUES:
        mmr_rankings: dict[int, list[int]] = {}
        for person_id, data in person_data.items():
            hobby_ids = data["hobby_ids"]
            scores = data["scores"]
            embs = data["embeddings"]

            if data["fallback"] or len(hobby_ids) <= max_k:
                mmr_rankings[person_id] = hobby_ids[:max_k]
                continue

            selected = mmr_rerank(hobby_ids, scores, embs, lambda_param=lam, top_k=max_k)
            mmr_rankings[person_id] = selected[:max_k]

        metrics = summarize_ranking_metrics(
            truth, mmr_rankings, config.eval.top_k,
            num_total_items=num_hobbies, item_popularity=popularity_counts,
            hobby_categories=hobby_categories, person_segments=person_segments,
        )

        ild_values: list[float] = []
        for person_id, data in person_data.items():
            ranked_names = mmr_rankings.get(person_id, [])[:max_k]
            if len(ranked_names) >= 2:
                ranked_names_str = [id_to_hobby.get(hid, "") for hid in ranked_names]
                indices = [hobby_id_to_embedding_idx[hid] for hid in ranked_names if hid in hobby_id_to_embedding_idx]
                if indices:
                    ild = compute_intra_list_diversity(
                        ranked_names_str,
                        embeddings=hobby_embeddings[indices],
                    )
                    ild_values.append(ild)

        avg_ild = float(np.mean(ild_values)) if ild_values else 0.0

        result = {
            "lambda": lam,
            "metrics": metrics,
            "avg_intra_list_diversity": avg_ild,
            "delta_recall@10": float(metrics.get("recall@10", 0)) - float(v2_metrics.get("recall@10", 0)),
            "delta_ndcg@10": float(metrics.get("ndcg@10", 0)) - float(v2_metrics.get("ndcg@10", 0)),
            "delta_coverage@10": float(metrics.get("catalog_coverage@10", 0)) - float(v2_metrics.get("catalog_coverage@10", 0)),
            "delta_novelty@10": float(metrics.get("novelty@10", 0)) - float(v2_metrics.get("novelty@10", 0)),
            "delta_ild@10": avg_ild - float(v2_metrics.get("intra_list_diversity@10", 0)) if "intra_list_diversity@10" in v2_metrics else avg_ild,
        }
        sweep_results.append(result)
        print(f"  λ={lam:.1f}: recall@10={metrics.get('recall@10', 0):.4f}, "
              f"ndcg@10={metrics.get('ndcg@10', 0):.4f}, "
              f"coverage@10={metrics.get('catalog_coverage@10', 0):.4f}, "
              f"ILD={avg_ild:.4f}")

    payload = {
        "split": args.split,
        "lambda_values": LAMBDA_VALUES,
        "v2_baseline": v2_metrics,
        "v2_fallback_count": v2_fallback_count,
        "sweep_results": sweep_results,
    }

    save_json(args.output_json, payload)
    print(f"\nSweep results saved: {args.output_json}")

    try:
        _plot_pareto(sweep_results, v2_metrics, args.output_plot)
    except Exception as e:
        print(f"Warning: could not generate Pareto plot: {e}")

    _recommend_best_lambda(sweep_results)


def _plot_pareto(sweep_results: list[dict[str, Any]], v2_metrics: dict[str, object], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lambdas = [r["lambda"] for r in sweep_results]
    recalls = [float(r["metrics"].get("recall@10", 0)) for r in sweep_results]
    coverages = [float(r["metrics"].get("catalog_coverage@10", 0)) for r in sweep_results]
    ilds = [r["avg_intra_list_diversity"] for r in sweep_results]

    v2_recall = float(v2_metrics.get("recall@10", 0))
    v2_coverage = float(v2_metrics.get("catalog_coverage@10", 0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(lambdas, recalls, "bo-", label="MMR")
    axes[0].axhline(y=v2_recall, color="r", linestyle="--", label="v2 baseline")
    axes[0].set_xlabel("Lambda")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_title("MMR: Recall vs Lambda")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(lambdas, coverages, "gs-", label="MMR")
    axes[1].axhline(y=v2_coverage, color="r", linestyle="--", label="v2 baseline")
    axes[1].set_xlabel("Lambda")
    axes[1].set_ylabel("Catalog Coverage@10")
    axes[1].set_title("MMR: Coverage vs Lambda")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].scatter([float(r["metrics"].get("recall@10", 0)) for r in sweep_results], ilds, c=lambdas, cmap="viridis", s=80)
    axes[2].set_xlabel("Recall@10")
    axes[2].set_ylabel("Avg Intra-List Diversity")
    axes[2].set_title("Accuracy vs Diversity Pareto")
    cbar = plt.colorbar(axes[2].collections[0], ax=axes[2])
    cbar.set_label("Lambda")
    axes[2].grid(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Pareto plot saved: {output_path}")


def _recommend_best_lambda(sweep_results: list[dict[str, Any]]) -> None:
    best = None
    best_score = float("-inf")
    for r in sweep_results:
        recall_d = r.get("delta_recall@10", 0)
        ndcg_d = r.get("delta_ndcg@10", 0)
        coverage_d = r.get("delta_coverage@10", 0)
        novelty_d = r.get("delta_novelty@10", 0)
        ild_d = r.get("delta_ild@10", 0)
        if recall_d < -0.002:
            continue
        diversity_improvements = sum(1 for d in [coverage_d, novelty_d, ild_d] if d > 0)
        if diversity_improvements < 2:
            continue
        score = coverage_d + ild_d
        if score > best_score:
            best_score = score
            best = r

    if best:
        print(f"\nRecommended λ={best['lambda']:.1f}: "
              f"recall@10 delta={best.get('delta_recall@10', 0):+.4f}, "
              f"ndcg@10 delta={best.get('delta_ndcg@10', 0):+.4f}, "
              f"coverage@10 delta={best.get('delta_coverage@10', 0):+.4f}, "
              f"ILD delta={best.get('delta_ild@10', 0):+.4f}")
    else:
        print("\nNo λ value passes the diversity gate (recall drop < 0.002 and 2+ diversity improvements).")


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
    person_ids,
    id_to_person: dict[int, str],
    contexts: dict[str, PersonContext],
) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for person_id in person_ids:
        person_uuid = id_to_person.get(person_id, "")
        ctx = contexts.get(person_uuid)
        if ctx is not None:
            result[person_id] = {"age_group": ctx.age_group, "sex": ctx.sex}
    return result


def _load_hobby_taxonomy(configured_path: Path, artifact_dir: Path) -> dict[str, object] | None:
    for path in (configured_path, artifact_dir / "hobby_taxonomy.json"):
        if path.exists():
            value = load_json(path)
            if isinstance(value, dict):
                return value
    return None


def _normalization_method(path: Path) -> str:
    if not path.exists():
        return "rank_percentile"
    value = load_json(path)
    if not isinstance(value, dict):
        return "rank_percentile"
    return str(value.get("method", "rank_percentile"))


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


if __name__ == "__main__":
    main()
