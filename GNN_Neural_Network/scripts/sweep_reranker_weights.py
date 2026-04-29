from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.config import load_config  # noqa: E402
from GNN_Neural_Network.gnn_recommender.data import PersonContext, load_json, load_person_contexts, save_json  # noqa: E402
from GNN_Neural_Network.gnn_recommender.metrics import summarize_ranking_metrics  # noqa: E402
from GNN_Neural_Network.gnn_recommender.model import LightGCN, build_normalized_adjacency, choose_device  # noqa: E402
from GNN_Neural_Network.gnn_recommender.baseline import build_cooccurrence_counts, build_popularity_counts  # noqa: E402
from GNN_Neural_Network.gnn_recommender.recommend import compute_lightgcn_embeddings, merge_candidates_by_hobby  # noqa: E402
from GNN_Neural_Network.gnn_recommender.rerank import HobbyCandidate, build_rerank_features, build_reranker_config, merge_stage1_candidates, score_rerank_features  # noqa: E402
from GNN_Neural_Network.scripts.evaluate_reranker import SELECTED_STAGE1_BASELINE, _expect_mapping, _normalization_method, _provider_candidates, _safe_torch_load, _selected_stage1_provider_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Stage2 reranker weights on validation only without retraining.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--split", choices=["validation"], default="validation")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    candidate_k = config.rerank.candidate_pool_size
    checkpoint = _safe_torch_load(config.paths.checkpoint)
    person_to_id = _expect_mapping(checkpoint.get("person_to_id"), "person_to_id")
    hobby_to_id = _expect_mapping(checkpoint.get("hobby_to_id"), "hobby_to_id")
    id_to_hobby = {value: key for key, value in hobby_to_id.items()}
    id_to_person = {value: key for key, value in person_to_id.items()}
    train_edges = _read_indexed_edges(config.paths.train_edges)
    target_edges = _read_indexed_edges(config.paths.validation_edges if args.split == "validation" else config.paths.test_edges)
    train_known = _known_from_edges(train_edges)
    truth = _known_from_edges(target_edges)
    contexts = load_person_contexts(config.paths.person_context_csv) if config.paths.person_context_csv.exists() else {}
    hobby_profile = load_json(config.paths.hobby_profile) if config.paths.hobby_profile.exists() else None
    normalization_method = _normalization_method(config.paths.score_normalization)
    if config.rerank.use_text_fit:
        raise ValueError("weight sweep requires use_text_fit=false to avoid text-leakage tuning")
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

    base_weights = {key: float(value) for key, value in asdict(build_reranker_config(config.rerank.use_text_fit, config.rerank.weights).weights).items()}
    search_space = list(
        product(
            [0.10, 0.15, 0.20, 0.25],
            [0.00, 0.10, 0.25, 0.50],
            [0.00, 0.03, 0.05],
        )
    )
    feature_contexts = _precompute_stage2_features(
        model=model,
        adjacency=adjacency,
        train_edges=train_edges,
        truth=truth,
        train_known=train_known,
        id_to_hobby=id_to_hobby,
        id_to_person=id_to_person,
        contexts=contexts,
        hobby_profile=hobby_profile if isinstance(hobby_profile, dict) else None,
        normalization_method=normalization_method,
        candidate_k=candidate_k,
        score_chunk_size=config.eval.score_chunk_size,
        device=device,
        person_embeddings=person_embeddings,
        hobby_embeddings=hobby_embeddings,
        popularity_counts=popularity_counts,
        cooccurrence_counts=cooccurrence_counts,
    )
    selected_stage1_rankings: dict[int, list[int]] = {}
    for item in feature_contexts:
        hobby_candidates = item.get("hobby_candidates")
        if not isinstance(hobby_candidates, list):
            continue
        person_id = _safe_int(item.get("person_id", -1))
        if person_id < 0:
            continue
        selected_stage1_rankings[person_id] = [candidate.hobby_id for candidate in hobby_candidates[: max(config.eval.top_k)]]
    selected_stage1_metrics = summarize_ranking_metrics(truth, selected_stage1_rankings, config.eval.top_k)
    results: list[dict[str, object]] = []
    for segment_weight, mismatch_weight, popularity_weight in tqdm(search_space, desc="rerank sweep"):
        weights = dict(base_weights)
        weights["segment_popularity_score"] = segment_weight
        weights["mismatch_penalty"] = mismatch_weight
        weights["popularity_prior"] = popularity_weight
        rankings = _rerank_rankings(
            top_k_values=config.eval.top_k,
            weights=weights,
            feature_contexts=feature_contexts,
        )
        metrics = summarize_ranking_metrics(truth, rankings, config.eval.top_k)
        results.append(
            {
                "weights": weights,
                "metrics": metrics,
                "delta_vs_selected_stage1": {
                    "recall@10": float(metrics.get("recall@10", 0.0)) - float(selected_stage1_metrics.get("recall@10", 0.0)),
                    "ndcg@10": float(metrics.get("ndcg@10", 0.0)) - float(selected_stage1_metrics.get("ndcg@10", 0.0)),
                },
            }
        )

    ranked = sorted(
        results,
        key=lambda item: (
            -_metric_value(item, "ndcg@10"),
            -_metric_value(item, "recall@10"),
        ),
    )
    payload = {
        "split": args.split,
        "candidate_k": candidate_k,
        "base_weights": base_weights,
        "selected_stage1_baseline": list(SELECTED_STAGE1_BASELINE),
        "selected_stage1_metrics": selected_stage1_metrics,
        "searched": len(results),
        "top_results": ranked[: args.top_n],
    }
    output_path = args.output or (config.paths.artifact_dir / f"rerank_sweep_{args.split}.json")
    save_json(output_path, payload)
    best = ranked[0]
    print(f"best_ndcg@10: {_metric_value(best, 'ndcg@10'):.6f}")
    print(f"best_recall@10: {_metric_value(best, 'recall@10'):.6f}")
    print(best["weights"])


def _rerank_rankings(
    *,
    top_k_values: tuple[int, ...],
    weights: dict[str, float],
    feature_contexts: list[dict[str, object]],
) -> dict[int, list[int]]:
    reranker_weights = build_reranker_config(False, weights).weights
    rerank_rankings: dict[int, list[int]] = {}
    max_k = max(top_k_values)
    for item in feature_contexts:
        person_id = _safe_int(item.get("person_id", -1))
        if person_id < 0:
            continue
        candidate_features = item.get("candidate_features", [])
        if not isinstance(candidate_features, list):
            continue
        ranked = sorted(
            (
                {
                    "hobby_id": entry["hobby_id"],
                    "score": score_rerank_features(entry["features"], reranker_weights),
                }
                for entry in candidate_features
                if isinstance(entry, dict) and isinstance(entry.get("features"), dict)
            ),
            key=lambda entry: (-float(entry["score"]), int(entry["hobby_id"])),
        )
        rerank_rankings[person_id] = [int(entry["hobby_id"]) for entry in ranked[:max_k]]
    return rerank_rankings


def _precompute_stage2_features(
    *,
    model: LightGCN,
    adjacency: torch.Tensor,
    train_edges: list[tuple[int, int]],
    truth: dict[int, set[int]],
    train_known: dict[int, set[int]],
    id_to_hobby: dict[int, str],
    id_to_person: dict[int, str],
    contexts: dict[str, PersonContext],
    hobby_profile: dict[str, object] | None,
    normalization_method: str,
    candidate_k: int,
    score_chunk_size: int,
    device: torch.device,
    person_embeddings: torch.Tensor,
    hobby_embeddings: torch.Tensor,
    popularity_counts: Counter[int],
    cooccurrence_counts: dict[int, Counter[int]],
) -> list[dict[str, object]]:
    precomputed: list[dict[str, object]] = []
    feature_config = build_reranker_config(False, {})
    if hobby_profile is None:
        raise ValueError("hobby_profile is required for Stage2 weight sweep")
    for person_id in tqdm(truth, desc="precompute stage1 candidates"):
        provider_candidates = _provider_candidates(
            model=model,
            adjacency=adjacency,
            train_edges=train_edges,
            person_id=person_id,
            known=train_known.get(person_id, set()),
            candidate_k=candidate_k,
            chunk_size=score_chunk_size,
            device=device,
            normalization_method=normalization_method,
            hobby_profile=hobby_profile,
            context=contexts.get(id_to_person.get(person_id, "")),
            person_embeddings=person_embeddings,
            hobby_embeddings=hobby_embeddings,
            popularity_counts=popularity_counts,
            cooccurrence_counts=cooccurrence_counts,
        )
        selected_provider_candidates = _selected_stage1_provider_candidates(provider_candidates)
        merged = merge_candidates_by_hobby(selected_provider_candidates, candidate_k)
        hobby_candidates = merge_stage1_candidates(merged, id_to_hobby)
        known_names = {id_to_hobby[hobby_id] for hobby_id in train_known.get(person_id, set()) if hobby_id in id_to_hobby}
        context = contexts.get(id_to_person.get(person_id, ""))
        if context is None:
            raise ValueError(f"missing PersonContext for person_id={person_id} during Stage2 weight sweep")
        candidate_features = [
            {
                "hobby_id": candidate.hobby_id,
                "features": build_rerank_features(context, candidate, hobby_profile, known_names, feature_config),
            }
            for candidate in hobby_candidates
        ]
        precomputed.append(
            {
                "person_id": person_id,
                "hobby_candidates": hobby_candidates,
                "candidate_features": candidate_features,
            }
        )
    return precomputed


def _read_indexed_edges(path: Path) -> list[tuple[int, int]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [(int(row["person_id"]), int(row["hobby_id"])) for row in reader]


def _known_from_edges(edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    known: dict[int, set[int]] = {}
    for person_id, hobby_id in edges:
        known.setdefault(person_id, set()).add(hobby_id)
    return known


def _metric_value(result: dict[str, object], key: str) -> float:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return 0.0
    value = metrics.get(key, 0.0)
    return float(value) if isinstance(value, int | float | str) else 0.0


def _safe_int(value: object) -> int:
    return int(value) if isinstance(value, int | float | str) else -1


if __name__ == "__main__":
    main()
