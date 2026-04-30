from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.config import load_config
from GNN_Neural_Network.gnn_recommender.data import load_json, load_person_contexts, save_json
from GNN_Neural_Network.gnn_recommender.metrics import summarize_ranking_metrics
from GNN_Neural_Network.gnn_recommender.model import LightGCN, build_normalized_adjacency, choose_device
from GNN_Neural_Network.gnn_recommender.recommend import Candidate, compute_lightgcn_embeddings, merge_candidates_by_hobby
from GNN_Neural_Network.scripts.evaluate_reranker import _expect_mapping, _known_from_edges, _normalization_method, _provider_candidates, _read_indexed_edges, _safe_torch_load


PROVIDER_ONLY: tuple[tuple[str, ...], ...] = (
    ("popularity",),
    ("cooccurrence",),
    ("bm25_itemknn",),
    ("idf_cooccurrence",),
    ("pop_capped_cooccurrence",),
    ("jaccard_itemknn",),
    ("pmi_itemknn",),
    ("segment_popularity",),
    ("lightgcn",),
)

PROVIDER_COMBINATIONS: tuple[tuple[str, ...], ...] = (
    ("popularity", "cooccurrence"),
    ("popularity", "idf_cooccurrence"),
    ("popularity", "pop_capped_cooccurrence"),
    ("popularity", "jaccard_itemknn"),
    ("popularity", "pmi_itemknn"),
    ("popularity", "cooccurrence", "lightgcn"),
    ("popularity", "idf_cooccurrence", "lightgcn"),
    ("popularity", "pop_capped_cooccurrence", "lightgcn"),
    ("popularity", "jaccard_itemknn", "lightgcn"),
    ("popularity", "pmi_itemknn", "lightgcn"),
    ("popularity", "bm25_itemknn"),
    ("popularity", "bm25_itemknn", "lightgcn"),
    ("segment_popularity", "popularity"),
    ("segment_popularity", "cooccurrence"),
    ("segment_popularity", "cooccurrence", "popularity"),
    ("segment_popularity", "cooccurrence", "popularity", "lightgcn"),
)

SELECTED_STAGE1_BASELINE: tuple[str, ...] = ("popularity", "cooccurrence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage1 provider-only and provider-combination baselines.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    candidate_k = args.candidate_k or config.rerank.candidate_pool_size
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

    popularity_counts = _build_popularity_counts_cached(train_edges)
    cooccurrence_counts = _build_cooccurrence_counts_cached(train_edges)
    bm25_counts = _build_bm25_itemknn_counts_cached(train_edges)
    idf_cooc_counts = _build_idf_cooc_counts_cached(train_edges)
    pop_capped_counts = _build_pop_capped_counts_cached(train_edges)
    jaccard_counts = _build_jaccard_counts_cached(train_edges)
    pmi_counts = _build_pmi_counts_cached(train_edges)
    provider_cache: dict[int, dict[str, list[Candidate]]] = {}
    for person_id in tqdm(truth, desc=f"stage1 ablation cache ({args.split})"):
        provider_cache[person_id] = _provider_candidates(
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
            context=contexts.get(id_to_person.get(person_id, "")),
            person_embeddings=person_embeddings,
            hobby_embeddings=hobby_embeddings,
            popularity_counts=popularity_counts,
            cooccurrence_counts=cooccurrence_counts,
            bm25_counts=bm25_counts,
            idf_cooc_counts=idf_cooc_counts,
            pop_capped_counts=pop_capped_counts,
            jaccard_counts=jaccard_counts,
            pmi_counts=pmi_counts,
        )

    provider_only_results = _evaluate_combinations(PROVIDER_ONLY, provider_cache, truth, config.eval.top_k, candidate_k)
    provider_combo_results = _evaluate_combinations(PROVIDER_COMBINATIONS, provider_cache, truth, config.eval.top_k, candidate_k)

    baseline_key = _combo_name(SELECTED_STAGE1_BASELINE)
    baseline_result = next(item for item in provider_combo_results if item["combo_name"] == baseline_key)
    _attach_delta(provider_only_results, baseline_result)
    _attach_delta(provider_combo_results, baseline_result)

    payload = {
        "split": args.split,
        "candidate_k": candidate_k,
        "masking": "train_known_only",
        "normalization_method": normalization_method,
        "selected_stage1_baseline": list(SELECTED_STAGE1_BASELINE),
        "provider_only": provider_only_results,
        "provider_combinations": provider_combo_results,
        "leakage_audit": load_json(config.paths.leakage_audit) if config.paths.leakage_audit.exists() else {"status": "missing"},
    }
    default_output = config.paths.artifact_dir / f"stage1_ablation_{args.split}.json"
    save_json(args.output or default_output, payload)


def _evaluate_combinations(
    combos: tuple[tuple[str, ...], ...],
    provider_cache: dict[int, dict[str, list[Candidate]]],
    truth: dict[int, set[int]],
    top_k_values: tuple[int, ...],
    candidate_k: int,
) -> list[dict[str, object]]:
    max_k = max(top_k_values)
    results: list[dict[str, object]] = []
    for combo in combos:
        rankings: dict[int, list[int]] = {}
        candidate_rankings: dict[int, list[int]] = {}
        for person_id, providers in provider_cache.items():
            subset: dict[str, list[Candidate]] = {provider: providers[provider] for provider in combo if provider in providers}
            merged = merge_candidates_by_hobby(subset, candidate_k)
            rankings[person_id] = [candidate.hobby_id for candidate in merged[:max_k]]
            candidate_rankings[person_id] = [candidate.hobby_id for candidate in merged]
        metrics = summarize_ranking_metrics(truth, rankings, top_k_values)
        candidate_metrics = summarize_ranking_metrics(truth, candidate_rankings, (candidate_k,))
        results.append(
            {
                "combo_name": _combo_name(combo),
                "providers": list(combo),
                "metrics": metrics,
                "candidate_recall": candidate_metrics,
            }
        )
    return results


def _attach_delta(results: list[dict[str, object]], baseline: dict[str, object]) -> None:
    baseline_metrics = baseline.get("metrics", {})
    baseline_candidate = baseline.get("candidate_recall", {})
    if not isinstance(baseline_metrics, dict) or not isinstance(baseline_candidate, dict):
        return
    baseline_recall = _metric_value(baseline_metrics, "recall@10")
    baseline_ndcg = _metric_value(baseline_metrics, "ndcg@10")
    baseline_hit = _metric_value(baseline_metrics, "hit_rate@10")
    baseline_candidate_recall = _metric_value(baseline_candidate, f"recall@{_candidate_k_from_candidate_metrics(baseline_candidate)}")
    for item in results:
        metrics = item.get("metrics", {})
        candidate_metrics = item.get("candidate_recall", {})
        if not isinstance(metrics, dict) or not isinstance(candidate_metrics, dict):
            continue
        item["delta_vs_selected_baseline"] = {
            "recall@10": _metric_value(metrics, "recall@10") - baseline_recall,
            "ndcg@10": _metric_value(metrics, "ndcg@10") - baseline_ndcg,
            "hit_rate@10": _metric_value(metrics, "hit_rate@10") - baseline_hit,
            "candidate_recall@50": _metric_value(candidate_metrics, f"recall@{_candidate_k_from_candidate_metrics(candidate_metrics)}") - baseline_candidate_recall,
        }


def _metric_value(metrics: dict[str, object], key: str) -> float:
    value = metrics.get(key, 0.0)
    return float(value) if isinstance(value, int | float | str) else 0.0


def _candidate_k_from_candidate_metrics(metrics: dict[str, object]) -> int:
    for key in metrics:
        if key.startswith("recall@"):
            suffix = key.split("@", maxsplit=1)[1]
            return int(suffix) if suffix.isdigit() else 50
    return 50


def _combo_name(combo: tuple[str, ...]) -> str:
    return "+".join(combo)


_popularity_counts_cache: Counter[int] | None = None
_cooccurrence_counts_cache: dict[int, Counter[int]] | None = None
_bm25_itemknn_counts_cache: dict[int, dict[int, float]] | None = None
_idf_cooc_counts_cache: dict[int, dict[int, float]] | None = None
_pop_capped_counts_cache: dict[int, dict[int, float]] | None = None
_jaccard_counts_cache: dict[int, dict[int, float]] | None = None
_pmi_counts_cache: dict[int, dict[int, float]] | None = None


def _build_popularity_counts_cached(train_edges: list[tuple[int, int]]) -> Counter[int]:
    from GNN_Neural_Network.gnn_recommender.baseline import build_popularity_counts

    global _popularity_counts_cache
    if _popularity_counts_cache is None:
        _popularity_counts_cache = build_popularity_counts(train_edges)
    return _popularity_counts_cache


def _build_cooccurrence_counts_cached(train_edges: list[tuple[int, int]]) -> dict[int, Counter[int]]:
    from GNN_Neural_Network.gnn_recommender.baseline import build_cooccurrence_counts

    global _cooccurrence_counts_cache
    if _cooccurrence_counts_cache is None:
        _cooccurrence_counts_cache = build_cooccurrence_counts(train_edges)
    return _cooccurrence_counts_cache


def _build_bm25_itemknn_counts_cached(train_edges: list[tuple[int, int]]) -> dict[int, dict[int, float]]:
    from GNN_Neural_Network.gnn_recommender.baseline import build_bm25_itemknn_counts

    global _bm25_itemknn_counts_cache
    if _bm25_itemknn_counts_cache is None:
        _bm25_itemknn_counts_cache = build_bm25_itemknn_counts(train_edges)
    return _bm25_itemknn_counts_cache


def _build_idf_cooc_counts_cached(train_edges: list[tuple[int, int]]) -> dict[int, dict[int, float]]:
    from GNN_Neural_Network.gnn_recommender.baseline import build_idf_weighted_cooccurrence_counts

    global _idf_cooc_counts_cache
    if _idf_cooc_counts_cache is None:
        _idf_cooc_counts_cache = build_idf_weighted_cooccurrence_counts(train_edges)
    return _idf_cooc_counts_cache


def _build_pop_capped_counts_cached(train_edges: list[tuple[int, int]]) -> dict[int, dict[int, float]]:
    from GNN_Neural_Network.gnn_recommender.baseline import build_pop_capped_cooccurrence_counts

    global _pop_capped_counts_cache
    if _pop_capped_counts_cache is None:
        _pop_capped_counts_cache = build_pop_capped_cooccurrence_counts(train_edges)
    return _pop_capped_counts_cache


def _build_jaccard_counts_cached(train_edges: list[tuple[int, int]]) -> dict[int, dict[int, float]]:
    from GNN_Neural_Network.gnn_recommender.baseline import build_jaccard_itemknn_counts

    global _jaccard_counts_cache
    if _jaccard_counts_cache is None:
        _jaccard_counts_cache = build_jaccard_itemknn_counts(train_edges)
    return _jaccard_counts_cache


def _build_pmi_counts_cached(train_edges: list[tuple[int, int]]) -> dict[int, dict[int, float]]:
    from GNN_Neural_Network.gnn_recommender.baseline import build_pmi_itemknn_counts

    global _pmi_counts_cache
    if _pmi_counts_cache is None:
        _pmi_counts_cache = build_pmi_itemknn_counts(train_edges)
    return _pmi_counts_cache


if __name__ == "__main__":
    main()
