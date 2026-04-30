from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.baseline import (  # noqa: E402
    build_bm25_itemknn_counts,
    build_cooccurrence_counts,
    build_idf_weighted_cooccurrence_counts,
    build_jaccard_itemknn_counts,
    build_pmi_itemknn_counts,
    build_pop_capped_cooccurrence_counts,
    build_popularity_counts,
    cooccurrence_candidate_provider,
    idf_weighted_cooccurrence_provider,
    jaccard_itemknn_candidate_provider,
    pmi_itemknn_candidate_provider,
    pop_capped_cooccurrence_provider,
    popularity_candidate_provider,
    segment_popularity_candidate_provider,
)
from GNN_Neural_Network.gnn_recommender.config import load_config  # noqa: E402
from GNN_Neural_Network.gnn_recommender.data import PersonContext, load_json, load_person_contexts, save_json  # noqa: E402
from GNN_Neural_Network.gnn_recommender.metrics import summarize_ranking_metrics  # noqa: E402
from GNN_Neural_Network.gnn_recommender.model import LightGCN, build_normalized_adjacency, choose_device  # noqa: E402
from GNN_Neural_Network.gnn_recommender.recommend import (  # noqa: E402
    Candidate,
    compute_lightgcn_embeddings,
    lightgcn_candidate_provider,
    merge_candidates_by_hobby,
    normalize_candidate_scores,
)
from GNN_Neural_Network.gnn_recommender.rerank import build_reranker_config, merge_stage1_candidates, rerank_candidates  # noqa: E402


SELECTED_STAGE1_BASELINE: tuple[str, ...] = ("popularity", "cooccurrence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic Stage 2 reranker without training.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    candidate_k = args.candidate_k or config.rerank.candidate_pool_size
    if candidate_k <= 0:
        raise ValueError("--candidate-k must be positive")
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
    hobby_taxonomy = _load_hobby_taxonomy(config.paths.hobby_taxonomy, config.paths.artifact_dir)
    normalization_method = _normalization_method(config.paths.score_normalization)
    reranker_config = build_reranker_config(config.rerank.use_text_fit, config.rerank.weights)
    effective_weights = {key: float(value) for key, value in asdict(reranker_config.weights).items()}

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
    idf_cooc_counts = build_idf_weighted_cooccurrence_counts(train_edges)
    pop_capped_counts = build_pop_capped_cooccurrence_counts(train_edges)
    jaccard_counts = build_jaccard_itemknn_counts(train_edges)
    pmi_counts = build_pmi_itemknn_counts(train_edges)

    lightgcn_rankings: dict[int, list[int]] = {}
    stage1_rankings: dict[int, list[int]] = {}
    candidate_rankings: dict[int, list[int]] = {}
    rerank_rankings: dict[int, list[int]] = {}
    stage2_fallback_count = 0
    max_k = max(config.eval.top_k)
    for person_id in tqdm(truth, desc=f"rerank eval ({args.split})"):
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
        lightgcn_rankings[person_id] = [candidate.hobby_id for candidate in provider_candidates["lightgcn"][:max_k]]
        selected_stage1_candidates = _selected_stage1_provider_candidates(provider_candidates)
        merged = merge_candidates_by_hobby(selected_stage1_candidates, candidate_k)
        candidate_rankings[person_id] = [candidate.hobby_id for candidate in merged]
        stage1_rankings[person_id] = [candidate.hobby_id for candidate in merged[:max_k]]
        hobby_candidates = merge_stage1_candidates(merged, id_to_hobby)
        known_names = {id_to_hobby[hobby_id] for hobby_id in train_known.get(person_id, set()) if hobby_id in id_to_hobby}
        reranked = rerank_candidates(
            contexts.get(id_to_person.get(person_id, "")),
            hobby_candidates,
            hobby_profile if isinstance(hobby_profile, dict) else None,
            known_names,
            reranker_config,
            hobby_taxonomy=hobby_taxonomy,
        )
        if reranked and reranked[0].reason_features.get("fallback") == "stage1_score_only":
            stage2_fallback_count += 1
        rerank_rankings[person_id] = [candidate.hobby_id for candidate in reranked[:max_k]]

    hobby_categories = _build_hobby_categories(id_to_hobby, hobby_taxonomy)
    person_segments = _build_person_segments(truth.keys(), id_to_person, contexts)

    lightgcn_metrics = summarize_ranking_metrics(
        truth, lightgcn_rankings, config.eval.top_k,
        num_total_items=model.num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, person_segments=person_segments,
    )
    selected_stage1_metrics = summarize_ranking_metrics(
        truth, stage1_rankings, config.eval.top_k,
        num_total_items=model.num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, person_segments=person_segments,
    )
    stage2_metrics = summarize_ranking_metrics(
        truth, rerank_rankings, config.eval.top_k,
        num_total_items=model.num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, candidate_pool_by_person=candidate_rankings,
        person_segments=person_segments,
    )
    candidate_recall_metrics = summarize_ranking_metrics(
        truth, candidate_rankings, (candidate_k,),
        num_total_items=model.num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories,
    )
    delta_vs_selected_stage1 = {
        "recall@10": _metric_value(stage2_metrics, "recall@10") - _metric_value(selected_stage1_metrics, "recall@10"),
        "ndcg@10": _metric_value(stage2_metrics, "ndcg@10") - _metric_value(selected_stage1_metrics, "ndcg@10"),
        "hit_rate@10": _metric_value(stage2_metrics, "hit_rate@10") - _metric_value(selected_stage1_metrics, "hit_rate@10"),
    }
    promotion_decision = _promotion_decision(args.split, delta_vs_selected_stage1)

    payload = {
        "split": args.split,
        "candidate_k": candidate_k,
        "reranker_mode": "stage2" if stage2_fallback_count == 0 else "stage1_fallback_for_missing_context_or_profile",
        "stage2_fallback_count": stage2_fallback_count,
        "normalization_method": normalization_method,
        "reranker_weights": effective_weights,
        "masking": "train_known_only",
        "lightgcn": lightgcn_metrics,
        "selected_stage1_baseline": {
            "providers": list(SELECTED_STAGE1_BASELINE),
            "metrics": selected_stage1_metrics,
            "candidate_recall": candidate_recall_metrics,
        },
        "stage1_multi_provider": {
            "status": "replaced_by_selected_stage1_baseline",
            "providers": list(SELECTED_STAGE1_BASELINE),
            "metrics": selected_stage1_metrics,
        },
        "stage2_reranker": {
            "metrics": stage2_metrics,
            "delta_vs_selected_stage1": delta_vs_selected_stage1,
        },
        "candidate_recall": candidate_recall_metrics,
        "promotion_decision": promotion_decision,
        "leakage_audit": load_json(config.paths.leakage_audit) if config.paths.leakage_audit.exists() else {"status": "missing"},
    }
    output_path = args.output or config.paths.rerank_metrics
    save_json(output_path, payload)
    save_json(
        config.paths.reranker_weights,
        {
            "source": "config.rerank.weights",
            "use_text_fit": config.rerank.use_text_fit,
            "configured_weights": config.rerank.weights,
            "effective_weights": effective_weights,
            "candidate_k": candidate_k,
            "split": args.split,
        },
    )
    for section in ("lightgcn", "selected_stage1_baseline", "stage2_reranker"):
        values = payload.get(section, {})
        if isinstance(values, dict):
            metric_source = values.get("metrics", values)
            if not isinstance(metric_source, dict):
                continue
            for key, value in sorted(metric_source.items()):
                if isinstance(value, int | float):
                    print(f"{section}_{key}: {value:.6f}")


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


def _provider_candidates(
    *,
    model: LightGCN,
    adjacency: torch.Tensor,
    train_edges: list[tuple[int, int]],
    person_id: int,
    known: set[int],
    candidate_k: int,
    chunk_size: int,
    device: torch.device,
    normalization_method: str,
    hobby_profile: dict[str, object] | None,
    context: PersonContext | None,
    person_embeddings: torch.Tensor,
    hobby_embeddings: torch.Tensor,
    popularity_counts: Counter[int],
    cooccurrence_counts: dict[int, Counter[int]],
    bm25_counts: dict[int, dict[int, float]] | None = None,
    idf_cooc_counts: dict[int, dict[int, float]] | None = None,
    pop_capped_counts: dict[int, dict[int, float]] | None = None,
    jaccard_counts: dict[int, dict[int, float]] | None = None,
    pmi_counts: dict[int, dict[int, float]] | None = None,
) -> dict[str, list[Candidate]]:
    from GNN_Neural_Network.gnn_recommender.baseline import bm25_itemknn_candidate_provider
    return {
        "lightgcn": normalize_candidate_scores(
            lightgcn_candidate_provider(
                model,
                adjacency,
                person_id,
                known,
                candidate_k,
                chunk_size,
                device,
                person_embeddings=person_embeddings,
                hobby_embeddings=hobby_embeddings,
            ),
            normalization_method,
        ),
        "cooccurrence": normalize_candidate_scores(
            cooccurrence_candidate_provider(train_edges, person_id, known, candidate_k, cooccurrence_counts=cooccurrence_counts),
            normalization_method,
        ),
        "popularity": normalize_candidate_scores(
            popularity_candidate_provider(train_edges, person_id, known, candidate_k, popularity_counts=popularity_counts),
            normalization_method,
        ),
        "bm25_itemknn": normalize_candidate_scores(
            bm25_itemknn_candidate_provider(train_edges, person_id, known, candidate_k, bm25_counts=bm25_counts),
            normalization_method,
        ),
        "idf_cooccurrence": normalize_candidate_scores(
            idf_weighted_cooccurrence_provider(train_edges, person_id, known, candidate_k, idf_cooc_counts=idf_cooc_counts),
            normalization_method,
        ),
        "pop_capped_cooccurrence": normalize_candidate_scores(
            pop_capped_cooccurrence_provider(train_edges, person_id, known, candidate_k, pop_capped_counts=pop_capped_counts),
            normalization_method,
        ),
        "jaccard_itemknn": normalize_candidate_scores(
            jaccard_itemknn_candidate_provider(train_edges, person_id, known, candidate_k, jaccard_counts=jaccard_counts),
            normalization_method,
        ),
        "pmi_itemknn": normalize_candidate_scores(
            pmi_itemknn_candidate_provider(train_edges, person_id, known, candidate_k, pmi_counts=pmi_counts),
            normalization_method,
        ),
        "segment_popularity": normalize_candidate_scores(
            segment_popularity_candidate_provider(hobby_profile, context, known, candidate_k),
            normalization_method,
        ),
    }


def _selected_stage1_provider_candidates(provider_candidates: dict[str, list[Candidate]]) -> dict[str, list[Candidate]]:
    return {
        provider: provider_candidates[provider]
        for provider in SELECTED_STAGE1_BASELINE
        if provider in provider_candidates
    }


def _metric_value(metrics: Mapping[str, object], key: str) -> float:
    value = metrics.get(key, 0.0)
    return float(value) if isinstance(value, int | float | str) else 0.0


def _promotion_decision(split: str, delta_vs_selected_stage1: dict[str, float]) -> dict[str, object]:
    recall_delta = float(delta_vs_selected_stage1.get("recall@10", 0.0))
    ndcg_delta = float(delta_vs_selected_stage1.get("ndcg@10", 0.0))
    eligible = recall_delta >= 0.0 or ndcg_delta >= 0.0
    if split == "validation":
        status = "eligible_for_test" if eligible else "blocked"
        reason = (
            "Stage2 matches or beats selected Stage1 baseline on validation recall@10 or ndcg@10"
            if eligible
            else "Stage2 is below selected Stage1 baseline on validation"
        )
    elif split == "test":
        status = "promoted" if eligible else "blocked"
        reason = (
            "Stage2 matches or beats selected Stage1 baseline on test recall@10 or ndcg@10"
            if eligible
            else "Stage2 is below selected Stage1 baseline on test"
        )
    else:
        status = "blocked"
        reason = "Unknown split"
    return {
        "status": status,
        "selected_stage1_baseline": list(SELECTED_STAGE1_BASELINE),
        "criteria": "recall@10 >= baseline OR ndcg@10 >= baseline on the evaluated split",
        "reason": reason,
    }


def _normalization_method(path: Path) -> str:
    if not path.exists():
        return "rank_percentile"
    value = load_json(path)
    if not isinstance(value, dict):
        return "rank_percentile"
    return str(value.get("method", "rank_percentile"))


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
    return {str(key): int(item) for key, item in value.items()}


def _read_indexed_edges(path: Path) -> list[tuple[int, int]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [(int(row["person_id"]), int(row["hobby_id"])) for row in reader]


def _known_from_edges(edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    known: dict[int, set[int]] = defaultdict(set)
    for person_id, hobby_id in edges:
        known[person_id].add(hobby_id)
    return dict(known)


if __name__ == "__main__":
    main()
