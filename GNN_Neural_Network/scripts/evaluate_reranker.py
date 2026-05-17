from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
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
ALL_STAGE1_PROVIDERS: tuple[str, ...] = (
    "lightgcn",
    "popularity",
    "cooccurrence",
    "bm25_itemknn",
    "idf_cooccurrence",
    "pop_capped_cooccurrence",
    "jaccard_itemknn",
    "pmi_itemknn",
    "segment_popularity",
)
LOGGER = logging.getLogger(__name__)
_WORKER_CONTEXT: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic Stage 2 reranker without training.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--providers",
        default=",".join(SELECTED_STAGE1_BASELINE),
        help="Comma-separated Stage1 providers to evaluate, or 'all'. Default: popularity,cooccurrence.",
    )
    parser.add_argument(
        "--include-lightgcn-metrics",
        action="store_true",
        help="Compute standalone LightGCN metrics even when lightgcn is not in --providers.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="CPU worker threads for non-LightGCN reranker evaluation. Use 0 to auto-detect, 1 to disable.",
    )
    parser.add_argument(
        "--worker-chunk-size",
        type=int,
        default=256,
        help="Person IDs per worker task when CPU parallel evaluation is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    config = load_config(args.config)
    candidate_k = args.candidate_k or config.rerank.candidate_pool_size
    if candidate_k <= 0:
        raise ValueError("--candidate-k must be positive")
    if args.worker_chunk_size <= 0:
        raise ValueError("--worker-chunk-size must be positive")
    active_providers = _parse_providers(args.providers)
    computed_providers = active_providers
    if args.include_lightgcn_metrics and "lightgcn" not in computed_providers:
        computed_providers = (*computed_providers, "lightgcn")
    needs_lightgcn = "lightgcn" in active_providers or args.include_lightgcn_metrics
    LOGGER.info(
        "Stage1 provider policy: providers=%s computed_providers=%s include_lightgcn_metrics=%s",
        ",".join(active_providers),
        ",".join(computed_providers),
        args.include_lightgcn_metrics,
    )
    checkpoint = _safe_torch_load(config.paths.checkpoint)
    person_to_id = _expect_mapping(checkpoint.get("person_to_id"), "person_to_id")
    hobby_to_id = _expect_mapping(checkpoint.get("hobby_to_id"), "hobby_to_id")
    num_hobbies = len(hobby_to_id)
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
    model: LightGCN | None = None
    adjacency: torch.Tensor | None = None
    person_embeddings: torch.Tensor | None = None
    hobby_embeddings: torch.Tensor | None = None
    if needs_lightgcn:
        LOGGER.info("Loading LightGCN checkpoint and computing embeddings for requested provider/metrics.")
        model = LightGCN(
            num_persons=int(checkpoint["num_persons"]),
            num_hobbies=int(checkpoint["num_hobbies"]),
            embedding_dim=int(checkpoint["embedding_dim"]),
            num_layers=int(checkpoint["num_layers"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        adjacency = build_normalized_adjacency(model.num_persons, model.num_hobbies, train_edges, device)
        person_embeddings, hobby_embeddings = compute_lightgcn_embeddings(model, adjacency)
    else:
        LOGGER.info("Skipping LightGCN embedding computation; provider and standalone metrics are disabled.")
    popularity_counts = build_popularity_counts(train_edges)
    ranked_popularity = popularity_counts.most_common() if "popularity" in active_providers else []
    cooccurrence_counts = build_cooccurrence_counts(train_edges) if "cooccurrence" in active_providers else None
    bm25_counts = build_bm25_itemknn_counts(train_edges) if "bm25_itemknn" in active_providers else None
    idf_cooc_counts = (
        build_idf_weighted_cooccurrence_counts(train_edges) if "idf_cooccurrence" in active_providers else None
    )
    pop_capped_counts = (
        build_pop_capped_cooccurrence_counts(train_edges) if "pop_capped_cooccurrence" in active_providers else None
    )
    jaccard_counts = build_jaccard_itemknn_counts(train_edges) if "jaccard_itemknn" in active_providers else None
    pmi_counts = build_pmi_itemknn_counts(train_edges) if "pmi_itemknn" in active_providers else None

    lightgcn_rankings: dict[int, list[int]] = {}
    stage1_rankings: dict[int, list[int]] = {}
    candidate_rankings: dict[int, list[int]] = {}
    rerank_rankings: dict[int, list[int]] = {}
    stage2_fallback_count = 0
    max_k = max(config.eval.top_k)
    eval_context = {
        "model": model,
        "adjacency": adjacency,
        "train_edges": train_edges,
        "train_known": train_known,
        "candidate_k": candidate_k,
        "chunk_size": config.eval.score_chunk_size,
        "device": device,
        "normalization_method": normalization_method,
        "hobby_profile": hobby_profile if isinstance(hobby_profile, dict) else None,
        "contexts": contexts,
        "id_to_person": id_to_person,
        "id_to_hobby": id_to_hobby,
        "person_embeddings": person_embeddings,
        "hobby_embeddings": hobby_embeddings,
        "popularity_counts": popularity_counts,
        "ranked_popularity": ranked_popularity,
        "cooccurrence_counts": cooccurrence_counts,
        "bm25_counts": bm25_counts,
        "idf_cooc_counts": idf_cooc_counts,
        "pop_capped_counts": pop_capped_counts,
        "jaccard_counts": jaccard_counts,
        "pmi_counts": pmi_counts,
        "computed_providers": computed_providers,
        "active_providers": active_providers,
        "needs_lightgcn": needs_lightgcn,
        "reranker_config": reranker_config,
        "hobby_taxonomy": hobby_taxonomy,
        "max_k": max_k,
    }
    worker_count = _resolve_worker_count(args.num_workers, len(truth), needs_lightgcn)
    LOGGER.info(
        "Reranker CPU resource plan: workers=%s logical_cpus=%s chunk_size=%s parallelism=%s",
        worker_count,
        os.cpu_count() or 1,
        args.worker_chunk_size,
        "thread_pool" if worker_count > 1 else "single_thread",
    )
    person_ids = list(truth)
    if worker_count > 1:
        chunks = list(_chunks(person_ids, args.worker_chunk_size))
        with ThreadPoolExecutor(
            max_workers=worker_count,
            initializer=_init_reranker_worker,
            initargs=(eval_context,),
        ) as executor:
            iterator = executor.map(_evaluate_person_chunk_worker, chunks)
            for chunk_result in tqdm(iterator, total=len(chunks), desc=f"rerank eval ({args.split})"):
                for result in chunk_result:
                    _merge_person_result(
                        result,
                        lightgcn_rankings,
                        stage1_rankings,
                        candidate_rankings,
                        rerank_rankings,
                    )
                    stage2_fallback_count += result["stage2_fallback_count"]
    else:
        for person_id in tqdm(person_ids, desc=f"rerank eval ({args.split})"):
            result = _evaluate_person(eval_context, person_id)
            _merge_person_result(
                result,
                lightgcn_rankings,
                stage1_rankings,
                candidate_rankings,
                rerank_rankings,
            )
            stage2_fallback_count += result["stage2_fallback_count"]

    hobby_categories = _build_hobby_categories(id_to_hobby, hobby_taxonomy)
    person_segments = _build_person_segments(truth.keys(), id_to_person, contexts)

    lightgcn_metrics: dict[str, object]
    if lightgcn_rankings:
        lightgcn_metrics = summarize_ranking_metrics(
            truth, lightgcn_rankings, config.eval.top_k,
            num_total_items=num_hobbies, item_popularity=popularity_counts,
            hobby_categories=hobby_categories, person_segments=person_segments,
        )
    else:
        lightgcn_metrics = {
            "status": "skipped",
            "reason": "lightgcn provider and --include-lightgcn-metrics are disabled",
        }
    selected_stage1_metrics = summarize_ranking_metrics(
        truth, stage1_rankings, config.eval.top_k,
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, person_segments=person_segments,
    )
    stage2_metrics = summarize_ranking_metrics(
        truth, rerank_rankings, config.eval.top_k,
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, candidate_pool_by_person=candidate_rankings,
        person_segments=person_segments,
    )
    candidate_recall_metrics = summarize_ranking_metrics(
        truth, candidate_rankings, (candidate_k,),
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories,
    )
    delta_vs_selected_stage1 = {
        "recall@10": _metric_value(stage2_metrics, "recall@10") - _metric_value(selected_stage1_metrics, "recall@10"),
        "ndcg@10": _metric_value(stage2_metrics, "ndcg@10") - _metric_value(selected_stage1_metrics, "ndcg@10"),
        "hit_rate@10": _metric_value(stage2_metrics, "hit_rate@10") - _metric_value(selected_stage1_metrics, "hit_rate@10"),
    }
    promotion_decision = _promotion_decision(args.split, delta_vs_selected_stage1, active_providers)

    payload = {
        "split": args.split,
        "candidate_k": candidate_k,
        "reranker_mode": "stage2" if stage2_fallback_count == 0 else "stage1_fallback_for_missing_context_or_profile",
        "stage2_fallback_count": stage2_fallback_count,
        "normalization_method": normalization_method,
        "reranker_weights": effective_weights,
        "masking": "train_known_only",
        "provider_policy": {
            "providers": list(active_providers),
            "computed_providers": list(computed_providers),
            "default_selected_stage1_baseline": list(SELECTED_STAGE1_BASELINE),
            "include_lightgcn_metrics": bool(args.include_lightgcn_metrics),
            "lightgcn_computation": "enabled" if needs_lightgcn else "skipped",
            "cpu_workers": worker_count,
            "worker_chunk_size": args.worker_chunk_size,
        },
        "lightgcn": lightgcn_metrics,
        "selected_stage1_baseline": {
            "providers": list(active_providers),
            "metrics": selected_stage1_metrics,
            "candidate_recall": candidate_recall_metrics,
        },
        "stage1_multi_provider": {
            "status": "active_provider_set",
            "providers": list(active_providers),
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


def _resolve_worker_count(requested: int, person_count: int, needs_lightgcn: bool) -> int:
    if person_count <= 1:
        return 1
    if requested < 0:
        raise ValueError("--num-workers must be >= 0")
    if needs_lightgcn:
        if requested > 1:
            LOGGER.info("Disabling thread workers because LightGCN provider/metrics require local model tensors.")
        return 1
    if requested == 1:
        return 1
    logical_cpus = os.cpu_count() or 1
    if requested > 1:
        return min(requested, person_count)
    return min(max(logical_cpus - 4, 1), 18, person_count)


def _chunks(values: list[int], chunk_size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), chunk_size):
        yield values[start:start + chunk_size]


def _init_reranker_worker(context: dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _evaluate_person_chunk_worker(person_ids: list[int]) -> list[dict[str, object]]:
    return [_evaluate_person(_WORKER_CONTEXT, person_id) for person_id in person_ids]


def _evaluate_person(context: dict[str, Any], person_id: int) -> dict[str, object]:
    train_known: dict[int, set[int]] = context["train_known"]
    id_to_person: dict[int, str] = context["id_to_person"]
    id_to_hobby: dict[int, str] = context["id_to_hobby"]
    contexts: dict[str, PersonContext] = context["contexts"]
    known = train_known.get(person_id, set())
    person_uuid = id_to_person.get(person_id, "")
    provider_candidates = _provider_candidates(
        model=context["model"],
        adjacency=context["adjacency"],
        train_edges=context["train_edges"],
        person_id=person_id,
        known=known,
        candidate_k=context["candidate_k"],
        chunk_size=context["chunk_size"],
        device=context["device"],
        normalization_method=context["normalization_method"],
        hobby_profile=context["hobby_profile"],
        context=contexts.get(person_uuid),
        person_embeddings=context["person_embeddings"],
        hobby_embeddings=context["hobby_embeddings"],
        popularity_counts=context["popularity_counts"],
        ranked_popularity=context["ranked_popularity"],
        cooccurrence_counts=context["cooccurrence_counts"],
        bm25_counts=context["bm25_counts"],
        idf_cooc_counts=context["idf_cooc_counts"],
        pop_capped_counts=context["pop_capped_counts"],
        jaccard_counts=context["jaccard_counts"],
        pmi_counts=context["pmi_counts"],
        providers=context["computed_providers"],
    )
    max_k = int(context["max_k"])
    lightgcn_ranking: list[int] = []
    if context["needs_lightgcn"] and "lightgcn" in provider_candidates:
        lightgcn_ranking = [candidate.hobby_id for candidate in provider_candidates["lightgcn"][:max_k]]
    selected_stage1_candidates = _selected_stage1_provider_candidates(provider_candidates, context["active_providers"])
    merged = merge_candidates_by_hobby(selected_stage1_candidates, int(context["candidate_k"]))
    candidate_ranking = [candidate.hobby_id for candidate in merged]
    stage1_ranking = candidate_ranking[:max_k]
    hobby_candidates = merge_stage1_candidates(merged, id_to_hobby)
    known_names = {id_to_hobby[hobby_id] for hobby_id in known if hobby_id in id_to_hobby}
    reranked = rerank_candidates(
        contexts.get(person_uuid),
        hobby_candidates,
        context["hobby_profile"],
        known_names,
        context["reranker_config"],
        hobby_taxonomy=context["hobby_taxonomy"],
    )
    fallback_count = 1 if reranked and reranked[0].reason_features.get("fallback") == "stage1_score_only" else 0
    return {
        "person_id": person_id,
        "lightgcn_ranking": lightgcn_ranking,
        "stage1_ranking": stage1_ranking,
        "candidate_ranking": candidate_ranking,
        "rerank_ranking": [candidate.hobby_id for candidate in reranked[:max_k]],
        "stage2_fallback_count": fallback_count,
    }


def _merge_person_result(
    result: dict[str, object],
    lightgcn_rankings: dict[int, list[int]],
    stage1_rankings: dict[int, list[int]],
    candidate_rankings: dict[int, list[int]],
    rerank_rankings: dict[int, list[int]],
) -> None:
    person_id = int(result["person_id"])
    lightgcn_ranking = result.get("lightgcn_ranking", [])
    if isinstance(lightgcn_ranking, list) and lightgcn_ranking:
        lightgcn_rankings[person_id] = [int(value) for value in lightgcn_ranking]
    stage1_rankings[person_id] = [int(value) for value in result.get("stage1_ranking", [])]
    candidate_rankings[person_id] = [int(value) for value in result.get("candidate_ranking", [])]
    rerank_rankings[person_id] = [int(value) for value in result.get("rerank_ranking", [])]


def _provider_candidates(
    *,
    model: LightGCN | None,
    adjacency: torch.Tensor | None,
    train_edges: list[tuple[int, int]],
    person_id: int,
    known: set[int],
    candidate_k: int,
    chunk_size: int,
    device: torch.device,
    normalization_method: str,
    hobby_profile: dict[str, object] | None,
    context: PersonContext | None,
    person_embeddings: torch.Tensor | None,
    hobby_embeddings: torch.Tensor | None,
    popularity_counts: Counter[int],
    ranked_popularity: list[tuple[int, int]],
    cooccurrence_counts: dict[int, Counter[int]] | None,
    bm25_counts: dict[int, dict[int, float]] | None = None,
    idf_cooc_counts: dict[int, dict[int, float]] | None = None,
    pop_capped_counts: dict[int, dict[int, float]] | None = None,
    jaccard_counts: dict[int, dict[int, float]] | None = None,
    pmi_counts: dict[int, dict[int, float]] | None = None,
    providers: tuple[str, ...] = SELECTED_STAGE1_BASELINE,
) -> dict[str, list[Candidate]]:
    from GNN_Neural_Network.gnn_recommender.baseline import bm25_itemknn_candidate_provider
    result: dict[str, list[Candidate]] = {}
    if "lightgcn" in providers:
        if model is None or adjacency is None or person_embeddings is None or hobby_embeddings is None:
            raise ValueError("lightgcn provider requires model, adjacency, person_embeddings, and hobby_embeddings")
        result["lightgcn"] = normalize_candidate_scores(
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
        )
    if "cooccurrence" in providers:
        result["cooccurrence"] = normalize_candidate_scores(
            cooccurrence_candidate_provider(train_edges, person_id, known, candidate_k, cooccurrence_counts=cooccurrence_counts),
            normalization_method,
        )
    if "popularity" in providers:
        result["popularity"] = normalize_candidate_scores(
            popularity_candidate_provider(
                ranked_popularity,
                person_id,
                known,
                candidate_k,
                popularity_counts=popularity_counts,
            ),
            normalization_method,
        )
    if "bm25_itemknn" in providers:
        result["bm25_itemknn"] = normalize_candidate_scores(
            bm25_itemknn_candidate_provider(train_edges, person_id, known, candidate_k, bm25_counts=bm25_counts),
            normalization_method,
        )
    if "idf_cooccurrence" in providers:
        result["idf_cooccurrence"] = normalize_candidate_scores(
            idf_weighted_cooccurrence_provider(train_edges, person_id, known, candidate_k, idf_cooc_counts=idf_cooc_counts),
            normalization_method,
        )
    if "pop_capped_cooccurrence" in providers:
        result["pop_capped_cooccurrence"] = normalize_candidate_scores(
            pop_capped_cooccurrence_provider(train_edges, person_id, known, candidate_k, pop_capped_counts=pop_capped_counts),
            normalization_method,
        )
    if "jaccard_itemknn" in providers:
        result["jaccard_itemknn"] = normalize_candidate_scores(
            jaccard_itemknn_candidate_provider(train_edges, person_id, known, candidate_k, jaccard_counts=jaccard_counts),
            normalization_method,
        )
    if "pmi_itemknn" in providers:
        result["pmi_itemknn"] = normalize_candidate_scores(
            pmi_itemknn_candidate_provider(train_edges, person_id, known, candidate_k, pmi_counts=pmi_counts),
            normalization_method,
        )
    if "segment_popularity" in providers:
        result["segment_popularity"] = normalize_candidate_scores(
            segment_popularity_candidate_provider(hobby_profile, context, known, candidate_k),
            normalization_method,
        )
    return result


def _selected_stage1_provider_candidates(
    provider_candidates: dict[str, list[Candidate]],
    providers: tuple[str, ...],
) -> dict[str, list[Candidate]]:
    return {
        provider: provider_candidates[provider]
        for provider in providers
        if provider in provider_candidates
    }


def _parse_providers(value: str) -> tuple[str, ...]:
    normalized = value.strip().lower()
    if normalized == "all":
        return ALL_STAGE1_PROVIDERS
    providers: list[str] = []
    for raw_provider in value.split(","):
        provider = raw_provider.strip()
        if not provider:
            continue
        if provider not in ALL_STAGE1_PROVIDERS:
            raise ValueError(
                f"Unknown provider {provider!r}. Use one of {', '.join(ALL_STAGE1_PROVIDERS)} or 'all'."
            )
        if provider not in providers:
            providers.append(provider)
    if not providers:
        raise ValueError("--providers must contain at least one provider")
    return tuple(providers)


def _metric_value(metrics: Mapping[str, object], key: str) -> float:
    value = metrics.get(key, 0.0)
    return float(value) if isinstance(value, int | float | str) else 0.0


def _promotion_decision(
    split: str,
    delta_vs_selected_stage1: dict[str, float],
    evaluated_providers: tuple[str, ...],
) -> dict[str, object]:
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
        "evaluated_stage1_providers": list(evaluated_providers),
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
