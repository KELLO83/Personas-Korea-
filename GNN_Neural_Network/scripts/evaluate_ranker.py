from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import time
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.baseline import (  # noqa: E402
    build_cooccurrence_counts,
    build_popularity_counts,
)
from GNN_Neural_Network.gnn_recommender.config import load_config  # noqa: E402
from GNN_Neural_Network.gnn_recommender.data import (
    LEAKAGE_TEXT_FIELDS,
    PersonContext,
    load_alias_map,
    load_json,
    load_person_contexts,
    normalize_hobby_name,
    save_json,
)  # noqa: E402
from GNN_Neural_Network.gnn_recommender.embedding_cache import (
    HobbyEmbeddingCache,
    PersonEmbeddingCache,
)  # noqa: E402
from GNN_Neural_Network.gnn_recommender.metrics import summarize_ranking_metrics  # noqa: E402
from GNN_Neural_Network.gnn_recommender.diversity import (
    compute_hobby_embeddings,
    dpp_rerank,
    mmr_rerank,
)
from GNN_Neural_Network.gnn_recommender.ranker import (
    LightGBMRanker,
    load_or_build_candidate_pool,
    get_candidate_pool_cache_key,
)  # noqa: E402
from GNN_Neural_Network.gnn_recommender.text_embedding import KURE_MODEL_NAME, mask_holdout_hobbies, post_mask_leakage_audit  # noqa: E402
from GNN_Neural_Network.gnn_recommender.rerank import (  # noqa: E402
    build_rerank_features,
    build_reranker_config,
    rerank_candidates,
)

RECALL_GATE = -0.002
NDCG_GATE = 0.005
NDCG_GATE_MMR = -0.002

PHASE5_RECALL_GATE = -0.002
PHASE5_NDCG_GATE = -0.002
PHASE5_DIVERSITY_PROBE_RECALL_GATE = -0.010
PHASE5_DIVERSITY_PROBE_NDCG_GATE = -0.010
PHASE5_DIVERSITY_PROBE_REVIEW_RECALL_GATE = -0.005
PHASE5_DIVERSITY_PROBE_REVIEW_NDCG_GATE = -0.005
PHASE5_CANDIDATE_RECALL_TOLERANCE = 1e-6
PHASE5_BASELINE_PATHS = {
    "validation": Path("GNN_Neural_Network/artifacts/experiments/phase2_5_num_leaves_31/validation_metrics.json"),
    "test": Path("GNN_Neural_Network/artifacts/experiments/phase2_5_num_leaves_31/test_metrics.json"),
}
PHASE5_DIVERSITY_KEYS = (
    "catalog_coverage@10",
    "novelty@10",
    "intra_list_diversity@10",
)
PHASE5_DIVERSITY_SCORE_WEIGHTS = {
    "catalog_coverage@10": 1.0,
    "novelty@10": 1.0,
    "intra_list_diversity@10": 1.0,
}
PHASE5_DIVERSITY_MIN_GAINS = {
    "catalog_coverage@10": 0.025,
    "novelty@10": 0.10,
    "intra_list_diversity@10": 0.02,
}

_TQDM_KWARGS = {
    "miniters": 200,
    "mininterval": 5.0,
    "maxinterval": 30.0,
    "dynamic_ncols": False,
    "ascii": True,
    "leave": False,
    "file": sys.stderr,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LightGBM ranker on a single split.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--use-mmr", action="store_true", help="Apply MMR diversity reordering after ranker scoring")
    parser.add_argument("--mmr-lambda", type=float, default=0.7, help="MMR lambda parameter (0=all diversity, 1=all relevance)")
    parser.add_argument("--use-dpp", action="store_true", help="Apply DPP diversity reordering after ranker scoring")
    parser.add_argument("--dpp-theta", type=float, default=0.5, help="DPP theta parameter (0=all relevance, 1=all diversity)")
    parser.add_argument(
        "--mmr-embedding-method",
        choices=["category_onehot", "kure"],
        default="category_onehot",
        help="Diversity embedding source for MMR/DPP (category_onehot or kure)",
    )
    parser.add_argument("--skip-v1", action="store_true", help="Skip v1 deterministic reranker evaluation")
    parser.add_argument("--pool-cache-dir", type=Path, default=None, help="Directory for candidate pool cache artifacts")
    parser.add_argument("--feature-cache-dir", type=Path, default=None, help="Directory for feature matrix cache artifacts")
    parser.add_argument("--embedding-cache-dir", type=Path, default=None, help="Directory for KURE hobby embedding cache")
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="Batch size for KURE hobby embeddings")
    parser.add_argument("--experiment-id", type=str, default="", help="Optional experiment identifier for artifact naming")
    parser.add_argument(
        "--phase5-kure-mmr",
        action="store_true",
        help="Apply Phase 5 KURE MMR baseline and promotion gates",
    )
    parser.add_argument(
        "--progress-mode",
        choices=["auto", "on", "off"],
        default="off",
        help="Progress output mode: off (default), auto (tty only), on (always).",
    )
    parser.add_argument("--progress-mininterval", type=float, default=5.0, help="Minimum seconds between progress updates")
    parser.add_argument("--progress-maxinterval", type=float, default=30.0, help="Maximum seconds between progress updates")
    parser.add_argument("--progress-miniters", type=int, default=200, help="Minimum updates between progress refresh")
    return parser.parse_args()


def _log_policy(args: argparse.Namespace) -> dict[str, object]:
    return {
        "progress_mode": args.progress_mode,
        "progress_mininterval": float(args.progress_mininterval),
        "progress_maxinterval": float(args.progress_maxinterval),
        "progress_miniters": int(args.progress_miniters),
        "tqdm_enabled": args.progress_mode != "off",
    }


def _command_signature() -> str:
    return " ".join([Path(sys.argv[0]).name, *(shlex.quote(arg) for arg in sys.argv[1:])])


def _iter_with_progress(
    args: argparse.Namespace,
    iterable: Iterable[Any],
    desc: str,
) -> Iterable[Any]:
    if args.progress_mode == "off":
        return iterable
    if args.progress_mode == "auto" and not sys.stderr.isatty():
        return iterable

    try:
        total = len(iterable)  # type: ignore[arg-type]
    except Exception:
        total = None

    kwargs = dict(_TQDM_KWARGS)
    kwargs.update(
        {
            "desc": desc,
            "total": total,
            "mininterval": float(args.progress_mininterval),
            "maxinterval": float(args.progress_maxinterval),
            "miniters": int(args.progress_miniters),
        },
    )
    return tqdm(iterable, **kwargs)


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
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

    input_config_summary = _input_config_summary(
        args.config,
        candidate_pool_size=candidate_k,
        score_normalization=normalization_method,
    )

    model_path = args.model_path or Path("GNN_Neural_Network/artifacts/ranker_model.txt")
    if not model_path.exists():
        raise FileNotFoundError(f"Ranker model not found: {model_path}. Run train_ranker.py first.")
    ranker = LightGBMRanker.load(model_path)
    model_feature_columns = ranker.feature_columns()
    print(f"Loaded ranker model: {model_path} (best_iteration={ranker.best_iteration})")

    popularity_counts = build_popularity_counts(train_edges)
    cooccurrence_counts = build_cooccurrence_counts(train_edges)
    max_k = max(config.eval.top_k)
    truth_person_ids = sorted(truth.keys())

    _write_status(
        args,
        "started",
        runtime_seconds=0.0,
        input_config_summary=input_config_summary,
        summary={
            "phase": "started",
            "split": args.split,
            "total_persons": len(truth_person_ids),
        },
    )

    stage1_rankings: dict[int, list[int]] = {}
    v1_rankings: dict[int, list[int]] = {}
    v2_rankings: dict[int, list[int]] = {}
    candidate_rankings: dict[int, list[int]] = {}
    v2_fallback_count = 0
    mmr_embedding_meta: dict[str, object] = {}

    all_hobby_names = list(hobby_to_id.keys())
    hobby_emb: np.ndarray | None = None
    hobby_id_to_emb_idx: dict[int, int] = {}

    if args.use_mmr and args.use_dpp:
        raise ValueError("--use-mmr and --use-dpp cannot be enabled at the same time")

    mmr_cache_dir = args.embedding_cache_dir or (config.paths.artifact_dir / "hobby_embedding_cache")
    if args.use_mmr or args.use_dpp:
        if args.mmr_embedding_method == "kure":
            hobby_cache = HobbyEmbeddingCache(
                mmr_cache_dir,
                model_name=KURE_MODEL_NAME,
                batch_size=args.embedding_batch_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            hobby_emb, mmr_embedding_meta = hobby_cache.load_or_build_matrix(all_hobby_names)
            mmr_embedding_meta = {
                "embedding_method": "kure",
                "cache_enabled": bool(mmr_embedding_meta.get("cache_enabled", False)),
                "cache_dir": str(mmr_cache_dir),
                "cache_key": str(mmr_embedding_meta.get("cache_key", "")),
                "model_name": str(mmr_embedding_meta.get("model_name", KURE_MODEL_NAME)),
                "batch_size": int(args.embedding_batch_size),
                "embedding_dim": int(mmr_embedding_meta.get("embedding_dim", 0)),
                "num_hobbies": int(mmr_embedding_meta.get("num_hobbies", len(all_hobby_names))),
                "hobby_names_hash": str(mmr_embedding_meta.get("hobby_names_hash", "")),
            }
        else:
            hobby_emb = compute_hobby_embeddings(all_hobby_names, hobby_taxonomy)
            mmr_embedding_meta = {
                "cache_enabled": False,
                "cache_dir": "",
                "cache_key": "",
                "model_name": "",
                "batch_size": None,
                "embedding_dim": int(hobby_emb.shape[1]) if hobby_emb.ndim > 1 else 0,
                "num_hobbies": int(len(all_hobby_names)),
                "embedding_method": "category_onehot",
                "hobby_names_hash": "",
            }
        hobby_id_to_emb_idx = {
            hid: idx for idx, name in enumerate(all_hobby_names) for hid in [hobby_to_id[name]]
        }

    pool_cache_dir = args.pool_cache_dir or config.paths.artifact_dir
    candidate_pool_cache_key = get_candidate_pool_cache_key(
        person_ids=truth_person_ids,
        train_edges=train_edges,
        id_to_hobby=id_to_hobby,
        candidate_k=candidate_k,
        normalization_method=normalization_method,
        label=args.split,
    )
    candidate_pool_cache_path = pool_cache_dir / "cache" / f"{candidate_pool_cache_key}.json"
    feature_cache_npz_path: Path | None = None
    feature_cache_meta_path: Path | None = None
    feature_cache_key: str = ""
    pools_by_person = load_or_build_candidate_pool(
        person_ids=truth_person_ids,
        train_edges=train_edges,
        train_known=train_known,
        candidate_k=candidate_k,
        id_to_hobby=id_to_hobby,
        popularity_counts=popularity_counts,
        cooccurrence_counts=cooccurrence_counts,
        normalization_method=normalization_method,
        cache_dir=pool_cache_dir,
        label=args.split,
        disable_progress=args.progress_mode == "off",
    )

    candidate_pool_policy = _candidate_pool_policy(
        pools_by_person,
        candidate_k=candidate_k,
        normalization_method=normalization_method,
        cache_key=candidate_pool_cache_key,
        cache_path=candidate_pool_cache_path,
    )

    if args.feature_cache_dir is not None:
        feature_cache_key = _feature_cache_key(
            args,
            truth_person_ids,
            pools_by_person,
            model_feature_columns,
            config.paths.person_context_csv,
            config.paths.hobby_profile,
            config.paths.hobby_taxonomy,
        )
        feature_cache_npz_path, feature_cache_meta_path = _feature_cache_paths(
            args,
            truth_person_ids,
            pools_by_person,
            model_feature_columns,
            config.paths.person_context_csv,
            config.paths.hobby_profile,
            config.paths.hobby_taxonomy,
        )

    include_text_embedding_feature = _feature_policy(model_feature_columns)["include_text_embedding_feature"]
    text_similarity_fn: Any = None
    text_embedding_audit: dict[str, object] = {
        "enabled": include_text_embedding_feature,
        "cache_dir": "",
        "known_hobbies_masked": False,
        "audit_pass": True,
        "passed_person_count": 0,
        "failed_person_count": 0,
    }

    if include_text_embedding_feature:
        hobby_aliases = _build_hobby_alias_map(config.paths.hobby_aliases, set(id_to_hobby.values())) if config.paths.hobby_aliases.exists() else {}
        text_cache_dir = args.embedding_cache_dir or (config.paths.artifact_dir / "text_embedding_cache")
        person_embedding_cache = PersonEmbeddingCache(text_cache_dir)
        hobby_embedding_cache = HobbyEmbeddingCache(
            text_cache_dir,
            model_name=KURE_MODEL_NAME,
            batch_size=max(1, int(args.embedding_batch_size)),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        text_prepare_payload = _prepare_text_leakage_context(
            person_ids=truth_person_ids,
            target_edges=target_edges,
            id_to_person=id_to_person,
            contexts=contexts,
            id_to_hobby=id_to_hobby,
            alias_map=hobby_aliases,
        )
        person_text_by_id = text_prepare_payload["person_text_by_id"]
        person_audit_pass = text_prepare_payload["person_audit_pass"]
        text_embedding_audit.update(text_prepare_payload["summary"])
        text_embedding_audit["cache_dir"] = str(text_cache_dir)
        text_embedding_audit["known_hobbies_masked"] = bool(len(text_prepare_payload["person_text_by_id"]) > 0)

        if person_text_by_id:
            text_similarity_fn = _make_text_similarity_fn(
                person_text_by_id=person_text_by_id,
                person_audit_pass=person_audit_pass,
                person_embedding_cache=person_embedding_cache,
                hobby_embedding_cache=hobby_embedding_cache,
            )
        else:
            text_similarity_fn = None
            text_embedding_audit["audit_pass"] = False

    for person_id in _iter_with_progress(args, truth_person_ids, desc=f"rank candidates ({args.split})"):
        hobby_candidates = pools_by_person.get(person_id, [])
        candidate_rankings[person_id] = [c.hobby_id for c in hobby_candidates]
        stage1_rankings[person_id] = candidate_rankings[person_id][:max_k]
        if args.skip_v1:
            v1_rankings[person_id] = []
            continue

        known = train_known.get(person_id, set())
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

    _write_status(
        args,
        "candidates_done",
        summary={
            "phase": "candidates_done",
            "split": args.split,
            "candidate_pool_person_count": len(truth_person_ids),
            "candidate_ranked_person_count": len(candidate_rankings),
        },
    )

    cached_features = _load_feature_cache(
        args,
        truth_person_ids,
        pools_by_person,
        model_feature_columns,
        config.paths.person_context_csv,
        config.paths.hobby_profile,
        config.paths.hobby_taxonomy,
    )
    feature_cache_hit = False
    if cached_features is not None:
        feature_cache_hit = True
        X, person_to_feature_slice, hobby_ids_by_person, fallback_person_ids = cached_features
        v2_fallback_count = len(fallback_person_ids)
    else:
        all_features: list[list[float]] = []
        person_to_feature_slice = {}
        hobby_ids_by_person = {}
        fallback_person_ids = []

        for person_id in _iter_with_progress(args, truth_person_ids, desc=f"features ({args.split})"):
            person_uuid = id_to_person.get(person_id, "")
            person_context = contexts.get(person_uuid)
            hobby_candidates = pools_by_person.get(person_id, [])
            known_names = {id_to_hobby[hid] for hid in train_known.get(person_id, set()) if hid in id_to_hobby}

            if person_context and hobby_candidates:
                start = len(all_features)
                hobby_ids_list: list[int] = []
                for candidate in hobby_candidates:
                    text_embedding_similarity = 0.0
                    if text_similarity_fn is not None:
                        try:
                            text_embedding_similarity = float(text_similarity_fn(person_id, candidate))
                        except Exception:
                            text_embedding_similarity = 0.0
                    features = build_rerank_features(
                        person_context,
                        candidate,
                        hobby_profile,
                        known_names,
                        reranker_config,
                        text_embedding_similarity=text_embedding_similarity,
                    )
                    features.pop("similar_person_score", None)
                    features.pop("persona_text_fit", None)
                    all_features.append([features.get(col, 0.0) for col in model_feature_columns])
                    hobby_ids_list.append(candidate.hobby_id)
                person_to_feature_slice[person_id] = (start, len(all_features))
                hobby_ids_by_person[person_id] = hobby_ids_list
            else:
                fallback_person_ids.append(person_id)
                v2_fallback_count += 1
        X = np.array(all_features, dtype=np.float32) if all_features else np.empty((0, len(model_feature_columns)), dtype=np.float32)
        _save_feature_cache(
            args,
            truth_person_ids,
            pools_by_person,
            model_feature_columns,
            config.paths.person_context_csv,
            config.paths.hobby_profile,
            config.paths.hobby_taxonomy,
            X,
            person_to_feature_slice,
            hobby_ids_by_person,
            fallback_person_ids,
        )

    feature_rows = int(X.shape[0]) if hasattr(X, "shape") else 0
    _write_status(
        args,
        "features_done",
        summary={
            "phase": "features_done",
            "split": args.split,
            "feature_cache_hit": feature_cache_hit,
            "fallback_person_count": len(fallback_person_ids),
            "feature_rows": feature_rows,
        },
    )

    if len(X) > 0:
        all_scores = ranker.predict(X)

        for person_id in _iter_with_progress(args, truth, desc=f"ranking ({args.split})"):
            if person_id in fallback_person_ids:
                v2_rankings[person_id] = stage1_rankings.get(person_id, [])
                continue
            start, end = person_to_feature_slice[person_id]
            scores = all_scores[start:end]
            hobby_ids = hobby_ids_by_person[person_id]
            sorted_indices = np.argsort(-scores)
            sorted_hobby_ids = [hobby_ids[int(i)] for i in sorted_indices]
            sorted_scores = scores[sorted_indices]

            if (not args.use_mmr and not args.use_dpp) or hobby_emb is None:
                v2_rankings[person_id] = sorted_hobby_ids[:max_k]
                continue

            rerank_hobby_ids: list[int] = []
            rerank_scores: list[float] = []
            rerank_emb_indices: list[int] = []
            for idx, hobby_id in enumerate(sorted_hobby_ids):
                emb_idx = hobby_id_to_emb_idx.get(hobby_id)
                if emb_idx is None:
                    continue
                rerank_hobby_ids.append(hobby_id)
                rerank_scores.append(float(sorted_scores[idx]))
                rerank_emb_indices.append(emb_idx)

            if not rerank_emb_indices:
                v2_rankings[person_id] = sorted_hobby_ids[:max_k]
                continue

            emb_subset = hobby_emb[rerank_emb_indices]
            if args.use_dpp:
                v2_rankings[person_id] = dpp_rerank(
                    rerank_hobby_ids,
                    np.asarray(rerank_scores, dtype=np.float32),
                    emb_subset,
                    theta=args.dpp_theta,
                    top_k=max_k,
                )
                continue

            v2_rankings[person_id] = mmr_rerank(
                rerank_hobby_ids,
                np.asarray(rerank_scores, dtype=np.float32),
                emb_subset,
                lambda_param=args.mmr_lambda,
                top_k=max_k,
            )

    _write_status(
        args,
        "v2_rankings_done",
        summary={
            "phase": "v2_rankings_done",
            "split": args.split,
            "ranked_person_count": len(v2_rankings),
            "fallback_count": v2_fallback_count,
        },
    )

    for person_id in truth_person_ids:
        v2_rankings.setdefault(person_id, stage1_rankings.get(person_id, []))

    hobby_categories = _build_hobby_categories(id_to_hobby, hobby_taxonomy)
    person_segments = _build_person_segments(truth.keys(), id_to_person, contexts)
    num_hobbies = len(hobby_to_id)
    cold_start_person_ids = [
        person_id for person_id in truth_person_ids if len(train_known.get(person_id, set())) <= 1
    ]

    cold_start_person_segments = {
        person_id: person_segments[person_id]
        for person_id in cold_start_person_ids
        if person_id in person_segments
    }

    cold_start_truth, cold_start_rankings = _split_person_subset(truth, v2_rankings, cold_start_person_ids)
    cold_start_stage1_truth, cold_start_stage1_rankings = _split_person_subset(truth, stage1_rankings, cold_start_person_ids)
    cold_start_v1_truth: dict[int, set[int]] = {}
    cold_start_v1_rankings: dict[int, list[int]] = {}
    if not args.skip_v1 and v1_rankings:
        cold_start_v1_truth, cold_start_v1_rankings = _split_person_subset(truth, v1_rankings, cold_start_person_ids)

    stage1_metrics = summarize_ranking_metrics(
        truth, stage1_rankings, config.eval.top_k,
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, person_segments=person_segments,
    )

    if not args.skip_v1:
        v1_metrics = summarize_ranking_metrics(
            truth, v1_rankings, config.eval.top_k,
            num_total_items=num_hobbies, item_popularity=popularity_counts,
            hobby_categories=hobby_categories, person_segments=person_segments,
        )
    else:
        v1_metrics = None

    v2_metrics = summarize_ranking_metrics(
        truth, v2_rankings, config.eval.top_k,
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, candidate_pool_by_person=candidate_rankings,
        person_segments=person_segments,
    )
    v2_metrics_cold_start = summarize_ranking_metrics(
        cold_start_truth,
        cold_start_rankings,
        config.eval.top_k,
        num_total_items=num_hobbies,
        item_popularity=popularity_counts,
        hobby_categories=hobby_categories,
        person_segments=cold_start_person_segments,
    )
    stage1_metrics_cold_start = summarize_ranking_metrics(
        cold_start_stage1_truth,
        cold_start_stage1_rankings,
        config.eval.top_k,
        num_total_items=num_hobbies,
        item_popularity=popularity_counts,
        hobby_categories=hobby_categories,
        person_segments=cold_start_person_segments,
    )
    cold_start_v1_metrics = summarize_ranking_metrics(
        cold_start_v1_truth,
        cold_start_v1_rankings,
        config.eval.top_k,
        num_total_items=num_hobbies,
        item_popularity=popularity_counts,
        hobby_categories=hobby_categories,
        person_segments=cold_start_person_segments,
    )
    candidate_recall_metrics = summarize_ranking_metrics(
        truth, candidate_rankings, (candidate_k,),
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories,
    )

    delta_v2_vs_v1 = {}
    delta_v2_vs_stage1 = {}
    phase5_evaluation: dict[str, object] | None = None
    if not args.skip_v1 and v1_metrics is not None:
        delta_v2_vs_v1 = {
            "recall@10": _metric_value(v2_metrics, "recall@10") - _metric_value(v1_metrics, "recall@10"),
            "ndcg@10": _metric_value(v2_metrics, "ndcg@10") - _metric_value(v1_metrics, "ndcg@10"),
            "hit_rate@10": _metric_value(v2_metrics, "hit_rate@10") - _metric_value(v1_metrics, "hit_rate@10"),
        }
        promotion = _promotion_decision(args.split, delta_v2_vs_v1, use_mmr=(args.use_mmr or args.use_dpp))
    else:
        promotion = {"status": "test_only", "gates": {}, "reason": "v1 skipped"}

    if args.phase5_kure_mmr and (args.use_mmr or args.use_dpp):
        phase5_baseline = _load_phase5_kure_baseline(args.split)
        if phase5_baseline is None:
            promotion = {
                "status": "blocked",
                "gates": {},
                "reason": "Phase 5 baseline artifacts not found. Run with completed phase2_5 defaults first.",
            }
        else:
            baseline_v2 = phase5_baseline.get("metrics", {})
            baseline_candidate_recall = phase5_baseline.get("candidate_recall", {})
            delta_v2_vs_phase5 = {
                "recall@10": _metric_value(v2_metrics, "recall@10") - _metric_value(cast(Mapping[str, object], baseline_v2), "recall@10"),
                "ndcg@10": _metric_value(v2_metrics, "ndcg@10") - _metric_value(cast(Mapping[str, object], baseline_v2), "ndcg@10"),
                "candidate_recall@50": _metric_value(candidate_recall_metrics, "recall@50") - _metric_value(
                    cast(Mapping[str, object], baseline_candidate_recall),
                    "recall@50",
                ),
                "coverage@10": _metric_value(v2_metrics, "catalog_coverage@10") - _metric_value(cast(Mapping[str, object], baseline_v2), "catalog_coverage@10"),
                "novelty@10": _metric_value(v2_metrics, "novelty@10") - _metric_value(cast(Mapping[str, object], baseline_v2), "novelty@10"),
                "intra_list_diversity@10": _metric_value(v2_metrics, "intra_list_diversity@10") - _metric_value(
                    cast(Mapping[str, object], baseline_v2),
                    "intra_list_diversity@10",
                ),
                "v2_fallback_count": v2_fallback_count,
            }
            phase5_promotion = _phase5_promotion_decision(
                split=args.split,
                delta_v2_vs_baseline=delta_v2_vs_phase5,
                candidate_recall_delta=_metric_value(candidate_recall_metrics, "recall@50")
                - _metric_value(cast(Mapping[str, object], baseline_candidate_recall), "recall@50"),
                v2_fallback_count=v2_fallback_count,
                mmr_embedding_meta=mmr_embedding_meta,
                baseline_path=phase5_baseline.get("source"),
            )
            phase5_probe = _phase5_diversity_probe_decision(
                split=args.split,
                delta_v2_vs_baseline=delta_v2_vs_phase5,
                candidate_recall_delta=_metric_value(candidate_recall_metrics, "recall@50")
                - _metric_value(cast(Mapping[str, object], baseline_candidate_recall), "recall@50"),
                v2_fallback_count=v2_fallback_count,
                mmr_embedding_meta=mmr_embedding_meta,
                baseline_path=phase5_baseline.get("source"),
            )
            promotion = phase5_promotion
            phase5_evaluation = {
                "mode": "phase5_kure_mmr",
                "baseline_path": str(phase5_baseline.get("source", PHASE5_BASELINE_PATHS.get(args.split, Path("")))),
                "delta_vs_closed_phase2_5": delta_v2_vs_phase5,
                "promotion": phase5_promotion,
                "diversity_probe": phase5_probe,
                "gates": phase5_promotion.get("gates", {}),
                "decision": phase5_promotion,
            }

    delta_v2_vs_stage1 = {
        "recall@10": _metric_value(v2_metrics, "recall@10") - _metric_value(stage1_metrics, "recall@10"),
        "ndcg@10": _metric_value(v2_metrics, "ndcg@10") - _metric_value(stage1_metrics, "ndcg@10"),
        "hit_rate@10": _metric_value(v2_metrics, "hit_rate@10") - _metric_value(stage1_metrics, "hit_rate@10"),
    }

    payload: dict[str, object] = {
        "split": args.split,
        "experiment_id": args.experiment_id,
        "phase5_mode": args.phase5_kure_mmr,
        "status": "validation_evaluated" if args.split == "validation" else "test_evaluated",
        "runtime_seconds": None,
        "model_path": str(model_path),
        "feature_policy": {
            "feature_columns": model_feature_columns,
            "include_source_features": _feature_policy(model_feature_columns)["include_source_features"],
            "include_text_embedding_feature": _feature_policy(model_feature_columns)["include_text_embedding_feature"],
        },
        "input_config_summary": input_config_summary,
        "candidate_pool_policy": candidate_pool_policy,
        "feature_cache_policy": {
            "cache_key": feature_cache_key,
            "cache_path": str(feature_cache_npz_path) if feature_cache_npz_path is not None else "",
            "metadata_path": str(feature_cache_meta_path) if feature_cache_meta_path is not None else "",
            "cache_hit": feature_cache_hit,
        },
        "v2_fallback_count": v2_fallback_count,
        "stage1_baseline": {
            "providers": ["popularity", "cooccurrence"],
            "metrics": stage1_metrics,
        },
        "v2_lightgbm_ranker": {
            "metrics": v2_metrics,
            "delta_vs_v1_reranker": delta_v2_vs_v1,
            "delta_vs_stage1": delta_v2_vs_stage1,
            "phase5_kure_mmr_gates": phase5_evaluation,
            "use_mmr": args.use_mmr,
            "use_dpp": args.use_dpp,
            "mmr_lambda": args.mmr_lambda if args.use_mmr else None,
            "dpp_theta": args.dpp_theta if args.use_dpp else None,
            "mmr_embedding_method": args.mmr_embedding_method if args.use_mmr else None,
            "dpp_embedding_method": args.mmr_embedding_method if args.use_dpp else None,
            "mmr_embedding_meta": mmr_embedding_meta if (args.use_mmr or args.use_dpp) else None,
            "mmr_embedding_batch_size": args.embedding_batch_size if (args.use_mmr or args.use_dpp) else None,
            "text_embedding_feature": {
                "enabled": include_text_embedding_feature,
                "audit": text_embedding_audit,
            },
            "cold_start_subset": {
                "person_count": len(cold_start_truth),
                "known_hobbies_leq": 1,
                "v2_metrics": v2_metrics_cold_start,
                "stage1_metrics": stage1_metrics_cold_start,
                "v1_metrics": None if v1_metrics is None else cold_start_v1_metrics,
            },
        },
        "candidate_recall": candidate_recall_metrics,
        "metrics_summary": {
            "stage1_recall@10": _metric_value(stage1_metrics, "recall@10"),
            "v2_recall@10": _metric_value(v2_metrics, "recall@10"),
            "stage1_ndcg@10": _metric_value(stage1_metrics, "ndcg@10"),
            "v2_ndcg@10": _metric_value(v2_metrics, "ndcg@10"),
            "delta_vs_stage1_recall@10": delta_v2_vs_stage1["recall@10"],
            "delta_vs_stage1_ndcg@10": delta_v2_vs_stage1["ndcg@10"],
            "v2_fallback_count": v2_fallback_count,
            "candidate_recall@50": _metric_value(candidate_recall_metrics, "recall@50"),
            "cold_start_recall@10": _metric_value(v2_metrics_cold_start, "recall@10"),
            "cold_start_ndcg@10": _metric_value(v2_metrics_cold_start, "ndcg@10"),
            "cold_start_coverage@10": _metric_value(v2_metrics_cold_start, "catalog_coverage@10"),
            "cold_start_novelty@10": _metric_value(v2_metrics_cold_start, "novelty@10"),
            "cold_start_intra_list_diversity@10": _metric_value(v2_metrics_cold_start, "intra_list_diversity@10"),
        },
        "promotion_decision": promotion,
    }
    if not args.skip_v1:
        payload["v1_deterministic_reranker"] = {
            "metrics": v1_metrics,
        }

    payload["runtime_seconds"] = time.perf_counter() - start_time

    output_path = args.output or Path("GNN_Neural_Network/artifacts/ranker_eval_metrics.json")
    save_json(output_path, payload)
    print(f"\nResults saved: {output_path}")
    status_summary = {
        "phase": "metrics_done",
        "split": args.split,
        "v2_recall@10": _metric_value(v2_metrics, "recall@10"),
        "v2_ndcg@10": _metric_value(v2_metrics, "ndcg@10"),
        "coverage@10": _metric_value(v2_metrics, "catalog_coverage@10"),
        "novelty@10": _metric_value(v2_metrics, "novelty@10"),
        "candidate_recall@50": _metric_value(candidate_recall_metrics, "recall@50"),
        "v2_fallback_count": v2_fallback_count,
        "promotion_status": str(promotion.get("status", "unknown")),
    }

    if args.phase5_kure_mmr and phase5_evaluation is not None:
        phase5_gates = phase5_evaluation.get("gates", {}) if isinstance(phase5_evaluation, dict) else {}
        phase5_diversity_probe = phase5_evaluation.get("diversity_probe", {}) if isinstance(phase5_evaluation, dict) else {}
        status_summary["phase5_delta_recall@10"] = float(phase5_evaluation.get("delta_vs_closed_phase2_5", {}).get("recall@10", 0.0)) if isinstance(
            phase5_evaluation, dict
        ) else 0.0
        status_summary["phase5_delta_ndcg@10"] = float(phase5_evaluation.get("delta_vs_closed_phase2_5", {}).get("ndcg@10", 0.0)) if isinstance(
            phase5_evaluation, dict
        ) else 0.0
        status_summary["phase5_candidate_recall@50_delta"] = float(
            phase5_evaluation.get("delta_vs_closed_phase2_5", {}).get("candidate_recall@50", 0.0),
        ) if isinstance(phase5_evaluation, dict) else 0.0
        status_summary["phase5_gates"] = phase5_gates
        status_summary["phase5_diversity_probe_status"] = str(
            phase5_diversity_probe.get("status", "not_recorded")
        )

    _write_status(
        args,
        "test_evaluated" if args.split == "test" else "validation_evaluated",
        runtime_seconds=time.perf_counter() - start_time,
        input_config_summary=input_config_summary,
        summary=status_summary,
    )

    print(f"\n{'='*60}")
    if args.use_dpp:
        mode_label = f"v2 LightGBM + DPP (theta={args.dpp_theta}, {args.mmr_embedding_method})"
    elif args.use_mmr:
        mode_label = f"v2 LightGBM + MMR (λ={args.mmr_lambda}, {args.mmr_embedding_method})"
    else:
        mode_label = "v2 LightGBM Ranker"
    print(f"  LightGBM Ranker Evaluation ({args.split})")
    print(f"  Mode: {mode_label}")
    print(f"{'='*60}")
    _print_section("Stage1 (pop+cooc)", stage1_metrics)
    if not args.skip_v1 and v1_metrics is not None:
        _print_section("v1 Deterministic Reranker", v1_metrics)
    _print_section("v2 LightGBM Ranker", v2_metrics)
    print(f"\n--- Delta v2 vs Stage1 ---")
    for key, val in delta_v2_vs_stage1.items():
        sign = "+" if val >= 0 else ""
        print(f"  {key}: {sign}{val:.6f}")
    if delta_v2_vs_v1:
        print(f"\n--- Delta v2 vs v1 ---")
        for key, val in delta_v2_vs_v1.items():
            sign = "+" if val >= 0 else ""
            print(f"  {key}: {sign}{val:.6f}")
    print(f"\n--- Promotion Gate ---")
    if args.phase5_kure_mmr and phase5_evaluation is not None:
        print(f"  Active mode: Phase 5 KURE MMR")
        probe = phase5_evaluation.get("diversity_probe", {}) if isinstance(phase5_evaluation, dict) else {}
        phase5_gates = phase5_evaluation.get("gates", {}) if isinstance(phase5_evaluation, dict) else {}
        for key, details in sorted(phase5_gates.items()):
            if isinstance(details, dict):
                actual = details.get("actual")
                delta = details.get("delta") if isinstance(key, str) else None
                if isinstance(actual, int | float) and (isinstance(delta, int | float) or key == "kure_cache_reusable"):
                    delta_text = f", delta={float(delta):.6f}" if isinstance(delta, int | float) else ""
                    print(f"  phase5 gate {key}: actual={float(actual):.6f}{delta_text}")
        print(f"  phase5 candidate_recall@50 gate: {phase5_evaluation.get('gates', {}).get('candidate_recall@50', {}).get('pass', None)}")
        if isinstance(probe, dict):
            print(f"  phase5 diversity probe status: {probe.get('status', 'not_recorded')}")
    print(f"  Decision: {promotion['status']}")
    if v2_fallback_count > 0:
        print(f"  v2 fallback (missing context): {v2_fallback_count}")


def _load_feature_cache(
    args: argparse.Namespace,
    person_ids: list[int],
    pools_by_person: dict[int, list[Any]],
    feature_columns: list[str],
    person_context_path: Path,
    hobby_profile_path: Path,
    hobby_taxonomy_path: Path,
) -> tuple[np.ndarray, dict[int, tuple[int, int]], dict[int, list[int]], list[int]] | None:
    if args.feature_cache_dir is None:
        return None
    npz_path, meta_path = _feature_cache_paths(
        args,
        person_ids,
        pools_by_person,
        feature_columns,
        person_context_path,
        hobby_profile_path,
        hobby_taxonomy_path,
    )
    if not npz_path.exists() or not meta_path.exists():
        return None

    metadata = _read_feature_cache_metadata(meta_path)
    if metadata is None or not _feature_cache_metadata_matches(metadata, args.split, feature_columns):
        return None

    data = np.load(npz_path)
    persons = [int(v) for v in data["person_ids"]]
    offsets = [int(v) for v in data["offsets"]]
    flat_hobbies = [int(v) for v in data["hobby_ids"]]
    fallback_person_ids = [int(v) for v in data["fallback_person_ids"]]

    person_to_feature_slice: dict[int, tuple[int, int]] = {}
    hobby_ids_by_person: dict[int, list[int]] = {}
    for idx, person_id in enumerate(persons):
        start = offsets[idx]
        end = offsets[idx + 1]
        if end > start:
            person_to_feature_slice[person_id] = (start, end)
            hobby_ids_by_person[person_id] = flat_hobbies[start:end]

    print(f"Loaded feature matrix from cache: {npz_path}")
    return data["X"].astype(np.float32), person_to_feature_slice, hobby_ids_by_person, fallback_person_ids


def _save_feature_cache(
    args: argparse.Namespace,
    person_ids: list[int],
    pools_by_person: dict[int, list[Any]],
    feature_columns: list[str],
    person_context_path: Path,
    hobby_profile_path: Path,
    hobby_taxonomy_path: Path,
    X: np.ndarray,
    person_to_feature_slice: dict[int, tuple[int, int]],
    hobby_ids_by_person: dict[int, list[int]],
    fallback_person_ids: list[int],
) -> None:
    if args.feature_cache_dir is None:
        return
    npz_path, meta_path = _feature_cache_paths(
        args,
        person_ids,
        pools_by_person,
        feature_columns,
        person_context_path,
        hobby_profile_path,
        hobby_taxonomy_path,
    )
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    offsets = [0]
    flat_hobby_ids: list[int] = []
    for person_id in person_ids:
        hobbies = hobby_ids_by_person.get(person_id, [])
        flat_hobby_ids.extend(hobbies)
        offsets.append(offsets[-1] + len(hobbies))

    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32),
        person_ids=np.array(person_ids, dtype=np.int64),
        offsets=np.array(offsets, dtype=np.int64),
        hobby_ids=np.array(flat_hobby_ids, dtype=np.int64),
        fallback_person_ids=np.array(fallback_person_ids, dtype=np.int64),
    )
    meta_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "experiment_id": args.experiment_id,
                "feature_columns": feature_columns,
                "feature_policy": _feature_policy(feature_columns),
                "num_rows": int(X.shape[0]),
                "num_persons": len(person_ids),
                "fallback_count": len(fallback_person_ids),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Feature matrix cached: {npz_path}")


def _feature_cache_paths(
    args: argparse.Namespace,
    person_ids: list[int],
    pools_by_person: dict[int, list[Any]],
    feature_columns: list[str],
    person_context_path: Path,
    hobby_profile_path: Path,
    hobby_taxonomy_path: Path,
) -> tuple[Path, Path]:
    if args.feature_cache_dir is None:
        raise ValueError("feature_cache_dir is required")
    key = _feature_cache_key(
        args,
        person_ids,
        pools_by_person,
        feature_columns,
        person_context_path,
        hobby_profile_path,
        hobby_taxonomy_path,
    )
    cache_dir = args.feature_cache_dir / "cache"
    return cache_dir / f"features_{key}.npz", cache_dir / f"features_{key}.json"


def _feature_cache_key(
    args: argparse.Namespace,
    person_ids: list[int],
    pools_by_person: dict[int, list[Any]],
    feature_columns: list[str],
    person_context_path: Path,
    hobby_profile_path: Path,
    hobby_taxonomy_path: Path,
) -> str:
    payload = {
        "split": args.split,
        "person_ids": sorted(person_ids),
        "feature_columns": feature_columns,
        "feature_policy": _feature_policy(feature_columns),
        "candidate_pool": _candidate_pool_fingerprint(person_ids, pools_by_person),
        "files": {
            "person_context": _file_fingerprint(person_context_path),
            "hobby_profile": _file_fingerprint(hobby_profile_path),
            "hobby_taxonomy": _file_fingerprint(hobby_taxonomy_path),
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _candidate_pool_fingerprint(
    person_ids: list[int],
    pools_by_person: dict[int, list[Any]],
) -> list[list[object]]:
    return [
        [
            person_id,
            [
                {
                    "hobby_id": int(candidate.hobby_id),
                    "source_scores": _sorted_float_items(candidate.source_scores),
                    "raw_source_scores": _sorted_float_items(candidate.raw_source_scores),
                }
                for candidate in pools_by_person.get(person_id, [])
            ],
        ]
        for person_id in sorted(person_ids)
    ]


def _sorted_float_items(values: dict[str, object]) -> list[list[object]]:
    return [[str(key), float(value)] for key, value in sorted(values.items()) if isinstance(value, int | float)]


def _feature_policy(feature_columns: list[str]) -> dict[str, bool]:
    return {
        "include_source_features": any(col.startswith("source_") for col in feature_columns),
        "include_text_embedding_feature": "text_embedding_similarity" in feature_columns,
    }


def _read_feature_cache_metadata(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _feature_cache_metadata_matches(
    metadata: dict[str, object],
    split: str,
    feature_columns: list[str],
) -> bool:
    return (
        metadata.get("split") == split
        and metadata.get("feature_columns") == feature_columns
        and metadata.get("feature_policy") == _feature_policy(feature_columns)
    )


def _file_fingerprint(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {"path": str(path), "exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_status(
    args: argparse.Namespace,
    status: str,
    runtime_seconds: float | None = None,
    input_config_summary: dict[str, object] | None = None,
    summary: dict[str, object] | None = None,
) -> None:
    output_path = args.output or Path("GNN_Neural_Network/artifacts/ranker_eval_metrics.json")
    status_path = output_path.with_suffix(".status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "experiment_id": args.experiment_id,
        "split": args.split,
        "status": status,
        "event_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command_signature": _command_signature(),
        "log_policy": _log_policy(args),
        "artifact_path": str(status_path),
    }
    if runtime_seconds is not None:
        payload["runtime_seconds"] = runtime_seconds
    if input_config_summary is not None:
        payload["input_config_summary"] = input_config_summary
    if summary is not None:
        payload["summary"] = summary
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _input_config_summary(
    config_path: Path,
    *,
    candidate_pool_size: int,
    score_normalization: str,
) -> dict[str, object]:
    return {
        "config_path": str(config_path),
        "candidate_pool_size": candidate_pool_size,
        "score_normalization": score_normalization,
    }


def _candidate_pool_policy(
    pools: dict[int, list[Any]],
    candidate_k: int,
    normalization_method: str,
    cache_key: str,
    cache_path: Path,
) -> dict[str, object]:
    providers: list[str] = []
    seen: set[str] = set()
    for candidates in pools.values():
        for candidate in candidates:
            for provider in candidate.source_scores:
                if provider not in seen:
                    seen.add(provider)
                    providers.append(provider)
    return {
        "providers": providers,
        "candidate_k": candidate_k,
        "normalization_method": normalization_method,
        "cache_key": cache_key,
        "cache_path": str(cache_path),
    }


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


def _split_person_subset(
    truth_by_person: dict[int, set[int]] | Mapping[int, set[int]],
    rankings_by_person: Mapping[int, list[int]],
    person_ids: Iterable[int],
) -> tuple[dict[int, set[int]], dict[int, list[int]]]:
    selected = set(person_ids)
    truth_subset = {person_id: set(truth_by_person[person_id]) for person_id in selected if person_id in truth_by_person}
    rankings_subset = {
        person_id: list(rankings_by_person.get(person_id, [])) for person_id in selected if person_id in rankings_by_person
    }
    return truth_subset, rankings_subset


def _safe_cosine_similarity(vector_a: Any, vector_b: Any) -> float:
    a = np.asarray(vector_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vector_b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if not norm_a or not norm_b:
        return 0.0
    value = float(np.dot(a, b) / (norm_a * norm_b))
    if value != value:
        return 0.0
    if value < 0.0:
        return 0.0
    return min(1.0, value)


def _build_hobby_alias_map(alias_map_path: Path, valid_hobby_names: set[str]) -> dict[str, list[str]]:
    normalized_valid = {normalize_hobby_name(value) for value in valid_hobby_names}
    raw_alias_map = load_alias_map(alias_map_path)
    canonical_to_aliases: dict[str, set[str]] = defaultdict(set)
    for raw_alias, canonical in raw_alias_map.items():
        normalized_alias = normalize_hobby_name(raw_alias)
        normalized_canonical = normalize_hobby_name(canonical)
        if normalized_canonical not in normalized_valid or not normalized_alias:
            continue
        canonical_to_aliases[normalized_canonical].add(normalized_alias)
    return {canonical: sorted(aliases) for canonical, aliases in canonical_to_aliases.items()}


def _prepare_text_leakage_context(
    person_ids: list[int],
    target_edges: list[tuple[int, int]],
    id_to_person: dict[int, str],
    contexts: dict[str, PersonContext],
    id_to_hobby: dict[int, str],
    alias_map: dict[str, list[str]],
) -> dict[str, object]:
    known_by_person: dict[int, set[int]] = defaultdict(set)
    for person_id, hobby_id in target_edges:
        known_by_person[person_id].add(hobby_id)

    person_text_by_id: dict[int, str] = {}
    person_audit_pass: dict[int, bool] = {}
    passed_person_ids: list[int] = []
    failed_person_ids: list[int] = []

    for person_id in person_ids:
        person_uuid = id_to_person.get(person_id, "")
        context = contexts.get(person_uuid)
        if not context:
            person_audit_pass[person_id] = False
            failed_person_ids.append(person_id)
            continue

        holdout_hobby_names = {
            normalize_hobby_name(id_to_hobby[hobby_id])
            for hobby_id in known_by_person.get(person_id, set())
            if hobby_id in id_to_hobby
        }

        field_texts: list[str] = []
        for field in LEAKAGE_TEXT_FIELDS:
            try:
                value = str(getattr(context, field, "") or "").strip()
            except Exception:
                value = ""
            if value:
                field_texts.append(value)

        masked = " ".join(field_texts)
        if holdout_hobby_names:
            masked = mask_holdout_hobbies(masked, holdout_hobby_names, alias_map=alias_map)

        audit_ok = post_mask_leakage_audit(masked, holdout_hobby_names, alias_map=alias_map)
        person_audit_pass[person_id] = bool(audit_ok)
        if audit_ok and masked:
            person_text_by_id[person_id] = masked
            passed_person_ids.append(person_id)
        else:
            failed_person_ids.append(person_id)

    return {
        "person_text_by_id": person_text_by_id,
        "person_audit_pass": person_audit_pass,
        "summary": {
            "audit_pass": not failed_person_ids,
            "passed_person_count": len(passed_person_ids),
            "failed_person_count": len(failed_person_ids),
        },
    }


def _make_text_similarity_fn(
    person_text_by_id: dict[int, str],
    person_audit_pass: dict[int, bool],
    person_embedding_cache: PersonEmbeddingCache,
    hobby_embedding_cache: HobbyEmbeddingCache,
):
    def _score(person_id: int, candidate: Any) -> float:
        if not person_audit_pass.get(person_id, False):
            return 0.0
        person_text = person_text_by_id.get(person_id, "")
        if not person_text:
            return 0.0
        candidate_name = str(getattr(candidate, "hobby_name", "") or "").strip()
        if not candidate_name:
            return 0.0
        person_embedding = person_embedding_cache.encode(person_text)
        hobby_embedding = hobby_embedding_cache.encode(candidate_name)
        return _safe_cosine_similarity(person_embedding, hobby_embedding)

    return _score


def _load_phase5_kure_baseline(split: str) -> dict[str, object] | None:
    path = PHASE5_BASELINE_PATHS.get(split)
    if path is None or not path.exists():
        return None

    raw = load_json(path)
    if not isinstance(raw, dict):
        return None

    v2_ranker = raw.get("v2_lightgbm_ranker")
    if not isinstance(v2_ranker, dict):
        return None

    v2_metrics = v2_ranker.get("metrics")
    if not isinstance(v2_metrics, dict):
        return None

    candidate_recall = raw.get("candidate_recall")
    if not isinstance(candidate_recall, dict):
        candidate_recall = {}

    return {
        "source": str(path),
        "metrics": v2_metrics,
        "candidate_recall": candidate_recall,
    }


def _phase5_promotion_decision(
    *,
    split: str,
    delta_v2_vs_baseline: dict[str, float],
    candidate_recall_delta: float,
    v2_fallback_count: int,
    mmr_embedding_meta: dict[str, object],
    baseline_path: object,
) -> dict[str, object]:
    recall_delta = float(delta_v2_vs_baseline.get("recall@10", 0.0))
    ndcg_delta = float(delta_v2_vs_baseline.get("ndcg@10", 0.0))
    coverage_delta = float(delta_v2_vs_baseline.get("coverage@10", 0.0))
    novelty_delta = float(delta_v2_vs_baseline.get("novelty@10", 0.0))
    ild_delta = float(delta_v2_vs_baseline.get("intra_list_diversity@10", 0.0))

    recall_pass = recall_delta >= PHASE5_RECALL_GATE
    ndcg_pass = ndcg_delta >= PHASE5_NDCG_GATE
    candidate_recall_pass = abs(candidate_recall_delta) <= PHASE5_CANDIDATE_RECALL_TOLERANCE
    fallback_pass = v2_fallback_count == 0
    cache_reusable = bool(mmr_embedding_meta.get("cache_enabled", True))
    diversity_weighted_score = 0.0
    improved_diversity = 0
    improved_metrics: list[str] = []
    for metric_key in PHASE5_DIVERSITY_KEYS:
        delta = float(delta_v2_vs_baseline.get(metric_key, 0.0))
        threshold = float(PHASE5_DIVERSITY_MIN_GAINS.get(metric_key, 0.0))
        weight = float(PHASE5_DIVERSITY_SCORE_WEIGHTS.get(metric_key, 1.0))
        if delta >= threshold:
            improved_diversity += 1
            improved_metrics.append(metric_key)
            diversity_weighted_score += weight

    gates: dict[str, object] = {
        "recall@10": {
            "baseline": "closed_phase2_5",
            "threshold": PHASE5_RECALL_GATE,
            "actual": recall_delta,
            "delta": recall_delta,
            "pass": recall_pass,
        },
        "ndcg@10": {
            "baseline": "closed_phase2_5",
            "threshold": PHASE5_NDCG_GATE,
            "actual": ndcg_delta,
            "delta": ndcg_delta,
            "pass": ndcg_pass,
        },
        "coverage@10": {
            "baseline": "closed_phase2_5",
            "actual": coverage_delta,
            "threshold": PHASE5_DIVERSITY_MIN_GAINS["catalog_coverage@10"],
            "delta": coverage_delta,
            "pass": coverage_delta >= PHASE5_DIVERSITY_MIN_GAINS["catalog_coverage@10"],
        },
        "novelty@10": {
            "baseline": "closed_phase2_5",
            "actual": novelty_delta,
            "threshold": PHASE5_DIVERSITY_MIN_GAINS["novelty@10"],
            "delta": novelty_delta,
            "pass": novelty_delta >= PHASE5_DIVERSITY_MIN_GAINS["novelty@10"],
        },
        "intra_list_diversity@10": {
            "baseline": "closed_phase2_5",
            "actual": ild_delta,
            "threshold": PHASE5_DIVERSITY_MIN_GAINS["intra_list_diversity@10"],
            "delta": ild_delta,
            "pass": ild_delta >= PHASE5_DIVERSITY_MIN_GAINS["intra_list_diversity@10"],
        },
        "candidate_recall@50": {
            "baseline": "closed_phase2_5",
            "tolerance": PHASE5_CANDIDATE_RECALL_TOLERANCE,
            "actual": candidate_recall_delta,
            "delta": candidate_recall_delta,
            "pass": candidate_recall_pass,
        },
        "v2_fallback_count": {
            "baseline": 0,
            "actual": v2_fallback_count,
            "delta": v2_fallback_count,
            "pass": fallback_pass,
        },
        "kure_cache_reusable": {
            "baseline": True,
            "actual": cache_reusable,
            "pass": cache_reusable,
        },
    }

    failed = [
        key for key, values in gates.items() if not bool(values.get("pass", False))
    ]
    diversity_pass = improved_diversity >= 2
    gate_pass = recall_pass and ndcg_pass and diversity_pass and candidate_recall_pass and fallback_pass and cache_reusable
    if split == "validation":
        status = "eligible_for_test" if gate_pass else "blocked"
        reason = "All Phase 5 gates pass" if gate_pass else f"Phase 5 blocked; failed: {', '.join(failed)}"
    elif split == "test":
        status = "promoted" if gate_pass else "blocked"
        reason = "Phase 5 criteria pass" if gate_pass else f"Phase 5 blocked; failed: {', '.join(failed)}"
    else:
        status = "blocked"
        reason = "Unknown split"

    return {
        "status": status,
        "mode": "phase5_kure_mmr",
        "baseline_split": split,
        "baseline_path": str(baseline_path or ""),
        "criteria": {
            "accuracy": {
                "recall_delta": recall_delta,
                "ndcg_delta": ndcg_delta,
            },
            "diversity": {
                "improved_metrics": improved_diversity,
                "improvement_score": diversity_weighted_score,
                "required_improvements": 2,
                "improved_metric_names": improved_metrics,
                "metric_thresholds": PHASE5_DIVERSITY_MIN_GAINS,
            },
            "diversity_improvements_required": "at least 2 of coverage, novelty, intra_list_diversity",
            "candidate_recall_tolerance": PHASE5_CANDIDATE_RECALL_TOLERANCE,
            "fallback_requirement": "zero",
            "kure_cache_reusable": True,
        },
        "gates": gates,
        "reason": reason,
    }


def _phase5_diversity_probe_decision(
    *,
    split: str,
    delta_v2_vs_baseline: dict[str, float],
    candidate_recall_delta: float,
    v2_fallback_count: int,
    mmr_embedding_meta: dict[str, object],
    baseline_path: object,
) -> dict[str, object]:
    recall_delta = float(delta_v2_vs_baseline.get("recall@10", 0.0))
    ndcg_delta = float(delta_v2_vs_baseline.get("ndcg@10", 0.0))
    candidate_recall_pass = abs(candidate_recall_delta) <= PHASE5_CANDIDATE_RECALL_TOLERANCE
    fallback_pass = v2_fallback_count == 0

    diversity_improvements: list[str] = []
    diversity_score = 0.0
    for metric_key in PHASE5_DIVERSITY_KEYS:
        delta = float(delta_v2_vs_baseline.get(metric_key, 0.0))
        threshold = float(PHASE5_DIVERSITY_MIN_GAINS.get(metric_key, 0.0))
        if delta >= threshold:
            diversity_improvements.append(metric_key)
            diversity_score += float(PHASE5_DIVERSITY_SCORE_WEIGHTS.get(metric_key, 1.0))

    recall_accuracy_pass = recall_delta >= PHASE5_DIVERSITY_PROBE_RECALL_GATE
    ndcg_accuracy_pass = ndcg_delta >= PHASE5_DIVERSITY_PROBE_NDCG_GATE
    review_threshold_triggered = (
        recall_delta < PHASE5_DIVERSITY_PROBE_REVIEW_RECALL_GATE
        or ndcg_delta < PHASE5_DIVERSITY_PROBE_REVIEW_NDCG_GATE
    )
    diversity_pass = len(diversity_improvements) >= 2

    gates: dict[str, object] = {
        "recall@10": {
            "baseline": "closed_phase2_5",
            "threshold": PHASE5_DIVERSITY_PROBE_RECALL_GATE,
            "actual": recall_delta,
            "pass": recall_accuracy_pass,
        },
        "ndcg@10": {
            "baseline": "closed_phase2_5",
            "threshold": PHASE5_DIVERSITY_PROBE_NDCG_GATE,
            "actual": ndcg_delta,
            "pass": ndcg_accuracy_pass,
        },
        "catalog_coverage@10": {
            "baseline": "closed_phase2_5",
            "threshold": PHASE5_DIVERSITY_MIN_GAINS["catalog_coverage@10"],
            "actual": float(delta_v2_vs_baseline.get("catalog_coverage@10", 0.0)),
            "pass": float(delta_v2_vs_baseline.get("catalog_coverage@10", 0.0))
            >= float(PHASE5_DIVERSITY_MIN_GAINS["catalog_coverage@10"]),
        },
        "novelty@10": {
            "baseline": "closed_phase2_5",
            "threshold": PHASE5_DIVERSITY_MIN_GAINS["novelty@10"],
            "actual": float(delta_v2_vs_baseline.get("novelty@10", 0.0)),
            "pass": float(delta_v2_vs_baseline.get("novelty@10", 0.0))
            >= float(PHASE5_DIVERSITY_MIN_GAINS["novelty@10"]),
        },
        "intra_list_diversity@10": {
            "baseline": "closed_phase2_5",
            "threshold": PHASE5_DIVERSITY_MIN_GAINS["intra_list_diversity@10"],
            "actual": float(delta_v2_vs_baseline.get("intra_list_diversity@10", 0.0)),
            "pass": float(delta_v2_vs_baseline.get("intra_list_diversity@10", 0.0))
            >= float(PHASE5_DIVERSITY_MIN_GAINS["intra_list_diversity@10"]),
        },
        "candidate_recall@50": {
            "baseline": "closed_phase2_5",
            "tolerance": PHASE5_CANDIDATE_RECALL_TOLERANCE,
            "actual": candidate_recall_delta,
            "pass": candidate_recall_pass,
        },
        "v2_fallback_count": {
            "baseline": 0,
            "actual": v2_fallback_count,
            "pass": fallback_pass,
        },
        "requires_additional_review": {
            "baseline": True,
            "actual": review_threshold_triggered,
            "pass": not review_threshold_triggered,
        },
    }

    gate_pass = (
        recall_accuracy_pass
        and ndcg_accuracy_pass
        and diversity_pass
        and candidate_recall_pass
        and fallback_pass
    )

    if split == "validation":
        if gate_pass:
            status = "requires_additional_review" if review_threshold_triggered else "passed"
            reason = (
                "Diversity probe passed" +
                (", requires additional review" if review_threshold_triggered else "")
            )
        else:
            reason = "Diversity probe accuracy/diversity/stability gates fail"
            status = "blocked"
    elif split == "test":
        if gate_pass:
            status = "needs_review" if review_threshold_triggered else "passed"
            reason = (
                "Diversity probe passed on test" +
                (", requires review" if review_threshold_triggered else "")
            )
        else:
            reason = "Diversity probe accuracy/diversity/stability gates fail"
            status = "blocked"
    else:
        status = "blocked"
        reason = "Unknown split"

    return {
        "status": status,
        "mode": "diversity_probe",
        "baseline_split": split,
        "baseline_path": str(baseline_path or ""),
        "accuracy": {
            "recall_delta": recall_delta,
            "ndcg_delta": ndcg_delta,
            "review_gate_recall": PHASE5_DIVERSITY_PROBE_REVIEW_RECALL_GATE,
            "review_gate_ndcg": PHASE5_DIVERSITY_PROBE_REVIEW_NDCG_GATE,
            "requires_additional_review": review_threshold_triggered,
        },
        "diversity": {
            "improved_metric_names": diversity_improvements,
            "improved_metric_count": len(diversity_improvements),
            "diversity_weighted_score": diversity_score,
            "required_improvements": 2,
            "metric_thresholds": PHASE5_DIVERSITY_MIN_GAINS,
        },
        "stability": {
            "candidate_recall_drift": candidate_recall_delta,
            "fallback_count": v2_fallback_count,
            "kure_cache_reusable": bool(mmr_embedding_meta.get("cache_enabled", True)),
        },
        "gates": gates,
        "reason": reason,
    }


def _promotion_decision(
    split: str,
    delta_v2_vs_v1: dict[str, float],
    *,
    use_mmr: bool = False,
) -> dict[str, object]:
    recall_delta = float(delta_v2_vs_v1.get("recall@10", 0.0))
    ndcg_delta = float(delta_v2_vs_v1.get("ndcg@10", 0.0))
    recall_pass = recall_delta >= RECALL_GATE
    ndcg_threshold = NDCG_GATE_MMR if split == "validation" and use_mmr else NDCG_GATE
    ndcg_pass = ndcg_delta >= ndcg_threshold
    gate_pass = recall_pass and ndcg_pass

    if split == "validation":
        if gate_pass:
            status = "eligible_for_test"
            reason = (
                "v2 passes both gates on validation "
                f"(recall@10 delta={recall_delta:+.6f} >= {RECALL_GATE}, "
                f"ndcg@10 delta={ndcg_delta:+.6f} >= {ndcg_threshold})"
            )
        else:
            failed_gates = []
            if not recall_pass:
                failed_gates.append(f"recall@10 delta={recall_delta:+.6f} < {RECALL_GATE}")
            if not ndcg_pass:
                failed_gates.append(f"ndcg@10 delta={ndcg_delta:+.6f} < {ndcg_threshold}")
            status = "blocked"
            reason = f"v2 fails gate(s) on validation: {'; '.join(failed_gates)}"
    elif split == "test":
        if gate_pass:
            status = "promoted"
            reason = (
                "v2 passes both gates on test "
                f"(recall@10 delta={recall_delta:+.6f} >= {RECALL_GATE}, "
                f"ndcg@10 delta={ndcg_delta:+.6f} >= {ndcg_threshold})"
            )
        else:
            failed_gates = []
            if not recall_pass:
                failed_gates.append(f"recall@10 delta={recall_delta:+.6f} < {RECALL_GATE}")
            if not ndcg_pass:
                failed_gates.append(f"ndcg@10 delta={ndcg_delta:+.6f} < {ndcg_threshold}")
            status = "blocked"
            reason = f"v2 fails gate(s) on test: {'; '.join(failed_gates)}"
    else:
        status = "blocked"
        reason = "Unknown split"

    return {
        "status": status,
        "gates": {
            "recall@10": {"threshold": RECALL_GATE, "actual": recall_delta, "pass": recall_pass},
            "ndcg@10": {"threshold": ndcg_threshold, "actual": ndcg_delta, "pass": ndcg_pass},
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
