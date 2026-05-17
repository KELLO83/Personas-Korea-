from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import shlex
import subprocess
import time
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import numpy as np
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.baseline import (  # noqa: E402
    build_cooccurrence_counts,
    build_popularity_counts,
)
from GNN_Neural_Network.gnn_recommender.config import load_config, validate_experimental_feature_policy  # noqa: E402
from GNN_Neural_Network.gnn_recommender.data import (
    LEAKAGE_TEXT_FIELDS,
    PersonContext,
    build_domain_persona_texts,
    build_domain_tagged_persona_text,
    empty_person_context,
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
    RANKER_DOMAIN_TEXT_FEATURE_COLUMNS,
    RANKER_TEXT_RANK_MARGIN_FEATURE_COLUMNS,
    build_text_rank_margin_lookup,
    load_or_build_candidate_pool,
    get_candidate_pool_cache_key,
    build_kure_semantic_candidate_scores,
)  # noqa: E402
from GNN_Neural_Network.gnn_recommender.text_embedding import KURE_MODEL_NAME, mask_holdout_hobbies, post_mask_leakage_audit  # noqa: E402
from GNN_Neural_Network.gnn_recommender.rerank import (  # noqa: E402
    build_rerank_features,
    build_reranker_config,
    rerank_candidates,
)

TEXT_EMBEDDING_PREPROCESSING_VERSION = "domain_tagged_masked_v1"

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
FEATURE_CACHE_VERSION = 2
_feature_worker_hobby_profile: dict[str, object] | None = None
_feature_worker_profile_cache: dict[str, dict[str, object]] = {}
_feature_worker_reranker_config: Any = None
_feature_worker_model_feature_columns: list[str] = []
_ranking_worker_train_known: dict[int, set[int]] = {}
_ranking_worker_id_to_hobby: dict[int, str] = {}
_ranking_worker_id_to_person: dict[int, str] = {}
_ranking_worker_contexts: dict[str, PersonContext] = {}
_ranking_worker_hobby_profile: dict[str, object] | None = None
_ranking_worker_reranker_config: Any = None
_ranking_worker_hobby_taxonomy: dict[str, object] | None = None
_ranking_worker_max_k = 10


def _torch_module() -> Any:
    import torch

    return torch


def _init_feature_worker(
    hobby_profile: dict[str, object],
    reranker_config: Any,
    model_feature_columns: list[str],
) -> None:
    global _feature_worker_hobby_profile
    global _feature_worker_profile_cache
    global _feature_worker_reranker_config
    global _feature_worker_model_feature_columns
    _feature_worker_hobby_profile = hobby_profile
    _feature_worker_profile_cache = _build_profile_feature_cache(hobby_profile)
    _feature_worker_reranker_config = reranker_config
    _feature_worker_model_feature_columns = model_feature_columns


def _build_feature_rows_for_person(
    payload: tuple[int, PersonContext, list[Any], set[str], dict[int, float], dict[int, dict[str, float]], dict[int, dict[str, float]]],
) -> tuple[int, list[list[float]], list[int], bool]:
    person_id, person_context, hobby_candidates, known_names, text_scores, domain_scores, rank_margin_scores = payload
    if not hobby_candidates:
        return person_id, [], [], True
    if _feature_worker_hobby_profile is None or _feature_worker_reranker_config is None:
        raise RuntimeError("feature worker was not initialized")

    rows: list[list[float]] = []
    hobby_ids: list[int] = []
    for candidate in hobby_candidates:
        features = _build_fast_rerank_features(
            person_context,
            candidate,
            known_names,
            _feature_worker_reranker_config,
            text_embedding_similarity=float(text_scores.get(candidate.hobby_id, 0.0)),
            domain_text_embedding_similarities=domain_scores.get(candidate.hobby_id, {}),
            text_rank_margin_features=rank_margin_scores.get(candidate.hobby_id, {}),
        )
        rows.append([features.get(col, 0.0) for col in _feature_worker_model_feature_columns])
        hobby_ids.append(candidate.hobby_id)
    return person_id, rows, hobby_ids, False


def _build_profile_feature_cache(hobby_profile: dict[str, object] | None) -> dict[str, dict[str, object]]:
    if not isinstance(hobby_profile, dict):
        return {}
    hobbies = hobby_profile.get("hobbies", {})
    if not isinstance(hobbies, dict):
        return {}
    max_popularity = 0.0
    for entry in hobbies.values():
        if isinstance(entry, dict):
            max_popularity = max(max_popularity, _safe_float(entry.get("train_popularity", 0.0)))

    cache: dict[str, dict[str, object]] = {}
    for hobby_name, entry in hobbies.items():
        if not isinstance(entry, dict):
            continue
        popularity = _safe_float(entry.get("train_popularity", 0.0))
        distributions = entry.get("distributions", {})
        if not isinstance(distributions, dict):
            distributions = {}
        cooccurring_pairs: list[tuple[str, float]] = []
        cooccurring_total = 0.0
        cooccurring = entry.get("cooccurring_hobbies", [])
        if isinstance(cooccurring, list):
            for item in cooccurring:
                if not isinstance(item, dict):
                    continue
                count = _safe_float(item.get("count", 0.0))
                cooccurring_total += count
                cooccurring_pairs.append((str(item.get("hobby_name", "")), count))
        popularity_penalty = (
            math.log1p(popularity) / math.log1p(max_popularity)
            if max_popularity > 0.0
            else 0.0
        )
        cache[str(hobby_name)] = {
            "distributions": distributions,
            "cooccurring_pairs": cooccurring_pairs,
            "cooccurring_total": cooccurring_total,
            "popularity_prior": popularity / max_popularity if max_popularity else 0.0,
            "popularity_penalty": popularity_penalty,
            "novelty_bonus": 1.0 - popularity_penalty,
        }
    return cache


def _build_fast_rerank_features(
    context: PersonContext,
    candidate: Any,
    known_hobby_names: set[str],
    config: Any,
    text_embedding_similarity: float = 0.0,
    domain_text_embedding_similarities: Mapping[str, float] | None = None,
    text_rank_margin_features: Mapping[str, float] | None = None,
) -> dict[str, float]:
    cached = _feature_worker_profile_cache.get(str(candidate.hobby_name), {})
    distributions = cached.get("distributions", {})
    distributions = distributions if isinstance(distributions, dict) else {}
    source_scores = candidate.source_scores or {}
    source_keys = set(source_scores)
    source_is_popularity = 1.0 if "popularity" in source_keys else 0.0
    source_is_cooccurrence = 1.0 if "cooccurrence" in source_keys else 0.0
    features = {
        "lightgcn_score": _safe_float(source_scores.get("lightgcn", 0.0)),
        "cooccurrence_score": _safe_float(source_scores.get("cooccurrence", 0.0)),
        "segment_popularity_score": _safe_float(source_scores.get("segment_popularity", 0.0)),
        "similar_person_score": 0.0,
        "persona_text_fit": 0.0,
        "known_hobby_compatibility": _known_hobby_compatibility_cached(cached, known_hobby_names),
        "age_group_fit": _distribution_fit_cached(distributions, "age_group", context.age_group),
        "occupation_fit": _distribution_fit_cached(distributions, "occupation", context.occupation),
        "region_fit": max(
            _distribution_fit_cached(distributions, "province", context.province),
            _distribution_fit_cached(distributions, "district", context.district),
        ),
        "popularity_prior": _safe_float(cached.get("popularity_prior", 0.0)),
        "mismatch_penalty": _mismatch_penalty_cached(distributions, context),
        "popularity_penalty": _safe_float(cached.get("popularity_penalty", 0.0)),
        "novelty_bonus": _safe_float(cached.get("novelty_bonus", 0.0)),
        "category_diversity_reward": 0.0,
        "is_cold_start": 1.0 if len(known_hobby_names) <= 1 else 0.0,
        "source_is_popularity": source_is_popularity,
        "source_is_cooccurrence": source_is_cooccurrence,
        "source_count": source_is_popularity + source_is_cooccurrence,
        "text_embedding_similarity": text_embedding_similarity,
    }
    if domain_text_embedding_similarities:
        for column in RANKER_DOMAIN_TEXT_FEATURE_COLUMNS:
            features[column] = _safe_float(domain_text_embedding_similarities.get(column, 0.0))
    if text_rank_margin_features:
        for column in RANKER_TEXT_RANK_MARGIN_FEATURE_COLUMNS:
            features[column] = _safe_float(text_rank_margin_features.get(column, 0.0))
    return features


def _distribution_fit_cached(distributions: dict[object, object], field: str, value: str) -> float:
    if not value:
        return 0.0
    distribution = distributions.get(field, {})
    if not isinstance(distribution, dict) or not distribution:
        return 0.0
    total = sum(_safe_float(count) for count in distribution.values())
    return _safe_float(distribution.get(value, 0.0)) / total if total else 0.0


def _known_hobby_compatibility_cached(cached: dict[str, object], known_hobby_names: set[str]) -> float:
    if not known_hobby_names:
        return 0.0
    pairs = cached.get("cooccurring_pairs", [])
    total = _safe_float(cached.get("cooccurring_total", 0.0))
    if not isinstance(pairs, list) or total <= 0.0:
        return 0.0
    matched = sum(count for name, count in pairs if name in known_hobby_names)
    return matched / total if total else 0.0


def _mismatch_penalty_cached(distributions: dict[object, object], context: PersonContext) -> float:
    fields = {
        "age_group": context.age_group,
        "occupation": context.occupation,
        "sex": context.sex,
    }
    penalties = []
    for field, value in fields.items():
        distribution = distributions.get(field, {})
        if value and isinstance(distribution, dict) and distribution:
            fit = _distribution_fit_cached(distributions, field, value)
            if fit < 0.05:
                penalties.append(1.0 - fit)
    return sum(penalties) / len(fields) if penalties else 0.0


def _safe_float(value: object) -> float:
    return float(value) if isinstance(value, int | float | str) else 0.0


def _init_ranking_worker(
    train_known: dict[int, set[int]],
    id_to_hobby: dict[int, str],
    id_to_person: dict[int, str],
    contexts: dict[str, PersonContext],
    hobby_profile: dict[str, object],
    reranker_config: Any,
    hobby_taxonomy: dict[str, object] | None,
    max_k: int,
) -> None:
    global _ranking_worker_train_known
    global _ranking_worker_id_to_hobby
    global _ranking_worker_id_to_person
    global _ranking_worker_contexts
    global _ranking_worker_hobby_profile
    global _ranking_worker_reranker_config
    global _ranking_worker_hobby_taxonomy
    global _ranking_worker_max_k
    _ranking_worker_train_known = train_known
    _ranking_worker_id_to_hobby = id_to_hobby
    _ranking_worker_id_to_person = id_to_person
    _ranking_worker_contexts = contexts
    _ranking_worker_hobby_profile = hobby_profile
    _ranking_worker_reranker_config = reranker_config
    _ranking_worker_hobby_taxonomy = hobby_taxonomy
    _ranking_worker_max_k = max_k


def _build_stage1_v1_rankings_for_person(payload: tuple[int, list[Any]]) -> tuple[int, list[int], list[int], list[int]]:
    if _ranking_worker_hobby_profile is None or _ranking_worker_reranker_config is None:
        raise RuntimeError("ranking worker was not initialized")
    person_id, hobby_candidates = payload
    candidate_ranking = [candidate.hobby_id for candidate in hobby_candidates]
    known = _ranking_worker_train_known.get(person_id, set())
    known_names = {
        _ranking_worker_id_to_hobby[hobby_id]
        for hobby_id in known
        if hobby_id in _ranking_worker_id_to_hobby
    }
    reranked = rerank_candidates(
        _ranking_worker_contexts.get(_ranking_worker_id_to_person.get(person_id, "")),
        hobby_candidates,
        _ranking_worker_hobby_profile,
        known_names,
        _ranking_worker_reranker_config,
        hobby_taxonomy=_ranking_worker_hobby_taxonomy,
    )
    return person_id, candidate_ranking, candidate_ranking[:_ranking_worker_max_k], [c.hobby_id for c in reranked[:_ranking_worker_max_k]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LightGBM ranker on a single split.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgbm_ranker.yaml"))
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
    parser.add_argument("--max-persons", type=int, default=0, help="Optional split-person cap for fast pilot evaluation")
    parser.add_argument("--stage1-kure-semantic-provider", action="store_true", help="Enable opt-in KURE-v1 Stage1 semantic candidate provider")
    parser.add_argument("--stage1-kure-score-batch-size", type=int, default=128, help="Person batch size for KURE Stage1 semantic scoring")
    parser.add_argument("--pool-cache-dir", type=Path, default=None, help="Directory for candidate pool cache artifacts")
    parser.add_argument("--feature-cache-dir", type=Path, default=None, help="Directory for feature matrix cache artifacts")
    parser.add_argument("--disable-feature-cache", action="store_true", help="Disable default feature matrix cache")
    parser.add_argument("--embedding-cache-dir", type=Path, default=None, help="Directory for KURE hobby embedding cache")
    parser.add_argument(
        "--text-embedding-model-name",
        type=str,
        default=KURE_MODEL_NAME,
        help="SentenceTransformer model name for text embedding features.",
    )
    parser.add_argument(
        "--text-embedding-model-revision",
        type=str,
        default="",
        help="Optional model revision used for text embedding cache identity.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for KURE embeddings. Use 0 to auto-size from available GPU VRAM.",
    )
    parser.add_argument(
        "--embedding-vram-utilization",
        type=float,
        default=0.85,
        help="Target fraction of currently free GPU VRAM to use when --embedding-batch-size=0.",
    )
    parser.add_argument(
        "--embedding-target-vram-mb",
        type=int,
        default=0,
        help="Absolute target GPU VRAM MB for KURE embedding auto batch. Overrides utilization when >0.",
    )
    parser.add_argument(
        "--candidate-text-builder",
        choices=["name_only", "name_plus_aliases", "name_plus_category", "name_plus_short_description"],
        default="name_only",
        help="Candidate hobby text builder for Stage2 embedding features.",
    )
    parser.add_argument("--experiment-id", type=str, default="", help="Optional experiment identifier for artifact naming")
    parser.add_argument(
        "--phase5-kure-mmr",
        action="store_true",
        help="Apply Phase 5 KURE MMR baseline and promotion gates",
    )
    parser.add_argument(
        "--progress-mode",
        choices=["auto", "on", "off"],
        default="on",
        help="Progress output mode: on (default), auto (tty only), off.",
    )
    parser.add_argument("--progress-mininterval", type=float, default=5.0, help="Minimum seconds between progress updates")
    parser.add_argument("--progress-maxinterval", type=float, default=30.0, help="Maximum seconds between progress updates")
    parser.add_argument("--progress-miniters", type=int, default=200, help="Minimum updates between progress refresh")
    parser.add_argument(
        "--cpu-thread-count",
        type=int,
        default=0,
        help="CPU threads for PyTorch/LightGBM predict. Use 0 to auto-detect logical CPUs.",
    )
    parser.add_argument(
        "--feature-build-parallelism",
        choices=["auto", "thread", "serial"],
        default="auto",
        help="Parallel backend for CPU-bound feature row construction. auto uses thread workers.",
    )
    parser.add_argument(
        "--ranking-build-parallelism",
        choices=["auto", "thread", "serial"],
        default="auto",
        help="Parallel backend for Stage1/V1 ranking construction. auto uses thread workers.",
    )
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


def _resolve_system_resource_plan(args: argparse.Namespace) -> dict[str, object]:
    logical_cpus = os.cpu_count() or 1
    default_cpu_threads = min(max(logical_cpus - 4, 1), 18)
    requested_threads = int(args.cpu_thread_count)
    cpu_threads = default_cpu_threads if requested_threads <= 0 else max(1, min(requested_threads, logical_cpus))
    memory_total_mb, memory_available_mb = _query_system_memory_mb()
    gpu_total_mb, gpu_used_mb, gpu_free_mb = _query_gpu_memory_mb()
    return {
        "logical_cpus": logical_cpus,
        "default_cpu_threads": default_cpu_threads,
        "requested_cpu_threads": requested_threads,
        "cpu_threads": cpu_threads,
        "system_memory_total_mb": memory_total_mb,
        "system_memory_available_mb": memory_available_mb,
        "gpu_total_vram_mb": gpu_total_mb,
        "gpu_used_vram_mb": gpu_used_mb,
        "gpu_free_vram_mb": gpu_free_mb,
        "feature_builder_parallelism": "auto_thread_pool",
        "ranking_builder_parallelism": "auto_thread_pool",
        "lightgbm_predict_threads": cpu_threads,
        "torch_threads": cpu_threads,
    }


def _resolve_parallel_backend(value: str) -> str:
    backend = str(value or "auto").strip().lower()
    if backend == "auto":
        return "thread"
    if backend not in {"thread", "serial"}:
        raise ValueError(f"Unsupported parallel backend: {value}")
    return backend


def _apply_cpu_resource_plan(plan: Mapping[str, object]) -> None:
    cpu_threads = int(plan.get("cpu_threads", 1) or 1)
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    try:
        torch_module = _torch_module()
        torch_module.set_num_threads(cpu_threads)
        torch_module.set_num_interop_threads(max(1, min(4, cpu_threads)))
    except RuntimeError:
        pass


def _query_system_memory_mb() -> tuple[int, int]:
    try:
        import psutil

        mem = psutil.virtual_memory()
        return int(mem.total // (1024 * 1024)), int(mem.available // (1024 * 1024))
    except Exception:
        return 0, 0


def _iter_with_progress(
    args: argparse.Namespace,
    iterable: Iterable[Any],
    desc: str,
    total_count: int | None = None,
) -> Iterable[Any]:
    if args.progress_mode == "off":
        return iterable
    if args.progress_mode == "auto" and not sys.stderr.isatty():
        return iterable

    if total_count is not None:
        total = total_count
    else:
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    _configure_third_party_logging()
    args = parse_args()
    text_embedding_model_name = str(args.text_embedding_model_name or KURE_MODEL_NAME).strip() or KURE_MODEL_NAME
    text_embedding_model_revision = str(args.text_embedding_model_revision or "").strip()
    system_resource_plan = _resolve_system_resource_plan(args)
    _apply_cpu_resource_plan(system_resource_plan)
    LOGGER.info(
        "Evaluation system resource plan: cpu_threads=%s, logical_cpus=%s, "
        "system_memory_total_mb=%s, system_memory_available_mb=%s, "
        "gpu_total_vram_mb=%s, gpu_free_vram_mb=%s",
        system_resource_plan["cpu_threads"],
        system_resource_plan["logical_cpus"],
        system_resource_plan["system_memory_total_mb"],
        system_resource_plan["system_memory_available_mb"],
        system_resource_plan["gpu_total_vram_mb"],
        system_resource_plan["gpu_free_vram_mb"],
    )
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
    hobby_aliases_for_text = (
        _build_hobby_alias_map(config.paths.hobby_aliases, set(id_to_hobby.values()))
        if config.paths.hobby_aliases.exists()
        else {}
    )
    candidate_text_by_id = _build_candidate_text_by_id(
        id_to_hobby=id_to_hobby,
        hobby_profile=load_json(config.paths.hobby_profile) if config.paths.hobby_profile.exists() else {},
        hobby_taxonomy=load_json(config.paths.hobby_taxonomy) if config.paths.hobby_taxonomy.exists() else {},
        alias_map=hobby_aliases_for_text,
        builder=args.candidate_text_builder,
    )

    train_edges = _read_indexed_edges(config.paths.train_edges)
    target_edges = _read_indexed_edges(
        config.paths.validation_edges if args.split == "validation" else config.paths.test_edges,
    )
    train_known = _known_from_edges(train_edges)
    truth = _known_from_edges(target_edges)
    if args.max_persons > 0 and len(truth) > args.max_persons:
        pilot_rng = random.Random(42)
        selected_persons = set(pilot_rng.sample(sorted(truth), args.max_persons))
        target_edges = [(pid, hid) for pid, hid in target_edges if pid in selected_persons]
        truth = {pid: hobbies for pid, hobbies in truth.items() if pid in selected_persons}

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
    if ranker.model is not None and sys.platform != "win32":
        try:
            ranker.model.reset_parameter({"num_threads": int(system_resource_plan["cpu_threads"])})
        except Exception as exc:
            LOGGER.warning("LightGBM thread reset failed; continuing with model defaults: %s", exc)
    model_feature_columns = ranker.feature_columns()
    model_feature_policy = _feature_policy(model_feature_columns)
    validate_experimental_feature_policy(
        config,
        use_kure_mmr=(args.use_mmr or args.use_dpp) and args.mmr_embedding_method == "kure",
        include_text_embedding_feature=model_feature_policy["include_text_embedding_feature"],
        use_stage1_kure_provider=args.stage1_kure_semantic_provider,
        include_source_features=model_feature_policy["include_source_features"],
    )
    print(f"Loaded ranker model: {model_path} (best_iteration={ranker.best_iteration})")
    if args.feature_cache_dir is None and not args.disable_feature_cache:
        args.feature_cache_dir = model_path.parent / "feature_cache"
        LOGGER.info("Default feature cache enabled: %s", args.feature_cache_dir)

    popularity_counts = build_popularity_counts(train_edges)
    cooccurrence_counts = build_cooccurrence_counts(train_edges)
    max_k = max(config.eval.top_k)
    truth_person_ids = sorted(truth.keys())

    stage1_provider_names: tuple[str, ...] = ("popularity", "cooccurrence")
    stage1_provider_cache_fingerprint = ""
    stage1_kure_semantic_scores: dict[int, dict[int, float]] | None = None
    stage1_kure_metadata: dict[str, object] = {"enabled": False}
    if args.stage1_kure_semantic_provider:
        stage1_provider_names = ("popularity", "cooccurrence", "kure_semantic")
        hobby_aliases = _build_hobby_alias_map(config.paths.hobby_aliases, set(id_to_hobby.values())) if config.paths.hobby_aliases.exists() else {}
        stage1_text_payload = _prepare_text_leakage_context(
            person_ids=truth_person_ids,
            target_edges=target_edges,
            id_to_person=id_to_person,
            contexts=contexts,
            id_to_hobby=id_to_hobby,
            alias_map=hobby_aliases,
        )
        stage1_text_audit = stage1_text_payload["summary"]
        if _text_audit_failure_rate(stage1_text_audit) > 0.05:
            raise ValueError("KURE Stage1 semantic provider blocked: post-mask leakage audit failure rate exceeds 0.05")
        stage1_embedding_plan = _resolve_embedding_resource_plan(args)
        stage1_embedding_batch_size = int(stage1_embedding_plan["effective_batch_size"])
        stage1_text_cache_dir = args.embedding_cache_dir or (config.paths.artifact_dir / "text_embedding_cache")
        stage1_torch = _torch_module()
        stage1_text_device = "cuda" if stage1_torch.cuda.is_available() else "cpu"
        stage1_person_embedding_cache = PersonEmbeddingCache(
            stage1_text_cache_dir,
            model_name=KURE_MODEL_NAME,
            preprocessing_version=TEXT_EMBEDDING_PREPROCESSING_VERSION,
            batch_size=stage1_embedding_batch_size,
            device=stage1_text_device,
        )
        stage1_hobby_embedding_cache = HobbyEmbeddingCache(
            stage1_text_cache_dir,
            model_name=KURE_MODEL_NAME,
            preprocessing_version=TEXT_EMBEDDING_PREPROCESSING_VERSION,
            batch_size=stage1_embedding_batch_size,
            device=stage1_text_device,
        )
        stage1_kure_semantic_scores, stage1_kure_metadata = build_kure_semantic_candidate_scores(
            cast(dict[int, str], stage1_text_payload["person_text_by_id"]),
            stage1_person_embedding_cache,
            stage1_hobby_embedding_cache,
            id_to_hobby,
            train_known,
            candidate_k,
            score_batch_size=args.stage1_kure_score_batch_size,
            show_progress_bar=args.progress_mode != "off",
            progress_desc=f"KURE Stage1 semantic scoring ({args.split})",
        )
        stage1_kure_metadata["text_audit"] = stage1_text_audit
        stage1_kure_metadata["resource_plan"] = stage1_embedding_plan
        stage1_provider_cache_fingerprint = str(stage1_kure_metadata.get("fingerprint", ""))

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
    mmr_embedding_plan: dict[str, object] = {}
    mmr_embedding_batch_size = max(1, int(args.embedding_batch_size))
    if args.use_mmr or args.use_dpp:
        if args.mmr_embedding_method == "kure":
            mmr_embedding_plan = _resolve_embedding_resource_plan(args)
            mmr_embedding_batch_size = int(mmr_embedding_plan["effective_batch_size"])
            hobby_cache = HobbyEmbeddingCache(
                mmr_cache_dir,
                model_name=KURE_MODEL_NAME,
                preprocessing_version=TEXT_EMBEDDING_PREPROCESSING_VERSION,
                batch_size=mmr_embedding_batch_size,
                device="cuda" if _torch_module().cuda.is_available() else "cpu",
            )
            hobby_emb, mmr_embedding_meta = hobby_cache.load_or_build_matrix(all_hobby_names)
            mmr_embedding_meta = {
                "embedding_method": "kure",
                "cache_enabled": bool(mmr_embedding_meta.get("cache_enabled", False)),
                "cache_dir": str(mmr_cache_dir),
                "cache_key": str(mmr_embedding_meta.get("cache_key", "")),
                "model_name": str(mmr_embedding_meta.get("model_name", KURE_MODEL_NAME)),
                "model_revision": str(mmr_embedding_meta.get("model_revision", "")),
                "preprocessing_version": str(mmr_embedding_meta.get("preprocessing_version", TEXT_EMBEDDING_PREPROCESSING_VERSION)),
                "batch_size": mmr_embedding_batch_size,
                "embedding_dim": int(mmr_embedding_meta.get("embedding_dim", 0)),
                "num_hobbies": int(mmr_embedding_meta.get("num_hobbies", len(all_hobby_names))),
                "hobby_names_hash": str(mmr_embedding_meta.get("hobby_names_hash", "")),
                "resource_plan": mmr_embedding_plan,
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
        providers=stage1_provider_names,
        provider_cache_fingerprint=stage1_provider_cache_fingerprint,
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
        stage1_providers=stage1_provider_names,
        semantic_scores_by_person=stage1_kure_semantic_scores,
        provider_cache_fingerprint=stage1_provider_cache_fingerprint,
    )

    candidate_pool_policy = _candidate_pool_policy(
        pools_by_person,
        candidate_k=candidate_k,
        normalization_method=normalization_method,
        cache_key=candidate_pool_cache_key,
        cache_path=candidate_pool_cache_path,
    )
    candidate_pool_person_count = len(pools_by_person)
    candidate_pool_row_count = sum(len(candidates) for candidates in pools_by_person.values())
    LOGGER.info(
        "Candidate pool ready: persons=%s candidate_rows=%s candidate_k=%s cache_path=%s",
        candidate_pool_person_count,
        candidate_pool_row_count,
        candidate_k,
        candidate_pool_cache_path,
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
            config.paths.hobby_aliases,
        )
        feature_cache_npz_path, feature_cache_meta_path = _feature_cache_paths(
            args,
            truth_person_ids,
            pools_by_person,
            model_feature_columns,
            config.paths.person_context_csv,
            config.paths.hobby_profile,
            config.paths.hobby_taxonomy,
            config.paths.hobby_aliases,
        )
        LOGGER.info(
            "Feature cache lookup prepared: key=%s npz=%s metadata=%s",
            feature_cache_key,
            feature_cache_npz_path,
            feature_cache_meta_path,
        )
    else:
        LOGGER.info("Feature cache disabled; feature rows will be rebuilt.")

    include_text_embedding_feature = model_feature_policy["include_text_embedding_feature"]
    include_domain_text_embedding_features = model_feature_policy["include_domain_text_embedding_features"]
    include_text_rank_margin_features = model_feature_policy["include_text_rank_margin_features"]
    embedding_resource_plan = _resolve_embedding_resource_plan(args)
    effective_embedding_batch_size = int(embedding_resource_plan["effective_batch_size"])
    text_cache_dir = args.embedding_cache_dir or (config.paths.artifact_dir / "text_embedding_cache")
    text_device = str(embedding_resource_plan.get("device", ""))
    text_similarity_fn: Any = None
    text_similarity_lookup: dict[int, dict[int, float]] = {}
    domain_similarity_lookup: dict[int, dict[int, dict[str, float]]] = {}
    text_rank_margin_lookup: dict[int, dict[int, dict[str, float]]] = {}
    LOGGER.info(
        "Checking feature cache before KURE preparation: split=%s include_text_embedding_feature=%s include_domain_text_embedding_features=%s",
        args.split,
        include_text_embedding_feature,
        include_domain_text_embedding_features,
    )
    cached_features = _load_feature_cache(
        args,
        truth_person_ids,
        pools_by_person,
        model_feature_columns,
        config.paths.person_context_csv,
        config.paths.hobby_profile,
        config.paths.hobby_taxonomy,
        config.paths.hobby_aliases,
    )
    feature_cache_hit = cached_features is not None
    if feature_cache_hit:
        LOGGER.info("Feature cache hit; skipping KURE text embedding prewarm and feature rebuild.")
    text_embedding_audit: dict[str, object] = {
        "enabled": include_text_embedding_feature or include_domain_text_embedding_features or include_text_rank_margin_features,
        "include_domain_text_embedding_features": include_domain_text_embedding_features,
        "include_text_rank_margin_features": include_text_rank_margin_features,
        "cache_dir": "",
        "resource_plan": embedding_resource_plan,
        "known_hobbies_masked": False,
        "audit_pass": True,
        "skipped_due_to_feature_cache_hit": feature_cache_hit,
        "passed_person_count": 0,
        "failed_person_count": 0,
    }

    if (include_text_embedding_feature or include_domain_text_embedding_features or include_text_rank_margin_features) and not feature_cache_hit:
        text_prepare_start = time.perf_counter()
        LOGGER.info(
            "KURE embedding resource plan: device=%s, requested_batch_size=%s, "
            "effective_batch_size=%s, gpu_total_vram_mb=%s, gpu_free_vram_mb=%s, "
            "target_vram_mb=%s, estimated_vram_mb=%s",
            embedding_resource_plan["device"],
            embedding_resource_plan["requested_batch_size"],
            embedding_resource_plan["effective_batch_size"],
            embedding_resource_plan["gpu_total_vram_mb"],
            embedding_resource_plan["gpu_free_vram_mb"],
            embedding_resource_plan["target_vram_mb"],
            embedding_resource_plan["estimated_vram_mb"],
        )
        LOGGER.info("Preparing KURE leakage-safe text context: persons=%s", len(truth_person_ids))
        hobby_aliases = _build_hobby_alias_map(config.paths.hobby_aliases, set(id_to_hobby.values())) if config.paths.hobby_aliases.exists() else {}
        text_torch = _torch_module()
        text_device = "cuda" if text_torch.cuda.is_available() else "cpu"
        LOGGER.info(
            "KURE model source prepared: model=%s device=%s embedding_cache_dir=%s huggingface_cache=%s",
            text_embedding_model_name,
            text_device,
            text_cache_dir / _safe_model_cache_name(text_embedding_model_name),
            _huggingface_model_cache_status(text_embedding_model_name),
        )
        person_embedding_cache = PersonEmbeddingCache(
            text_cache_dir,
            model_name=text_embedding_model_name,
            model_revision=text_embedding_model_revision,
            preprocessing_version=TEXT_EMBEDDING_PREPROCESSING_VERSION,
            batch_size=effective_embedding_batch_size,
            device=text_device,
        )
        hobby_embedding_cache = HobbyEmbeddingCache(
            text_cache_dir,
            model_name=text_embedding_model_name,
            model_revision=text_embedding_model_revision,
            preprocessing_version=TEXT_EMBEDDING_PREPROCESSING_VERSION,
            batch_size=effective_embedding_batch_size,
            device=text_device,
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
        person_domain_texts_by_id = text_prepare_payload["person_domain_texts_by_id"]
        person_audit_pass = text_prepare_payload["person_audit_pass"]
        text_embedding_audit.update(text_prepare_payload["summary"])
        text_embedding_audit["cache_dir"] = str(text_cache_dir)
        text_embedding_audit["known_hobbies_masked"] = bool(len(text_prepare_payload["person_text_by_id"]) > 0)
        LOGGER.info(
            "KURE leakage-safe text context ready: eligible=%s passed=%s failed=%s missing_context=%s seconds=%.3f cache_dir=%s",
            text_embedding_audit.get("audit_eligible_person_count", 0),
            text_embedding_audit.get("passed_person_count", 0),
            text_embedding_audit.get("failed_person_count", 0),
            text_embedding_audit.get("missing_context_person_count", 0),
            time.perf_counter() - text_prepare_start,
            text_cache_dir,
        )

        if person_text_by_id:
            LOGGER.info(
                "Starting KURE embedding prewarm and similarity lookup: persons_with_text=%s candidate_rows=%s",
                len(person_text_by_id),
                candidate_pool_row_count,
            )
            _prewarm_text_embedding_caches(
                person_text_by_id=person_text_by_id,
                person_domain_texts_by_id=person_domain_texts_by_id if include_domain_text_embedding_features else None,
                person_embedding_cache=person_embedding_cache,
                hobby_embedding_cache=hobby_embedding_cache,
                candidate_pools=pools_by_person,
                candidate_text_by_id=candidate_text_by_id,
                show_progress_bar=args.progress_mode != "off",
                split=args.split,
            )
            text_similarity_lookup = _build_text_similarity_lookup(
                person_text_by_id=person_text_by_id,
                person_audit_pass=person_audit_pass,
                person_embedding_cache=person_embedding_cache,
                hobby_embedding_cache=hobby_embedding_cache,
                candidate_pools=pools_by_person,
                candidate_text_by_id=candidate_text_by_id,
            )
            if include_domain_text_embedding_features:
                domain_similarity_lookup = _build_domain_similarity_lookup(
                    person_domain_texts_by_id=person_domain_texts_by_id,
                    person_audit_pass=person_audit_pass,
                    person_embedding_cache=person_embedding_cache,
                    hobby_embedding_cache=hobby_embedding_cache,
                    candidate_pools=pools_by_person,
                    candidate_text_by_id=candidate_text_by_id,
                )
            if include_text_rank_margin_features:
                text_rank_margin_lookup = build_text_rank_margin_lookup(pools_by_person, text_similarity_lookup)
            text_embedding_audit["similarity_lookup_person_count"] = len(text_similarity_lookup)
            text_embedding_audit["similarity_lookup_pair_count"] = sum(len(values) for values in text_similarity_lookup.values())
            text_embedding_audit["domain_similarity_lookup_person_count"] = len(domain_similarity_lookup)
            text_embedding_audit["domain_similarity_lookup_pair_count"] = sum(len(values) for values in domain_similarity_lookup.values())
            text_embedding_audit["text_rank_margin_lookup_person_count"] = len(text_rank_margin_lookup)
            text_embedding_audit["text_rank_margin_lookup_pair_count"] = sum(len(values) for values in text_rank_margin_lookup.values())
            text_similarity_fn = None
            LOGGER.info(
                "KURE text similarity lookup ready: persons=%s pairs=%s dynamic_fallback=%s",
                text_embedding_audit["similarity_lookup_person_count"],
                text_embedding_audit["similarity_lookup_pair_count"],
                False,
            )
        else:
            text_similarity_fn = None
            text_embedding_audit["audit_pass"] = False
            text_embedding_audit["disable_reason"] = "no eligible non-empty masked person text"
            LOGGER.warning("KURE text feature has no eligible non-empty masked person text; evaluation will be disabled.")

    text_audit_failure_rate = _text_audit_failure_rate(text_embedding_audit)
    text_has_eligible_persons = int(text_embedding_audit.get("audit_eligible_person_count", 0)) > 0
    text_has_passed_persons = int(text_embedding_audit.get("passed_person_count", 0)) > 0
    should_disable_text_eval = (
        (include_text_embedding_feature or include_domain_text_embedding_features or include_text_rank_margin_features)
        and not feature_cache_hit
        and (
            text_audit_failure_rate > 0.05
            or not text_has_eligible_persons
            or not text_has_passed_persons
        )
    )
    if should_disable_text_eval:
        runtime_seconds = time.perf_counter() - start_time
        disable_reason = str(text_embedding_audit.get("disable_reason", ""))
        if not disable_reason:
            if not text_has_eligible_persons:
                disable_reason = "no audit-eligible person text after masking"
            elif not text_has_passed_persons:
                disable_reason = "no person text passed leakage audit after masking"
            else:
                disable_reason = "post-mask leakage audit failed above threshold"
        disabled_summary = {
            "reason": disable_reason,
            "threshold": 0.05,
            "failure_rate": text_audit_failure_rate,
            "passed_person_count": int(text_embedding_audit.get("passed_person_count", 0)),
            "failed_person_count": int(text_embedding_audit.get("failed_person_count", 0)),
            "audit_eligible_person_count": int(text_embedding_audit.get("audit_eligible_person_count", 0)),
            "cache_dir": str(text_embedding_audit.get("cache_dir", "")),
        }
        result = {
            "split": args.split,
            "experiment_id": args.experiment_id,
            "status": "disabled",
            "runtime_seconds": runtime_seconds,
            "model_path": str(model_path),
            "feature_policy": {
                "feature_columns": model_feature_columns,
                "include_source_features": model_feature_policy["include_source_features"],
                "include_text_embedding_feature": include_text_embedding_feature,
                "include_domain_text_embedding_features": include_domain_text_embedding_features,
            },
            "text_embedding_audit": text_embedding_audit,
            "disabled_summary": disabled_summary,
        }
        embedding_model_metadata = _embedding_model_metadata(
            enabled=True,
            model_name=text_embedding_model_name,
            model_revision=text_embedding_model_revision,
            cache_dir=text_cache_dir,
            batch_size=effective_embedding_batch_size,
            device=text_device,
            resource_plan=embedding_resource_plan,
        )
        result["embedding_model_metadata"] = embedding_model_metadata
        if args.output is not None:
            save_json(args.output.with_name("embedding_model_metadata.json"), embedding_model_metadata)
            result["embedding_model_metadata_path"] = str(args.output.with_name("embedding_model_metadata.json"))
            save_json(args.output, result)
        _write_status(
            args,
            "disabled",
            runtime_seconds=runtime_seconds,
            summary=disabled_summary,
        )
        print(
            "Text embedding evaluation disabled: "
            f"{disable_reason} (failure_rate={disabled_summary['failure_rate']:.4f})"
        )
        return

    ranking_parallelism = _resolve_parallel_backend(args.ranking_build_parallelism)
    LOGGER.info(
        "Starting Stage1/V1 ranking build: persons=%s skip_v1=%s workers=%s backend=%s",
        len(truth_person_ids),
        args.skip_v1,
        int(system_resource_plan["cpu_threads"]) if not args.skip_v1 else 1,
        ranking_parallelism,
    )
    candidate_rank_start = time.perf_counter()
    if args.skip_v1:
        for person_id in _iter_with_progress(args, truth_person_ids, desc=f"rank candidates ({args.split})"):
            candidate_rankings[person_id] = [c.hobby_id for c in pools_by_person.get(person_id, [])]
            stage1_rankings[person_id] = candidate_rankings[person_id][:max_k]
            v1_rankings[person_id] = []
    else:
        ranking_worker_count = max(1, int(system_resource_plan["cpu_threads"]))
        if ranking_parallelism == "serial":
            ranking_worker_count = 1
        ranking_payloads = [(person_id, pools_by_person.get(person_id, [])) for person_id in truth_person_ids]
        ranking_chunksize = max(1, min(64, len(ranking_payloads) // (ranking_worker_count * 4) if ranking_worker_count else 1))
        if ranking_worker_count > 1:
            with ThreadPoolExecutor(
                max_workers=ranking_worker_count,
                initializer=_init_ranking_worker,
                initargs=(
                    train_known,
                    id_to_hobby,
                    id_to_person,
                    contexts,
                    hobby_profile,
                    reranker_config,
                    hobby_taxonomy,
                    max_k,
                ),
            ) as executor:
                results = executor.map(_build_stage1_v1_rankings_for_person, ranking_payloads, chunksize=ranking_chunksize)
                for person_id, candidate_ranking, stage1_ranking, v1_ranking in _iter_with_progress(
                    args,
                    results,
                    desc=f"rank candidates ({args.split})",
                    total_count=len(ranking_payloads),
                ):
                    candidate_rankings[person_id] = candidate_ranking
                    stage1_rankings[person_id] = stage1_ranking
                    v1_rankings[person_id] = v1_ranking
        else:
            _init_ranking_worker(
                train_known,
                id_to_hobby,
                id_to_person,
                contexts,
                hobby_profile,
                reranker_config,
                hobby_taxonomy,
                max_k,
            )
            for person_id, candidate_ranking, stage1_ranking, v1_ranking in _iter_with_progress(
                args,
                (_build_stage1_v1_rankings_for_person(payload) for payload in ranking_payloads),
                desc=f"rank candidates ({args.split})",
                total_count=len(ranking_payloads),
            ):
                candidate_rankings[person_id] = candidate_ranking
                stage1_rankings[person_id] = stage1_ranking
                v1_rankings[person_id] = v1_ranking
    LOGGER.info(
        "Stage1/V1 ranking build done: persons=%s seconds=%.3f",
        len(candidate_rankings),
        time.perf_counter() - candidate_rank_start,
    )

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

    if cached_features is not None:
        feature_matrix, person_to_feature_slice, hobby_ids_by_person, fallback_person_ids = cached_features
        v2_fallback_count = len(fallback_person_ids)
        LOGGER.info(
            "Using cached feature matrix: rows=%s columns=%s fallback_persons=%s",
            int(feature_matrix.shape[0]) if hasattr(feature_matrix, "shape") else 0,
            int(feature_matrix.shape[1]) if hasattr(feature_matrix, "shape") and len(feature_matrix.shape) > 1 else len(model_feature_columns),
            v2_fallback_count,
        )
    else:
        feature_matrix = np.empty((candidate_pool_row_count, len(model_feature_columns)), dtype=np.float32)
        feature_row_offset = 0
        person_to_feature_slice = {}
        hobby_ids_by_person = {}
        fallback_person_ids = []

        feature_worker_count = max(1, int(system_resource_plan["cpu_threads"]))
        feature_parallelism = _resolve_parallel_backend(args.feature_build_parallelism)
        if feature_parallelism == "serial":
            feature_worker_count = 1
        use_parallel_feature_build = feature_worker_count > 1 and feature_parallelism == "thread"
        LOGGER.info(
            "Starting feature row build: persons=%s candidate_rows=%s feature_columns=%s text_lookup_pairs=%s parallel=%s workers=%s backend=%s",
            len(truth_person_ids),
            candidate_pool_row_count,
            len(model_feature_columns),
            sum(len(values) for values in text_similarity_lookup.values()),
            use_parallel_feature_build,
            feature_worker_count if use_parallel_feature_build else 1,
            feature_parallelism,
        )
        feature_build_start = time.perf_counter()
        if use_parallel_feature_build:
            feature_payloads = [
                (
                    person_id,
                    contexts.get(id_to_person.get(person_id, "")) or empty_person_context(id_to_person.get(person_id, "")),
                    pools_by_person.get(person_id, []),
                    {id_to_hobby[hid] for hid in train_known.get(person_id, set()) if hid in id_to_hobby},
                    text_similarity_lookup.get(person_id, {}),
                    domain_similarity_lookup.get(person_id, {}),
                    text_rank_margin_lookup.get(person_id, {}),
                )
                for person_id in truth_person_ids
            ]
            chunksize = max(1, min(64, len(feature_payloads) // (feature_worker_count * 4) if feature_worker_count else 1))
            _init_feature_worker(hobby_profile, reranker_config, model_feature_columns)
            with ThreadPoolExecutor(max_workers=feature_worker_count) as executor:
                results = executor.map(_build_feature_rows_for_person, feature_payloads, chunksize=chunksize)
                for person_id, rows, hobby_ids_list, is_fallback in _iter_with_progress(
                    args,
                    results,
                    desc=f"features ({args.split})",
                    total_count=len(feature_payloads),
                ):
                    if is_fallback:
                        fallback_person_ids.append(person_id)
                        v2_fallback_count += 1
                        continue
                    start = feature_row_offset
                    row_count = len(rows)
                    if row_count:
                        feature_matrix[start : start + row_count] = np.asarray(rows, dtype=np.float32)
                        feature_row_offset += row_count
                    person_to_feature_slice[person_id] = (start, feature_row_offset)
                    hobby_ids_by_person[person_id] = hobby_ids_list
        else:
            _init_feature_worker(hobby_profile, reranker_config, model_feature_columns)
            for person_id in _iter_with_progress(args, truth_person_ids, desc=f"features ({args.split})"):
                person_uuid = id_to_person.get(person_id, "")
                person_context = contexts.get(person_uuid) or empty_person_context(person_uuid)
                hobby_candidates = pools_by_person.get(person_id, [])
                known_names = {id_to_hobby[hid] for hid in train_known.get(person_id, set()) if hid in id_to_hobby}

                if hobby_candidates:
                    start = feature_row_offset
                    hobby_ids_list: list[int] = []
                    for candidate in hobby_candidates:
                        text_embedding_similarity = 0.0
                        if text_similarity_lookup:
                            text_embedding_similarity = text_similarity_lookup.get(person_id, {}).get(candidate.hobby_id, 0.0)
                        elif text_similarity_fn is not None:
                            try:
                                text_embedding_similarity = float(text_similarity_fn(person_id, candidate))
                            except Exception:
                                text_embedding_similarity = 0.0
                        features = _build_fast_rerank_features(
                            person_context,
                            candidate,
                            known_names,
                            reranker_config,
                            text_embedding_similarity=text_embedding_similarity,
                            domain_text_embedding_similarities=domain_similarity_lookup.get(person_id, {}).get(candidate.hobby_id, {}),
                            text_rank_margin_features=text_rank_margin_lookup.get(person_id, {}).get(candidate.hobby_id, {}),
                        )
                        feature_matrix[feature_row_offset] = np.asarray(
                            [features.get(col, 0.0) for col in model_feature_columns],
                            dtype=np.float32,
                        )
                        feature_row_offset += 1
                        hobby_ids_list.append(candidate.hobby_id)
                    person_to_feature_slice[person_id] = (start, feature_row_offset)
                    hobby_ids_by_person[person_id] = hobby_ids_list
                else:
                    fallback_person_ids.append(person_id)
                    v2_fallback_count += 1
        feature_matrix = feature_matrix[:feature_row_offset]
        LOGGER.info(
            "Feature row build done: rows=%s persons_with_features=%s fallback_persons=%s seconds=%.3f",
            int(feature_matrix.shape[0]),
            len(person_to_feature_slice),
            len(fallback_person_ids),
            time.perf_counter() - feature_build_start,
        )
        LOGGER.info("Saving feature cache if enabled.")
        _save_feature_cache(
            args,
            truth_person_ids,
            pools_by_person,
            model_feature_columns,
            config.paths.person_context_csv,
            config.paths.hobby_profile,
            config.paths.hobby_taxonomy,
            config.paths.hobby_aliases,
            feature_matrix,
            person_to_feature_slice,
            hobby_ids_by_person,
            fallback_person_ids,
        )

    feature_rows = int(feature_matrix.shape[0]) if hasattr(feature_matrix, "shape") else 0
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

    if len(feature_matrix) > 0:
        LOGGER.info("Starting LightGBM prediction: rows=%s columns=%s", int(feature_matrix.shape[0]), int(feature_matrix.shape[1]))
        predict_start = time.perf_counter()
        all_scores = ranker.predict(feature_matrix)
        LOGGER.info("LightGBM prediction done: scores=%s seconds=%.3f", len(all_scores), time.perf_counter() - predict_start)

        LOGGER.info("Starting v2 ranking assembly: persons=%s", len(truth))
        ranking_start = time.perf_counter()
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
        LOGGER.info("V2 ranking assembly done: persons=%s seconds=%.3f", len(v2_rankings), time.perf_counter() - ranking_start)
    else:
        LOGGER.info("No feature rows available; v2 rankings will fall back to Stage1 rankings.")

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
            "include_source_features": model_feature_policy["include_source_features"],
            "include_text_embedding_feature": model_feature_policy["include_text_embedding_feature"],
        },
        "input_config_summary": input_config_summary,
        "max_persons": args.max_persons,
        "candidate_pool_policy": candidate_pool_policy,
        "feature_cache_policy": {
            "cache_key": feature_cache_key,
            "cache_path": str(feature_cache_npz_path) if feature_cache_npz_path is not None else "",
            "metadata_path": str(feature_cache_meta_path) if feature_cache_meta_path is not None else "",
            "cache_hit": feature_cache_hit,
        },
        "resource_policy": system_resource_plan,
        "v2_fallback_count": v2_fallback_count,
        "stage1_baseline": {
            "providers": list(stage1_provider_names),
            "metrics": stage1_metrics,
            "kure_semantic_provider": stage1_kure_metadata,
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
                "enabled": include_text_embedding_feature or include_domain_text_embedding_features,
                "include_domain_text_embedding_features": include_domain_text_embedding_features,
                "effective_embedding_batch_size": effective_embedding_batch_size,
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
    text_embedding_enabled = include_text_embedding_feature or include_domain_text_embedding_features
    embedding_model_metadata = _embedding_model_metadata(
        enabled=text_embedding_enabled or args.stage1_kure_semantic_provider or (
            (args.use_mmr or args.use_dpp) and args.mmr_embedding_method == "kure"
        ),
        model_name=text_embedding_model_name,
        model_revision=text_embedding_model_revision,
        cache_dir=text_cache_dir if (text_embedding_enabled or args.stage1_kure_semantic_provider) else mmr_cache_dir,
        batch_size=effective_embedding_batch_size if (text_embedding_enabled or args.stage1_kure_semantic_provider) else mmr_embedding_batch_size,
        device=text_device if (text_embedding_enabled or args.stage1_kure_semantic_provider) else str(mmr_embedding_plan.get("device", "")),
        resource_plan=embedding_resource_plan if (text_embedding_enabled or args.stage1_kure_semantic_provider) else mmr_embedding_plan,
    )
    payload["embedding_model_metadata"] = embedding_model_metadata
    if not args.skip_v1:
        payload["v1_deterministic_reranker"] = {
            "metrics": v1_metrics,
        }

    payload["runtime_seconds"] = time.perf_counter() - start_time

    output_path = args.output or Path("GNN_Neural_Network/artifacts/ranker_eval_metrics.json")
    embedding_metadata_path = output_path.with_name("embedding_model_metadata.json")
    save_json(embedding_metadata_path, embedding_model_metadata)
    payload["embedding_model_metadata_path"] = str(embedding_metadata_path)
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
    _print_section(f"Stage1 ({'+'.join(stage1_provider_names)})", stage1_metrics)
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
    hobby_aliases_path: Path,
) -> tuple[np.ndarray, dict[int, tuple[int, int]], dict[int, list[int]], list[int]] | None:
    if args.feature_cache_dir is None:
        LOGGER.info("Feature cache lookup skipped: cache_dir is not set.")
        return None
    npz_path, meta_path = _feature_cache_paths(
        args,
        person_ids,
        pools_by_person,
        feature_columns,
        person_context_path,
        hobby_profile_path,
        hobby_taxonomy_path,
        hobby_aliases_path,
    )
    if not npz_path.exists() or not meta_path.exists():
        LOGGER.info(
            "Feature cache miss: cache files not found npz_exists=%s metadata_exists=%s npz=%s metadata=%s",
            npz_path.exists(),
            meta_path.exists(),
            npz_path,
            meta_path,
        )
        return None

    metadata = _read_feature_cache_metadata(meta_path)
    if metadata is None or not _feature_cache_metadata_matches(metadata, args.split, feature_columns):
        LOGGER.info("Feature cache miss: metadata missing or incompatible metadata=%s", meta_path)
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

    LOGGER.info(
        "Feature cache hit: rows=%s persons=%s fallback_persons=%s npz=%s",
        int(data["X"].shape[0]),
        len(persons),
        len(fallback_person_ids),
        npz_path,
    )
    return data["X"].astype(np.float32), person_to_feature_slice, hobby_ids_by_person, fallback_person_ids


def _save_feature_cache(
    args: argparse.Namespace,
    person_ids: list[int],
    pools_by_person: dict[int, list[Any]],
    feature_columns: list[str],
    person_context_path: Path,
    hobby_profile_path: Path,
    hobby_taxonomy_path: Path,
    hobby_aliases_path: Path,
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
        hobby_aliases_path,
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
                "cache_version": FEATURE_CACHE_VERSION,
                "split": args.split,
                "experiment_id": args.experiment_id,
                "feature_columns": feature_columns,
                "feature_policy": _feature_policy(feature_columns),
                "text_embedding": {
                    "model_name": str(getattr(args, "text_embedding_model_name", KURE_MODEL_NAME) or KURE_MODEL_NAME)
                    if _feature_policy(feature_columns)["include_text_embedding_feature"]
                    else "",
                    "model_revision": "",
                    "preprocessing_version": TEXT_EMBEDDING_PREPROCESSING_VERSION
                    if _feature_policy(feature_columns)["include_text_embedding_feature"]
                    else "",
                    "text_builder": (
                        "build_domain_tagged_persona_text+build_domain_persona_texts"
                        if _feature_policy(feature_columns)["include_domain_text_embedding_features"]
                        else "build_domain_tagged_persona_text"
                    )
                    if _feature_policy(feature_columns)["include_text_embedding_feature"]
                    else "",
                    "candidate_text_builder": str(getattr(args, "candidate_text_builder", "name_only") or "name_only")
                    if _feature_policy(feature_columns)["include_text_embedding_feature"]
                    else "",
                    "masking": "mask_holdout_hobbies",
                    "similarity": "precomputed_lookup",
                },
                "num_rows": int(X.shape[0]),
                "num_persons": len(person_ids),
                "fallback_count": len(fallback_person_ids),
                "files": {
                    "person_context": _file_fingerprint(person_context_path),
                    "hobby_profile": _file_fingerprint(hobby_profile_path),
                    "hobby_taxonomy": _file_fingerprint(hobby_taxonomy_path),
                    "hobby_aliases": _file_fingerprint(hobby_aliases_path),
                },
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
    hobby_aliases_path: Path,
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
        hobby_aliases_path,
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
    hobby_aliases_path: Path,
) -> str:
    payload = {
        "cache_version": FEATURE_CACHE_VERSION,
        "split": args.split,
        "person_ids": sorted(person_ids),
        "feature_columns": feature_columns,
        "feature_policy": _feature_policy(feature_columns),
        "candidate_pool": _candidate_pool_fingerprint(person_ids, pools_by_person),
        "files": {
            "person_context": _file_fingerprint(person_context_path),
            "hobby_profile": _file_fingerprint(hobby_profile_path),
            "hobby_taxonomy": _file_fingerprint(hobby_taxonomy_path),
            "hobby_aliases": _file_fingerprint(hobby_aliases_path),
        },
        "text_embedding": {
            "model_name": str(getattr(args, "text_embedding_model_name", KURE_MODEL_NAME) or KURE_MODEL_NAME)
            if _feature_policy(feature_columns)["include_text_embedding_feature"]
            else "",
            "model_revision": str(getattr(args, "text_embedding_model_revision", "") or "")
            if _feature_policy(feature_columns)["include_text_embedding_feature"]
            else "",
            "preprocessing_version": TEXT_EMBEDDING_PREPROCESSING_VERSION
            if _feature_policy(feature_columns)["include_text_embedding_feature"]
            else "",
            "text_builder": (
                "build_domain_tagged_persona_text+build_domain_persona_texts"
                if _feature_policy(feature_columns)["include_domain_text_embedding_features"]
                else "build_domain_tagged_persona_text"
            )
            if _feature_policy(feature_columns)["include_text_embedding_feature"]
            else "",
            "candidate_text_builder": str(getattr(args, "candidate_text_builder", "name_only") or "name_only")
            if _feature_policy(feature_columns)["include_text_embedding_feature"]
            else "",
            "masking": "mask_holdout_hobbies",
            "similarity": "precomputed_lookup",
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
        "include_domain_text_embedding_features": any(col in feature_columns for col in RANKER_DOMAIN_TEXT_FEATURE_COLUMNS),
        "include_text_rank_margin_features": any(col in feature_columns for col in RANKER_TEXT_RANK_MARGIN_FEATURE_COLUMNS),
    }


def _resolve_embedding_resource_plan(args: argparse.Namespace) -> dict[str, object]:
    requested = int(args.embedding_batch_size)
    torch_module = _torch_module()
    device = "cuda" if torch_module.cuda.is_available() else "cpu"
    total_mb, used_mb, free_mb = _query_gpu_memory_mb()
    if requested > 0:
        estimated_mb = _estimate_embedding_vram_mb(requested)
        return {
            "device": device,
            "requested_batch_size": requested,
            "effective_batch_size": requested,
            "gpu_total_vram_mb": total_mb,
            "gpu_used_vram_mb": used_mb,
            "gpu_free_vram_mb": free_mb,
            "vram_utilization_target": float(args.embedding_vram_utilization),
            "target_vram_mb": 0,
            "estimated_mb_per_text": _estimated_mb_per_text(),
            "estimated_vram_mb": estimated_mb,
            "mode": "manual",
        }
    if device != "cuda":
        batch_size = 32
        return {
            "device": device,
            "requested_batch_size": requested,
            "effective_batch_size": batch_size,
            "gpu_total_vram_mb": total_mb,
            "gpu_used_vram_mb": used_mb,
            "gpu_free_vram_mb": free_mb,
            "vram_utilization_target": float(args.embedding_vram_utilization),
            "target_vram_mb": 0,
            "estimated_mb_per_text": _estimated_mb_per_text(),
            "estimated_vram_mb": _estimate_embedding_vram_mb(batch_size),
            "mode": "cpu_default",
        }

    requested_target_mb = max(0, int(args.embedding_target_vram_mb))
    if requested_target_mb > 0:
        usable_mb = min(requested_target_mb, max(512, int(total_mb * 0.95)))
        mode = "auto_target_vram"
    else:
        usable_mb = max(512, int(free_mb * max(0.1, min(float(args.embedding_vram_utilization), 0.95))))
        mode = "auto_vram"
    batch_size = max(64, min(1024, usable_mb // _estimated_mb_per_text()))
    estimated_mb = _estimate_embedding_vram_mb(batch_size)
    return {
        "device": device,
        "requested_batch_size": requested,
        "effective_batch_size": int(batch_size),
        "gpu_total_vram_mb": total_mb,
        "gpu_used_vram_mb": used_mb,
        "gpu_free_vram_mb": free_mb,
        "vram_utilization_target": float(args.embedding_vram_utilization),
        "requested_target_vram_mb": requested_target_mb,
        "target_vram_mb": usable_mb,
        "estimated_mb_per_text": _estimated_mb_per_text(),
        "estimated_vram_mb": estimated_mb,
        "mode": mode,
    }


def _estimated_mb_per_text() -> int:
    return 18


def _estimate_embedding_vram_mb(batch_size: int) -> int:
    return int(max(1, batch_size) * _estimated_mb_per_text())


def _query_gpu_memory_mb() -> tuple[int, int, int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return 0, 0, 0
    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 3:
        return 0, 0, 0
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return 0, 0, 0


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
        metadata.get("cache_version") == FEATURE_CACHE_VERSION
        and metadata.get("split") == split
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


def _embedding_model_metadata(
    *,
    enabled: bool,
    model_name: str,
    model_revision: str,
    cache_dir: Path,
    batch_size: int,
    device: str,
    resource_plan: dict[str, object],
) -> dict[str, object]:
    return {
        "enabled": enabled,
        "model_name": model_name if enabled else "",
        "model_revision": model_revision if enabled else "",
        "preprocessing_version": TEXT_EMBEDDING_PREPROCESSING_VERSION if enabled else "",
        "text_builder": "build_domain_tagged_persona_text" if enabled else "",
        "cache_dir": str(cache_dir) if enabled else "",
        "batch_size": batch_size,
        "device": device,
        "resource_plan": resource_plan,
        "cache_key_policy": "model_name|model_revision|preprocessing_version|text",
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


def _text_audit_failure_rate(text_embedding_audit: dict[str, object]) -> float:
    failed = int(text_embedding_audit.get("failed_person_count", 0) or 0)
    passed = int(text_embedding_audit.get("passed_person_count", 0) or 0)
    total = failed + passed
    if total <= 0:
        return 0.0
    return failed / total


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


def _build_candidate_text_by_id(
    *,
    id_to_hobby: dict[int, str],
    hobby_profile: dict[str, object],
    hobby_taxonomy: dict[str, object],
    alias_map: dict[str, list[str]],
    builder: str,
) -> dict[int, str]:
    categories = _build_hobby_categories(id_to_hobby, hobby_taxonomy)
    hobbies_profile = hobby_profile.get("hobbies", {}) if isinstance(hobby_profile, dict) else {}
    hobbies_profile = hobbies_profile if isinstance(hobbies_profile, dict) else {}
    output: dict[int, str] = {}
    for hobby_id, hobby_name in id_to_hobby.items():
        parts = [str(hobby_name)]
        normalized_name = normalize_hobby_name(hobby_name)
        if builder in {"name_plus_aliases", "name_plus_short_description"}:
            aliases = [alias for alias in alias_map.get(normalized_name, []) if alias and alias != normalized_name]
            if aliases:
                parts.append("aliases: " + ", ".join(aliases[:8]))
        if builder in {"name_plus_category", "name_plus_short_description"}:
            category = categories.get(hobby_id, "")
            if category:
                parts.append("category: " + category)
        if builder == "name_plus_short_description":
            profile_entry = hobbies_profile.get(hobby_name, {})
            if isinstance(profile_entry, dict):
                description = (
                    profile_entry.get("short_description")
                    or profile_entry.get("description")
                    or profile_entry.get("summary")
                    or ""
                )
                if description:
                    parts.append("description: " + str(description))
        output[hobby_id] = " | ".join(part for part in parts if part)
    return output


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
    person_domain_texts_by_id: dict[int, dict[str, str]] = {}
    person_audit_pass: dict[int, bool] = {}
    passed_person_ids: list[int] = []
    failed_person_ids: list[int] = []
    missing_context_person_ids: list[int] = []

    for person_id in person_ids:
        person_uuid = id_to_person.get(person_id, "")
        context = contexts.get(person_uuid)
        if not context:
            person_audit_pass[person_id] = True
            missing_context_person_ids.append(person_id)
            continue

        holdout_hobby_names = {
            normalize_hobby_name(id_to_hobby[hobby_id])
            for hobby_id in known_by_person.get(person_id, set())
            if hobby_id in id_to_hobby
        }

        masked_field_values: dict[str, str] = {}
        for field in LEAKAGE_TEXT_FIELDS:
            try:
                value = str(getattr(context, field, "") or "").strip()
            except Exception:
                value = ""
            if value:
                masked_field_values[field] = (
                    mask_holdout_hobbies(value, holdout_hobby_names, alias_map=alias_map)
                    if holdout_hobby_names else value
                )

        masked = build_domain_tagged_persona_text(context, masked_field_values)
        if not masked:
            person_audit_pass[person_id] = True
            missing_context_person_ids.append(person_id)
            continue

        audit_ok = post_mask_leakage_audit(masked, holdout_hobby_names, alias_map=alias_map)
        person_audit_pass[person_id] = bool(audit_ok)
        if audit_ok and masked:
            person_text_by_id[person_id] = masked
            person_domain_texts_by_id[person_id] = build_domain_persona_texts(context, masked_field_values)
            passed_person_ids.append(person_id)
        else:
            failed_person_ids.append(person_id)

    return {
        "person_text_by_id": person_text_by_id,
        "person_domain_texts_by_id": person_domain_texts_by_id,
        "person_audit_pass": person_audit_pass,
        "summary": {
            "audit_pass": not failed_person_ids,
            "text_builder": "build_domain_tagged_persona_text",
            "preprocessing_version": TEXT_EMBEDDING_PREPROCESSING_VERSION,
            "passed_person_count": len(passed_person_ids),
            "failed_person_count": len(failed_person_ids),
            "missing_context_person_count": len(missing_context_person_ids),
            "audit_eligible_person_count": len(passed_person_ids) + len(failed_person_ids),
            "passed_person_id_sample": passed_person_ids[:100],
            "failed_person_id_sample": failed_person_ids[:100],
            "missing_context_person_id_sample": missing_context_person_ids[:100],
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


def _prewarm_text_embedding_caches(
    *,
    person_text_by_id: dict[int, str],
    person_domain_texts_by_id: dict[int, dict[str, str]] | None = None,
    person_embedding_cache: PersonEmbeddingCache,
    hobby_embedding_cache: HobbyEmbeddingCache,
    candidate_pools: dict[int, list[Any]],
    candidate_text_by_id: dict[int, str],
    show_progress_bar: bool = False,
    split: str = "",
) -> None:
    prewarm_start = time.perf_counter()
    person_texts = list(person_text_by_id.values())
    if person_domain_texts_by_id:
        for domain_texts in person_domain_texts_by_id.values():
            person_texts.extend(domain_texts.values())
    if person_texts:
        person_start = time.perf_counter()
        LOGGER.info("Prewarming KURE persona embeddings for evaluation: %s unique texts", len(set(person_texts)))
        person_embedding_cache.encode_batch(
            person_texts,
            show_progress_bar=show_progress_bar,
            progress_desc=f"KURE persona embeddings ({split})" if split else "KURE persona embeddings",
        )
        LOGGER.info("KURE persona embedding prewarm done: seconds=%.3f", time.perf_counter() - person_start)

    hobby_names: set[str] = set()
    for candidates in candidate_pools.values():
        for candidate in candidates:
            candidate_name = str(getattr(candidate, "hobby_name", "") or "").strip()
            if candidate_name:
                hobby_names.add(candidate_name)
    if hobby_names:
        hobby_start = time.perf_counter()
        LOGGER.info("Prewarming KURE hobby embeddings for evaluation: %s unique candidate hobbies", len(hobby_names))
        hobby_embedding_cache.encode_batch(
            sorted(hobby_names),
            show_progress_bar=show_progress_bar,
            progress_desc=f"KURE hobby embeddings ({split})" if split else "KURE hobby embeddings",
        )
        LOGGER.info("KURE hobby embedding prewarm done: seconds=%.3f", time.perf_counter() - hobby_start)
    LOGGER.info("KURE embedding prewarm done: seconds=%.3f", time.perf_counter() - prewarm_start)


def _build_text_similarity_lookup(
    *,
    person_text_by_id: dict[int, str],
    person_audit_pass: dict[int, bool],
    person_embedding_cache: PersonEmbeddingCache,
    hobby_embedding_cache: HobbyEmbeddingCache,
    candidate_pools: dict[int, list[Any]],
    candidate_text_by_id: dict[int, str],
) -> dict[int, dict[int, float]]:
    start_time = time.perf_counter()
    lookup: dict[int, dict[int, float]] = {}
    person_vectors: dict[int, np.ndarray] = {}
    hobby_vectors: dict[int, np.ndarray] = {}

    for person_id, person_text in person_text_by_id.items():
        if person_audit_pass.get(person_id, False) and person_text:
            vector = person_embedding_cache.get(person_text)
            if vector is not None:
                person_vectors[person_id] = _normalize_vector_np(vector)

    for candidates in candidate_pools.values():
        for candidate in candidates:
            candidate_id = int(getattr(candidate, "hobby_id", -1))
            candidate_name = str(candidate_text_by_id.get(candidate_id, getattr(candidate, "hobby_name", "")) or "").strip()
            if candidate_id < 0 or candidate_id in hobby_vectors or not candidate_name:
                continue
            vector = hobby_embedding_cache.get(candidate_name)
            if vector is None:
                vector = hobby_embedding_cache.encode(candidate_name)
            if vector is not None:
                hobby_vectors[candidate_id] = _normalize_vector_np(vector)

    pair_count = 0
    for person_id, candidates in candidate_pools.items():
        person_vector = person_vectors.get(person_id)
        if person_vector is None:
            continue
        person_lookup: dict[int, float] = {}
        for candidate in candidates:
            candidate_id = int(getattr(candidate, "hobby_id", -1))
            hobby_vector = hobby_vectors.get(candidate_id)
            if hobby_vector is None:
                continue
            person_lookup[candidate_id] = max(0.0, min(1.0, float(np.dot(person_vector, hobby_vector))))
        if person_lookup:
            pair_count += len(person_lookup)
            lookup[person_id] = person_lookup

    LOGGER.info(
        "Built KURE similarity lookup: persons=%s, hobbies=%s, pairs=%s, seconds=%.3f",
        len(lookup),
        len(hobby_vectors),
        pair_count,
        time.perf_counter() - start_time,
    )
    return lookup


def _build_domain_similarity_lookup(
    *,
    person_domain_texts_by_id: dict[int, dict[str, str]],
    person_audit_pass: dict[int, bool],
    person_embedding_cache: PersonEmbeddingCache,
    hobby_embedding_cache: HobbyEmbeddingCache,
    candidate_pools: dict[int, list[Any]],
    candidate_text_by_id: dict[int, str],
) -> dict[int, dict[int, dict[str, float]]]:
    start_time = time.perf_counter()
    lookup: dict[int, dict[int, dict[str, float]]] = {}
    person_vectors: dict[int, dict[str, np.ndarray]] = {}
    hobby_vectors: dict[int, np.ndarray] = {}

    domain_to_feature = {
        "professional": "e5_professional_similarity",
        "sports": "e5_sports_similarity",
        "arts": "e5_arts_similarity",
        "travel": "e5_travel_similarity",
        "food": "e5_food_similarity",
        "family": "e5_family_similarity",
    }

    for person_id, domain_texts in person_domain_texts_by_id.items():
        if not person_audit_pass.get(person_id, False):
            continue
        vectors: dict[str, np.ndarray] = {}
        for domain, text in domain_texts.items():
            vector = person_embedding_cache.get(text)
            if vector is not None:
                vectors[domain] = _normalize_vector_np(vector)
        if vectors:
            person_vectors[person_id] = vectors

    for candidates in candidate_pools.values():
        for candidate in candidates:
            candidate_id = int(getattr(candidate, "hobby_id", -1))
            candidate_name = str(candidate_text_by_id.get(candidate_id, getattr(candidate, "hobby_name", "")) or "").strip()
            if candidate_id < 0 or candidate_id in hobby_vectors or not candidate_name:
                continue
            vector = hobby_embedding_cache.get(candidate_name)
            if vector is None:
                vector = hobby_embedding_cache.encode(candidate_name)
            if vector is not None:
                hobby_vectors[candidate_id] = _normalize_vector_np(vector)

    pair_count = 0
    for person_id, candidates in candidate_pools.items():
        domain_vectors = person_vectors.get(person_id)
        if not domain_vectors:
            continue
        person_lookup: dict[int, dict[str, float]] = {}
        for candidate in candidates:
            candidate_id = int(getattr(candidate, "hobby_id", -1))
            hobby_vector = hobby_vectors.get(candidate_id)
            if hobby_vector is None:
                continue
            scores: dict[str, float] = {}
            for domain, person_vector in domain_vectors.items():
                feature_name = domain_to_feature.get(domain)
                if not feature_name:
                    continue
                scores[feature_name] = max(0.0, min(1.0, float(np.dot(person_vector, hobby_vector))))
            if scores:
                person_lookup[candidate_id] = scores
        if person_lookup:
            pair_count += len(person_lookup)
            lookup[person_id] = person_lookup

    LOGGER.info(
        "Built domain similarity lookup: persons=%s, hobbies=%s, pairs=%s, seconds=%.3f",
        len(lookup),
        len(hobby_vectors),
        pair_count,
        time.perf_counter() - start_time,
    )
    return lookup


def _normalize_vector_np(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return array
    return array / norm


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
def _safe_torch_load(path: Path) -> dict[str, Any]:
    torch_module = _torch_module()
    try:
        value = torch_module.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch_module.load(path, map_location="cpu")
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


def _configure_third_party_logging() -> None:
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "sentence_transformers",
        "sentence_transformers.base",
        "sentence_transformers.base.model",
        "transformers",
        "urllib3",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _safe_model_cache_name(model_name: str) -> str:
    return model_name.replace("\\", "__").replace("/", "__").replace(":", "__")


def _huggingface_model_cache_status(model_name: str) -> str:
    model_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return "not_found_download_if_needed"
    try:
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    except OSError:
        return "unknown"
    return "local_snapshot_available" if snapshots else "not_found_download_if_needed"


if __name__ == "__main__":
    main()
