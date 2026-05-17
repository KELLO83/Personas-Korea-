from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

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
    build_domain_persona_texts,
    build_domain_tagged_persona_text,
    load_alias_map,
    normalize_hobby_name,
    load_json,
    load_person_contexts,
    save_json,
)  # noqa: E402
from GNN_Neural_Network.gnn_recommender.embedding_cache import HobbyEmbeddingCache, PersonEmbeddingCache  # noqa: E402
from GNN_Neural_Network.gnn_recommender.ranker import (
    LightGBMRanker,
    RANKER_DOMAIN_TEXT_FEATURE_COLUMNS,
    RANKER_TEXT_RANK_MARGIN_FEATURE_COLUMNS,
    build_ranker_dataset,
    build_text_rank_margin_lookup,
    create_lambda_rank_dataset,
    load_or_build_candidate_pool,
    get_candidate_pool_cache_key,
    build_kure_semantic_candidate_scores,
)  # noqa: E402
from GNN_Neural_Network.gnn_recommender.rerank import HobbyCandidate, build_reranker_config  # noqa: E402
from GNN_Neural_Network.gnn_recommender.text_embedding import KURE_MODEL_NAME, mask_holdout_hobbies, post_mask_leakage_audit  # noqa: E402

TEXT_EMBEDDING_PREPROCESSING_VERSION = "domain_tagged_masked_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM learned ranker with a single config.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgbm_ranker.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("GNN_Neural_Network/artifacts"))
    parser.add_argument("--neg-ratio", type=int, default=4)
    parser.add_argument("--hard-ratio", type=float, default=0.8)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--ranker-val-ratio", type=float, default=0.2)
    parser.add_argument("--max-persons", type=int, default=0, help="Optional validation-person cap for fast pilot runs")
    parser.add_argument("--include-source-features", action="store_true")
    parser.add_argument("--include-text-embedding-feature", action="store_true", help="Enable leakage-safe text embedding similarity feature")
    parser.add_argument(
        "--include-domain-text-embedding-features",
        action="store_true",
        help="Enable E5 domain-specific Stage2 text similarity features in addition to text_embedding_similarity.",
    )
    parser.add_argument(
        "--include-text-rank-margin-features",
        action="store_true",
        help="Enable within-candidate-pool E5 similarity rank, percentile, and gap features.",
    )
    parser.add_argument(
        "--candidate-text-builder",
        choices=["name_only", "name_plus_aliases", "name_plus_category", "name_plus_short_description"],
        default="name_only",
        help="Candidate hobby text builder for Stage2 embedding features.",
    )
    parser.add_argument("--stage1-kure-semantic-provider", action="store_true", help="Enable opt-in KURE-v1 Stage1 semantic candidate provider")
    parser.add_argument("--stage1-kure-score-batch-size", type=int, default=128, help="Person batch size for KURE Stage1 semantic scoring")
    parser.add_argument("--text-embedding-cache-dir", type=Path, default=None, help="Directory for persona/hobby KURE embedding cache")
    parser.add_argument(
        "--text-embedding-model-name",
        type=str,
        default=KURE_MODEL_NAME,
        help="SentenceTransformer model name for Stage2 text embedding features.",
    )
    parser.add_argument(
        "--text-embedding-model-revision",
        type=str,
        default="",
        help="Optional model revision recorded in embedding cache metadata.",
    )
    parser.add_argument(
        "--text-embedding-batch-size",
        "--embedding-batch-size",
        dest="text_embedding_batch_size",
        type=int,
        default=32,
        help="KURE batch size. Use 0 to auto-size from available GPU VRAM.",
    )
    parser.add_argument(
        "--text-embedding-vram-utilization",
        "--embedding-vram-utilization",
        dest="text_embedding_vram_utilization",
        type=float,
        default=0.85,
        help="Target fraction of currently free GPU VRAM to use when --text-embedding-batch-size=0.",
    )
    parser.add_argument(
        "--text-embedding-target-vram-mb",
        "--embedding-target-vram-mb",
        dest="text_embedding_target_vram_mb",
        type=int,
        default=0,
        help="Absolute target GPU VRAM MB for KURE embedding auto batch. Overrides utilization when >0.",
    )
    parser.add_argument("--text-embedding-device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for KURE embedding")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-leaves", type=int, default=None, help="LightGBM num_leaves")
    parser.add_argument("--min-data-in-leaf", type=int, default=None, help="LightGBM min_data_in_leaf")
    parser.add_argument("--learning-rate", type=float, default=None, help="LightGBM learning_rate")
    parser.add_argument("--feature-fraction", type=float, default=None, help="LightGBM feature_fraction")
    parser.add_argument("--bagging-fraction", type=float, default=None, help="LightGBM bagging_fraction")
    parser.add_argument("--bagging-freq", type=int, default=None, help="LightGBM bagging_freq")
    parser.add_argument("--reg-alpha", type=float, default=None, help="LightGBM reg_alpha (L1)")
    parser.add_argument("--reg-lambda", type=float, default=None, help="LightGBM reg_lambda (L2)")
    parser.add_argument(
        "--objective",
        choices=["binary", "lambdarank"],
        default="binary",
        help="Training objective: binary (default) or lambdarank",
    )
    parser.add_argument(
        "--ndcg-eval-at",
        type=int,
        default=10,
        help="NDCG eval k when objective is lambdarank",
    )
    parser.add_argument("--experiment-id", type=str, default="", help="Optional experiment identifier")
    parser.add_argument("--pool-cache-dir", type=Path, default=None, help="Directory for candidate pool cache artifacts")
    parser.add_argument(
        "--cpu-thread-count",
        type=int,
        default=0,
        help="CPU threads for LightGBM training and thread workers. Use 0 for the laptop default policy, currently up to 18.",
    )
    parser.add_argument(
        "--progress-mode",
        choices=["auto", "on", "off"],
        default="on",
        help="Progress output mode for long train steps. Default on so batch progress remains visible.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    _configure_third_party_logging()
    args = parse_args()
    if args.include_domain_text_embedding_features or args.include_text_rank_margin_features:
        args.include_text_embedding_feature = True
    start_time = time.perf_counter()
    text_embedding_model_name = str(args.text_embedding_model_name or KURE_MODEL_NAME).strip() or KURE_MODEL_NAME
    text_embedding_model_revision = str(args.text_embedding_model_revision or "").strip()
    show_progress = args.progress_mode == "on" or (args.progress_mode == "auto" and sys.stderr.isatty())
    logical_cpus = os.cpu_count() or 1
    default_cpu_threads = min(max(logical_cpus - 4, 1), 18)
    requested_cpu_threads = int(args.cpu_thread_count)
    cpu_threads = default_cpu_threads if requested_cpu_threads <= 0 else max(1, min(requested_cpu_threads, logical_cpus))
    data_split = "validation_internal_ranker_split"
    config = load_config(args.config)
    validate_experimental_feature_policy(
        config,
        include_text_embedding_feature=args.include_text_embedding_feature,
        use_stage1_kure_provider=args.stage1_kure_semantic_provider,
        include_source_features=args.include_source_features,
    )
    candidate_k = config.rerank.candidate_pool_size
    if candidate_k <= 0:
        raise ValueError("candidate_pool_size must be positive")

    checkpoint = _safe_torch_load(config.paths.checkpoint)
    person_to_id = _expect_mapping(checkpoint.get("person_to_id"), "person_to_id")
    hobby_to_id = _expect_mapping(checkpoint.get("hobby_to_id"), "hobby_to_id")
    id_to_hobby = {v: k for k, v in hobby_to_id.items()}
    id_to_person = {v: k for k, v in person_to_id.items()}
    hobby_profile_for_text = load_json(config.paths.hobby_profile) if config.paths.hobby_profile.exists() else {}
    hobby_taxonomy_for_text = load_json(config.paths.hobby_taxonomy) if config.paths.hobby_taxonomy.exists() else {}
    hobby_aliases_for_text = (
        _build_hobby_alias_map(config.paths.hobby_aliases, set(id_to_hobby.values()))
        if config.paths.hobby_aliases.exists()
        else {}
    )
    candidate_text_by_id = _build_candidate_text_by_id(
        id_to_hobby=id_to_hobby,
        hobby_profile=hobby_profile_for_text,
        hobby_taxonomy=hobby_taxonomy_for_text,
        alias_map=hobby_aliases_for_text,
        builder=args.candidate_text_builder,
    )

    train_edges = _read_indexed_edges(config.paths.train_edges)
    val_edges = _read_indexed_edges(config.paths.validation_edges)
    train_known = _known_from_edges(train_edges)
    val_person_ids = sorted({pid for pid, _ in val_edges})
    if args.max_persons > 0 and len(val_person_ids) > args.max_persons:
        pilot_rng = random.Random(args.seed)
        val_person_ids = sorted(pilot_rng.sample(val_person_ids, args.max_persons))
        pilot_person_set = set(val_person_ids)
        val_edges = [(pid, hid) for pid, hid in val_edges if pid in pilot_person_set]
    normalization_method = _normalization_method(config.paths.score_normalization)

    input_config_summary = _input_config_summary(
        args.config,
        candidate_pool_size=candidate_k,
        score_normalization=normalization_method,
    )

    _write_status(
        args,
        "started",
        data_split=data_split,
        runtime_seconds=0.0,
        input_config_summary=input_config_summary,
        summary={
            "text_embedding_enabled": args.include_text_embedding_feature,
            "text_embedding_model_name": text_embedding_model_name if args.include_text_embedding_feature else "",
            "text_embedding_model_revision": text_embedding_model_revision if args.include_text_embedding_feature else "",
            "text_embedding_audit_path": str(args.output_dir / "text_leakage_audit.json"),
        },
    )

    contexts = load_person_contexts(config.paths.person_context_csv) if config.paths.person_context_csv.exists() else {}
    hobby_profile = load_json(config.paths.hobby_profile) if config.paths.hobby_profile.exists() else None
    if not isinstance(hobby_profile, dict):
        raise ValueError("hobby_profile.json required")

    reranker_config = build_reranker_config(config.rerank.use_text_fit, config.rerank.weights)

    text_embedding_cache_dir = args.text_embedding_cache_dir or (config.paths.artifact_dir / "text_embedding_cache")
    text_similarity_fn: Callable[[int, HobbyCandidate], float] | None = None
    person_embedding_cache: PersonEmbeddingCache | None = None
    hobby_embedding_cache: HobbyEmbeddingCache | None = None
    person_masked_text: dict[int, str] = {}
    person_domain_texts: dict[int, dict[str, str]] = {}
    person_audit_pass: dict[int, bool] = {}
    text_similarity_lookup: dict[int, dict[int, float]] = {}
    domain_similarity_lookup: dict[int, dict[int, dict[str, float]]] = {}
    text_rank_margin_lookup: dict[int, dict[int, dict[str, float]]] = {}
    kure_device = ""
    embedding_resource_plan: dict[str, object] = {}
    effective_text_batch_size = 0
    text_leakage_audit = {
        "enabled": args.include_text_embedding_feature,
        "include_text_embedding_feature": args.include_text_embedding_feature,
        "pass": True,
        "passed_person_count": 0,
        "failed_person_count": 0,
        "failed_person_ids": [],
        "masked_text_fields": LEAKAGE_TEXT_FIELDS,
        "alias_map_path": str(config.paths.hobby_aliases) if config.paths.hobby_aliases is not None else "",
        "text_embedding_cache_dir": str(text_embedding_cache_dir),
        "text_embedding_model_name": text_embedding_model_name if args.include_text_embedding_feature else "",
        "text_embedding_model_revision": text_embedding_model_revision if args.include_text_embedding_feature else "",
    }

    if args.include_text_embedding_feature:
        hobby_aliases = {}
        if config.paths.hobby_aliases.exists():
            hobby_aliases = _build_hobby_alias_map(config.paths.hobby_aliases, set(id_to_hobby.values()))
        kure_device = _select_kure_device(args.text_embedding_device)
        embedding_resource_plan = _resolve_text_embedding_resource_plan(args, kure_device)
        effective_text_batch_size = int(embedding_resource_plan["effective_batch_size"])
        LOGGER.info(
            "Text embedding resource plan: model=%s revision=%s device=%s, requested_batch_size=%s, "
            "effective_batch_size=%s, gpu_total_vram_mb=%s, gpu_free_vram_mb=%s, "
            "target_vram_mb=%s, estimated_vram_mb=%s",
            text_embedding_model_name,
            text_embedding_model_revision,
            embedding_resource_plan["device"],
            embedding_resource_plan["requested_batch_size"],
            embedding_resource_plan["effective_batch_size"],
            embedding_resource_plan["gpu_total_vram_mb"],
            embedding_resource_plan["gpu_free_vram_mb"],
            embedding_resource_plan["target_vram_mb"],
            embedding_resource_plan["estimated_vram_mb"],
        )
        person_embedding_cache = PersonEmbeddingCache(
            text_embedding_cache_dir,
            model_name=text_embedding_model_name,
            model_revision=text_embedding_model_revision,
            preprocessing_version=TEXT_EMBEDDING_PREPROCESSING_VERSION,
            batch_size=effective_text_batch_size,
            device=kure_device,
        )
        LOGGER.info(
            "Text embedding model source prepared: model=%s revision=%s device=%s embedding_cache_dir=%s huggingface_cache=%s",
            text_embedding_model_name,
            text_embedding_model_revision,
            kure_device,
            text_embedding_cache_dir / _safe_model_cache_name(text_embedding_model_name),
            _huggingface_model_cache_status(text_embedding_model_name),
        )
        hobby_embedding_cache = HobbyEmbeddingCache(
            text_embedding_cache_dir,
            model_name=text_embedding_model_name,
            model_revision=text_embedding_model_revision,
            preprocessing_version=TEXT_EMBEDDING_PREPROCESSING_VERSION,
            batch_size=effective_text_batch_size,
            device=kure_device,
        )
        text_leakage_audit["resource_plan"] = embedding_resource_plan
        text_leakage_payload = _prepare_text_leakage_context(
            person_ids=val_person_ids,
            split_edges=val_edges,
            id_to_person=id_to_person,
            contexts=contexts,
            id_to_hobby=id_to_hobby,
            alias_map=hobby_aliases,
        )
        person_masked_text = text_leakage_payload["person_masked_text"]
        person_domain_texts = text_leakage_payload["person_domain_texts"]
        person_audit_pass = text_leakage_payload["person_audit_pass"]
        text_leakage_audit.update(text_leakage_payload["summary"])

        if person_masked_text:
            text_similarity_fn = _make_text_similarity_fn(
                person_masked_text=person_masked_text,
                person_audit_pass=person_audit_pass,
                person_embedding_cache=person_embedding_cache,
                hobby_embedding_cache=hobby_embedding_cache,
            )
        else:
            print("Warning: no leakage-safe text contexts available. text embedding feature will be zero.")

    else:
        text_leakage_audit["pass"] = True

    text_leakage_audit_path = args.output_dir / "text_leakage_audit.json"
    save_json(text_leakage_audit_path, text_leakage_audit)
    if args.include_text_embedding_feature and _text_audit_failure_rate(text_leakage_audit) > 0.05:
        runtime_seconds = time.perf_counter() - start_time
        disabled_summary = {
            "reason": "post-mask leakage audit failed above threshold",
            "threshold": 0.05,
            "failure_rate": _text_audit_failure_rate(text_leakage_audit),
            "text_embedding_audit_path": str(text_leakage_audit_path),
            "passed_person_count": int(text_leakage_audit.get("passed_person_count", 0)),
            "failed_person_count": int(text_leakage_audit.get("failed_person_count", 0)),
        }
        embedding_model_metadata = _embedding_model_metadata(
            enabled=True,
            model_name=text_embedding_model_name,
            model_revision=text_embedding_model_revision,
            cache_dir=text_embedding_cache_dir,
            batch_size=effective_text_batch_size,
            device=kure_device,
            resource_plan=embedding_resource_plan,
        )
        save_json(args.output_dir / "embedding_model_metadata.json", embedding_model_metadata)
        save_json(args.output_dir / "ranker_params.json", {
            "experiment_id": args.experiment_id,
            "status": "disabled",
            "runtime_seconds": runtime_seconds,
            "data_split": data_split,
            "input_config_summary": input_config_summary,
            "feature_policy": {
                "include_source_features": args.include_source_features,
                "include_text_embedding_feature": True,
                "include_domain_text_embedding_features": args.include_domain_text_embedding_features,
                "include_text_rank_margin_features": args.include_text_rank_margin_features,
                "candidate_text_builder": args.candidate_text_builder,
            },
            "text_leakage_audit_path": str(text_leakage_audit_path),
            "text_leakage_audit": disabled_summary,
            "embedding_model_metadata_path": str(args.output_dir / "embedding_model_metadata.json"),
            "embedding_model_metadata": embedding_model_metadata,
        })
        _write_status(
            args,
            "disabled",
            runtime_seconds=runtime_seconds,
            data_split=data_split,
            input_config_summary=input_config_summary,
            summary=disabled_summary,
        )
        print(
            "Text embedding experiment disabled: "
            f"post-mask leakage audit failure rate {disabled_summary['failure_rate']:.4f} exceeds 0.05"
        )
        return

    stage1_provider_names: tuple[str, ...] = ("popularity", "cooccurrence")
    stage1_provider_cache_fingerprint = ""
    stage1_kure_semantic_scores: dict[int, dict[int, float]] | None = None
    stage1_kure_metadata: dict[str, object] = {"enabled": False}
    if args.stage1_kure_semantic_provider:
        stage1_provider_names = ("popularity", "cooccurrence", "kure_semantic")
        print("[progress] Preparing leakage-safe KURE Stage1 persona text...", flush=True)
        if not person_masked_text:
            hobby_aliases = {}
            if config.paths.hobby_aliases.exists():
                hobby_aliases = _build_hobby_alias_map(config.paths.hobby_aliases, set(id_to_hobby.values()))
            text_leakage_payload = _prepare_text_leakage_context(
                person_ids=val_person_ids,
                split_edges=val_edges,
                id_to_person=id_to_person,
                contexts=contexts,
                id_to_hobby=id_to_hobby,
                alias_map=hobby_aliases,
            )
            person_masked_text = text_leakage_payload["person_masked_text"]
            person_domain_texts = text_leakage_payload["person_domain_texts"]
            person_audit_pass = text_leakage_payload["person_audit_pass"]
            text_leakage_audit.update(text_leakage_payload["summary"])
        if _text_audit_failure_rate(text_leakage_audit) > 0.05:
            raise ValueError("KURE Stage1 semantic provider blocked: post-mask leakage audit failure rate exceeds 0.05")
        kure_device = _select_kure_device(args.text_embedding_device)
        embedding_resource_plan = _resolve_text_embedding_resource_plan(args, kure_device)
        effective_text_batch_size = int(embedding_resource_plan["effective_batch_size"])
        if person_embedding_cache is None:
            person_embedding_cache = PersonEmbeddingCache(
                text_embedding_cache_dir,
                model_name=text_embedding_model_name,
                model_revision=text_embedding_model_revision,
                preprocessing_version=TEXT_EMBEDDING_PREPROCESSING_VERSION,
                batch_size=effective_text_batch_size,
                device=kure_device,
            )
        if hobby_embedding_cache is None:
            hobby_embedding_cache = HobbyEmbeddingCache(
                text_embedding_cache_dir,
                model_name=text_embedding_model_name,
                model_revision=text_embedding_model_revision,
                preprocessing_version=TEXT_EMBEDDING_PREPROCESSING_VERSION,
                batch_size=effective_text_batch_size,
                device=kure_device,
            )
        print("[progress] Encoding KURE embeddings and scoring Stage1 semantic candidates...", flush=True)
        stage1_kure_semantic_scores, stage1_kure_metadata = build_kure_semantic_candidate_scores(
            person_masked_text,
            person_embedding_cache,
            hobby_embedding_cache,
            id_to_hobby,
            train_known,
            candidate_k,
            score_batch_size=args.stage1_kure_score_batch_size,
            show_progress_bar=show_progress,
            progress_desc="KURE Stage1 semantic scoring (train)",
        )
        stage1_provider_cache_fingerprint = str(stage1_kure_metadata.get("fingerprint", ""))
        text_leakage_audit["stage1_kure_semantic_provider"] = stage1_kure_metadata

    popularity_counts = build_popularity_counts(train_edges)
    cooccurrence_counts = build_cooccurrence_counts(train_edges)
    all_hobby_ids = list(hobby_to_id.values())

    pool_cache_dir = args.pool_cache_dir or config.paths.artifact_dir
    pool_cache_key = get_candidate_pool_cache_key(
        person_ids=val_person_ids,
        train_edges=train_edges,
        id_to_hobby=id_to_hobby,
        candidate_k=candidate_k,
        normalization_method=normalization_method,
        label=data_split,
        providers=stage1_provider_names,
        provider_cache_fingerprint=stage1_provider_cache_fingerprint,
    )
    pool_cache_path = pool_cache_dir / "cache" / f"{pool_cache_key}.json"
    rng = random.Random(args.seed)
    shuffled = list(val_person_ids)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1.0 - args.ranker_val_ratio)))
    ranker_train_persons = set(shuffled[:split_idx])
    ranker_val_persons = set(shuffled[split_idx:])
    print(f"Val persons split: {len(ranker_train_persons)} ranker-train, {len(ranker_val_persons)} ranker-val")

    ranker_train_edges = [(pid, hid) for pid, hid in val_edges if pid in ranker_train_persons]
    ranker_val_edges = [(pid, hid) for pid, hid in val_edges if pid in ranker_val_persons]

    pools = load_or_build_candidate_pool(
        person_ids=val_person_ids,
        train_edges=train_edges,
        train_known=train_known,
        candidate_k=candidate_k,
        id_to_hobby=id_to_hobby,
        popularity_counts=popularity_counts,
        cooccurrence_counts=cooccurrence_counts,
        normalization_method=normalization_method,
        cache_dir=pool_cache_dir,
        label=data_split,
        disable_progress=not show_progress,
        stage1_providers=stage1_provider_names,
        semantic_scores_by_person=stage1_kure_semantic_scores,
        provider_cache_fingerprint=stage1_provider_cache_fingerprint,
    )

    if args.include_text_embedding_feature and text_similarity_fn is not None:
        _prewarm_text_embedding_caches(
            person_masked_text=person_masked_text,
            person_domain_texts=person_domain_texts if args.include_domain_text_embedding_features else None,
            person_embedding_cache=person_embedding_cache,
            hobby_embedding_cache=hobby_embedding_cache,
            id_to_hobby=id_to_hobby,
            candidate_text_by_id=candidate_text_by_id,
            candidate_pools=pools,
            show_progress_bar=show_progress,
        )
        if person_embedding_cache is not None and hobby_embedding_cache is not None:
            text_similarity_lookup = _build_text_similarity_lookup(
                person_masked_text=person_masked_text,
                person_audit_pass=person_audit_pass,
                person_embedding_cache=person_embedding_cache,
                hobby_embedding_cache=hobby_embedding_cache,
                candidate_pools=pools,
                candidate_text_by_id=candidate_text_by_id,
            )
            if args.include_domain_text_embedding_features:
                domain_similarity_lookup = _build_domain_similarity_lookup(
                    person_domain_texts=person_domain_texts,
                    person_audit_pass=person_audit_pass,
                    person_embedding_cache=person_embedding_cache,
                    hobby_embedding_cache=hobby_embedding_cache,
                    candidate_pools=pools,
                    candidate_text_by_id=candidate_text_by_id,
                )
                text_leakage_audit["domain_similarity_lookup_person_count"] = len(domain_similarity_lookup)
                text_leakage_audit["domain_similarity_lookup_pair_count"] = sum(
                    len(values) for values in domain_similarity_lookup.values()
                )
            if args.include_text_rank_margin_features:
                text_rank_margin_lookup = build_text_rank_margin_lookup(pools, text_similarity_lookup)
                text_leakage_audit["text_rank_margin_lookup_person_count"] = len(text_rank_margin_lookup)
                text_leakage_audit["text_rank_margin_lookup_pair_count"] = sum(
                    len(values) for values in text_rank_margin_lookup.values()
                )

            def _lookup_text_similarity(person_id: int, candidate: HobbyCandidate) -> float:
                return text_similarity_lookup.get(person_id, {}).get(candidate.hobby_id, 0.0)

            text_similarity_fn = _lookup_text_similarity

    candidate_pool_policy = _candidate_pool_policy(
        pools,
        candidate_k=candidate_k,
        normalization_method=normalization_method,
        cache_key=pool_cache_key,
        cache_path=pool_cache_path,
    )

    params = dict(LightGBMRanker.DEFAULT_PARAMS)
    if args.num_leaves is not None:
        params["num_leaves"] = args.num_leaves
    if args.min_data_in_leaf is not None:
        params["min_data_in_leaf"] = args.min_data_in_leaf
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    params["objective"] = args.objective
    params["num_threads"] = cpu_threads
    if args.objective == "lambdarank":
        params["metric"] = "ndcg"
        params["ndcg_eval_at"] = [int(args.ndcg_eval_at)]
    if args.feature_fraction is not None:
        params["feature_fraction"] = args.feature_fraction
    if args.bagging_fraction is not None:
        params["bagging_fraction"] = args.bagging_fraction
    if args.bagging_freq is not None:
        params["bagging_freq"] = args.bagging_freq
    if args.reg_alpha is not None:
        params["reg_alpha"] = args.reg_alpha
    if args.reg_lambda is not None:
        params["reg_lambda"] = args.reg_lambda

    print(
        f"[progress] Building ranker train dataset with thread workers={cpu_threads} "
        f"(neg_ratio={args.neg_ratio}, hard_ratio={args.hard_ratio})...",
        flush=True,
    )
    train_ds = build_ranker_dataset(
        ranker_train_edges, pools, all_hobby_ids, train_known,
        id_to_hobby, contexts, id_to_person, hobby_profile, reranker_config,
        neg_ratio=args.neg_ratio, hard_ratio=args.hard_ratio, seed=args.seed,
        include_source_features=args.include_source_features,
        include_text_embedding_feature=args.include_text_embedding_feature,
        include_domain_text_embedding_features=args.include_domain_text_embedding_features,
        include_text_rank_margin_features=args.include_text_rank_margin_features,
        text_similarity_fn=None,
        text_similarity_lookup=text_similarity_lookup,
        domain_similarity_lookup=domain_similarity_lookup,
        text_rank_margin_lookup=text_rank_margin_lookup,
        parallel_workers=cpu_threads,
        parallel_backend="thread",
        show_progress=True,
        progress_desc="ranker train rows",
    )
    train_pos = sum(1 for r in train_ds.rows if r.label == 1)
    print(f"  rows={len(train_ds.rows)} pos={train_pos} neg={len(train_ds.rows) - train_pos}")

    print(f"[progress] Building ranker val dataset with thread workers={cpu_threads}...", flush=True)
    val_ds = build_ranker_dataset(
        ranker_val_edges, pools, all_hobby_ids, train_known,
        id_to_hobby, contexts, id_to_person, hobby_profile, reranker_config,
        neg_ratio=args.neg_ratio, hard_ratio=args.hard_ratio, seed=args.seed + 1,
        include_source_features=args.include_source_features,
        include_text_embedding_feature=args.include_text_embedding_feature,
        include_domain_text_embedding_features=args.include_domain_text_embedding_features,
        include_text_rank_margin_features=args.include_text_rank_margin_features,
        text_similarity_fn=None,
        text_similarity_lookup=text_similarity_lookup,
        domain_similarity_lookup=domain_similarity_lookup,
        text_rank_margin_lookup=text_rank_margin_lookup,
        parallel_workers=cpu_threads,
        parallel_backend="thread",
        show_progress=True,
        progress_desc="ranker validation rows",
    )
    val_pos = sum(1 for r in val_ds.rows if r.label == 1)
    print(f"  rows={len(val_ds.rows)} pos={val_pos} neg={len(val_ds.rows) - val_pos}")

    use_listwise = params["objective"] == "lambdarank"
    if use_listwise:
        _, _, train_group_sizes = create_lambda_rank_dataset(train_ds)
        _, _, val_group_sizes = create_lambda_rank_dataset(val_ds)
        print(f"Using LambdaRank objective; train groups={len(train_group_sizes)}, val groups={len(val_group_sizes)}")

    train_lgb = train_ds.to_lgb_dataset(group_by_person=use_listwise)
    val_lgb = val_ds.to_lgb_dataset(reference=train_lgb, group_by_person=use_listwise)

    print(f"[progress] Training LightGBM with params: {params}", flush=True)
    ranker = LightGBMRanker(params=params)
    metadata = ranker.fit(
        train_lgb, val_lgb,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ranker.save(output_dir / "ranker_model.txt")
    runtime_seconds = time.perf_counter() - start_time
    embedding_model_metadata = _embedding_model_metadata(
        enabled=args.include_text_embedding_feature or args.stage1_kure_semantic_provider,
        model_name=text_embedding_model_name,
        model_revision=text_embedding_model_revision,
        cache_dir=text_embedding_cache_dir,
        batch_size=effective_text_batch_size if (args.include_text_embedding_feature or args.stage1_kure_semantic_provider) else 0,
        device=kure_device if (args.include_text_embedding_feature or args.stage1_kure_semantic_provider) else "",
        resource_plan=embedding_resource_plan if (args.include_text_embedding_feature or args.stage1_kure_semantic_provider) else {},
    )
    save_json(output_dir / "embedding_model_metadata.json", embedding_model_metadata)

    save_json(output_dir / "ranker_params.json", {
        "best_iteration": metadata["best_iteration"],
        "best_score": metadata["best_score"],
        "best_metric": metadata.get("best_metric", "auc"),
        "params": params,
        "neg_ratio": args.neg_ratio,
        "hard_ratio": args.hard_ratio,
        "ranker_val_ratio": args.ranker_val_ratio,
        "seed": args.seed,
        "ranker_train_persons": len(ranker_train_persons),
        "ranker_val_persons": len(ranker_val_persons),
        "train_rows": len(train_ds.rows),
        "val_rows": len(val_ds.rows),
        "feature_columns": train_ds.feature_columns,
        "include_source_features": args.include_source_features,
        "candidate_text_builder": args.candidate_text_builder,
        "experiment_id": args.experiment_id,
        "status": "trained",
        "runtime_seconds": runtime_seconds,
        "data_split": data_split,
        "input_config_summary": input_config_summary,
        "max_persons": args.max_persons,
        "model_path": str(output_dir / "ranker_model.txt"),
        "lightgbm_params": params,
        "text_leakage_audit_path": str(text_leakage_audit_path),
        "text_leakage_audit": {
            "enabled": args.include_text_embedding_feature or args.include_domain_text_embedding_features or args.include_text_rank_margin_features,
            "include_domain_text_embedding_features": args.include_domain_text_embedding_features,
            "include_text_rank_margin_features": args.include_text_rank_margin_features,
            "pass": bool(text_leakage_audit.get("pass", False)),
            "failed_person_count": int(text_leakage_audit.get("failed_person_count", 0)),
            "passed_person_count": int(text_leakage_audit.get("passed_person_count", 0)),
        },
        "feature_policy": {
            "include_source_features": args.include_source_features,
            "include_text_embedding_feature": "text_embedding_similarity" in train_ds.feature_columns,
            "include_domain_text_embedding_features": any(
                column in train_ds.feature_columns for column in RANKER_DOMAIN_TEXT_FEATURE_COLUMNS
            ),
            "include_text_rank_margin_features": any(
                column in train_ds.feature_columns for column in RANKER_TEXT_RANK_MARGIN_FEATURE_COLUMNS
            ),
            "candidate_text_builder": args.candidate_text_builder,
        },
        "candidate_pool_policy": candidate_pool_policy,
        "stage1_kure_semantic_provider": stage1_kure_metadata,
        "resource_policy": {
            "logical_cpus": logical_cpus,
            "default_cpu_threads": default_cpu_threads,
            "requested_cpu_threads": requested_cpu_threads,
            "cpu_threads": cpu_threads,
            "lightgbm_train_threads": cpu_threads,
            "ranker_dataset_parallel_backend": "thread",
            "ranker_dataset_thread_workers": cpu_threads,
        },
        "embedding_model_metadata_path": str(output_dir / "embedding_model_metadata.json"),
        "embedding_model_metadata": embedding_model_metadata,
    })
    save_json(output_dir / "ranker_feature_importance.json", metadata["feature_importance"])
    _write_status(
        args,
        "trained",
        runtime_seconds=runtime_seconds,
        data_split=data_split,
        input_config_summary=input_config_summary,
        summary={
            "text_embedding_enabled": args.include_text_embedding_feature,
            "domain_text_embedding_enabled": args.include_domain_text_embedding_features,
            "text_embedding_audit_path": str(text_leakage_audit_path),
            "text_embedding_audit_pass": bool(text_leakage_audit.get("pass", False)),
        },
    )

    best_metric = str(metadata.get("best_metric", "auc"))
    print(f"\nBest iteration: {metadata['best_iteration']}")
    print(f"Best {best_metric}: {metadata['best_score']:.6f}")
    print(f"Model: {output_dir / 'ranker_model.txt'}")
    for name, imp in sorted(metadata["feature_importance"].items(), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")


def _write_status(
    args: argparse.Namespace,
    status: str,
    runtime_seconds: float | None = None,
    data_split: str | None = None,
    input_config_summary: dict[str, object] | None = None,
    summary: dict[str, object] | None = None,
) -> None:
    status_path = args.output_dir / "ranker_train.status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "experiment_id": args.experiment_id,
        "status": status,
    }
    if data_split is not None:
        payload["data_split"] = data_split
    if input_config_summary is not None:
        payload["input_config_summary"] = input_config_summary
    if summary is not None:
        payload["summary"] = summary
    if runtime_seconds is not None:
        payload["runtime_seconds"] = runtime_seconds
    save_json(status_path, payload)


def _candidate_pool_policy(
    pools: dict[int, list[HobbyCandidate]],
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


def _select_kure_device(preference: str) -> str:
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preference == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preference == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported text embedding device: {preference}")


def _resolve_text_embedding_resource_plan(args: argparse.Namespace, device: str) -> dict[str, object]:
    requested = int(args.text_embedding_batch_size)
    total_mb, used_mb, free_mb = _query_gpu_memory_mb()
    if requested > 0:
        return {
            "device": device,
            "requested_batch_size": requested,
            "effective_batch_size": requested,
            "gpu_total_vram_mb": total_mb,
            "gpu_used_vram_mb": used_mb,
            "gpu_free_vram_mb": free_mb,
            "vram_utilization_target": float(args.text_embedding_vram_utilization),
            "requested_target_vram_mb": max(0, int(args.text_embedding_target_vram_mb)),
            "target_vram_mb": 0,
            "estimated_mb_per_text": _estimated_embedding_mb_per_text(),
            "estimated_vram_mb": _estimate_embedding_vram_mb(requested),
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
            "vram_utilization_target": float(args.text_embedding_vram_utilization),
            "requested_target_vram_mb": max(0, int(args.text_embedding_target_vram_mb)),
            "target_vram_mb": 0,
            "estimated_mb_per_text": _estimated_embedding_mb_per_text(),
            "estimated_vram_mb": _estimate_embedding_vram_mb(batch_size),
            "mode": "cpu_default",
        }

    requested_target_mb = max(0, int(args.text_embedding_target_vram_mb))
    if requested_target_mb > 0:
        usable_mb = min(requested_target_mb, max(512, int(total_mb * 0.95)))
        mode = "auto_target_vram"
    else:
        usable_mb = max(512, int(free_mb * max(0.1, min(float(args.text_embedding_vram_utilization), 0.95))))
        mode = "auto_vram"
    batch_size = max(64, min(1024, usable_mb // _estimated_embedding_mb_per_text()))
    return {
        "device": device,
        "requested_batch_size": requested,
        "effective_batch_size": int(batch_size),
        "gpu_total_vram_mb": total_mb,
        "gpu_used_vram_mb": used_mb,
        "gpu_free_vram_mb": free_mb,
        "vram_utilization_target": float(args.text_embedding_vram_utilization),
        "requested_target_vram_mb": requested_target_mb,
        "target_vram_mb": usable_mb,
        "estimated_mb_per_text": _estimated_embedding_mb_per_text(),
        "estimated_vram_mb": _estimate_embedding_vram_mb(batch_size),
        "mode": mode,
    }


def _estimated_embedding_mb_per_text() -> int:
    return 18


def _estimate_embedding_vram_mb(batch_size: int) -> int:
    return int(max(1, batch_size) * _estimated_embedding_mb_per_text())


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


def _build_hobby_categories(
    id_to_hobby: dict[int, str],
    hobby_taxonomy: dict[str, object],
) -> dict[int, str]:
    taxonomy_map = hobby_taxonomy.get("taxonomy", {}) if isinstance(hobby_taxonomy, dict) else {}
    rules = hobby_taxonomy.get("rules", []) if isinstance(hobby_taxonomy, dict) else []
    result: dict[int, str] = {}
    for hobby_id, hobby_name in id_to_hobby.items():
        category = ""
        subcategory = ""
        if isinstance(taxonomy_map, dict):
            entry = taxonomy_map.get(hobby_name, {})
            if isinstance(entry, dict):
                category = str(entry.get("category", "") or "")
                subcategory = str(entry.get("subcategory", "") or "")
        if not category and isinstance(rules, list):
            for rule in rules:
                if isinstance(rule, dict) and rule.get("canonical_hobby") == hobby_name:
                    tax = rule.get("taxonomy", {})
                    if isinstance(tax, dict):
                        category = str(tax.get("category", "") or "")
                        subcategory = str(tax.get("subcategory", "") or "")
                    break
        value = " > ".join(part for part in [category, subcategory] if part)
        if value:
            result[hobby_id] = value
    return result


def _safe_cosine_similarity(vector_a: Any, vector_b: Any) -> float:
    arr_a = np.asarray(vector_a, dtype=np.float32).reshape(-1)
    arr_b = np.asarray(vector_b, dtype=np.float32).reshape(-1)
    if arr_a.size == 0 or arr_b.size == 0:
        return 0.0
    norm_a = float(np.linalg.norm(arr_a))
    norm_b = float(np.linalg.norm(arr_b))
    if not norm_a or not norm_b:
        return 0.0
    value = float(np.dot(arr_a, arr_b) / (norm_a * norm_b))
    if value != value:
        return 0.0
    if value < 0.0:
        return 0.0
    return min(1.0, value)


def _text_audit_failure_rate(text_leakage_audit: dict[str, Any]) -> float:
    failed = int(text_leakage_audit.get("failed_person_count", 0) or 0)
    passed = int(text_leakage_audit.get("passed_person_count", 0) or 0)
    total = failed + passed
    if total <= 0:
        return 0.0
    return failed / total


def _make_text_similarity_fn(
    person_masked_text: dict[int, str],
    person_audit_pass: dict[int, bool],
    person_embedding_cache: PersonEmbeddingCache,
    hobby_embedding_cache: HobbyEmbeddingCache,
) -> Callable[[int, HobbyCandidate], float]:

    def _score(person_id: int, candidate: HobbyCandidate) -> float:
        if not person_audit_pass.get(person_id, False):
            return 0.0
        person_text = person_masked_text.get(person_id, "")
        if not person_text:
            return 0.0

        hobby_name = (candidate.hobby_name or "").strip()
        if not hobby_name:
            return 0.0

        person_embedding = person_embedding_cache.encode(person_text)
        hobby_embedding = hobby_embedding_cache.encode(hobby_name)
        return _safe_cosine_similarity(person_embedding, hobby_embedding)

    return _score


def _prewarm_text_embedding_caches(
    *,
    person_masked_text: dict[int, str],
    person_domain_texts: dict[int, dict[str, str]] | None = None,
    person_embedding_cache: PersonEmbeddingCache,
    hobby_embedding_cache: HobbyEmbeddingCache,
    id_to_hobby: dict[int, str],
    candidate_text_by_id: dict[int, str],
    candidate_pools: dict[int, list[HobbyCandidate]],
    show_progress_bar: bool = False,
) -> None:
    person_texts = list(person_masked_text.values())
    if person_domain_texts:
        for domain_texts in person_domain_texts.values():
            person_texts.extend(domain_texts.values())
    if person_texts:
        LOGGER.info("Prewarming text persona embeddings: %s unique texts", len(set(person_texts)))
        person_embedding_cache.encode_batch(
            person_texts,
            show_progress_bar=show_progress_bar,
            progress_desc="Text persona embeddings (train)",
        )

    hobby_names = {candidate_text_by_id.get(hobby_id, hobby_name) for hobby_id, hobby_name in id_to_hobby.items()}
    for candidates in candidate_pools.values():
        for candidate in candidates:
            candidate_text = candidate_text_by_id.get(candidate.hobby_id, candidate.hobby_name)
            if candidate_text:
                hobby_names.add(candidate_text)
    if hobby_names:
        LOGGER.info("Prewarming text hobby embeddings: %s unique hobbies", len(hobby_names))
        hobby_embedding_cache.encode_batch(
            sorted(hobby_names),
            show_progress_bar=show_progress_bar,
            progress_desc="Text hobby embeddings (train)",
        )


def _build_text_similarity_lookup(
    *,
    person_masked_text: dict[int, str],
    person_audit_pass: dict[int, bool],
    person_embedding_cache: PersonEmbeddingCache,
    hobby_embedding_cache: HobbyEmbeddingCache,
    candidate_pools: dict[int, list[HobbyCandidate]],
    candidate_text_by_id: dict[int, str],
) -> dict[int, dict[int, float]]:
    start_time = time.perf_counter()
    lookup: dict[int, dict[int, float]] = {}
    person_vectors: dict[int, np.ndarray] = {}
    hobby_vectors: dict[int, np.ndarray] = {}

    for person_id, person_text in person_masked_text.items():
        if person_audit_pass.get(person_id, False) and person_text:
            vector = person_embedding_cache.get(person_text)
            if vector is not None:
                person_vectors[person_id] = _normalize_vector_np(vector)

    for candidates in candidate_pools.values():
        for candidate in candidates:
            candidate_name = (candidate_text_by_id.get(candidate.hobby_id, candidate.hobby_name) or "").strip()
            if candidate.hobby_id in hobby_vectors or not candidate_name:
                continue
            vector = hobby_embedding_cache.get(candidate_name)
            if vector is not None:
                hobby_vectors[candidate.hobby_id] = _normalize_vector_np(vector)

    pair_count = 0
    for person_id, candidates in candidate_pools.items():
        person_vector = person_vectors.get(person_id)
        if person_vector is None:
            continue
        person_lookup: dict[int, float] = {}
        for candidate in candidates:
            hobby_vector = hobby_vectors.get(candidate.hobby_id)
            if hobby_vector is None:
                continue
            person_lookup[candidate.hobby_id] = max(0.0, min(1.0, float(np.dot(person_vector, hobby_vector))))
        if person_lookup:
            pair_count += len(person_lookup)
            lookup[person_id] = person_lookup

    LOGGER.info(
        "Built KURE training similarity lookup: persons=%s hobbies=%s pairs=%s seconds=%.3f",
        len(lookup),
        len(hobby_vectors),
        pair_count,
        time.perf_counter() - start_time,
    )
    return lookup


def _build_domain_similarity_lookup(
    *,
    person_domain_texts: dict[int, dict[str, str]],
    person_audit_pass: dict[int, bool],
    person_embedding_cache: PersonEmbeddingCache,
    hobby_embedding_cache: HobbyEmbeddingCache,
    candidate_pools: dict[int, list[HobbyCandidate]],
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

    for person_id, domain_text_by_name in person_domain_texts.items():
        if not person_audit_pass.get(person_id, False):
            continue
        vectors: dict[str, np.ndarray] = {}
        for domain, text in domain_text_by_name.items():
            vector = person_embedding_cache.get(text)
            if vector is not None:
                vectors[domain] = _normalize_vector_np(vector)
        if vectors:
            person_vectors[person_id] = vectors

    for candidates in candidate_pools.values():
        for candidate in candidates:
            candidate_name = (candidate_text_by_id.get(candidate.hobby_id, candidate.hobby_name) or "").strip()
            if candidate.hobby_id in hobby_vectors or not candidate_name:
                continue
            vector = hobby_embedding_cache.get(candidate_name)
            if vector is not None:
                hobby_vectors[candidate.hobby_id] = _normalize_vector_np(vector)

    pair_count = 0
    for person_id, candidates in candidate_pools.items():
        domain_vectors = person_vectors.get(person_id)
        if not domain_vectors:
            continue
        person_lookup: dict[int, dict[str, float]] = {}
        for candidate in candidates:
            hobby_vector = hobby_vectors.get(candidate.hobby_id)
            if hobby_vector is None:
                continue
            scores: dict[str, float] = {}
            for domain, person_vector in domain_vectors.items():
                feature_name = domain_to_feature.get(domain)
                if not feature_name:
                    continue
                scores[feature_name] = max(0.0, min(1.0, float(np.dot(person_vector, hobby_vector))))
            if scores:
                person_lookup[candidate.hobby_id] = scores
        if person_lookup:
            pair_count += len(person_lookup)
            lookup[person_id] = person_lookup

    LOGGER.info(
        "Built domain training similarity lookup: persons=%s hobbies=%s pairs=%s seconds=%.3f",
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


def _prepare_text_leakage_context(
    person_ids: list[int],
    split_edges: list[tuple[int, int]],
    id_to_person: dict[int, str],
    contexts: dict[str, Any],
    id_to_hobby: dict[int, str],
    alias_map: dict[str, list[str]],
) -> dict[str, object]:
    known_by_person: dict[int, set[int]] = defaultdict(set)
    for person_id, hobby_id in split_edges:
        known_by_person[person_id].add(hobby_id)

    person_masked_text: dict[int, str] = {}
    person_domain_texts: dict[int, dict[str, str]] = {}
    person_audit_pass: dict[int, bool] = {}
    passed: list[int] = []
    failed: list[int] = []
    missing_context: list[int] = []

    for person_id in person_ids:
        person_uuid = id_to_person.get(person_id, "")
        context = contexts.get(person_uuid)
        if not context:
            person_audit_pass[person_id] = True
            missing_context.append(person_id)
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
            missing_context.append(person_id)
            continue

        audit_ok = post_mask_leakage_audit(masked, holdout_hobby_names, alias_map=alias_map)
        person_audit_pass[person_id] = bool(audit_ok)
        if audit_ok:
            passed.append(person_id)
            if masked:
                person_masked_text[person_id] = masked
                person_domain_texts[person_id] = build_domain_persona_texts(context, masked_field_values)
        else:
            failed.append(person_id)

    return {
        "person_masked_text": person_masked_text,
        "person_domain_texts": person_domain_texts,
        "person_audit_pass": person_audit_pass,
        "summary": {
            "pass": not failed,
            "text_builder": "build_domain_tagged_persona_text",
            "preprocessing_version": TEXT_EMBEDDING_PREPROCESSING_VERSION,
            "passed_person_count": len(passed),
            "failed_person_count": len(failed),
            "missing_context_person_count": len(missing_context),
            "audit_eligible_person_count": len(passed) + len(failed),
            "failed_person_id_sample": failed[:100],
            "passed_person_id_sample": passed[:100],
            "missing_context_person_id_sample": missing_context[:100],
        },
    }


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
