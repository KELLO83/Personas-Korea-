from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import torch
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.config import load_config, validate_experimental_feature_policy  # noqa: E402
from GNN_Neural_Network.gnn_recommender.data import (  # noqa: E402
    LEAKAGE_TEXT_FIELDS,
    build_domain_tagged_persona_text,
    load_alias_map,
    load_json,
    load_person_contexts,
    normalize_hobby_name,
    save_json,
)
from GNN_Neural_Network.gnn_recommender.embedding_cache import HobbyEmbeddingCache, PersonEmbeddingCache  # noqa: E402
from GNN_Neural_Network.gnn_recommender.metrics import oracle_recall_at_k, summarize_ranking_metrics  # noqa: E402
from GNN_Neural_Network.gnn_recommender.ranker import LightGBMRanker, RANKER_BASE_FEATURE_COLUMNS  # noqa: E402
from GNN_Neural_Network.gnn_recommender.text_embedding import KURE_MODEL_NAME, mask_holdout_hobbies, post_mask_leakage_audit  # noqa: E402


TEXT_EMBEDDING_PREPROCESSING_VERSION = "domain_tagged_masked_v1"
TEXT_FEATURE_NAME = "text_embedding_similarity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate KURE Stage2 feature on the preserved closed-SOTA candidate feature cache.",
    )
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument(
        "--sota-dir",
        type=Path,
        default=Path("GNN_Neural_Network/artifacts/experiments/phase2_5_num_leaves_31"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("GNN_Neural_Network/artifacts/experiments/phase5_c_text_embedding/sota_pool_kure_stage2_feature"),
    )
    parser.add_argument("--split", choices=["validation", "test", "both"], default="validation")
    parser.add_argument("--run-test-if-validation-beats-sota", action="store_true")
    parser.add_argument("--cpu-thread-count", type=int, default=0)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--ranker-val-ratio", type=float, default=0.2)
    parser.add_argument("--neg-ratio", type=int, default=4)
    parser.add_argument("--hard-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-embedding-cache-dir", type=Path, default=None)
    parser.add_argument("--text-embedding-batch-size", type=int, default=32)
    parser.add_argument("--text-embedding-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--force-text-feature", action="store_true")
    parser.add_argument("--progress-mode", choices=["on", "off"], default="on")
    parser.add_argument(
        "--min-sota-reproduction-candidate-recall",
        type=float,
        default=0.95,
        help="Abort if the preserved cache does not reproduce a high-recall SOTA candidate pool.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    start = time.perf_counter()
    show_progress = args.progress_mode == "on"
    cpu_threads = _resolve_cpu_threads(args.cpu_thread_count)
    _apply_cpu_threads(cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    validate_experimental_feature_policy(config, include_text_embedding_feature=True)
    checkpoint = _safe_torch_load(config.paths.checkpoint)
    person_to_id = _expect_mapping(checkpoint.get("person_to_id"), "person_to_id")
    hobby_to_id = _expect_mapping(checkpoint.get("hobby_to_id"), "hobby_to_id")
    id_to_person = {int(v): str(k) for k, v in person_to_id.items()}
    id_to_hobby = {int(v): str(k) for k, v in hobby_to_id.items()}

    train_edges = _read_indexed_edges(config.paths.train_edges)
    train_known = _known_from_edges(train_edges)
    validation_edges = _read_indexed_edges(config.paths.validation_edges)
    test_edges = _read_indexed_edges(config.paths.test_edges)

    validation_cache = _find_feature_cache(args.sota_dir, "validation")
    test_cache = _find_feature_cache(args.sota_dir, "test")
    validation_bundle = _load_feature_cache(validation_cache)
    test_bundle = _load_feature_cache(test_cache)

    feature_columns = list(validation_bundle["metadata"].get("feature_columns", []))
    if feature_columns != RANKER_BASE_FEATURE_COLUMNS:
        raise ValueError(f"Unexpected SOTA feature columns: {feature_columns}")

    baseline_model_path = args.sota_dir / "ranker_model.txt"
    baseline_ranker = LightGBMRanker.load(baseline_model_path)
    validation_baseline = _evaluate_model(
        model=baseline_ranker.model,
        feature_matrix=validation_bundle["X"],
        person_ids=validation_bundle["person_ids"],
        offsets=validation_bundle["offsets"],
        hobby_ids=validation_bundle["hobby_ids"],
        truth=_known_from_edges(validation_edges),
        top_k=tuple(config.eval.top_k),
        num_hobbies=len(id_to_hobby),
        model_best_iteration=baseline_ranker.best_iteration,
        show_progress=show_progress,
        desc="Reproduce SOTA validation",
    )
    _write_json(args.output_dir / "sota_reproduction_validation.json", validation_baseline)
    LOGGER.info(
        "SOTA validation reproduction: recall@10=%.6f ndcg@10=%.6f candidate_recall@50=%.6f",
        validation_baseline["metrics"].get("recall@10", 0.0),
        validation_baseline["metrics"].get("ndcg@10", 0.0),
        validation_baseline["metrics"].get("candidate_recall@50", 0.0),
    )
    reproduced_candidate_recall = float(validation_baseline["metrics"].get("candidate_recall@50", 0.0))
    if reproduced_candidate_recall < float(args.min_sota_reproduction_candidate_recall):
        failure_payload = {
            "status": "blocked_sota_reproduction_failed",
            "reason": (
                "Preserved feature cache does not match the current split/truth labels. "
                "Do not use this run for default promotion decisions."
            ),
            "feature_cache": validation_bundle["path"],
            "candidate_recall@50": reproduced_candidate_recall,
            "min_required_candidate_recall@50": float(args.min_sota_reproduction_candidate_recall),
            "person_count_in_cache": int(len(validation_bundle["person_ids"])),
            "protocol": "preserved_closed_sota_candidate_pool_plus_stage2_kure_feature",
        }
        _write_json(args.output_dir / "run_status.json", failure_payload)
        _write_json(args.output_dir / "validation_metrics.json", failure_payload)
        print(
            "blocked: SOTA candidate pool reproduction failed "
            f"(candidate_recall@50={reproduced_candidate_recall:.6f}, "
            f"required>={float(args.min_sota_reproduction_candidate_recall):.6f})"
        )
        return

    text_cache_dir = args.text_embedding_cache_dir or (config.paths.artifact_dir / "text_embedding_cache")
    device = _select_device(args.text_embedding_device)
    batch_size = max(1, int(args.text_embedding_batch_size))
    contexts = load_person_contexts(config.paths.person_context_csv) if config.paths.person_context_csv.exists() else {}
    alias_map = _build_hobby_alias_map(config.paths.hobby_aliases, set(id_to_hobby.values())) if config.paths.hobby_aliases.exists() else {}
    person_cache = PersonEmbeddingCache(
        text_cache_dir,
        model_name=KURE_MODEL_NAME,
        preprocessing_version=TEXT_EMBEDDING_PREPROCESSING_VERSION,
        batch_size=batch_size,
        device=device,
    )
    hobby_cache = HobbyEmbeddingCache(
        text_cache_dir,
        model_name=KURE_MODEL_NAME,
        preprocessing_version=TEXT_EMBEDDING_PREPROCESSING_VERSION,
        batch_size=batch_size,
        device=device,
    )

    validation_text = _load_or_build_text_feature(
        split="validation",
        output_dir=args.output_dir,
        bundle=validation_bundle,
        target_edges=validation_edges,
        id_to_person=id_to_person,
        id_to_hobby=id_to_hobby,
        contexts=contexts,
        alias_map=alias_map,
        person_cache=person_cache,
        hobby_cache=hobby_cache,
        force=args.force_text_feature,
        show_progress=show_progress,
    )
    validation_X_text = np.column_stack([validation_bundle["X"], validation_text]).astype(np.float32)

    no_text_model, no_text_train_meta = _train_cached_ranker(
        X=validation_bundle["X"],
        person_ids=validation_bundle["person_ids"],
        offsets=validation_bundle["offsets"],
        hobby_ids=validation_bundle["hobby_ids"],
        target_edges=validation_edges,
        train_known=train_known,
        feature_columns=RANKER_BASE_FEATURE_COLUMNS,
        args=args,
        cpu_threads=cpu_threads,
        show_progress=show_progress,
        desc="Build no-text train rows",
    )
    kure_model, kure_train_meta = _train_cached_ranker(
        X=validation_X_text,
        person_ids=validation_bundle["person_ids"],
        offsets=validation_bundle["offsets"],
        hobby_ids=validation_bundle["hobby_ids"],
        target_edges=validation_edges,
        train_known=train_known,
        feature_columns=RANKER_BASE_FEATURE_COLUMNS + [TEXT_FEATURE_NAME],
        args=args,
        cpu_threads=cpu_threads,
        show_progress=show_progress,
        desc="Build KURE train rows",
    )

    no_text_validation = _evaluate_model(
        model=no_text_model,
        feature_matrix=validation_bundle["X"],
        person_ids=validation_bundle["person_ids"],
        offsets=validation_bundle["offsets"],
        hobby_ids=validation_bundle["hobby_ids"],
        truth=_known_from_edges(validation_edges),
        top_k=tuple(config.eval.top_k),
        num_hobbies=len(id_to_hobby),
        model_best_iteration=0,
        show_progress=show_progress,
        desc="Evaluate no-text validation",
    )
    kure_validation = _evaluate_model(
        model=kure_model,
        feature_matrix=validation_X_text,
        person_ids=validation_bundle["person_ids"],
        offsets=validation_bundle["offsets"],
        hobby_ids=validation_bundle["hobby_ids"],
        truth=_known_from_edges(validation_edges),
        top_k=tuple(config.eval.top_k),
        num_hobbies=len(id_to_hobby),
        model_best_iteration=0,
        show_progress=show_progress,
        desc="Evaluate KURE validation",
    )

    validation_result = _comparison_payload(
        split="validation",
        sota_reproduction=validation_baseline,
        no_text=no_text_validation,
        kure=kure_validation,
        no_text_train_meta=no_text_train_meta,
        kure_train_meta=kure_train_meta,
        feature_cache=validation_cache,
        cpu_threads=cpu_threads,
        device=device,
        batch_size=batch_size,
    )
    _write_json(args.output_dir / "validation_metrics.json", validation_result)
    _save_lgb_model(no_text_model, args.output_dir / "no_text_ranker_model.txt")
    _save_lgb_model(kure_model, args.output_dir / "kure_ranker_model.txt")
    _write_json(args.output_dir / "kure_ranker_feature_importance.json", _feature_importance(kure_model))

    validation_beats_sota = (
        float(kure_validation["metrics"].get("recall@10", 0.0)) > float(validation_baseline["metrics"].get("recall@10", 0.0))
        and float(kure_validation["metrics"].get("ndcg@10", 0.0)) > float(validation_baseline["metrics"].get("ndcg@10", 0.0))
    )
    test_requested = args.split in {"test", "both"} or (
        args.run_test_if_validation_beats_sota and validation_beats_sota
    )
    if test_requested:
        test_result = _run_test(
            args=args,
            config_top_k=tuple(config.eval.top_k),
            id_to_hobby=id_to_hobby,
            id_to_person=id_to_person,
            contexts=contexts,
            alias_map=alias_map,
            person_cache=person_cache,
            hobby_cache=hobby_cache,
            test_bundle=test_bundle,
            test_cache=test_cache,
            test_edges=test_edges,
            baseline_ranker=baseline_ranker,
            no_text_model=no_text_model,
            kure_model=kure_model,
            device=device,
            batch_size=batch_size,
            cpu_threads=cpu_threads,
            show_progress=show_progress,
        )
        _write_json(args.output_dir / "test_metrics.json", test_result)
    else:
        _write_json(
            args.output_dir / "test_gate.json",
            {
                "status": "not_run",
                "reason": "validation KURE did not beat reproduced closed SOTA on both recall@10 and ndcg@10",
                "validation_beats_sota": validation_beats_sota,
            },
        )

    _write_json(
        args.output_dir / "run_status.json",
        {
            "status": "completed",
            "runtime_seconds": time.perf_counter() - start,
            "cpu_threads": cpu_threads,
            "text_embedding": {
                "model_name": KURE_MODEL_NAME,
                "preprocessing_version": TEXT_EMBEDDING_PREPROCESSING_VERSION,
                "device": device,
                "batch_size": batch_size,
                "cache_dir": str(text_cache_dir),
            },
            "validation_beats_sota": validation_beats_sota,
        },
    )
    print(
        "validation recall@10 "
        f"sota={validation_baseline['metrics'].get('recall@10', 0.0):.6f} "
        f"no_text={no_text_validation['metrics'].get('recall@10', 0.0):.6f} "
        f"kure={kure_validation['metrics'].get('recall@10', 0.0):.6f}"
    )
    print(
        "validation ndcg@10 "
        f"sota={validation_baseline['metrics'].get('ndcg@10', 0.0):.6f} "
        f"no_text={no_text_validation['metrics'].get('ndcg@10', 0.0):.6f} "
        f"kure={kure_validation['metrics'].get('ndcg@10', 0.0):.6f}"
    )


def _run_test(
    *,
    args: argparse.Namespace,
    config_top_k: tuple[int, ...],
    id_to_hobby: dict[int, str],
    id_to_person: dict[int, str],
    contexts: dict[str, Any],
    alias_map: dict[str, list[str]],
    person_cache: PersonEmbeddingCache,
    hobby_cache: HobbyEmbeddingCache,
    test_bundle: dict[str, Any],
    test_cache: Path,
    test_edges: list[tuple[int, int]],
    baseline_ranker: LightGBMRanker,
    no_text_model: lgb.Booster,
    kure_model: lgb.Booster,
    device: str,
    batch_size: int,
    cpu_threads: int,
    show_progress: bool,
) -> dict[str, Any]:
    test_text = _load_or_build_text_feature(
        split="test",
        output_dir=args.output_dir,
        bundle=test_bundle,
        target_edges=test_edges,
        id_to_person=id_to_person,
        id_to_hobby=id_to_hobby,
        contexts=contexts,
        alias_map=alias_map,
        person_cache=person_cache,
        hobby_cache=hobby_cache,
        force=args.force_text_feature,
        show_progress=show_progress,
    )
    test_X_text = np.column_stack([test_bundle["X"], test_text]).astype(np.float32)
    truth = _known_from_edges(test_edges)
    sota_test = _evaluate_model(
        model=baseline_ranker.model,
        feature_matrix=test_bundle["X"],
        person_ids=test_bundle["person_ids"],
        offsets=test_bundle["offsets"],
        hobby_ids=test_bundle["hobby_ids"],
        truth=truth,
        top_k=config_top_k,
        num_hobbies=len(id_to_hobby),
        model_best_iteration=baseline_ranker.best_iteration,
        show_progress=show_progress,
        desc="Reproduce SOTA test",
    )
    no_text_test = _evaluate_model(
        model=no_text_model,
        feature_matrix=test_bundle["X"],
        person_ids=test_bundle["person_ids"],
        offsets=test_bundle["offsets"],
        hobby_ids=test_bundle["hobby_ids"],
        truth=truth,
        top_k=config_top_k,
        num_hobbies=len(id_to_hobby),
        model_best_iteration=0,
        show_progress=show_progress,
        desc="Evaluate no-text test",
    )
    kure_test = _evaluate_model(
        model=kure_model,
        feature_matrix=test_X_text,
        person_ids=test_bundle["person_ids"],
        offsets=test_bundle["offsets"],
        hobby_ids=test_bundle["hobby_ids"],
        truth=truth,
        top_k=config_top_k,
        num_hobbies=len(id_to_hobby),
        model_best_iteration=0,
        show_progress=show_progress,
        desc="Evaluate KURE test",
    )
    return _comparison_payload(
        split="test",
        sota_reproduction=sota_test,
        no_text=no_text_test,
        kure=kure_test,
        no_text_train_meta={},
        kure_train_meta={},
        feature_cache=test_cache,
        cpu_threads=cpu_threads,
        device=device,
        batch_size=batch_size,
    )


def _load_or_build_text_feature(
    *,
    split: str,
    output_dir: Path,
    bundle: dict[str, Any],
    target_edges: list[tuple[int, int]],
    id_to_person: dict[int, str],
    id_to_hobby: dict[int, str],
    contexts: dict[str, Any],
    alias_map: dict[str, list[str]],
    person_cache: PersonEmbeddingCache,
    hobby_cache: HobbyEmbeddingCache,
    force: bool,
    show_progress: bool,
) -> np.ndarray:
    feature_path = output_dir / f"{split}_kure_text_feature.npy"
    audit_path = output_dir / f"{split}_text_leakage_audit.json"
    if feature_path.exists() and audit_path.exists() and not force:
        LOGGER.info("Text feature cache hit: %s", feature_path)
        return np.load(feature_path).astype(np.float32)

    person_ids = [int(v) for v in bundle["person_ids"]]
    hobby_ids = [int(v) for v in np.unique(bundle["hobby_ids"])]
    text_payload = _prepare_texts(
        person_ids=person_ids,
        target_edges=target_edges,
        id_to_person=id_to_person,
        id_to_hobby=id_to_hobby,
        contexts=contexts,
        alias_map=alias_map,
    )
    audit = dict(text_payload["summary"])
    if _audit_failure_rate(audit) > 0.05:
        _write_json(audit_path, audit)
        raise ValueError(f"{split} text leakage audit failed: failure_rate>{0.05}")

    person_text_by_id = text_payload["person_text_by_id"]
    person_cache.encode_batch(
        list(person_text_by_id.values()),
        show_progress_bar=show_progress,
        progress_desc=f"KURE persona embeddings ({split})",
    )
    hobby_names = [id_to_hobby[hobby_id] for hobby_id in hobby_ids if hobby_id in id_to_hobby]
    hobby_cache.encode_batch(
        hobby_names,
        show_progress_bar=show_progress,
        progress_desc=f"KURE hobby embeddings ({split})",
    )

    person_vectors: dict[int, np.ndarray] = {}
    for person_id, text in tqdm(
        person_text_by_id.items(),
        desc=f"Load persona vectors ({split})",
        unit="person",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        vector = person_cache.get(text)
        if vector is not None:
            person_vectors[person_id] = _normalize(vector)
    hobby_vectors: dict[int, np.ndarray] = {}
    for hobby_id in tqdm(
        hobby_ids,
        desc=f"Load hobby vectors ({split})",
        unit="hobby",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        name = id_to_hobby.get(hobby_id, "")
        vector = hobby_cache.get(name) if name else None
        if vector is not None:
            hobby_vectors[hobby_id] = _normalize(vector)

    person_ids_arr = bundle["person_ids"]
    offsets = bundle["offsets"]
    candidate_hobby_ids = bundle["hobby_ids"]
    sims = np.zeros(candidate_hobby_ids.shape[0], dtype=np.float32)
    for index, person_id in enumerate(
        tqdm(person_ids_arr, desc=f"Build KURE feature ({split})", unit="person", dynamic_ncols=True, disable=not show_progress),
    ):
        person_vector = person_vectors.get(int(person_id))
        if person_vector is None:
            continue
        start = int(offsets[index])
        end = int(offsets[index + 1])
        for row in range(start, end):
            hobby_vector = hobby_vectors.get(int(candidate_hobby_ids[row]))
            if hobby_vector is None:
                continue
            sims[row] = max(0.0, min(1.0, float(np.dot(person_vector, hobby_vector))))

    np.save(feature_path, sims)
    audit.update(
        {
            "feature_path": str(feature_path),
            "row_count": int(sims.shape[0]),
            "nonzero_count": int(np.count_nonzero(sims)),
            "mean_similarity": float(np.mean(sims)) if sims.size else 0.0,
        },
    )
    _write_json(audit_path, audit)
    return sims


def _train_cached_ranker(
    *,
    X: np.ndarray,
    person_ids: np.ndarray,
    offsets: np.ndarray,
    hobby_ids: np.ndarray,
    target_edges: list[tuple[int, int]],
    train_known: dict[int, set[int]],
    feature_columns: list[str],
    args: argparse.Namespace,
    cpu_threads: int,
    show_progress: bool,
    desc: str,
) -> tuple[lgb.Booster, dict[str, Any]]:
    rng = random.Random(args.seed)
    positives_by_person = _known_from_edges(target_edges)
    person_order = [int(pid) for pid in person_ids if int(pid) in positives_by_person]
    rng.shuffle(person_order)
    val_count = max(1, int(len(person_order) * float(args.ranker_val_ratio)))
    val_persons = set(person_order[:val_count])
    train_persons = set(person_order[val_count:])

    train_rows, train_labels, train_missing = _select_training_rows(
        person_ids=person_ids,
        offsets=offsets,
        hobby_ids=hobby_ids,
        positives_by_person=positives_by_person,
        train_known=train_known,
        selected_persons=train_persons,
        neg_ratio=int(args.neg_ratio),
        rng=rng,
        show_progress=show_progress,
        desc=f"{desc} train",
    )
    val_rows, val_labels, val_missing = _select_training_rows(
        person_ids=person_ids,
        offsets=offsets,
        hobby_ids=hobby_ids,
        positives_by_person=positives_by_person,
        train_known=train_known,
        selected_persons=val_persons,
        neg_ratio=int(args.neg_ratio),
        rng=rng,
        show_progress=show_progress,
        desc=f"{desc} val",
    )
    train_dataset = lgb.Dataset(
        X[train_rows],
        label=np.asarray(train_labels, dtype=np.float32),
        feature_name=feature_columns,
        categorical_feature=[feature_columns.index("is_cold_start")] if "is_cold_start" in feature_columns else "auto",
        free_raw_data=False,
    )
    val_dataset = lgb.Dataset(
        X[val_rows],
        label=np.asarray(val_labels, dtype=np.float32),
        feature_name=feature_columns,
        categorical_feature=[feature_columns.index("is_cold_start")] if "is_cold_start" in feature_columns else "auto",
        reference=train_dataset,
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "learning_rate": 0.05,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "seed": int(args.seed),
        "num_threads": int(cpu_threads),
    }
    LOGGER.info("Training LightGBM: features=%s train_rows=%s val_rows=%s", len(feature_columns), len(train_rows), len(val_rows))
    model = lgb.train(
        params=params,
        train_set=train_dataset,
        num_boost_round=int(args.num_boost_round),
        valid_sets=[train_dataset, val_dataset],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=int(args.early_stopping), verbose=True),
            lgb.log_evaluation(period=25),
        ],
    )
    return model, {
        "params": params,
        "best_iteration": int(model.best_iteration or 0),
        "train_persons": len(train_persons),
        "val_persons": len(val_persons),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "missing_positive_rows_train": train_missing,
        "missing_positive_rows_val": val_missing,
        "sampling_policy": "positives_present_in_preserved_sota_pool_plus_candidate_pool_negatives",
        "neg_ratio": int(args.neg_ratio),
    }


def _select_training_rows(
    *,
    person_ids: np.ndarray,
    offsets: np.ndarray,
    hobby_ids: np.ndarray,
    positives_by_person: dict[int, set[int]],
    train_known: dict[int, set[int]],
    selected_persons: set[int],
    neg_ratio: int,
    rng: random.Random,
    show_progress: bool,
    desc: str,
) -> tuple[list[int], list[int], int]:
    rows: list[int] = []
    labels: list[int] = []
    missing_positive_rows = 0
    for index, raw_person_id in enumerate(
        tqdm(person_ids, desc=desc, unit="person", dynamic_ncols=True, disable=not show_progress),
    ):
        person_id = int(raw_person_id)
        if person_id not in selected_persons:
            continue
        positives = positives_by_person.get(person_id, set())
        if not positives:
            continue
        start = int(offsets[index])
        end = int(offsets[index + 1])
        row_hobbies = [int(hobby_ids[row]) for row in range(start, end)]
        row_by_hobby = {hobby_id: row for row, hobby_id in zip(range(start, end), row_hobbies, strict=False)}
        positive_rows = [row_by_hobby[hobby_id] for hobby_id in positives if hobby_id in row_by_hobby]
        missing_positive_rows += len(positives) - len(positive_rows)
        if not positive_rows:
            continue
        negative_candidates = [
            row_by_hobby[hobby_id]
            for hobby_id in row_hobbies
            if hobby_id not in positives and hobby_id not in train_known.get(person_id, set())
        ]
        negative_count = min(len(negative_candidates), max(1, neg_ratio * len(positive_rows)))
        negative_rows = rng.sample(negative_candidates, negative_count) if negative_count else []
        rows.extend(positive_rows)
        labels.extend([1] * len(positive_rows))
        rows.extend(negative_rows)
        labels.extend([0] * len(negative_rows))
    return rows, labels, missing_positive_rows


def _evaluate_model(
    *,
    model: lgb.Booster | None,
    feature_matrix: np.ndarray,
    person_ids: np.ndarray,
    offsets: np.ndarray,
    hobby_ids: np.ndarray,
    truth: dict[int, set[int]],
    top_k: tuple[int, ...],
    num_hobbies: int,
    model_best_iteration: int,
    show_progress: bool,
    desc: str,
) -> dict[str, Any]:
    if model is None:
        raise ValueError("LightGBM model is not loaded")
    recommendations: dict[int, list[int]] = {}
    candidate_pool: dict[int, list[int]] = {}
    for index, raw_person_id in enumerate(
        tqdm(person_ids, desc=desc, unit="person", dynamic_ncols=True, disable=not show_progress),
    ):
        person_id = int(raw_person_id)
        start = int(offsets[index])
        end = int(offsets[index + 1])
        candidates = [int(value) for value in hobby_ids[start:end]]
        candidate_pool[person_id] = candidates
        if person_id not in truth:
            continue
        scores = model.predict(
            feature_matrix[start:end],
            num_iteration=model_best_iteration if model_best_iteration > 0 else None,
        )
        ranked = [
            hobby_id
            for hobby_id, _score in sorted(
                zip(candidates, scores, strict=False),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ]
        recommendations[person_id] = ranked
    metrics = summarize_ranking_metrics(
        truth,
        recommendations,
        top_k,
        num_total_items=num_hobbies,
        candidate_pool_by_person=candidate_pool,
    )
    metrics["candidate_recall@50"] = oracle_recall_at_k(truth, candidate_pool, 50)
    return {
        "metrics": metrics,
        "person_count": len([person_id for person_id in truth if truth[person_id]]),
        "candidate_person_count": len(candidate_pool),
    }


def _prepare_texts(
    *,
    person_ids: list[int],
    target_edges: list[tuple[int, int]],
    id_to_person: dict[int, str],
    id_to_hobby: dict[int, str],
    contexts: dict[str, Any],
    alias_map: dict[str, list[str]],
) -> dict[str, Any]:
    holdout_by_person = _known_from_edges(target_edges)
    person_text_by_id: dict[int, str] = {}
    passed: list[int] = []
    failed: list[int] = []
    missing: list[int] = []
    for person_id in tqdm(person_ids, desc="Prepare masked KURE text", unit="person", dynamic_ncols=True):
        person_uuid = id_to_person.get(person_id, "")
        context = contexts.get(person_uuid)
        if context is None:
            missing.append(person_id)
            continue
        holdout_names = {
            normalize_hobby_name(id_to_hobby[hobby_id])
            for hobby_id in holdout_by_person.get(person_id, set())
            if hobby_id in id_to_hobby
        }
        masked_fields: dict[str, str] = {}
        for field in LEAKAGE_TEXT_FIELDS:
            value = str(getattr(context, field, "") or "").strip()
            if value:
                masked_fields[field] = mask_holdout_hobbies(value, holdout_names, alias_map=alias_map) if holdout_names else value
        masked_text = build_domain_tagged_persona_text(context, masked_fields)
        if not masked_text:
            missing.append(person_id)
            continue
        if post_mask_leakage_audit(masked_text, holdout_names, alias_map=alias_map):
            person_text_by_id[person_id] = masked_text
            passed.append(person_id)
        else:
            failed.append(person_id)
    return {
        "person_text_by_id": person_text_by_id,
        "summary": {
            "audit_pass": not failed,
            "text_builder": "build_domain_tagged_persona_text",
            "masking": "mask_holdout_hobbies",
            "preprocessing_version": TEXT_EMBEDDING_PREPROCESSING_VERSION,
            "passed_person_count": len(passed),
            "failed_person_count": len(failed),
            "missing_context_person_count": len(missing),
            "failed_person_id_sample": failed[:100],
            "missing_context_person_id_sample": missing[:100],
        },
    }


def _find_feature_cache(sota_dir: Path, split: str) -> Path:
    cache_dirs = [
        sota_dir / "cache" / "cache",
        Path("GNN_Neural_Network/artifacts/cache/cache"),
        sota_dir / "feature_cache" / "cache",
    ]
    candidates: list[tuple[int, Path]] = []
    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue
        for meta_path in cache_dir.glob("features_*.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("split") != split:
                continue
            npz_path = meta_path.with_suffix(".npz")
            if not npz_path.exists():
                continue
            experiment_id = str(meta.get("experiment_id", ""))
            priority = 0
            if cache_dir.as_posix().endswith("artifacts/cache/cache"):
                priority += 100
            if npz_path.stem in {"features_ac22205dddbdfaba", "features_550d33f5a0031157"}:
                priority += 100
            if "phase2_5" in experiment_id:
                priority += 10
            if "cold_start_baseline" in experiment_id:
                priority += 5
            candidates.append((priority, npz_path))
    if not candidates:
        searched = ", ".join(str(path) for path in cache_dirs)
        raise FileNotFoundError(f"No preserved SOTA feature cache found for split={split}; searched={searched}")
    return sorted(candidates, key=lambda item: (-item[0], item[1].name))[0][1]


def _load_feature_cache(path: Path) -> dict[str, Any]:
    data = np.load(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return {
        "X": data["X"].astype(np.float32),
        "person_ids": data["person_ids"].astype(np.int64),
        "offsets": data["offsets"].astype(np.int64),
        "hobby_ids": data["hobby_ids"].astype(np.int64),
        "metadata": metadata,
        "path": str(path),
    }


def _comparison_payload(
    *,
    split: str,
    sota_reproduction: dict[str, Any],
    no_text: dict[str, Any],
    kure: dict[str, Any],
    no_text_train_meta: dict[str, Any],
    kure_train_meta: dict[str, Any],
    feature_cache: Path,
    cpu_threads: int,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    no_text_metrics = no_text["metrics"]
    kure_metrics = kure["metrics"]
    sota_metrics = sota_reproduction["metrics"]
    return {
        "split": split,
        "status": f"{split}_evaluated",
        "feature_cache": str(feature_cache),
        "protocol": "preserved_closed_sota_candidate_pool_plus_stage2_kure_feature",
        "resource_plan": {
            "cpu_threads": cpu_threads,
            "text_embedding_device": device,
            "text_embedding_batch_size": batch_size,
        },
        "sota_reproduction": sota_reproduction,
        "no_text_retrained_same_cache_protocol": no_text,
        "kure_stage2_feature": kure,
        "deltas": {
            "kure_minus_no_text_recall@10": float(kure_metrics.get("recall@10", 0.0)) - float(no_text_metrics.get("recall@10", 0.0)),
            "kure_minus_no_text_ndcg@10": float(kure_metrics.get("ndcg@10", 0.0)) - float(no_text_metrics.get("ndcg@10", 0.0)),
            "kure_minus_sota_reproduction_recall@10": float(kure_metrics.get("recall@10", 0.0)) - float(sota_metrics.get("recall@10", 0.0)),
            "kure_minus_sota_reproduction_ndcg@10": float(kure_metrics.get("ndcg@10", 0.0)) - float(sota_metrics.get("ndcg@10", 0.0)),
        },
        "train_metadata": {
            "no_text": no_text_train_meta,
            "kure": kure_train_meta,
        },
    }


def _known_from_edges(edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    known: dict[int, set[int]] = defaultdict(set)
    for person_id, hobby_id in edges:
        known[int(person_id)].add(int(hobby_id))
    return dict(known)


def _read_indexed_edges(path: Path) -> list[tuple[int, int]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [(int(row["person_id"]), int(row["hobby_id"])) for row in reader]


def _safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint must be a dict: {path}")
    return value


def _expect_mapping(value: Any, name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint missing mapping: {name}")
    return {str(k): int(v) for k, v in value.items()}


def _build_hobby_alias_map(alias_map_path: Path, valid_hobby_names: set[str]) -> dict[str, list[str]]:
    normalized_valid = {normalize_hobby_name(value) for value in valid_hobby_names}
    raw_alias_map = load_alias_map(alias_map_path)
    result: dict[str, set[str]] = defaultdict(set)
    for raw_alias, canonical in raw_alias_map.items():
        normalized_alias = normalize_hobby_name(raw_alias)
        normalized_canonical = normalize_hobby_name(canonical)
        if normalized_canonical in normalized_valid and normalized_alias:
            result[normalized_canonical].add(normalized_alias)
    return {key: sorted(values) for key, values in result.items()}


def _normalize(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    return array if norm <= 0.0 else (array / norm).astype(np.float32)


def _audit_failure_rate(audit: dict[str, Any]) -> float:
    failed = int(audit.get("failed_person_count", 0) or 0)
    passed = int(audit.get("passed_person_count", 0) or 0)
    total = failed + passed
    return 0.0 if total <= 0 else failed / total


def _select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_cpu_threads(requested: int) -> int:
    logical = os.cpu_count() or 1
    if requested > 0:
        return max(1, min(int(requested), logical))
    return min(max(logical - 4, 1), 18)


def _apply_cpu_threads(cpu_threads: int) -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(cpu_threads)
    try:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
    except Exception:
        pass


def _feature_importance(model: lgb.Booster) -> dict[str, float]:
    importance = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    return {str(name): float(value) for name, value in zip(names, importance, strict=False)}


def _save_lgb_model(model: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, payload)


if __name__ == "__main__":
    main()
