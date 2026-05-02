from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

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
    load_alias_map,
    normalize_hobby_name,
    load_json,
    load_person_contexts,
    save_json,
)  # noqa: E402
from GNN_Neural_Network.gnn_recommender.embedding_cache import HobbyEmbeddingCache, PersonEmbeddingCache  # noqa: E402
from GNN_Neural_Network.gnn_recommender.ranker import (
    LightGBMRanker,
    build_ranker_dataset,
    create_lambda_rank_dataset,
    load_or_build_candidate_pool,
    get_candidate_pool_cache_key,
)  # noqa: E402
from GNN_Neural_Network.gnn_recommender.rerank import HobbyCandidate, build_reranker_config  # noqa: E402
from GNN_Neural_Network.gnn_recommender.text_embedding import KURE_MODEL_NAME, mask_holdout_hobbies, post_mask_leakage_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM learned ranker with a single config.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("GNN_Neural_Network/artifacts"))
    parser.add_argument("--neg-ratio", type=int, default=4)
    parser.add_argument("--hard-ratio", type=float, default=0.8)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--ranker-val-ratio", type=float, default=0.2)
    parser.add_argument("--include-source-features", action="store_true")
    parser.add_argument("--include-text-embedding-feature", action="store_true", help="Enable leakage-safe text embedding similarity feature")
    parser.add_argument("--text-embedding-cache-dir", type=Path, default=None, help="Directory for persona/hobby KURE embedding cache")
    parser.add_argument("--text-embedding-batch-size", type=int, default=32, help="KURE batch size for hobby embedding cache")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    data_split = "validation_internal_ranker_split"
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
    train_known = _known_from_edges(train_edges)
    val_person_ids = sorted({pid for pid, _ in val_edges})
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
    }

    if args.include_text_embedding_feature:
        hobby_aliases = {}
        if config.paths.hobby_aliases.exists():
            hobby_aliases = _build_hobby_alias_map(config.paths.hobby_aliases, set(id_to_hobby.values()))
        person_embedding_cache = PersonEmbeddingCache(text_embedding_cache_dir)
        hobby_embedding_cache = HobbyEmbeddingCache(
            text_embedding_cache_dir,
            model_name=KURE_MODEL_NAME,
            batch_size=max(1, int(args.text_embedding_batch_size)),
            device=_select_kure_device(args.text_embedding_device),
        )
        text_leakage_payload = _prepare_text_leakage_context(
            person_ids=val_person_ids,
            split_edges=val_edges,
            id_to_person=id_to_person,
            contexts=contexts,
            id_to_hobby=id_to_hobby,
            alias_map=hobby_aliases,
        )
        person_masked_text = text_leakage_payload["person_masked_text"]
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
    )

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

    print(f"Building ranker train dataset (neg_ratio={args.neg_ratio}, hard_ratio={args.hard_ratio})...")
    train_ds = build_ranker_dataset(
        ranker_train_edges, pools, all_hobby_ids, train_known,
        id_to_hobby, contexts, id_to_person, hobby_profile, reranker_config,
        neg_ratio=args.neg_ratio, hard_ratio=args.hard_ratio, seed=args.seed,
        include_source_features=args.include_source_features,
        include_text_embedding_feature=args.include_text_embedding_feature,
        text_similarity_fn=text_similarity_fn,
    )
    train_pos = sum(1 for r in train_ds.rows if r.label == 1)
    print(f"  rows={len(train_ds.rows)} pos={train_pos} neg={len(train_ds.rows) - train_pos}")

    print("Building ranker val dataset...")
    val_ds = build_ranker_dataset(
        ranker_val_edges, pools, all_hobby_ids, train_known,
        id_to_hobby, contexts, id_to_person, hobby_profile, reranker_config,
        neg_ratio=args.neg_ratio, hard_ratio=args.hard_ratio, seed=args.seed + 1,
        include_source_features=args.include_source_features,
        include_text_embedding_feature=args.include_text_embedding_feature,
        text_similarity_fn=text_similarity_fn,
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

    print(f"Training LightGBM with params: {params}")
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
        "experiment_id": args.experiment_id,
        "status": "trained",
        "runtime_seconds": runtime_seconds,
        "data_split": data_split,
        "input_config_summary": input_config_summary,
        "model_path": str(output_dir / "ranker_model.txt"),
        "lightgbm_params": params,
        "text_leakage_audit_path": str(text_leakage_audit_path),
        "text_leakage_audit": {
            "enabled": args.include_text_embedding_feature,
            "pass": bool(text_leakage_audit.get("pass", False)),
            "failed_person_count": int(text_leakage_audit.get("failed_person_count", 0)),
            "passed_person_count": int(text_leakage_audit.get("passed_person_count", 0)),
        },
        "feature_policy": {
            "include_source_features": args.include_source_features,
            "include_text_embedding_feature": "text_embedding_similarity" in train_ds.feature_columns,
        },
        "candidate_pool_policy": candidate_pool_policy,
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
    person_audit_pass: dict[int, bool] = {}
    passed: list[int] = []
    failed: list[int] = []

    for person_id in person_ids:
        person_uuid = id_to_person.get(person_id, "")
        context = contexts.get(person_uuid)
        if not context:
            person_audit_pass[person_id] = False
            failed.append(person_id)
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
        if audit_ok:
            passed.append(person_id)
            if masked:
                person_masked_text[person_id] = masked
        else:
            failed.append(person_id)

    return {
        "person_masked_text": person_masked_text,
        "person_audit_pass": person_audit_pass,
        "summary": {
            "pass": not failed,
            "passed_person_count": len(passed),
            "failed_person_count": len(failed),
            "failed_person_ids": failed,
            "passed_person_ids": passed,
        },
    }


def _normalization_method(path: Path) -> str:
    if not path.exists():
        return "rank_percentile"
    value = load_json(path)
    if not isinstance(value, dict):
        return "rank_percentile"
    return str(value.get("method", "rank_percentile"))


if __name__ == "__main__":
    main()
