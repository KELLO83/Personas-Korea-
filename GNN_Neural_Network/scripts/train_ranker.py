from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.baseline import (  # noqa: E402
    build_cooccurrence_counts,
    build_popularity_counts,
)
from GNN_Neural_Network.gnn_recommender.config import load_config  # noqa: E402
from GNN_Neural_Network.gnn_recommender.data import load_json, load_person_contexts, save_json  # noqa: E402
from GNN_Neural_Network.gnn_recommender.ranker import (
    LightGBMRanker,
    build_ranker_dataset,
    load_or_build_candidate_pool,
    get_candidate_pool_cache_key,
)  # noqa: E402
from GNN_Neural_Network.gnn_recommender.rerank import HobbyCandidate, build_reranker_config  # noqa: E402


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-leaves", type=int, default=None, help="LightGBM num_leaves")
    parser.add_argument("--min-data-in-leaf", type=int, default=None, help="LightGBM min_data_in_leaf")
    parser.add_argument("--learning-rate", type=float, default=None, help="LightGBM learning_rate")
    parser.add_argument("--feature-fraction", type=float, default=None, help="LightGBM feature_fraction")
    parser.add_argument("--bagging-fraction", type=float, default=None, help="LightGBM bagging_fraction")
    parser.add_argument("--bagging-freq", type=int, default=None, help="LightGBM bagging_freq")
    parser.add_argument("--reg-alpha", type=float, default=None, help="LightGBM reg_alpha (L1)")
    parser.add_argument("--reg-lambda", type=float, default=None, help="LightGBM reg_lambda (L2)")
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
    normalization_method = _normalization_method(config.paths.score_normalization)

    input_config_summary = _input_config_summary(
        args.config,
        candidate_pool_size=candidate_k,
        score_normalization=normalization_method,
    )

    _write_status(args, "started", data_split=data_split, runtime_seconds=0.0, input_config_summary=input_config_summary)

    contexts = load_person_contexts(config.paths.person_context_csv) if config.paths.person_context_csv.exists() else {}
    hobby_profile = load_json(config.paths.hobby_profile) if config.paths.hobby_profile.exists() else None
    if not isinstance(hobby_profile, dict):
        raise ValueError("hobby_profile.json required")

    reranker_config = build_reranker_config(config.rerank.use_text_fit, config.rerank.weights)

    popularity_counts = build_popularity_counts(train_edges)
    cooccurrence_counts = build_cooccurrence_counts(train_edges)
    all_hobby_ids = list(hobby_to_id.values())

    val_person_ids = sorted({pid for pid, _ in val_edges})
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
    )
    train_pos = sum(1 for r in train_ds.rows if r.label == 1)
    print(f"  rows={len(train_ds.rows)} pos={train_pos} neg={len(train_ds.rows) - train_pos}")

    print("Building ranker val dataset...")
    val_ds = build_ranker_dataset(
        ranker_val_edges, pools, all_hobby_ids, train_known,
        id_to_hobby, contexts, id_to_person, hobby_profile, reranker_config,
        neg_ratio=args.neg_ratio, hard_ratio=args.hard_ratio, seed=args.seed + 1,
        include_source_features=args.include_source_features,
    )
    val_pos = sum(1 for r in val_ds.rows if r.label == 1)
    print(f"  rows={len(val_ds.rows)} pos={val_pos} neg={len(val_ds.rows) - val_pos}")

    train_lgb = train_ds.to_lgb_dataset()
    val_lgb = val_ds.to_lgb_dataset(reference=train_lgb)

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
        "feature_policy": {
            "include_source_features": args.include_source_features,
            "include_text_embedding_feature": False,
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
    )

    print(f"\nBest iteration: {metadata['best_iteration']}")
    print(f"Best AUC: {metadata['best_score']:.6f}")
    print(f"Model: {output_dir / 'ranker_model.txt'}")
    for name, imp in sorted(metadata["feature_importance"].items(), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")


def _write_status(
    args: argparse.Namespace,
    status: str,
    runtime_seconds: float | None = None,
    data_split: str | None = None,
    input_config_summary: dict[str, object] | None = None,
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


def _normalization_method(path: Path) -> str:
    if not path.exists():
        return "rank_percentile"
    value = load_json(path)
    if not isinstance(value, dict):
        return "rank_percentile"
    return str(value.get("method", "rank_percentile"))


if __name__ == "__main__":
    main()
