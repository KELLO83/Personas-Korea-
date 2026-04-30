from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.baseline import (  # noqa: E402
    build_cooccurrence_counts,
    build_popularity_counts,
    cooccurrence_candidate_provider,
    popularity_candidate_provider,
)
from GNN_Neural_Network.gnn_recommender.config import load_config  # noqa: E402
from GNN_Neural_Network.gnn_recommender.data import load_json, load_person_contexts, save_json  # noqa: E402
from GNN_Neural_Network.gnn_recommender.ranker import LightGBMRanker, build_ranker_dataset  # noqa: E402
from GNN_Neural_Network.gnn_recommender.recommend import merge_candidates_by_hobby, normalize_candidate_scores  # noqa: E402
from GNN_Neural_Network.gnn_recommender.rerank import HobbyCandidate, build_reranker_config, merge_stage1_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM learned ranker.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("GNN_Neural_Network/artifacts"))
    parser.add_argument("--neg-ratio", type=int, default=4)
    parser.add_argument("--hard-ratio", type=float, default=0.8)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--ranker-val-ratio", type=float, default=0.2)
    parser.add_argument("--include-source-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    contexts = load_person_contexts(config.paths.person_context_csv) if config.paths.person_context_csv.exists() else {}
    hobby_profile = load_json(config.paths.hobby_profile) if config.paths.hobby_profile.exists() else None
    if not isinstance(hobby_profile, dict):
        raise ValueError("hobby_profile.json required")

    normalization_method = _normalization_method(config.paths.score_normalization)
    reranker_config = build_reranker_config(config.rerank.use_text_fit, config.rerank.weights)

    popularity_counts = build_popularity_counts(train_edges)
    cooccurrence_counts = build_cooccurrence_counts(train_edges)
    all_hobby_ids = list(hobby_to_id.values())

    val_person_ids = sorted({pid for pid, _ in val_edges})
    rng = random.Random(args.seed)
    shuffled = list(val_person_ids)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1.0 - args.ranker_val_ratio)))
    ranker_train_persons = set(shuffled[:split_idx])
    ranker_val_persons = set(shuffled[split_idx:])
    print(f"Val persons split: {len(ranker_train_persons)} ranker-train, {len(ranker_val_persons)} ranker-val")

    ranker_train_edges = [(pid, hid) for pid, hid in val_edges if pid in ranker_train_persons]
    ranker_val_edges = [(pid, hid) for pid, hid in val_edges if pid in ranker_val_persons]

    print(f"Generating candidate pools for {len(val_person_ids)} val persons...")
    pools = _generate_candidate_pools(
        val_person_ids, train_edges, train_known, candidate_k,
        id_to_hobby, popularity_counts, cooccurrence_counts, normalization_method,
    )

    print("Building ranker train dataset...")
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

    print("Training LightGBM...")
    ranker = LightGBMRanker()
    metadata = ranker.fit(
        train_lgb, val_lgb,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ranker.save(output_dir / "ranker_model.txt")
    save_json(output_dir / "ranker_params.json", {
        "best_iteration": metadata["best_iteration"],
        "best_score": metadata["best_score"],
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
    })
    save_json(output_dir / "ranker_feature_importance.json", metadata["feature_importance"])

    print(f"\nBest iteration: {metadata['best_iteration']}")
    print(f"Best AUC: {metadata['best_score']:.6f}")
    print(f"Model: {output_dir / 'ranker_model.txt'}")
    for name, imp in sorted(metadata["feature_importance"].items(), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")


def _generate_candidate_pools(
    person_ids: list[int],
    train_edges: list[tuple[int, int]],
    train_known: dict[int, set[int]],
    candidate_k: int,
    id_to_hobby: dict[int, str],
    popularity_counts: Counter[int],
    cooccurrence_counts: dict[int, Counter[int]],
    normalization_method: str,
) -> dict[int, list[HobbyCandidate]]:
    pools: dict[int, list[HobbyCandidate]] = {}
    for person_id in tqdm(person_ids, desc="candidate pools"):
        known = train_known.get(person_id, set())
        pop = normalize_candidate_scores(
            popularity_candidate_provider(train_edges, person_id, known, candidate_k, popularity_counts=popularity_counts),
            normalization_method,
        )
        cooc = normalize_candidate_scores(
            cooccurrence_candidate_provider(train_edges, person_id, known, candidate_k, cooccurrence_counts=cooccurrence_counts),
            normalization_method,
        )
        merged = merge_candidates_by_hobby({"popularity": pop, "cooccurrence": cooc}, candidate_k)
        pools[person_id] = merge_stage1_candidates(merged, id_to_hobby)
    return pools


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
