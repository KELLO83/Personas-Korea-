from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.baseline import baseline_ranking_metrics  # noqa: E402
from GNN_Neural_Network.gnn_recommender.config import load_config  # noqa: E402
from GNN_Neural_Network.gnn_recommender.metrics import summarize_ranking_metrics  # noqa: E402
from GNN_Neural_Network.gnn_recommender.model import (  # noqa: E402
    LightGCN,
    build_normalized_adjacency,
    choose_device,
)
from GNN_Neural_Network.gnn_recommender.recommend import batch_recommend_hobbies  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained LightGCN checkpoint. Training is not triggered here.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = _safe_torch_load(config.paths.checkpoint)
    train_edges = _read_indexed_edges(config.paths.train_edges)
    target_edges = _read_indexed_edges(config.paths.validation_edges if args.split == "validation" else config.paths.test_edges)
    train_known = _known_from_edges(train_edges)
    truth = _known_from_edges(target_edges)
    device = choose_device(config.train.device)
    model = LightGCN(
        num_persons=int(checkpoint["num_persons"]),
        num_hobbies=int(checkpoint["num_hobbies"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        num_layers=int(checkpoint["num_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    adjacency = build_normalized_adjacency(model.num_persons, model.num_hobbies, train_edges, device)
    max_k = max(config.eval.top_k)
    recommendations = batch_recommend_hobbies(
        model=model,
        adjacency=adjacency,
        person_ids=list(truth),
        known_by_person=train_known,
        top_k=max_k,
        chunk_size=config.eval.score_chunk_size,
        device=device,
        show_progress=True,
    )
    metrics = summarize_ranking_metrics(truth, recommendations, config.eval.top_k)
    baseline_metrics = baseline_ranking_metrics(
        train_edges=train_edges,
        target_edges=target_edges,
        known_by_person=train_known,
        top_k_values=config.eval.top_k,
    )
    for key, value in sorted(metrics.items()):
        print(f"lightgcn_{key}: {value:.6f}")
    for baseline_name, values in sorted(baseline_metrics.items()):
        for key, value in sorted(values.items()):
            print(f"{baseline_name}_{key}: {value:.6f}")


def _safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint {path} must contain a dictionary")
    return value


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
