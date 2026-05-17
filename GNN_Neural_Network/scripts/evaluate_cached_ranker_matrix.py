from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.config import load_config  # noqa: E402
from GNN_Neural_Network.gnn_recommender.data import save_json  # noqa: E402
from GNN_Neural_Network.gnn_recommender.metrics import oracle_recall_at_k, summarize_ranking_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LightGBM ranker directly from a cached feature matrix.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/kure_text_optin_ranker.yaml"))
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", type=str, default="")
    parser.add_argument("--cpu-thread-count", type=int, default=0)
    parser.add_argument("--progress-mode", choices=["on", "off"], default="on")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpu_threads = _resolve_cpu_threads(args.cpu_thread_count)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(cpu_threads)

    config = load_config(args.config)
    edges_path = config.paths.validation_edges if args.split == "validation" else config.paths.test_edges
    truth = _known_from_edges(_read_indexed_edges(edges_path))
    data = np.load(args.feature_cache)
    X = data["X"].astype(np.float32)
    person_ids = data["person_ids"].astype(np.int64)
    offsets = data["offsets"].astype(np.int64)
    hobby_ids = data["hobby_ids"].astype(np.int64)
    metadata_path = args.feature_cache.with_suffix(".json")
    feature_metadata: dict[str, Any] = {}
    if metadata_path.exists():
        feature_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    model = lgb.Booster(model_file=str(args.model_path))

    show_progress = args.progress_mode == "on"
    print(f"Batch scoring cached matrix rows={X.shape[0]} features={X.shape[1]}", flush=True)
    all_scores = model.predict(X, num_iteration=model.best_iteration if model.best_iteration else None)
    recommendations: dict[int, list[int]] = {}
    candidate_pool: dict[int, list[int]] = {}
    for index, raw_person_id in enumerate(
        tqdm(person_ids, desc=f"Score cached ranker ({args.split})", unit="person", dynamic_ncols=True, disable=not show_progress),
    ):
        person_id = int(raw_person_id)
        start = int(offsets[index])
        end = int(offsets[index + 1])
        candidates = [int(value) for value in hobby_ids[start:end]]
        candidate_pool[person_id] = candidates
        if person_id not in truth:
            continue
        scores = all_scores[start:end]
        recommendations[person_id] = [
            hobby_id
            for hobby_id, _score in sorted(
                zip(candidates, scores, strict=False),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ]

    metrics = summarize_ranking_metrics(
        truth,
        recommendations,
        tuple(config.eval.top_k),
        candidate_pool_by_person=candidate_pool,
    )
    metrics["candidate_recall@50"] = oracle_recall_at_k(truth, candidate_pool, 50)
    payload = {
        "split": args.split,
        "experiment_id": args.experiment_id,
        "status": f"{args.split}_evaluated",
        "model_path": str(args.model_path),
        "feature_cache": str(args.feature_cache),
        "feature_cache_metadata": feature_metadata,
        "resource_policy": {
            "cpu_threads": cpu_threads,
            "progress_mode": args.progress_mode,
        },
        "metrics": metrics,
        "person_count": len([person_id for person_id, relevant in truth.items() if relevant]),
        "candidate_person_count": len(candidate_pool),
    }
    save_json(args.output, payload)
    print(
        f"{args.split} recall@10={float(metrics.get('recall@10', 0.0)):.6f} "
        f"ndcg@10={float(metrics.get('ndcg@10', 0.0)):.6f} "
        f"candidate_recall@50={float(metrics.get('candidate_recall@50', 0.0)):.6f}"
    )


def _read_indexed_edges(path: Path) -> list[tuple[int, int]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [(int(row["person_id"]), int(row["hobby_id"])) for row in reader]


def _known_from_edges(edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for person_id, hobby_id in edges:
        result[person_id].add(hobby_id)
    return dict(result)


def _resolve_cpu_threads(requested: int) -> int:
    logical = os.cpu_count() or 1
    if requested > 0:
        return max(1, min(int(requested), logical))
    return min(max(logical - 4, 1), 18)


if __name__ == "__main__":
    main()
