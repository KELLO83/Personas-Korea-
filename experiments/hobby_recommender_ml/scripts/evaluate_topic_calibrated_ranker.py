from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import lightgbm as lgb
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hobby_recommender_ml.hobby_recommender.config import load_config  # noqa: E402
from experiments.hobby_recommender_ml.hobby_recommender.data import load_json, save_json  # noqa: E402
from experiments.hobby_recommender_ml.hobby_recommender.diversity import _get_category  # noqa: E402
from experiments.hobby_recommender_ml.hobby_recommender.metrics import oracle_recall_at_k, summarize_ranking_metrics  # noqa: E402
from experiments.hobby_recommender_ml.hobby_recommender.phase6 import topic_calibrated_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate post-ranker hobby topic calibration from a cached LightGBM feature matrix.",
    )
    parser.add_argument("--config", type=Path, default=Path("experiments/hobby_recommender_ml/configs/kure_text_optin_ranker.yaml"))
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", type=str, default="")
    parser.add_argument("--calibration-lambda", type=float, default=0.1)
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
    train_known = _known_from_edges(_read_indexed_edges(config.paths.train_edges))

    data = np.load(args.feature_cache)
    X = data["X"].astype(np.float32)
    person_ids = data["person_ids"].astype(np.int64)
    offsets = data["offsets"].astype(np.int64)
    hobby_ids = data["hobby_ids"].astype(np.int64)
    feature_metadata = _load_optional_json(args.feature_cache.with_suffix(".json"))

    id_to_hobby = _load_id_to_hobby(config.paths.hobby_mapping)
    taxonomy = _load_optional_json(config.paths.hobby_taxonomy)
    hobby_categories = _build_hobby_category_map(id_to_hobby, taxonomy)
    global_distribution = _global_topic_distribution(train_known, hobby_categories)

    model = lgb.Booster(model_file=str(args.model_path))
    all_scores = model.predict(X, num_iteration=model.best_iteration if model.best_iteration else None)

    show_progress = args.progress_mode == "on"
    baseline_recommendations: dict[int, list[int]] = {}
    calibrated_recommendations: dict[int, list[int]] = {}
    candidate_pool: dict[int, list[int]] = {}

    print(f"Batch scoring cached matrix rows={X.shape[0]} features={X.shape[1]}", flush=True)
    for index, raw_person_id in enumerate(
        tqdm(
            person_ids,
            desc=f"Topic calibrate ranker ({args.split})",
            unit="person",
            dynamic_ncols=True,
            disable=not show_progress,
        ),
    ):
        person_id = int(raw_person_id)
        start = int(offsets[index])
        end = int(offsets[index + 1])
        candidates = [int(value) for value in hobby_ids[start:end]]
        candidate_pool[person_id] = candidates
        if person_id not in truth:
            continue

        scores = [float(value) for value in all_scores[start:end]]
        baseline_recommendations[person_id] = _rank_by_score(candidates, scores)
        target_distribution = _person_topic_distribution(
            train_known.get(person_id, set()),
            hobby_categories,
            fallback=global_distribution,
        )
        sorted_candidates = [
            (
                str(hobby_id),
                score,
                hobby_categories.get(hobby_id, "unknown"),
            )
            for hobby_id, score in sorted(
                zip(candidates, scores, strict=False),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ]
        calibrated = topic_calibrated_scores(
            sorted_candidates,
            target_distribution,
            calibration_lambda=args.calibration_lambda,
        )
        calibrated_recommendations[person_id] = [
            int(hobby_id)
            for hobby_id, _score in sorted(calibrated, key=lambda item: float(item[1]), reverse=True)
        ]

    evaluated_truth = {
        person_id: truth[person_id]
        for person_id in candidate_pool
        if person_id in truth
    }
    baseline_metrics = summarize_ranking_metrics(
        evaluated_truth,
        baseline_recommendations,
        tuple(config.eval.top_k),
        hobby_categories=hobby_categories,
        candidate_pool_by_person=candidate_pool,
    )
    calibrated_metrics = summarize_ranking_metrics(
        evaluated_truth,
        calibrated_recommendations,
        tuple(config.eval.top_k),
        hobby_categories=hobby_categories,
        candidate_pool_by_person=candidate_pool,
    )
    calibrated_metrics["candidate_recall@50"] = oracle_recall_at_k(evaluated_truth, candidate_pool, 50)

    payload = {
        "split": args.split,
        "experiment_id": args.experiment_id,
        "status": f"{args.split}_topic_calibrated",
        "model_path": str(args.model_path),
        "feature_cache": str(args.feature_cache),
        "feature_cache_metadata": feature_metadata,
        "topic_policy": {
            "source": str(config.paths.hobby_taxonomy),
            "target_distribution": "person_train_history_category_distribution",
            "fallback_distribution": global_distribution,
            "calibration_lambda": args.calibration_lambda,
            "unknown_category_policy": "unknown",
        },
        "resource_policy": {
            "cpu_threads": cpu_threads,
            "progress_mode": args.progress_mode,
        },
        "metrics": {
            "baseline": baseline_metrics,
            "topic_calibrated": calibrated_metrics,
            "delta": _metric_delta(calibrated_metrics, baseline_metrics),
        },
        "person_count": len([person_id for person_id, relevant in evaluated_truth.items() if relevant]),
        "candidate_person_count": len(candidate_pool),
    }
    save_json(args.output, payload)
    print(
        f"{args.split} baseline recall@10={float(baseline_metrics.get('recall@10', 0.0)):.6f} "
        f"calibrated recall@10={float(calibrated_metrics.get('recall@10', 0.0)):.6f} "
        f"baseline ild@10={float(baseline_metrics.get('intra_list_diversity@10', 0.0)):.6f} "
        f"calibrated ild@10={float(calibrated_metrics.get('intra_list_diversity@10', 0.0)):.6f}"
    )


def _rank_by_score(candidates: list[int], scores: list[float]) -> list[int]:
    return [
        hobby_id
        for hobby_id, _score in sorted(
            zip(candidates, scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
    ]


def _read_indexed_edges(path: Path) -> list[tuple[int, int]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [(int(row["person_id"]), int(row["hobby_id"])) for row in reader]


def _known_from_edges(edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for person_id, hobby_id in edges:
        result[person_id].add(hobby_id)
    return dict(result)


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = load_json(path)
    return value if isinstance(value, dict) else {}


def _load_id_to_hobby(path: Path) -> dict[int, str]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"hobby mapping must be a JSON object: {path}")
    return {int(index): str(hobby_name) for hobby_name, index in raw.items()}


def _build_hobby_category_map(id_to_hobby: Mapping[int, str], taxonomy: dict[str, Any]) -> dict[int, str]:
    return {
        int(hobby_id): (_get_category(hobby_name, taxonomy) or "unknown")
        for hobby_id, hobby_name in id_to_hobby.items()
    }


def _person_topic_distribution(
    known_hobbies: set[int],
    hobby_categories: Mapping[int, str],
    *,
    fallback: Mapping[str, float],
) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for hobby_id in known_hobbies:
        counts[hobby_categories.get(hobby_id, "unknown")] += 1
    if not counts:
        return dict(fallback)
    total = sum(counts.values())
    return {topic: count / total for topic, count in counts.items()}


def _global_topic_distribution(
    known_by_person: Mapping[int, set[int]],
    hobby_categories: Mapping[int, str],
) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for hobby_ids in known_by_person.values():
        for hobby_id in hobby_ids:
            counts[hobby_categories.get(hobby_id, "unknown")] += 1
    total = sum(counts.values())
    if total <= 0:
        return {"unknown": 1.0}
    return {topic: count / total for topic, count in sorted(counts.items())}


def _metric_delta(current: Mapping[str, object], baseline: Mapping[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in current.items():
        if isinstance(value, (int, float)) and isinstance(baseline.get(key), (int, float)):
            result[key] = float(value) - float(baseline[key])
    return result


def _resolve_cpu_threads(requested: int) -> int:
    logical = os.cpu_count() or 1
    if requested > 0:
        return max(1, min(int(requested), logical))
    return min(max(logical - 4, 1), 18)


if __name__ == "__main__":
    main()
