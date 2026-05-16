from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.persona_similarity.scripts.common import file_sha256, load_config, mark_cache_hit, should_use_cache, stable_json_hash
from experiments.persona_similarity.scripts.evaluation_utils import (
    add_diversity_rerank_score,
    evaluate_score_column,
    load_test_features,
    topk_overlap_at_k,
    write_manual_review,
    write_metrics,
)
from experiments.persona_similarity.scripts.experiment_specs import metrics_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/persona_similarity/configs/lightgbm_reranker.yaml")
    parser.add_argument("--base-score", default="fastrp_score")
    parser.add_argument("--experiment-name", default="diversity_rerank_fastrp")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    diversity_config = config["evaluation"].get("diversity_rerank", {})
    lambda_values = [float(value) for value in diversity_config.get("lambda", [0.05, 0.1, 0.2])]
    penalty_columns = [str(value) for value in diversity_config.get("penalty_columns", ["target_occupation", "target_province", "target_community_id"])]
    features_path = config["paths"]["features"]
    cache_metadata = {
        "stage": "evaluate_diversity_rerank",
        "experiment_name": args.experiment_name,
        "base_score": args.base_score,
        "features_path": features_path,
        "features_hash": file_sha256(features_path),
        "config_hash": stable_json_hash({"evaluation": config["evaluation"], "lambda": lambda_values, "penalty_columns": penalty_columns}),
    }
    use_cache, cache_reason = should_use_cache(metrics_path(args.experiment_name), metrics_path(args.experiment_name), cache_metadata, args.force)
    if use_cache:
        mark_cache_hit(metrics_path(args.experiment_name), cache_metadata, metrics_path(args.experiment_name))
        return

    start_time = time.perf_counter()
    test = load_test_features(config)
    top_k_values = [int(value) for value in config["evaluation"]["top_k"]]
    results = {}
    review_scores = []
    review_frame = test.copy()
    for lambda_value in lambda_values:
        score_column = f"diversity_lambda_{lambda_value:g}"
        reranked = add_diversity_rerank_score(
            test,
            base_score=args.base_score,
            output_score=score_column,
            diversity_lambda=lambda_value,
            penalty_columns=penalty_columns,
        )
        results[score_column] = {
            "metrics": evaluate_score_column(reranked, score_column, top_k_values),
            "overlap_vs_base": {
                f"overlap@{k}": topk_overlap_at_k(reranked, args.base_score, score_column, k, progress=True)
                for k in top_k_values
            },
        }
        review_frame = review_frame.merge(
            reranked[["source_uuid", "target_uuid", score_column]],
            on=["source_uuid", "target_uuid"],
            how="left",
        )
        review_scores.append(score_column)

    write_manual_review(review_frame, args.experiment_name, review_scores, int(config["evaluation"].get("manual_review_size", 200)))
    write_metrics(
        args.experiment_name,
        {
            "experiment_name": args.experiment_name,
            **cache_metadata,
            "cache_hit": False,
            "cache_reason": cache_reason,
            "diversity_rerank": results,
            "test_rows": int(len(test)),
            "test_sources": int(test["source_uuid"].nunique()),
            "evaluation_seconds": time.perf_counter() - start_time,
        },
    )


if __name__ == "__main__":
    main()
