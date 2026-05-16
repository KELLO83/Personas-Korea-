from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from experiments.persona_similarity.scripts.common import ensure_parent, file_sha256, load_config, mark_cache_hit, should_use_cache, stable_json_hash
from experiments.persona_similarity.scripts.feature_builder import FEATURE_COLUMNS


def dcg(labels: list[float], k: int) -> float:
    return sum(((2**label - 1) / math.log2(index + 2)) for index, label in enumerate(labels[:k]))


def ndcg_at_k(frame: pd.DataFrame, score_column: str, k: int) -> float:
    values: list[float] = []
    for _, group in frame.groupby("source_uuid", sort=False):
        ranked = group.sort_values(score_column, ascending=False)["label"].astype(float).tolist()
        ideal = sorted(group["label"].astype(float).tolist(), reverse=True)
        ideal_dcg = dcg(ideal, k)
        if ideal_dcg > 0:
            values.append(dcg(ranked, k) / ideal_dcg)
    return float(sum(values) / len(values)) if values else 0.0


def explanation_coverage_at_k(frame: pd.DataFrame, score_column: str, k: int) -> float:
    covered = 0
    total = 0
    for _, group in frame.groupby("source_uuid", sort=False):
        top = group.sort_values(score_column, ascending=False).head(k)
        covered += int((top["explanation_feature_count"] > 0).sum())
        total += len(top)
    return covered / total if total else 0.0


def strong_reason_coverage_at_k(frame: pd.DataFrame, score_column: str, k: int) -> float:
    strong_columns = ["same_occupation", "same_district", "same_education", "same_field", "shared_hobby_count", "shared_skill_count"]
    covered = 0
    total = 0
    for _, group in frame.groupby("source_uuid", sort=False):
        top = group.sort_values(score_column, ascending=False).head(k)
        covered += int((top[strong_columns].sum(axis=1) > 0).sum())
        total += len(top)
    return covered / total if total else 0.0


def low_information_dominance_at_k(frame: pd.DataFrame, score_column: str, k: int) -> float:
    low_columns = ["same_sex", "same_marital", "same_province", "same_community"]
    strong_columns = ["same_occupation", "same_district", "same_education", "same_field", "shared_hobby_count", "shared_skill_count"]
    dominated = 0
    total = 0
    for _, group in frame.groupby("source_uuid", sort=False):
        top = group.sort_values(score_column, ascending=False).head(k)
        low_signal = top[low_columns].sum(axis=1) > 0
        strong_signal = top[strong_columns].sum(axis=1) > 0
        dominated += int((low_signal & ~strong_signal).sum())
        total += len(top)
    return dominated / total if total else 0.0


def average_reason_count_at_k(frame: pd.DataFrame, score_column: str, k: int) -> float:
    values: list[float] = []
    for _, group in frame.groupby("source_uuid", sort=False):
        top = group.sort_values(score_column, ascending=False).head(k)
        if not top.empty:
            values.append(float(top["explanation_feature_count"].mean()))
    return float(sum(values) / len(values)) if values else 0.0


def unique_target_rate_at_k(frame: pd.DataFrame, score_column: str, k: int) -> float:
    total = 0
    targets: set[str] = set()
    for _, group in frame.groupby("source_uuid", sort=False):
        top = group.sort_values(score_column, ascending=False).head(k)
        total += len(top)
        targets.update(top["target_uuid"].astype(str).tolist())
    return len(targets) / total if total else 0.0


def evaluate(frame: pd.DataFrame, score_column: str, top_k_values: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in top_k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(frame, score_column, k)
        metrics[f"explanation_coverage@{k}"] = explanation_coverage_at_k(frame, score_column, k)
        metrics[f"strong_reason_coverage@{k}"] = strong_reason_coverage_at_k(frame, score_column, k)
        metrics[f"low_information_dominance@{k}"] = low_information_dominance_at_k(frame, score_column, k)
        metrics[f"average_reason_count@{k}"] = average_reason_count_at_k(frame, score_column, k)
        metrics[f"unique_target_rate@{k}"] = unique_target_rate_at_k(frame, score_column, k)
    return metrics


def min_max_normalize(series: pd.Series) -> pd.Series:
    min_value = float(series.min())
    max_value = float(series.max())
    if max_value <= min_value:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_value) / (max_value - min_value)


def add_hybrid_scores(frame: pd.DataFrame, alphas: list[float]) -> list[str]:
    score_columns: list[str] = []
    for _, group_index in frame.groupby("source_uuid", sort=False).groups.items():
        frame.loc[group_index, "_norm_fastrp_score"] = min_max_normalize(frame.loc[group_index, "fastrp_score"])
        frame.loc[group_index, "_norm_reranker_score"] = min_max_normalize(frame.loc[group_index, "reranker_score"])
    for alpha in alphas:
        column = f"hybrid_alpha_{alpha:g}"
        frame[column] = alpha * frame["_norm_reranker_score"] + (1.0 - alpha) * frame["_norm_fastrp_score"]
        score_columns.append(column)
    return score_columns


def topk_overlap_at_k(frame: pd.DataFrame, left_score: str, right_score: str, k: int) -> float:
    overlaps: list[float] = []
    for _, group in frame.groupby("source_uuid", sort=False):
        left = set(group.sort_values(left_score, ascending=False).head(k)["target_uuid"].astype(str))
        right = set(group.sort_values(right_score, ascending=False).head(k)["target_uuid"].astype(str))
        if left or right:
            overlaps.append(len(left & right) / max(1, k))
    return float(sum(overlaps) / len(overlaps)) if overlaps else 0.0


def write_manual_review_sample(frame: pd.DataFrame, config: dict[str, Any], score_columns: list[str]) -> None:
    review_size = int(config["evaluation"].get("manual_review_size", 200))
    review_rows: list[pd.DataFrame] = []
    for model_name in score_columns:
        ranked = frame.sort_values(["source_uuid", model_name], ascending=[True, False]).groupby("source_uuid", sort=False).head(5).copy()
        ranked["model"] = model_name
        review_rows.append(ranked)
    if not review_rows:
        return
    review = pd.concat(review_rows, ignore_index=True).head(review_size)
    columns = [
        "model",
        "source_uuid",
        "target_uuid",
        "label",
        "fastrp_score",
        "deterministic_score",
        "reranker_score",
        "explanation_feature_count",
        "same_occupation",
        "same_district",
        "same_province",
        "same_education",
        "same_field",
        "same_age_group",
        "same_community",
        "shared_hobby_count",
        "shared_skill_count",
    ]
    output_path = ensure_parent(config["paths"]["manual_review"])
    review[[column for column in columns if column in review.columns]].to_csv(output_path, index=False, encoding="utf-8-sig")


def write_decision_artifacts(config: dict[str, Any], metrics: dict[str, Any]) -> None:
    baseline = metrics["baseline_fastrp"]
    deterministic = metrics["deterministic_score"]
    reranker = metrics.get("lightgbm_reranker")
    decision = {
        "status": "experimental",
        "production_default": "FastRP/KNN SIMILAR_TO",
        "selected_model": None,
        "reason": "Offline experiment artifact only. Production promotion requires manual review and integration decision.",
        "metrics_path": config["paths"]["metrics"],
    }
    if reranker:
        k = int(config["evaluation"]["top_k"][0])
        baseline_ndcg = baseline.get(f"ndcg@{k}", 0.0)
        reranker_ndcg = reranker.get(f"ndcg@{k}", 0.0)
        deterministic_ndcg = deterministic.get(f"ndcg@{k}", 0.0)
        if reranker_ndcg >= baseline_ndcg * 0.99 and reranker_ndcg >= deterministic_ndcg:
            decision["selected_model"] = "lightgbm_reranker_candidate"
            decision["reason"] = "Reranker passed weak-metric precheck; manual review is still required."
    output_path = ensure_parent(config["paths"]["decision"])
    output_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = ensure_parent(config["paths"]["run_summary"])
    summary_path.write_text(
        "\n".join(
            [
                "# Persona Similarity Experiment Run Summary",
                "",
                f"Status: {decision['status']}",
                f"Production default: {decision['production_default']}",
                f"Selected model: {decision['selected_model'] or 'none'}",
                f"Reason: {decision['reason']}",
                "",
                "This artifact is generated by `evaluate_reranker.py` and does not promote a model by itself.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/persona_similarity/configs/lightgbm_reranker.yaml")
    parser.add_argument("--allow-missing-model", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    cache_metadata = {
        "stage": "evaluate_reranker_legacy",
        "features_path": config["paths"]["features"],
        "features_hash": file_sha256(config["paths"]["features"]),
        "model_path": config["paths"]["model"],
        "model_hash": file_sha256(config["paths"]["model"]),
        "config_hash": stable_json_hash({"evaluation": config["evaluation"], "feature_columns": FEATURE_COLUMNS}),
    }
    use_cache, cache_reason = should_use_cache(config["paths"]["metrics"], config["paths"]["metrics"], cache_metadata, args.force)
    if use_cache:
        mark_cache_hit(config["paths"]["metrics"], cache_metadata, config["paths"]["metrics"])
        return

    start_time = time.perf_counter()
    features = pd.read_parquet(PROJECT_ROOT / config["paths"]["features"])
    test = features[features["split"] == "test"].copy()
    model_path = PROJECT_ROOT / config["paths"]["model"]
    inference_seconds = None
    if model_path.exists():
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise SystemExit("lightgbm is required to evaluate the persona similarity reranker.") from exc

        model = lgb.Booster(model_file=str(model_path))
        start_time = time.perf_counter()
        test["reranker_score"] = model.predict(test[FEATURE_COLUMNS])
        inference_seconds = time.perf_counter() - start_time
    elif args.allow_missing_model:
        test["reranker_score"] = test["deterministic_score"]
    else:
        raise SystemExit(f"Missing model artifact: {model_path}. Train first or pass --allow-missing-model.")

    top_k_values = [int(value) for value in config["evaluation"]["top_k"]]
    hybrid_columns = add_hybrid_scores(test, [float(value) for value in config["evaluation"].get("hybrid_alpha", [])])
    metrics: dict[str, Any] = {
        "baseline_fastrp": evaluate(test, "fastrp_score", top_k_values),
        **cache_metadata,
        "cache_hit": False,
        "cache_reason": cache_reason,
        "deterministic_score": evaluate(test, "deterministic_score", top_k_values),
        "lightgbm_reranker": evaluate(test, "reranker_score", top_k_values),
        "hybrid": {column: evaluate(test, column, top_k_values) for column in hybrid_columns},
        "topk_overlap_vs_fastrp": {
            score_column: {f"overlap@{k}": topk_overlap_at_k(test, "fastrp_score", score_column, k) for k in top_k_values}
            for score_column in ["deterministic_score", "reranker_score", *hybrid_columns]
        },
        "test_rows": int(len(test)),
        "test_sources": int(test["source_uuid"].nunique()),
        "inference_seconds": inference_seconds,
        "evaluation_seconds": time.perf_counter() - start_time,
    }
    write_manual_review_sample(test, config, ["fastrp_score", "deterministic_score", "reranker_score", *hybrid_columns])
    output_path = ensure_parent(config["paths"]["metrics"])
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_decision_artifacts(config, metrics)


if __name__ == "__main__":
    main()
