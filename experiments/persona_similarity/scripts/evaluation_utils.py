from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.persona_similarity.scripts.common import PROJECT_ROOT, ensure_parent
from experiments.persona_similarity.scripts.common import file_sha256, mark_cache_hit, should_use_cache, stable_json_hash
from experiments.persona_similarity.scripts.experiment_specs import manual_review_path, metrics_path, model_path, train_metadata_path
from experiments.persona_similarity.scripts.training_utils import load_feature_columns_from_metadata


def dcg(labels: list[float], k: int) -> float:
    return sum(((2**label - 1) / math.log2(index + 2)) for index, label in enumerate(labels[:k]))


def iter_groups(frame: pd.DataFrame, description: str, progress: bool = True) -> Any:
    groups = list(frame.groupby("source_uuid", sort=False))
    if not progress:
        return groups
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return groups
    return tqdm(groups, desc=description, unit="source")


def ndcg_at_k(frame: pd.DataFrame, score_column: str, k: int, progress: bool = False) -> float:
    values: list[float] = []
    for _, group in iter_groups(frame, f"ndcg@{k}:{score_column}", progress):
        ranked = group.sort_values(score_column, ascending=False)["label"].astype(float).tolist()
        ideal = sorted(group["label"].astype(float).tolist(), reverse=True)
        ideal_dcg = dcg(ideal, k)
        if ideal_dcg > 0:
            values.append(dcg(ranked, k) / ideal_dcg)
    return float(sum(values) / len(values)) if values else 0.0


def explanation_coverage_at_k(frame: pd.DataFrame, score_column: str, k: int, progress: bool = False) -> float:
    covered = 0
    total = 0
    for _, group in iter_groups(frame, f"explanation@{k}:{score_column}", progress):
        top = group.sort_values(score_column, ascending=False).head(k)
        covered += int((top["explanation_feature_count"] > 0).sum())
        total += len(top)
    return covered / total if total else 0.0


def strong_reason_coverage_at_k(frame: pd.DataFrame, score_column: str, k: int, progress: bool = False) -> float:
    strong_columns = ["same_occupation", "same_district", "same_education", "same_field", "shared_hobby_count", "shared_skill_count"]
    covered = 0
    total = 0
    for _, group in iter_groups(frame, f"strong_reason@{k}:{score_column}", progress):
        top = group.sort_values(score_column, ascending=False).head(k)
        covered += int((top[strong_columns].sum(axis=1) > 0).sum())
        total += len(top)
    return covered / total if total else 0.0


def low_information_dominance_at_k(frame: pd.DataFrame, score_column: str, k: int, progress: bool = False) -> float:
    low_columns = ["same_sex", "same_marital", "same_province", "same_community"]
    strong_columns = ["same_occupation", "same_district", "same_education", "same_field", "shared_hobby_count", "shared_skill_count"]
    dominated = 0
    total = 0
    for _, group in iter_groups(frame, f"low_info@{k}:{score_column}", progress):
        top = group.sort_values(score_column, ascending=False).head(k)
        low_signal = top[low_columns].sum(axis=1) > 0
        strong_signal = top[strong_columns].sum(axis=1) > 0
        dominated += int((low_signal & ~strong_signal).sum())
        total += len(top)
    return dominated / total if total else 0.0


def average_reason_count_at_k(frame: pd.DataFrame, score_column: str, k: int, progress: bool = False) -> float:
    values: list[float] = []
    for _, group in iter_groups(frame, f"reason_count@{k}:{score_column}", progress):
        top = group.sort_values(score_column, ascending=False).head(k)
        if not top.empty:
            values.append(float(top["explanation_feature_count"].mean()))
    return float(sum(values) / len(values)) if values else 0.0


def unique_target_rate_at_k(frame: pd.DataFrame, score_column: str, k: int, progress: bool = False) -> float:
    total = 0
    targets: set[str] = set()
    for _, group in iter_groups(frame, f"unique_target@{k}:{score_column}", progress):
        top = group.sort_values(score_column, ascending=False).head(k)
        total += len(top)
        targets.update(top["target_uuid"].astype(str).tolist())
    return len(targets) / total if total else 0.0


def attribute_diversity_at_k(frame: pd.DataFrame, score_column: str, attribute_column: str, k: int, progress: bool = False) -> float:
    if attribute_column not in frame.columns:
        return 0.0
    values: list[float] = []
    for _, group in iter_groups(frame, f"{attribute_column}_diversity@{k}:{score_column}", progress):
        top = group.sort_values(score_column, ascending=False).head(k)
        if not top.empty:
            values.append(float(top[attribute_column].fillna("").astype(str).nunique() / len(top)))
    return float(sum(values) / len(values)) if values else 0.0


def demographic_only_rate_at_k(frame: pd.DataFrame, score_column: str, k: int, progress: bool = False) -> float:
    low_columns = ["same_sex", "same_marital", "same_province", "same_community"]
    strong_columns = ["same_occupation", "same_district", "same_education", "same_field", "shared_hobby_count", "shared_skill_count"]
    text_columns = [column for column in frame.columns if column.endswith("_text_cosine")]
    count = 0
    total = 0
    for _, group in iter_groups(frame, f"demographic_only@{k}:{score_column}", progress):
        top = group.sort_values(score_column, ascending=False).head(k)
        low_signal = top[low_columns].sum(axis=1) > 0
        strong_signal = top[strong_columns].sum(axis=1) > 0
        if text_columns:
            text_signal = top[text_columns].max(axis=1) >= 0.5
        else:
            text_signal = pd.Series([False] * len(top), index=top.index)
        count += int((low_signal & ~strong_signal & ~text_signal).sum())
        total += len(top)
    return count / total if total else 0.0


def topk_overlap_at_k(frame: pd.DataFrame, left_score: str, right_score: str, k: int, progress: bool = False) -> float:
    overlaps: list[float] = []
    for _, group in iter_groups(frame, f"overlap@{k}:{left_score}->{right_score}", progress):
        left = set(group.sort_values(left_score, ascending=False).head(k)["target_uuid"].astype(str))
        right = set(group.sort_values(right_score, ascending=False).head(k)["target_uuid"].astype(str))
        if left or right:
            overlaps.append(len(left & right) / max(1, k))
    return float(sum(overlaps) / len(overlaps)) if overlaps else 0.0


def evaluate_score_column(frame: pd.DataFrame, score_column: str, top_k_values: list[int], progress: bool = True) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in top_k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(frame, score_column, k, progress=progress)
        metrics[f"explanation_coverage@{k}"] = explanation_coverage_at_k(frame, score_column, k, progress=progress)
        metrics[f"strong_reason_coverage@{k}"] = strong_reason_coverage_at_k(frame, score_column, k, progress=progress)
        metrics[f"low_information_dominance@{k}"] = low_information_dominance_at_k(frame, score_column, k, progress=progress)
        metrics[f"average_reason_count@{k}"] = average_reason_count_at_k(frame, score_column, k, progress=progress)
        metrics[f"unique_target_rate@{k}"] = unique_target_rate_at_k(frame, score_column, k, progress=progress)
        metrics[f"occupation_diversity@{k}"] = attribute_diversity_at_k(frame, score_column, "target_occupation", k, progress=progress)
        metrics[f"province_diversity@{k}"] = attribute_diversity_at_k(frame, score_column, "target_province", k, progress=progress)
        metrics[f"community_diversity@{k}"] = attribute_diversity_at_k(frame, score_column, "target_community_id", k, progress=progress)
        metrics[f"demographic_only_rate@{k}"] = demographic_only_rate_at_k(frame, score_column, k, progress=progress)
    return metrics


def load_test_features(config: dict[str, Any], features_path: str | None = None) -> pd.DataFrame:
    input_features_path = Path(features_path or config["paths"]["features"])
    if not input_features_path.is_absolute():
        input_features_path = PROJECT_ROOT / input_features_path
    features = pd.read_parquet(input_features_path)
    return features[features["split"] == "test"].copy()


def write_metrics(experiment_name: str, payload: dict[str, Any]) -> None:
    output_path = ensure_parent(metrics_path(experiment_name))
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_manual_review(frame: pd.DataFrame, experiment_name: str, score_columns: list[str], review_size: int) -> None:
    review_rows: list[pd.DataFrame] = []
    for score_column in score_columns:
        ranked = frame.sort_values(["source_uuid", score_column], ascending=[True, False]).groupby("source_uuid", sort=False).head(5).copy()
        ranked["model"] = score_column
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
        "model_score",
        "source_occupation",
        "target_occupation",
        "source_province",
        "target_province",
        "source_district",
        "target_district",
        "source_community_id",
        "target_community_id",
        "source_age_group",
        "target_age_group",
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
    output_path = ensure_parent(manual_review_path(experiment_name))
    review[[column for column in columns if column in review.columns]].to_csv(output_path, index=False, encoding="utf-8-sig")


def evaluate_existing_score(
    config: dict[str, Any],
    experiment_name: str,
    score_column: str,
    features_path: str | None = None,
    force: bool = False,
) -> None:
    resolved_features_path = Path(features_path or config["paths"]["features"])
    if not resolved_features_path.is_absolute():
        resolved_features_path = PROJECT_ROOT / resolved_features_path
    cache_metadata = {
        "stage": "evaluate_existing_score",
        "experiment_name": experiment_name,
        "score_column": score_column,
        "features_path": str(resolved_features_path.relative_to(PROJECT_ROOT)),
        "features_hash": file_sha256(resolved_features_path),
        "config_hash": stable_json_hash({"evaluation": config["evaluation"]}),
    }
    use_cache, cache_reason = should_use_cache(metrics_path(experiment_name), metrics_path(experiment_name), cache_metadata, force)
    if use_cache:
        mark_cache_hit(metrics_path(experiment_name), cache_metadata, metrics_path(experiment_name))
        return

    start_time = time.perf_counter()
    test = load_test_features(config, str(resolved_features_path))
    top_k_values = [int(value) for value in config["evaluation"]["top_k"]]
    metrics = {
        "experiment_name": experiment_name,
        **cache_metadata,
        "cache_hit": False,
        "cache_reason": cache_reason,
        "score_column": score_column,
        "metrics": evaluate_score_column(test, score_column, top_k_values),
        "test_rows": int(len(test)),
        "test_sources": int(test["source_uuid"].nunique()),
        "evaluation_seconds": time.perf_counter() - start_time,
    }
    write_manual_review(test, experiment_name, [score_column], int(config["evaluation"].get("manual_review_size", 200)))
    write_metrics(experiment_name, metrics)


def evaluate_model(config: dict[str, Any], experiment_name: str, features_path: str | None = None, force: bool = False) -> None:
    feature_columns = load_feature_columns_from_metadata(train_metadata_path(experiment_name))
    resolved_features_path = Path(features_path or config["paths"]["features"])
    if not resolved_features_path.is_absolute():
        resolved_features_path = PROJECT_ROOT / resolved_features_path
    cache_metadata = {
        "stage": "evaluate_model",
        "experiment_name": experiment_name,
        "model_path": str(model_path(experiment_name).relative_to(PROJECT_ROOT)),
        "model_hash": file_sha256(model_path(experiment_name)),
        "features_path": str(resolved_features_path.relative_to(PROJECT_ROOT)),
        "features_hash": file_sha256(resolved_features_path),
        "config_hash": stable_json_hash({"evaluation": config["evaluation"], "feature_columns": feature_columns}),
    }
    use_cache, cache_reason = should_use_cache(metrics_path(experiment_name), metrics_path(experiment_name), cache_metadata, force)
    if use_cache:
        mark_cache_hit(metrics_path(experiment_name), cache_metadata, metrics_path(experiment_name))
        return

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("lightgbm is required to evaluate persona similarity model experiments.") from exc

    start_time = time.perf_counter()
    test = load_test_features(config, str(resolved_features_path))
    model = lgb.Booster(model_file=str(model_path(experiment_name)))
    predict_start = time.perf_counter()
    test["model_score"] = model.predict(test[feature_columns])
    inference_seconds = time.perf_counter() - predict_start
    top_k_values = [int(value) for value in config["evaluation"]["top_k"]]
    metrics = {
        "experiment_name": experiment_name,
        **cache_metadata,
        "cache_hit": False,
        "cache_reason": cache_reason,
        "model_path": str(model_path(experiment_name).relative_to(PROJECT_ROOT)),
        "feature_columns": feature_columns,
        "metrics": evaluate_score_column(test, "model_score", top_k_values),
        "overlap_vs_fastrp": {f"overlap@{k}": topk_overlap_at_k(test, "fastrp_score", "model_score", k, progress=True) for k in top_k_values},
        "test_rows": int(len(test)),
        "test_sources": int(test["source_uuid"].nunique()),
        "inference_seconds": inference_seconds,
        "evaluation_seconds": time.perf_counter() - start_time,
    }
    write_manual_review(test, experiment_name, ["model_score"], int(config["evaluation"].get("manual_review_size", 200)))
    write_metrics(experiment_name, metrics)


def min_max_normalize(series: pd.Series) -> pd.Series:
    min_value = float(series.min())
    max_value = float(series.max())
    if max_value <= min_value:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_value) / (max_value - min_value)


def add_diversity_rerank_score(
    frame: pd.DataFrame,
    base_score: str,
    output_score: str,
    diversity_lambda: float,
    penalty_columns: list[str] | None = None,
) -> pd.DataFrame:
    penalties = penalty_columns or ["target_occupation", "target_province", "target_community_id"]
    reranked_groups: list[pd.DataFrame] = []
    for _, group in frame.groupby("source_uuid", sort=False):
        remaining = group.copy()
        selected_rows: list[pd.Series] = []
        seen: dict[str, set[str]] = {column: set() for column in penalties if column in remaining.columns}
        while not remaining.empty:
            best_index = None
            best_score = -float("inf")
            for index, row in remaining.iterrows():
                duplicate_penalty = 0.0
                for column, seen_values in seen.items():
                    value = str(row.get(column, ""))
                    if value and value in seen_values:
                        duplicate_penalty += 1.0
                adjusted_score = float(row[base_score]) - (diversity_lambda * duplicate_penalty)
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_index = index
            if best_index is None:
                break
            selected = remaining.loc[best_index].copy()
            selected[output_score] = len(group) - len(selected_rows)
            selected_rows.append(selected)
            for column, seen_values in seen.items():
                value = str(selected.get(column, ""))
                if value:
                    seen_values.add(value)
            remaining = remaining.drop(index=best_index)
        if selected_rows:
            reranked_groups.append(pd.DataFrame(selected_rows))
    if not reranked_groups:
        result = frame.copy()
        result[output_score] = result[base_score]
        return result
    return pd.concat(reranked_groups, ignore_index=True)


def evaluate_hybrid(
    config: dict[str, Any],
    source_experiment_name: str,
    hybrid_experiment_name: str,
    features_path: str | None = None,
    force: bool = False,
) -> None:
    feature_columns = load_feature_columns_from_metadata(train_metadata_path(source_experiment_name))
    resolved_features_path = Path(features_path or config["paths"]["features"])
    if not resolved_features_path.is_absolute():
        resolved_features_path = PROJECT_ROOT / resolved_features_path
    cache_metadata = {
        "stage": "evaluate_hybrid",
        "experiment_name": hybrid_experiment_name,
        "source_experiment_name": source_experiment_name,
        "model_path": str(model_path(source_experiment_name).relative_to(PROJECT_ROOT)),
        "model_hash": file_sha256(model_path(source_experiment_name)),
        "features_path": str(resolved_features_path.relative_to(PROJECT_ROOT)),
        "features_hash": file_sha256(resolved_features_path),
        "config_hash": stable_json_hash({"evaluation": config["evaluation"], "feature_columns": feature_columns}),
    }
    use_cache, cache_reason = should_use_cache(metrics_path(hybrid_experiment_name), metrics_path(hybrid_experiment_name), cache_metadata, force)
    if use_cache:
        mark_cache_hit(metrics_path(hybrid_experiment_name), cache_metadata, metrics_path(hybrid_experiment_name))
        return

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("lightgbm is required to evaluate hybrid persona similarity experiments.") from exc

    start_time = time.perf_counter()
    test = load_test_features(config, str(resolved_features_path))
    model = lgb.Booster(model_file=str(model_path(source_experiment_name)))
    test["model_score"] = model.predict(test[feature_columns])

    for _, group_index in test.groupby("source_uuid", sort=False).groups.items():
        test.loc[group_index, "_norm_fastrp_score"] = min_max_normalize(test.loc[group_index, "fastrp_score"])
        test.loc[group_index, "_norm_model_score"] = min_max_normalize(test.loc[group_index, "model_score"])

    top_k_values = [int(value) for value in config["evaluation"]["top_k"]]
    alpha_values = [float(value) for value in config["evaluation"].get("hybrid_alpha", [])]
    score_columns: list[str] = []
    for alpha in alpha_values:
        column = f"hybrid_alpha_{alpha:g}"
        test[column] = alpha * test["_norm_model_score"] + (1.0 - alpha) * test["_norm_fastrp_score"]
        score_columns.append(column)

    metrics = {
        "experiment_name": hybrid_experiment_name,
        **cache_metadata,
        "cache_hit": False,
        "cache_reason": cache_reason,
        "source_experiment_name": source_experiment_name,
        "hybrid": {column: evaluate_score_column(test, column, top_k_values) for column in score_columns},
        "overlap_vs_fastrp": {
            column: {f"overlap@{k}": topk_overlap_at_k(test, "fastrp_score", column, k, progress=True) for k in top_k_values}
            for column in score_columns
        },
        "test_rows": int(len(test)),
        "test_sources": int(test["source_uuid"].nunique()),
        "evaluation_seconds": time.perf_counter() - start_time,
    }
    write_manual_review(test, hybrid_experiment_name, score_columns, int(config["evaluation"].get("manual_review_size", 200)))
    write_metrics(hybrid_experiment_name, metrics)
