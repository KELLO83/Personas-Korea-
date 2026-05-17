from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl

from experiments.persona_similarity.scripts.common import PROJECT_ROOT, ensure_parent
from experiments.persona_similarity.scripts.common import file_sha256, mark_cache_hit, should_use_cache, stable_json_hash
from experiments.persona_similarity.scripts.common import resolve_worker_count
from experiments.persona_similarity.scripts.experiment_specs import manual_review_path, metrics_path, model_path, train_metadata_path
from experiments.persona_similarity.scripts.training_utils import load_feature_columns_from_metadata


def _to_polars_frame(frame: Any) -> pl.DataFrame:
    return frame if isinstance(frame, pl.DataFrame) else pl.from_pandas(frame)


def dcg(labels: list[float], k: int) -> float:
    return sum(((2**label - 1) / math.log2(index + 2)) for index, label in enumerate(labels[:k]))


def iter_groups(frame: Any, description: str, progress: bool = True) -> Any:
    polars_frame = _to_polars_frame(frame)
    groups = [(str(group["source_uuid"][0]), group) for group in polars_frame.partition_by("source_uuid", maintain_order=True)]
    if not progress:
        return groups
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return groups
    return tqdm(groups, desc=description, unit="source")


def ndcg_at_k(frame: pl.DataFrame, score_column: str, k: int, progress: bool = False) -> float:
    values: list[float] = []
    for _, group in iter_groups(frame, f"ndcg@{k}:{score_column}", progress):
        ranked = group.sort(score_column, descending=True)["label"].cast(pl.Float64).to_list()
        ideal = sorted(group["label"].cast(pl.Float64).to_list(), reverse=True)
        ideal_dcg = dcg(ideal, k)
        if ideal_dcg > 0:
            values.append(dcg(ranked, k) / ideal_dcg)
    return float(sum(values) / len(values)) if values else 0.0


def explanation_coverage_at_k(frame: pl.DataFrame, score_column: str, k: int, progress: bool = False) -> float:
    covered = 0
    total = 0
    for _, group in iter_groups(frame, f"explanation@{k}:{score_column}", progress):
        top = group.sort(score_column, descending=True).head(k)
        covered += int((top["explanation_feature_count"] > 0).sum())
        total += top.height
    return covered / total if total else 0.0


def strong_reason_coverage_at_k(frame: pl.DataFrame, score_column: str, k: int, progress: bool = False) -> float:
    strong_columns = ["same_occupation", "same_district", "same_education", "same_field", "shared_hobby_count", "shared_skill_count"]
    covered = 0
    total = 0
    for _, group in iter_groups(frame, f"strong_reason@{k}:{score_column}", progress):
        top = group.sort(score_column, descending=True).head(k)
        covered += int(top.select((pl.sum_horizontal(strong_columns) > 0).sum()).item())
        total += top.height
    return covered / total if total else 0.0


def low_information_dominance_at_k(frame: pl.DataFrame, score_column: str, k: int, progress: bool = False) -> float:
    low_columns = ["same_sex", "same_marital", "same_province", "same_community"]
    strong_columns = ["same_occupation", "same_district", "same_education", "same_field", "shared_hobby_count", "shared_skill_count"]
    dominated = 0
    total = 0
    for _, group in iter_groups(frame, f"low_info@{k}:{score_column}", progress):
        top = group.sort(score_column, descending=True).head(k)
        dominated += int(top.select(((pl.sum_horizontal(low_columns) > 0) & ~(pl.sum_horizontal(strong_columns) > 0)).sum()).item())
        total += top.height
    return dominated / total if total else 0.0


def average_reason_count_at_k(frame: pl.DataFrame, score_column: str, k: int, progress: bool = False) -> float:
    values: list[float] = []
    for _, group in iter_groups(frame, f"reason_count@{k}:{score_column}", progress):
        top = group.sort(score_column, descending=True).head(k)
        if top.height:
            values.append(float(top["explanation_feature_count"].mean()))
    return float(sum(values) / len(values)) if values else 0.0


def unique_target_rate_at_k(frame: pl.DataFrame, score_column: str, k: int, progress: bool = False) -> float:
    total = 0
    targets: set[str] = set()
    for _, group in iter_groups(frame, f"unique_target@{k}:{score_column}", progress):
        top = group.sort(score_column, descending=True).head(k)
        total += top.height
        targets.update(str(value) for value in top["target_uuid"].to_list())
    return len(targets) / total if total else 0.0


def attribute_diversity_at_k(frame: pl.DataFrame, score_column: str, attribute_column: str, k: int, progress: bool = False) -> float:
    if attribute_column not in frame.columns:
        return 0.0
    values: list[float] = []
    for _, group in iter_groups(frame, f"{attribute_column}_diversity@{k}:{score_column}", progress):
        top = group.sort(score_column, descending=True).head(k)
        if top.height:
            values.append(float(top[attribute_column].fill_null("").cast(pl.String).n_unique() / top.height))
    return float(sum(values) / len(values)) if values else 0.0


def demographic_only_rate_at_k(frame: pl.DataFrame, score_column: str, k: int, progress: bool = False) -> float:
    low_columns = ["same_sex", "same_marital", "same_province", "same_community"]
    strong_columns = ["same_occupation", "same_district", "same_education", "same_field", "shared_hobby_count", "shared_skill_count"]
    text_columns = [column for column in frame.columns if column.endswith("_text_cosine")]
    count = 0
    total = 0
    for _, group in iter_groups(frame, f"demographic_only@{k}:{score_column}", progress):
        top = group.sort(score_column, descending=True).head(k)
        low_signal = pl.sum_horizontal(low_columns) > 0
        strong_signal = pl.sum_horizontal(strong_columns) > 0
        if text_columns:
            text_signal = pl.max_horizontal(text_columns) >= 0.5
        else:
            text_signal = pl.lit(False)
        count += int(top.select((low_signal & ~strong_signal & ~text_signal).sum()).item())
        total += top.height
    return count / total if total else 0.0


def topk_overlap_at_k(frame: pl.DataFrame, left_score: str, right_score: str, k: int, progress: bool = False) -> float:
    overlaps: list[float] = []
    for _, group in iter_groups(frame, f"overlap@{k}:{left_score}->{right_score}", progress):
        left = set(str(value) for value in group.sort(left_score, descending=True).head(k)["target_uuid"].to_list())
        right = set(str(value) for value in group.sort(right_score, descending=True).head(k)["target_uuid"].to_list())
        if left or right:
            overlaps.append(len(left & right) / max(1, k))
    return float(sum(overlaps) / len(overlaps)) if overlaps else 0.0


def _evaluate_group_for_score(payload: tuple[pl.DataFrame, str, tuple[int, ...]]) -> dict[int, dict[str, Any]]:
    group, score_column, top_k_values = payload
    ranked = group.sort(score_column, descending=True)
    labels = ranked["label"].cast(pl.Float64).to_list()
    ideal = sorted(group["label"].cast(pl.Float64).to_list(), reverse=True)
    strong_columns = ["same_occupation", "same_district", "same_education", "same_field", "shared_hobby_count", "shared_skill_count"]
    low_columns = ["same_sex", "same_marital", "same_province", "same_community"]
    text_columns = [column for column in group.columns if column.endswith("_text_cosine")]

    result: dict[int, dict[str, Any]] = {}
    for k in top_k_values:
        top = ranked.head(k)
        ideal_dcg = dcg(ideal, k)
        ndcg_value = dcg(labels, k) / ideal_dcg if ideal_dcg > 0 else None
        strong_signal = pl.sum_horizontal(strong_columns) > 0
        low_signal = pl.sum_horizontal(low_columns) > 0
        if text_columns:
            text_signal = pl.max_horizontal(text_columns) >= 0.5
        else:
            text_signal = pl.lit(False)
        total = top.height
        result[k] = {
            "ndcg": ndcg_value,
            "explanation_covered": int((top["explanation_feature_count"] > 0).sum()),
            "strong_covered": int(top.select(strong_signal.sum()).item()),
            "low_dominated": int(top.select((low_signal & ~strong_signal).sum()).item()),
            "reason_count": float(top["explanation_feature_count"].mean()) if total else None,
            "targets": set(str(value) for value in top["target_uuid"].to_list()),
            "total": total,
            "occupation_diversity": float(top["target_occupation"].fill_null("").cast(pl.String).n_unique() / total) if total and "target_occupation" in top.columns else None,
            "province_diversity": float(top["target_province"].fill_null("").cast(pl.String).n_unique() / total) if total and "target_province" in top.columns else None,
            "community_diversity": float(top["target_community_id"].fill_null("").cast(pl.String).n_unique() / total) if total and "target_community_id" in top.columns else None,
            "demographic_only": int(top.select((low_signal & ~strong_signal & ~text_signal).sum()).item()),
        }
    return result


def evaluate_score_column(
    frame: Any,
    score_column: str,
    top_k_values: list[int],
    progress: bool = True,
    workers: int | None = None,
) -> dict[str, float]:
    polars_frame = _to_polars_frame(frame)
    groups = list(polars_frame.partition_by("source_uuid", maintain_order=True))
    worker_count = resolve_worker_count(workers)
    payloads = [(group, score_column, tuple(top_k_values)) for group in groups]
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            iterator = executor.map(_evaluate_group_for_score, payloads)
            if progress:
                try:
                    from tqdm.auto import tqdm
                    iterator = tqdm(iterator, total=len(payloads), desc=f"metrics:{score_column}", unit="source")
                except ImportError:
                    pass
            results = list(iterator)
    else:
        iterator = (_evaluate_group_for_score(payload) for payload in payloads)
        if progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, total=len(payloads), desc=f"metrics:{score_column}", unit="source")
            except ImportError:
                pass
        results = list(iterator)

    metrics: dict[str, float] = {}
    for k in top_k_values:
        total = sum(int(item[k]["total"]) for item in results)
        ndcg_values = [float(item[k]["ndcg"]) for item in results if item[k]["ndcg"] is not None]
        reason_values = [float(item[k]["reason_count"]) for item in results if item[k]["reason_count"] is not None]
        occupation_values = [float(item[k]["occupation_diversity"]) for item in results if item[k]["occupation_diversity"] is not None]
        province_values = [float(item[k]["province_diversity"]) for item in results if item[k]["province_diversity"] is not None]
        community_values = [float(item[k]["community_diversity"]) for item in results if item[k]["community_diversity"] is not None]
        targets: set[str] = set()
        for item in results:
            targets.update(item[k]["targets"])
        metrics[f"ndcg@{k}"] = float(sum(ndcg_values) / len(ndcg_values)) if ndcg_values else 0.0
        metrics[f"explanation_coverage@{k}"] = sum(int(item[k]["explanation_covered"]) for item in results) / total if total else 0.0
        metrics[f"strong_reason_coverage@{k}"] = sum(int(item[k]["strong_covered"]) for item in results) / total if total else 0.0
        metrics[f"low_information_dominance@{k}"] = sum(int(item[k]["low_dominated"]) for item in results) / total if total else 0.0
        metrics[f"average_reason_count@{k}"] = float(sum(reason_values) / len(reason_values)) if reason_values else 0.0
        metrics[f"unique_target_rate@{k}"] = len(targets) / total if total else 0.0
        metrics[f"occupation_diversity@{k}"] = float(sum(occupation_values) / len(occupation_values)) if occupation_values else 0.0
        metrics[f"province_diversity@{k}"] = float(sum(province_values) / len(province_values)) if province_values else 0.0
        metrics[f"community_diversity@{k}"] = float(sum(community_values) / len(community_values)) if community_values else 0.0
        metrics[f"demographic_only_rate@{k}"] = sum(int(item[k]["demographic_only"]) for item in results) / total if total else 0.0
    return metrics


def load_test_features(config: dict[str, Any], features_path: str | None = None) -> pl.DataFrame:
    input_features_path = Path(features_path or config["paths"]["features"])
    if not input_features_path.is_absolute():
        input_features_path = PROJECT_ROOT / input_features_path
    features = pl.read_parquet(input_features_path)
    return features.filter(pl.col("split") == "test")


def write_metrics(experiment_name: str, payload: dict[str, Any]) -> None:
    output_path = ensure_parent(metrics_path(experiment_name))
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_manual_review(frame: pl.DataFrame, experiment_name: str, score_columns: list[str], review_size: int) -> None:
    review_rows: list[pl.DataFrame] = []
    for score_column in score_columns:
        ranked = (
            frame.sort(["source_uuid", score_column], descending=[False, True])
            .group_by("source_uuid", maintain_order=True)
            .head(5)
            .with_columns(pl.lit(score_column).alias("model"))
        )
        review_rows.append(ranked)
    if not review_rows:
        return
    review = pl.concat(review_rows).head(review_size)
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
    review.select([column for column in columns if column in review.columns]).write_csv(output_path, include_bom=True)


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
        "test_rows": int(test.height),
        "test_sources": int(test["source_uuid"].n_unique()),
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
    test = test.with_columns(pl.Series("model_score", model.predict(test.select(feature_columns).to_numpy())))
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
        "test_rows": int(test.height),
        "test_sources": int(test["source_uuid"].n_unique()),
        "inference_seconds": inference_seconds,
        "evaluation_seconds": time.perf_counter() - start_time,
    }
    write_manual_review(test, experiment_name, ["model_score"], int(config["evaluation"].get("manual_review_size", 200)))
    write_metrics(experiment_name, metrics)


def min_max_values(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value <= min_value:
        return [0.0] * len(values)
    return [(value - min_value) / (max_value - min_value) for value in values]


def min_max_normalize(series: pl.Series) -> pl.Series:
    min_value = float(series.min())
    max_value = float(series.max())
    if max_value <= min_value:
        return pl.Series([0.0] * len(series))
    return (series - min_value) / (max_value - min_value)


def add_diversity_rerank_score(
    frame: Any,
    base_score: str,
    output_score: str,
    diversity_lambda: float,
    penalty_columns: list[str] | None = None,
) -> Any:
    return_pandas = not isinstance(frame, pl.DataFrame)
    polars_frame = _to_polars_frame(frame)
    penalties = penalty_columns or ["target_occupation", "target_province", "target_community_id"]
    reranked_rows: list[dict[str, Any]] = []
    for group in polars_frame.partition_by("source_uuid", maintain_order=True):
        remaining = group.to_dicts()
        selected_rows: list[dict[str, Any]] = []
        seen: dict[str, set[str]] = {column: set() for column in penalties if column in group.columns}
        while remaining:
            best_index = -1
            best_score = -float("inf")
            for index, row in enumerate(remaining):
                duplicate_penalty = 0.0
                for column, seen_values in seen.items():
                    value = str(row.get(column, ""))
                    if value and value in seen_values:
                        duplicate_penalty += 1.0
                adjusted_score = float(row[base_score]) - (diversity_lambda * duplicate_penalty)
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_index = index
            if best_index < 0:
                break
            selected = dict(remaining.pop(best_index))
            selected[output_score] = group.height - len(selected_rows)
            selected_rows.append(selected)
            for column, seen_values in seen.items():
                value = str(selected.get(column, ""))
                if value:
                    seen_values.add(value)
        reranked_rows.extend(selected_rows)
    if not reranked_rows:
        result = polars_frame.with_columns(pl.col(base_score).alias(output_score))
    else:
        result = pl.DataFrame(reranked_rows)
    return result.to_pandas() if return_pandas else result


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
    test = test.with_columns(pl.Series("model_score", model.predict(test.select(feature_columns).to_numpy())))
    normalized_groups: list[pl.DataFrame] = []
    for group in test.partition_by("source_uuid", maintain_order=True):
        normalized_groups.append(
            group.with_columns(
                pl.Series("_norm_fastrp_score", min_max_values([float(value) for value in group["fastrp_score"].to_list()])),
                pl.Series("_norm_model_score", min_max_values([float(value) for value in group["model_score"].to_list()])),
            )
        )
    test = pl.concat(normalized_groups) if normalized_groups else test

    top_k_values = [int(value) for value in config["evaluation"]["top_k"]]
    alpha_values = [float(value) for value in config["evaluation"].get("hybrid_alpha", [])]
    score_columns: list[str] = []
    for alpha in alpha_values:
        column = f"hybrid_alpha_{alpha:g}"
        test = test.with_columns((alpha * pl.col("_norm_model_score") + (1.0 - alpha) * pl.col("_norm_fastrp_score")).alias(column))
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
        "test_rows": int(test.height),
        "test_sources": int(test["source_uuid"].n_unique()),
        "evaluation_seconds": time.perf_counter() - start_time,
    }
    write_manual_review(test, hybrid_experiment_name, score_columns, int(config["evaluation"].get("manual_review_size", 200)))
    write_metrics(hybrid_experiment_name, metrics)
