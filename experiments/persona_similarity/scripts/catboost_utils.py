from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import polars as pl

from experiments.persona_similarity.scripts.common import PROJECT_ROOT, ensure_parent, file_sha256, mark_cache_hit, should_use_cache, stable_json_hash, write_json
from experiments.persona_similarity.scripts.evaluation_utils import evaluate_score_column, load_test_features, topk_overlap_at_k, write_manual_review, write_metrics
from experiments.persona_similarity.scripts.experiment_specs import metrics_path, model_path, train_metadata_path


def train_catboost_experiment(
    config: dict[str, Any],
    experiment_name: str,
    feature_columns: list[str],
    features_path: str | Path | None = None,
    force: bool = False,
) -> None:
    input_features_path = Path(features_path) if features_path is not None else PROJECT_ROOT / config["paths"]["features"]
    if not input_features_path.is_absolute():
        input_features_path = PROJECT_ROOT / input_features_path
    catboost_config = config.get("catboost", {})
    cache_metadata = {
        "stage": "train_catboost",
        "experiment_name": experiment_name,
        "features_path": str(input_features_path.relative_to(PROJECT_ROOT)),
        "features_hash": file_sha256(input_features_path),
        "config_hash": stable_json_hash({"catboost": catboost_config, "feature_columns": feature_columns}),
        "feature_columns": feature_columns,
    }
    use_cache, cache_reason = should_use_cache(model_path(experiment_name), train_metadata_path(experiment_name), cache_metadata, force)
    if use_cache:
        mark_cache_hit(train_metadata_path(experiment_name), cache_metadata, model_path(experiment_name))
        return

    try:
        from catboost import CatBoostRanker, Pool
    except ImportError as exc:
        raise SystemExit("catboost is required to train CatBoost persona similarity experiments.") from exc

    features = pl.read_parquet(input_features_path)
    train = features.filter(pl.col("split") == "train").sort(["source_uuid", "fastrp_score"], descending=[False, True])
    valid = features.filter(pl.col("split") == "valid").sort(["source_uuid", "fastrp_score"], descending=[False, True])

    train_pool = Pool(train.select(feature_columns).to_numpy(), label=train["label"].to_numpy(), group_id=train["source_uuid"].to_list())
    valid_pool = Pool(valid.select(feature_columns).to_numpy(), label=valid["label"].to_numpy(), group_id=valid["source_uuid"].to_list())
    params = {
        key: value
        for key, value in catboost_config.items()
        if key not in {"early_stopping_rounds", "verbose"}
    }
    start_time = time.perf_counter()
    model = CatBoostRanker(**params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=int(catboost_config.get("early_stopping_rounds", 30)),
        verbose=catboost_config.get("verbose", 50),
    )
    train_seconds = time.perf_counter() - start_time

    output_model_path = ensure_parent(model_path(experiment_name))
    model.save_model(str(output_model_path))
    write_json(
        train_metadata_path(experiment_name),
        {
            "experiment_name": experiment_name,
            **cache_metadata,
            "cache_hit": False,
            "cache_reason": cache_reason,
            "model_path": str(output_model_path.relative_to(PROJECT_ROOT)),
            "train_rows": int(train.height),
            "valid_rows": int(valid.height),
            "train_sources": int(train["source_uuid"].n_unique()),
            "valid_sources": int(valid["source_uuid"].n_unique()),
            "best_iteration": int(model.get_best_iteration() or 0),
            "best_score": json.loads(json.dumps(model.get_best_score(), default=str)),
            "train_seconds": train_seconds,
            "feature_importance": dict(zip(feature_columns, model.get_feature_importance(train_pool).astype(float).tolist(), strict=True)),
        },
    )


def evaluate_catboost_experiment(
    config: dict[str, Any],
    experiment_name: str,
    features_path: str | None = None,
    force: bool = False,
) -> None:
    metadata = json.loads(train_metadata_path(experiment_name).read_text(encoding="utf-8"))
    feature_columns = [str(column) for column in metadata["feature_columns"]]
    resolved_features_path = Path(features_path or config["paths"]["features"])
    if not resolved_features_path.is_absolute():
        resolved_features_path = PROJECT_ROOT / resolved_features_path
    cache_metadata = {
        "stage": "evaluate_catboost",
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
        from catboost import CatBoostRanker
    except ImportError as exc:
        raise SystemExit("catboost is required to evaluate CatBoost persona similarity experiments.") from exc

    start_time = time.perf_counter()
    test = load_test_features(config, str(resolved_features_path))
    model = CatBoostRanker()
    model.load_model(str(model_path(experiment_name)))
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
