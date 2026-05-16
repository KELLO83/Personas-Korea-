from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.persona_similarity.scripts.common import PROJECT_ROOT, ensure_parent, write_json
from experiments.persona_similarity.scripts.common import file_sha256, mark_cache_hit, should_use_cache, stable_json_hash
from experiments.persona_similarity.scripts.experiment_specs import model_path, train_metadata_path


def group_sizes(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("source_uuid", sort=False).size().astype(int).tolist()


def train_lightgbm_experiment(
    config: dict[str, Any],
    experiment_name: str,
    objective: str,
    feature_columns: list[str],
    features_path: str | Path | None = None,
    force: bool = False,
) -> None:
    input_features_path = Path(features_path) if features_path is not None else PROJECT_ROOT / config["paths"]["features"]
    if not input_features_path.is_absolute():
        input_features_path = PROJECT_ROOT / input_features_path
    cache_metadata = {
        "stage": "train_lightgbm",
        "experiment_name": experiment_name,
        "objective": objective,
        "features_path": str(input_features_path.relative_to(PROJECT_ROOT)),
        "features_hash": file_sha256(input_features_path),
        "config_hash": stable_json_hash({"lightgbm": config["lightgbm"], "feature_columns": feature_columns}),
        "feature_columns": feature_columns,
    }
    use_cache, cache_reason = should_use_cache(model_path(experiment_name), train_metadata_path(experiment_name), cache_metadata, force)
    if use_cache:
        mark_cache_hit(train_metadata_path(experiment_name), cache_metadata, model_path(experiment_name))
        return

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("lightgbm is required to train persona similarity experiments.") from exc

    features = pd.read_parquet(input_features_path)
    train = features[features["split"] == "train"].sort_values(["source_uuid", "fastrp_score"], ascending=[True, False])
    valid = features[features["split"] == "valid"].sort_values(["source_uuid", "fastrp_score"], ascending=[True, False])

    train_set = lgb.Dataset(train[feature_columns], label=train["label"], group=group_sizes(train), feature_name=feature_columns)
    valid_set = lgb.Dataset(valid[feature_columns], label=valid["label"], group=group_sizes(valid), feature_name=feature_columns, reference=train_set)

    lgb_config: dict[str, Any] = config["lightgbm"]
    params = {
        key: value
        for key, value in lgb_config.items()
        if key not in {"num_boost_round", "early_stopping_rounds", "log_period", "objective"}
    }
    params["objective"] = objective

    start_time = time.perf_counter()
    model = lgb.train(
        params,
        train_set,
        num_boost_round=int(lgb_config["num_boost_round"]),
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(int(lgb_config["early_stopping_rounds"])),
            lgb.log_evaluation(period=int(lgb_config.get("log_period", 10))),
        ],
    )
    train_seconds = time.perf_counter() - start_time

    output_model_path = ensure_parent(model_path(experiment_name))
    model.save_model(str(output_model_path))
    importance = dict(zip(feature_columns, model.feature_importance(importance_type="gain").astype(float).tolist(), strict=True))
    write_json(
        train_metadata_path(experiment_name),
        {
            "experiment_name": experiment_name,
            **cache_metadata,
            "cache_hit": False,
            "cache_reason": cache_reason,
            "model_path": str(output_model_path.relative_to(PROJECT_ROOT)),
            "objective": objective,
            "metric": params.get("metric"),
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "train_sources": int(train["source_uuid"].nunique()),
            "valid_sources": int(valid["source_uuid"].nunique()),
            "best_iteration": int(model.best_iteration or 0),
            "best_score": json.loads(json.dumps(model.best_score)),
            "train_seconds": train_seconds,
            "feature_importance_gain": importance,
        },
    )


def load_feature_columns_from_metadata(path: Path) -> list[str]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return [str(column) for column in metadata["feature_columns"]]
