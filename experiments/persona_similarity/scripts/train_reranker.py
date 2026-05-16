from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from experiments.persona_similarity.scripts.common import (
    ensure_parent,
    file_sha256,
    load_config,
    mark_cache_hit,
    should_use_cache,
    stable_json_hash,
    write_json,
)
from experiments.persona_similarity.scripts.feature_builder import FEATURE_COLUMNS


def group_sizes(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("source_uuid", sort=False).size().astype(int).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/persona_similarity/configs/lightgbm_reranker.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    cache_metadata = {
        "stage": "train_reranker_legacy",
        "features_path": config["paths"]["features"],
        "features_hash": file_sha256(config["paths"]["features"]),
        "model_path": config["paths"]["model"],
        "config_hash": stable_json_hash({"lightgbm": config["lightgbm"], "feature_columns": FEATURE_COLUMNS}),
        "feature_columns": FEATURE_COLUMNS,
    }
    use_cache, cache_reason = should_use_cache(config["paths"]["model"], config["paths"]["train_metadata"], cache_metadata, args.force)
    if use_cache:
        mark_cache_hit(config["paths"]["train_metadata"], cache_metadata, config["paths"]["model"])
        return

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("lightgbm is required to train the persona similarity reranker.") from exc

    features = pd.read_parquet(PROJECT_ROOT / config["paths"]["features"])
    train = features[features["split"] == "train"].sort_values(["source_uuid", "fastrp_score"], ascending=[True, False])
    valid = features[features["split"] == "valid"].sort_values(["source_uuid", "fastrp_score"], ascending=[True, False])

    train_set = lgb.Dataset(train[FEATURE_COLUMNS], label=train["label"], group=group_sizes(train), feature_name=FEATURE_COLUMNS)
    valid_set = lgb.Dataset(valid[FEATURE_COLUMNS], label=valid["label"], group=group_sizes(valid), feature_name=FEATURE_COLUMNS, reference=train_set)

    lgb_config: dict[str, Any] = config["lightgbm"]
    params = {key: value for key, value in lgb_config.items() if key not in {"num_boost_round", "early_stopping_rounds", "log_period"}}
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
    model_path = ensure_parent(config["paths"]["model"])
    model.save_model(str(model_path))
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importance(importance_type="gain").astype(float).tolist(), strict=True))
    write_json(
        config["paths"]["train_metadata"],
        {
            **cache_metadata,
            "cache_hit": False,
            "cache_reason": cache_reason,
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "objective": params.get("objective"),
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


if __name__ == "__main__":
    main()
