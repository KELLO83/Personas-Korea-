from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
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
from experiments.persona_similarity.scripts.experiment_specs import TEXT_FEATURE_COLUMNS
from experiments.persona_similarity.scripts.text_feature_builder import TEXT_FEATURE_BY_DOMAIN, cosine_similarity, embedding_key


def load_embedding_map(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    keys = payload["keys"].astype(str)
    embeddings = payload["embeddings"]
    return {key: embeddings[index] for index, key in enumerate(keys)}


def iter_records_with_progress(records: list[dict[str, Any]]) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return records
    return tqdm(records, desc="building text pair features", unit="pair")


def build_text_feature_frame(features: pd.DataFrame, embedding_map: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    zero = np.zeros(1, dtype=np.float32)
    for row in iter_records_with_progress(features[["source_uuid", "target_uuid"]].to_dict(orient="records")):
        source_uuid = str(row["source_uuid"])
        target_uuid = str(row["target_uuid"])
        output: dict[str, float] = {}
        for domain, feature_name in TEXT_FEATURE_BY_DOMAIN.items():
            left = embedding_map.get(embedding_key(source_uuid, domain), zero)
            right = embedding_map.get(embedding_key(target_uuid, domain), zero)
            output[feature_name] = cosine_similarity(left, right) if left.shape == right.shape else 0.0
        rows.append(output)
    return pd.DataFrame(rows, columns=TEXT_FEATURE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/persona_similarity/configs/lightgbm_reranker.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    cache_metadata = {
        "stage": "build_text_features",
        "features_path": config["paths"]["features"],
        "features_hash": file_sha256(config["paths"]["features"]),
        "embeddings_path": config["paths"]["text_embeddings"],
        "embeddings_hash": file_sha256(config["paths"]["text_embeddings"]),
        "config_hash": stable_json_hash({"text_feature_columns": TEXT_FEATURE_COLUMNS}),
    }
    use_cache, cache_reason = should_use_cache(config["paths"]["features_with_text"], config["paths"]["text_feature_status"], cache_metadata, args.force)
    if use_cache:
        mark_cache_hit(config["paths"]["text_feature_status"], cache_metadata, config["paths"]["features_with_text"])
        return

    start_time = time.perf_counter()
    features = pd.read_parquet(PROJECT_ROOT / config["paths"]["features"])
    embedding_map = load_embedding_map(PROJECT_ROOT / config["paths"]["text_embeddings"])
    text_features = build_text_feature_frame(features, embedding_map)
    merged = pd.concat([features.reset_index(drop=True), text_features], axis=1)
    output_path = ensure_parent(config["paths"]["features_with_text"])
    merged.to_parquet(output_path, index=False)
    write_json(
        config["paths"]["text_feature_status"],
        {
            **cache_metadata,
            "cache_hit": False,
            "cache_reason": cache_reason,
            "rows": int(len(merged)),
            "text_feature_columns": TEXT_FEATURE_COLUMNS,
            "build_seconds": time.perf_counter() - start_time,
            "nonzero_counts": {column: int((merged[column] != 0).sum()) for column in TEXT_FEATURE_COLUMNS},
        },
    )


if __name__ == "__main__":
    main()
