from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl

from experiments.persona_similarity.scripts.common import (
    ensure_parent,
    file_sha256,
    load_config,
    mark_cache_hit,
    resolve_worker_count,
    should_use_cache,
    stable_json_hash,
    write_json,
)
from experiments.persona_similarity.scripts.experiment_specs import TEXT_FEATURE_COLUMNS
from experiments.persona_similarity.scripts.text_feature_builder import TEXT_FEATURE_BY_DOMAIN, cosine_similarity, embedding_key

_text_feature_worker_embedding_map: dict[str, np.ndarray] = {}
_text_feature_worker_zero: np.ndarray | None = None


def load_embedding_map(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    keys = payload["keys"].astype(str)
    embeddings = payload["embeddings"]
    return {key: embeddings[index] for index, key in enumerate(keys)}


def iter_records_with_progress(records: Any, total: int | None = None) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return records
    return tqdm(records, desc="building text pair features", unit="pair", total=total)


def build_text_feature_row(row: dict[str, Any], embedding_map: dict[str, np.ndarray], zero: np.ndarray) -> dict[str, float]:
    source_uuid = str(row["source_uuid"])
    target_uuid = str(row["target_uuid"])
    output: dict[str, float] = {}
    for domain, feature_name in TEXT_FEATURE_BY_DOMAIN.items():
        left = embedding_map.get(embedding_key(source_uuid, domain), zero)
        right = embedding_map.get(embedding_key(target_uuid, domain), zero)
        output[feature_name] = cosine_similarity(left, right) if left.shape == right.shape else 0.0
    return output


def init_text_feature_worker(embedding_map: dict[str, np.ndarray], zero: np.ndarray) -> None:
    global _text_feature_worker_embedding_map
    global _text_feature_worker_zero
    _text_feature_worker_embedding_map = embedding_map
    _text_feature_worker_zero = zero


def build_text_feature_row_worker(row: dict[str, Any]) -> dict[str, float]:
    if _text_feature_worker_zero is None:
        raise RuntimeError("text feature worker was not initialized")
    return build_text_feature_row(row, _text_feature_worker_embedding_map, _text_feature_worker_zero)


def build_text_feature_frame(features: pl.DataFrame, embedding_map: dict[str, np.ndarray], workers: int) -> pl.DataFrame:
    zero = np.zeros(1, dtype=np.float32)
    records = features.select(["source_uuid", "target_uuid"]).to_dicts()
    if workers > 1:
        chunksize = max(1, min(256, len(records) // (workers * 4) if workers else 1))
        with ThreadPoolExecutor(
            max_workers=workers,
            initializer=init_text_feature_worker,
            initargs=(embedding_map, zero),
        ) as executor:
            iterator = executor.map(build_text_feature_row_worker, records, chunksize=chunksize)
            rows = list(iter_records_with_progress(iterator, total=len(records)))
    else:
        rows = [
            build_text_feature_row(row, embedding_map, zero)
            for row in iter_records_with_progress(records, total=len(records))
        ]
    return pl.DataFrame(rows).select(TEXT_FEATURE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/persona_similarity/configs/lightgbm_reranker.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu-thread-count", type=int, default=0, help="Thread workers for Python-heavy text feature building. 0 uses laptop default.")
    parser.add_argument("--parallel-backend", choices=["auto", "thread", "serial"], default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    workers = resolve_worker_count(args.cpu_thread_count)
    parallel_backend = "thread" if args.parallel_backend == "auto" else args.parallel_backend
    if parallel_backend == "serial":
        workers = 1
    cache_metadata = {
        "stage": "build_text_features",
        "features_path": config["paths"]["features"],
        "features_hash": file_sha256(config["paths"]["features"]),
        "embeddings_path": config["paths"]["text_embeddings"],
        "embeddings_hash": file_sha256(config["paths"]["text_embeddings"]),
        "config_hash": stable_json_hash(
            {
                "text_feature_columns": TEXT_FEATURE_COLUMNS,
                "parallel_backend": parallel_backend,
                "workers": workers,
            }
        ),
        "parallel_backend": parallel_backend,
        "workers": workers,
    }
    use_cache, cache_reason = should_use_cache(config["paths"]["features_with_text"], config["paths"]["text_feature_status"], cache_metadata, args.force)
    if use_cache:
        mark_cache_hit(config["paths"]["text_feature_status"], cache_metadata, config["paths"]["features_with_text"])
        return

    start_time = time.perf_counter()
    features = pl.read_parquet(PROJECT_ROOT / config["paths"]["features"])
    embedding_map = load_embedding_map(PROJECT_ROOT / config["paths"]["text_embeddings"])
    text_features = build_text_feature_frame(features, embedding_map, workers)
    merged = pl.concat([features, text_features], how="horizontal")
    output_path = ensure_parent(config["paths"]["features_with_text"])
    merged.write_parquet(output_path)
    write_json(
        config["paths"]["text_feature_status"],
        {
            **cache_metadata,
            "cache_hit": False,
            "cache_reason": cache_reason,
            "rows": int(merged.height),
            "text_feature_columns": TEXT_FEATURE_COLUMNS,
            "build_seconds": time.perf_counter() - start_time,
            "parallel_backend": parallel_backend,
            "workers": workers,
            "nonzero_counts": {column: int((merged[column] != 0).sum()) for column in TEXT_FEATURE_COLUMNS},
        },
    )


if __name__ == "__main__":
    main()
