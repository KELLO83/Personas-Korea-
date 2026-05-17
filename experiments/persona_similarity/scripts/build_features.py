from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from experiments.persona_similarity.scripts.feature_builder import (
    FEATURE_COLUMNS,
    DeterministicScoreWeights,
    WeakLabelWeights,
    build_pair_features,
    deterministic_score,
    weak_label,
)

AUDIT_COLUMNS = [
    "source_age_group",
    "target_age_group",
    "source_sex",
    "target_sex",
    "source_province",
    "target_province",
    "source_district",
    "target_district",
    "source_occupation",
    "target_occupation",
    "source_education",
    "target_education",
    "source_field",
    "target_field",
    "source_marital",
    "target_marital",
    "source_family",
    "target_family",
    "source_housing",
    "target_housing",
    "source_community_id",
    "target_community_id",
    "shared_hobbies",
    "shared_skills",
]


def split_sources(source_uuids: list[str], split_config: dict[str, Any]) -> dict[str, list[str]]:
    unique_sources = sorted(set(source_uuids))
    rng = random.Random(int(split_config["seed"]))
    rng.shuffle(unique_sources)

    train_end = int(len(unique_sources) * float(split_config["train_ratio"]))
    valid_end = train_end + int(len(unique_sources) * float(split_config["valid_ratio"]))
    return {
        "train": unique_sources[:train_end],
        "valid": unique_sources[train_end:valid_end],
        "test": unique_sources[valid_end:],
    }


def iter_records_with_progress(records: Any, total: int | None = None) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return records
    return tqdm(records, desc="building pair features", unit="pair", total=total)


def build_feature_row(
    row: dict[str, Any],
    weak_weights: WeakLabelWeights,
    deterministic_weights: DeterministicScoreWeights,
) -> dict[str, Any]:
    features = build_pair_features(row)
    return {
        "source_uuid": row["source_uuid"],
        "target_uuid": row["target_uuid"],
        "label": weak_label(features, weak_weights),
        "deterministic_score": deterministic_score(features, deterministic_weights),
        **{column: row.get(column) for column in AUDIT_COLUMNS},
        **features,
    }


def build_feature_row_from_payload(payload: tuple[dict[str, Any], WeakLabelWeights, DeterministicScoreWeights]) -> dict[str, Any]:
    row, weak_weights, deterministic_weights = payload
    return build_feature_row(row, weak_weights, deterministic_weights)


def build_feature_frame(
    pairs: Any,
    weak_weights: WeakLabelWeights,
    deterministic_weights: DeterministicScoreWeights,
    workers: int = 1,
) -> Any:
    return_pandas = not isinstance(pairs, pl.DataFrame)
    pair_frame = pairs if isinstance(pairs, pl.DataFrame) else pl.from_pandas(pairs)
    records = pair_frame.to_dicts()
    if workers > 1:
        payloads = [(row, weak_weights, deterministic_weights) for row in records]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(build_feature_row_from_payload, payloads, chunksize=max(1, min(256, len(payloads) // (workers * 4) if workers else 1)))
            rows = list(iter_records_with_progress(iterator, total=len(records)))
    else:
        rows = [
            build_feature_row(row, weak_weights, deterministic_weights)
            for row in iter_records_with_progress(records, total=len(records))
        ]
    frame = pl.DataFrame(rows).select(["source_uuid", "target_uuid", "label", "deterministic_score", *AUDIT_COLUMNS, *FEATURE_COLUMNS])
    return frame.to_pandas() if return_pandas else frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/persona_similarity/configs/lightgbm_reranker.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu-thread-count", type=int, default=0, help="Thread workers for Python-heavy feature building. 0 uses laptop default.")
    parser.add_argument("--parallel-backend", choices=["auto", "thread", "serial"], default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    workers = resolve_worker_count(args.cpu_thread_count)
    parallel_backend = "thread" if args.parallel_backend == "auto" else args.parallel_backend
    if parallel_backend == "serial":
        workers = 1

    candidate_path = PROJECT_ROOT / config["paths"]["candidate_pairs"]
    cache_metadata = {
        "stage": "build_features",
        "input_path": config["paths"]["candidate_pairs"],
        "input_hash": file_sha256(candidate_path),
        "config_hash": stable_json_hash(
            {
                "weak_label": config["weak_label"],
                "deterministic_score": config.get("deterministic_score", {}),
                "split": config["split"],
                "feature_columns": FEATURE_COLUMNS,
                "audit_columns": AUDIT_COLUMNS,
                "parallel_backend": parallel_backend,
                "workers": workers,
            }
        ),
        "parallel_backend": parallel_backend,
        "workers": workers,
    }
    use_cache, cache_reason = should_use_cache(config["paths"]["features"], config["paths"]["feature_status"], cache_metadata, args.force)
    if use_cache:
        mark_cache_hit(config["paths"]["feature_status"], cache_metadata, config["paths"]["features"])
        return

    pairs = pl.read_parquet(candidate_path)
    weak_weights = WeakLabelWeights.from_config(config["weak_label"])
    deterministic_weights = DeterministicScoreWeights.from_config(config.get("deterministic_score", {}))
    start_time = time.perf_counter()
    features = build_feature_frame(pairs, weak_weights, deterministic_weights, workers)
    build_seconds = time.perf_counter() - start_time
    splits = split_sources(features["source_uuid"].tolist(), config["split"])

    for split_name, source_uuids in splits.items():
        features = features.with_columns(
            pl.when(pl.col("source_uuid").is_in(source_uuids))
            .then(pl.lit(split_name))
            .otherwise(pl.col("split") if "split" in features.columns else pl.lit(None))
            .alias("split")
        )

    output_path = ensure_parent(config["paths"]["features"])
    features.write_parquet(output_path)
    split_path = ensure_parent(config["paths"]["splits"])
    split_path.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(
        config["paths"]["feature_status"],
        {
            "rows": int(features.height),
            **cache_metadata,
            "cache_hit": False,
            "cache_reason": cache_reason,
            "source_count": int(features["source_uuid"].n_unique()) if features.height else 0,
            "target_count": int(features["target_uuid"].n_unique()) if features.height else 0,
            "feature_columns": FEATURE_COLUMNS,
            "audit_columns": AUDIT_COLUMNS,
            "split_counts": {name: len(values) for name, values in splits.items()},
            "build_seconds": build_seconds,
            "parallel_backend": parallel_backend,
            "workers": workers,
        },
    )


if __name__ == "__main__":
    main()
