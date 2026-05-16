from __future__ import annotations

import argparse
import json
import random
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


def iter_records_with_progress(records: list[dict[str, Any]]) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return records
    return tqdm(records, desc="building pair features", unit="pair")


def build_feature_frame(
    pairs: pd.DataFrame,
    weak_weights: WeakLabelWeights,
    deterministic_weights: DeterministicScoreWeights,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    records = pairs.to_dict(orient="records")
    for row in iter_records_with_progress(records):
        features = build_pair_features(row)
        output = {
            "source_uuid": row["source_uuid"],
            "target_uuid": row["target_uuid"],
            "label": weak_label(features, weak_weights),
            "deterministic_score": deterministic_score(features, deterministic_weights),
            **{column: row.get(column) for column in AUDIT_COLUMNS},
            **features,
        }
        rows.append(output)
    return pd.DataFrame(rows, columns=["source_uuid", "target_uuid", "label", "deterministic_score", *AUDIT_COLUMNS, *FEATURE_COLUMNS])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/persona_similarity/configs/lightgbm_reranker.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

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
            }
        ),
    }
    use_cache, cache_reason = should_use_cache(config["paths"]["features"], config["paths"]["feature_status"], cache_metadata, args.force)
    if use_cache:
        mark_cache_hit(config["paths"]["feature_status"], cache_metadata, config["paths"]["features"])
        return

    pairs = pd.read_parquet(candidate_path)
    weak_weights = WeakLabelWeights.from_config(config["weak_label"])
    deterministic_weights = DeterministicScoreWeights.from_config(config.get("deterministic_score", {}))
    start_time = time.perf_counter()
    features = build_feature_frame(pairs, weak_weights, deterministic_weights)
    build_seconds = time.perf_counter() - start_time
    splits = split_sources(features["source_uuid"].tolist(), config["split"])

    for split_name, source_uuids in splits.items():
        features.loc[features["source_uuid"].isin(source_uuids), "split"] = split_name

    output_path = ensure_parent(config["paths"]["features"])
    features.to_parquet(output_path, index=False)
    split_path = ensure_parent(config["paths"]["splits"])
    split_path.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(
        config["paths"]["feature_status"],
        {
            "rows": int(len(features)),
            **cache_metadata,
            "cache_hit": False,
            "cache_reason": cache_reason,
            "source_count": int(features["source_uuid"].nunique()) if not features.empty else 0,
            "target_count": int(features["target_uuid"].nunique()) if not features.empty else 0,
            "feature_columns": FEATURE_COLUMNS,
            "audit_columns": AUDIT_COLUMNS,
            "split_counts": {name: len(values) for name, values in splits.items()},
            "build_seconds": build_seconds,
        },
    )


if __name__ == "__main__":
    main()
