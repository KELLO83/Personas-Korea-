from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.persona_similarity.scripts.common import ensure_parent, file_sha256, load_config, resolve_path, write_json
from experiments.persona_similarity.scripts.experiment_specs import metrics_path, train_metadata_path


IDENTIFIER_MARKERS = ("uuid", "display_name", "name")
BASELINE_EXPERIMENTS = ["fastrp_baseline", "deterministic_baseline"]
MODEL_EXPERIMENTS = [
    "lambdarank",
    "rank_xendcg",
    "text_all_lambdarank",
    "text_only_lambdarank",
    "structured_all_text_lambdarank",
    "structured_text_lambdarank",
    "structured_text_rank_xendcg",
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _split_sources(features: pl.DataFrame) -> dict[str, set[str]]:
    return {
        split: set(features.filter(pl.col("split") == split)["source_uuid"].cast(pl.String).to_list())
        for split in ["train", "valid", "test"]
    }


def _split_report(features: pl.DataFrame) -> dict[str, Any]:
    sources = _split_sources(features)
    overlaps = {
        "train_valid": len(sources["train"] & sources["valid"]),
        "train_test": len(sources["train"] & sources["test"]),
        "valid_test": len(sources["valid"] & sources["test"]),
    }
    return {
        "passed": sum(overlaps.values()) == 0,
        "source_counts": {split: len(values) for split, values in sources.items()},
        "overlaps": overlaps,
    }


def _candidate_width_report(features: pl.DataFrame, expected_top_k: int) -> dict[str, Any]:
    group_width = features.group_by("source_uuid").len()
    min_width = int(group_width["len"].min()) if group_width.height else 0
    max_width = int(group_width["len"].max()) if group_width.height else 0
    return {
        "passed": min_width >= expected_top_k,
        "expected_top_k": expected_top_k,
        "min_candidates_per_source": min_width,
        "max_candidates_per_source": max_width,
        "source_count": int(group_width.height),
    }


def _features_hash_from_metadata(name: str) -> str | None:
    metadata = _load_json(train_metadata_path(name))
    if not metadata:
        return None
    value = metadata.get("features_hash")
    return str(value) if value is not None else None


def _identifier_feature_report() -> dict[str, Any]:
    experiments: dict[str, Any] = {}
    for name in MODEL_EXPERIMENTS:
        metadata = _load_json(train_metadata_path(name))
        feature_columns = list(metadata.get("feature_columns", [])) if metadata else []
        identifier_like = [
            column
            for column in feature_columns
            if any(marker in column.lower() for marker in IDENTIFIER_MARKERS)
        ]
        experiments[name] = {
            "exists": metadata is not None,
            "passed": metadata is not None and not identifier_like,
            "feature_columns": feature_columns,
            "identifier_like_features": identifier_like,
            "features_hash": metadata.get("features_hash") if metadata else None,
        }
    return {
        "passed": all(item["passed"] for item in experiments.values()),
        "experiments": experiments,
    }


def _metric_report(features_hash: str | None, text_features_hash: str | None) -> dict[str, Any]:
    experiments: dict[str, Any] = {}
    for name in [*BASELINE_EXPERIMENTS, *MODEL_EXPERIMENTS]:
        metrics = _load_json(metrics_path(name))
        expected_hash = text_features_hash if name in {
            "text_all_lambdarank",
            "text_only_lambdarank",
            "structured_all_text_lambdarank",
            "structured_text_lambdarank",
            "structured_text_rank_xendcg",
        } else features_hash
        observed_hash = metrics.get("features_hash") if metrics else None
        experiments[name] = {
            "exists": metrics is not None,
            "features_hash": observed_hash,
            "test_rows": metrics.get("test_rows") if metrics else None,
            "test_sources": metrics.get("test_sources") if metrics else None,
            "passed": metrics is not None and (expected_hash is None or observed_hash == expected_hash),
        }
    return {
        "passed": all(item["passed"] for item in experiments.values()),
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/persona_similarity/configs/lightgbm_reranker.yaml")
    parser.add_argument("--output", default="experiments/persona_similarity/artifacts/metrics/promotion_gate_status.json")
    parser.add_argument("--expected-candidate-top-k", type=int, default=50)
    parser.add_argument("--manual-review-approved", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    features_path = resolve_path(config["paths"]["features"])
    text_features_path = resolve_path(config["paths"]["features_with_text"])
    features = pl.read_parquet(features_path)
    text_features = pl.read_parquet(text_features_path)
    features_hash = file_sha256(features_path)
    text_features_hash = file_sha256(text_features_path)

    candidate_width = _candidate_width_report(features, int(args.expected_candidate_top_k))
    split_source_disjoint = _split_report(features)
    text_split_source_disjoint = _split_report(text_features)
    raw_identifier_features = _identifier_feature_report()
    same_split_metric_comparison = _metric_report(features_hash, text_features_hash)
    rollback_path = {
        "passed": "fastrp_score" in features.columns,
        "path": "raw FastRP/KNN SIMILAR_TO ordering via fastrp_score",
    }

    automatic_reports = [
        candidate_width,
        split_source_disjoint,
        text_split_source_disjoint,
        raw_identifier_features,
        same_split_metric_comparison,
        rollback_path,
    ]
    passed_automatic_checks = all(report["passed"] for report in automatic_reports)
    payload = {
        "stage": "promotion_gate_status",
        "features_path": _relative(features_path),
        "features_hash": features_hash,
        "features_with_text_path": _relative(text_features_path),
        "features_with_text_hash": text_features_hash,
        "candidate_width": candidate_width,
        "split_source_disjoint": split_source_disjoint,
        "text_split_source_disjoint": text_split_source_disjoint,
        "raw_identifier_features": raw_identifier_features,
        "same_split_metric_comparison": same_split_metric_comparison,
        "rollback_path": rollback_path,
        "manual_review": {
            "passed": bool(args.manual_review_approved),
            "reason": "Human/manual quality review is still required before promotion."
            if not args.manual_review_approved
            else "Manual review approved by operator flag.",
        },
        "passed_automatic_checks": passed_automatic_checks,
    }
    write_json(ensure_parent(args.output), payload)


if __name__ == "__main__":
    main()
