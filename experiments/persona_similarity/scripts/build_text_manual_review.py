from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.persona_similarity.scripts.common import ensure_parent, load_config, resolve_path, write_json
from experiments.persona_similarity.scripts.experiment_specs import model_path, train_metadata_path
from experiments.persona_similarity.scripts.training_utils import load_feature_columns_from_metadata


DEFAULT_EXPERIMENTS = [
    "text_only_lambdarank",
    "structured_text_lambdarank",
    "structured_text_rank_xendcg",
]
TEXT_COLUMNS = [
    "all_text_cosine",
    "persona_text_cosine",
    "professional_text_cosine",
    "hobbies_text_cosine",
    "skills_text_cosine",
    "career_text_cosine",
    "family_text_cosine",
    "lifestyle_text_cosine",
]
TEXT_SNIPPET_COLUMNS = [
    "persona",
    "professional_persona",
    "hobbies_and_interests",
    "skills_and_expertise",
    "career_goals_and_ambitions",
    "family_persona",
]


def _predict_model_score(frame: pl.DataFrame, experiment_name: str) -> pl.Series:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("lightgbm is required to build text manual review samples.") from exc

    metadata_path = train_metadata_path(experiment_name)
    model_file = model_path(experiment_name)
    if not metadata_path.exists() or not model_file.exists():
        raise FileNotFoundError(f"Missing model or metadata for {experiment_name}")
    feature_columns = load_feature_columns_from_metadata(metadata_path)
    model = lgb.Booster(model_file=str(model_file))
    return pl.Series("model_score", model.predict(frame.select(feature_columns).to_numpy()))


def _snippet_expr(column: str, prefix: str, chars: int) -> pl.Expr:
    source = pl.col(column).cast(pl.String).fill_null("")
    return (
        pl.when(source.str.len_chars() > chars)
        .then(source.str.slice(0, chars) + "...")
        .otherwise(source)
        .alias(f"{prefix}_{column}")
    )


def _load_text_lookup(config: dict[str, Any], snippet_chars: int) -> tuple[tuple[pl.DataFrame, pl.DataFrame] | None, list[str]]:
    path = resolve_path(config["paths"]["persona_texts"])
    if not path.exists():
        return None, []
    frame = pl.read_parquet(path)
    if "uuid" not in frame.columns:
        return None, []
    snippet_columns = [column for column in TEXT_SNIPPET_COLUMNS if column in frame.columns]
    source = frame.select(
        pl.col("uuid").alias("source_uuid"),
        *[_snippet_expr(column, "source", snippet_chars) for column in snippet_columns],
    )
    target = frame.select(
        pl.col("uuid").alias("target_uuid"),
        *[_snippet_expr(column, "target", snippet_chars) for column in snippet_columns],
    )
    return (source, target), [*source.columns[1:], *target.columns[1:]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/persona_similarity/configs/lightgbm_reranker.yaml")
    parser.add_argument("--output", default="experiments/persona_similarity/artifacts/metrics/e5_text_manual_review_samples.csv")
    parser.add_argument("--status-output", default="experiments/persona_similarity/artifacts/metrics/e5_text_manual_review_status.json")
    parser.add_argument("--review-size", type=int, default=200)
    parser.add_argument("--snippet-chars", type=int, default=180)
    parser.add_argument("--features-path", default=None)
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_config(args.config)
    features_path = args.features_path or config["paths"]["features_with_text"]
    features = pl.read_parquet(resolve_path(features_path)).filter(pl.col("split") == "test")

    sample_frames: list[pl.DataFrame] = []
    for experiment_name in args.experiments:
        scored = features.with_columns(_predict_model_score(features, experiment_name))
        top = (
            scored.sort(["source_uuid", "model_score"], descending=[False, True])
            .group_by("source_uuid", maintain_order=True)
            .head(5)
            .with_columns(
                pl.lit(experiment_name).alias("experiment"),
                pl.lit("model_score").alias("model"),
            )
        )
        sample_frames.append(top)

    review = pl.concat(sample_frames).head(int(args.review_size))
    text_lookup, text_snippet_columns = _load_text_lookup(config, int(args.snippet_chars))
    if text_lookup is not None:
        source, target = text_lookup
        review = review.join(source, on="source_uuid", how="left").join(target, on="target_uuid", how="left")

    columns = [
        "experiment",
        "model",
        "source_uuid",
        "target_uuid",
        "label",
        "fastrp_score",
        "deterministic_score",
        "model_score",
        *[column for column in TEXT_COLUMNS if column in review.columns],
        "source_occupation",
        "target_occupation",
        "source_province",
        "target_province",
        "source_district",
        "target_district",
        "source_community_id",
        "target_community_id",
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
        *text_snippet_columns,
    ]
    output_path = ensure_parent(args.output)
    review.select([column for column in columns if column in review.columns]).write_csv(output_path, include_bom=True)
    write_json(
        args.status_output,
        {
            "stage": "build_text_manual_review",
            "output": str(output_path.relative_to(PROJECT_ROOT)),
            "experiments": args.experiments,
            "rows": int(review.height),
            "features_path": str(resolve_path(features_path).relative_to(PROJECT_ROOT)),
            "snippet_chars": int(args.snippet_chars),
            "text_feature_columns": TEXT_COLUMNS,
            "runtime_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
