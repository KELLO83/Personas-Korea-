from __future__ import annotations

from pathlib import Path

from experiments.persona_similarity.scripts.common import PROJECT_ROOT
from experiments.persona_similarity.scripts.feature_builder import FEATURE_COLUMNS


TEXT_FEATURE_COLUMNS = [
    "all_text_cosine",
    "persona_text_cosine",
    "professional_text_cosine",
    "hobbies_text_cosine",
    "skills_text_cosine",
    "career_text_cosine",
    "family_text_cosine",
    "lifestyle_text_cosine",
]


FEATURE_EXCLUSION_SETS: dict[str, tuple[str, ...]] = {
    "without_fastrp": ("fastrp_score",),
    "without_low_info": ("same_sex", "same_marital", "same_community"),
    "without_location": ("same_province", "same_district"),
    "without_hobby": ("shared_hobby_count",),
}


def feature_columns(exclude: tuple[str, ...] = ()) -> list[str]:
    excluded = set(exclude)
    return [column for column in FEATURE_COLUMNS if column not in excluded]


def text_feature_columns() -> list[str]:
    return list(TEXT_FEATURE_COLUMNS)


def structured_text_feature_columns(exclude: tuple[str, ...] = ()) -> list[str]:
    return [*feature_columns(exclude), *TEXT_FEATURE_COLUMNS]


def artifact_path(kind: str, experiment_name: str, suffix: str) -> Path:
    if kind == "model":
        return PROJECT_ROOT / "experiments" / "persona_similarity" / "artifacts" / "models" / f"{experiment_name}{suffix}"
    if kind == "metrics":
        return PROJECT_ROOT / "experiments" / "persona_similarity" / "artifacts" / "metrics" / f"{experiment_name}{suffix}"
    raise ValueError(f"Unsupported artifact kind: {kind}")


def model_path(experiment_name: str) -> Path:
    return artifact_path("model", experiment_name, ".txt")


def train_metadata_path(experiment_name: str) -> Path:
    return artifact_path("metrics", experiment_name, "_train_metadata.json")


def metrics_path(experiment_name: str) -> Path:
    return artifact_path("metrics", experiment_name, "_metrics.json")


def manual_review_path(experiment_name: str) -> Path:
    return artifact_path("metrics", experiment_name, "_manual_review.csv")
