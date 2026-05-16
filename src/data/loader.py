from pathlib import Path
from typing import Any

import polars as pl
from datasets import load_dataset as load_hf_dataset

from src.config import settings

REQUIRED_COLUMNS = {
    "uuid",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "family_persona",
    "persona",
    "cultural_background",
    "skills_and_expertise",
    "hobbies_and_interests",
    "career_goals_and_ambitions",
    "skills_and_expertise_list",
    "hobbies_and_interests_list",
    "sex",
    "age",
    "marital_status",
    "military_status",
    "family_type",
    "housing_type",
    "education_level",
    "bachelors_field",
    "occupation",
    "district",
    "province",
    "country",
}


_UNSET = object()


def _build_candidate_paths(data_dir: Path, data_file: str) -> list[Path]:
    configured_path = data_dir / data_file
    base_name = configured_path.stem if configured_path.suffix else configured_path.name

    candidates = [
        data_dir / f"{base_name}.parquet",
        data_dir / f"{base_name}.csv",
    ]

    if configured_path not in candidates:
        candidates.insert(0, configured_path)

    parquet_candidates = [path for path in candidates if path.suffix.lower() == ".parquet"]
    csv_candidates = [path for path in candidates if path.suffix.lower() == ".csv"]
    other_candidates = [path for path in candidates if path.suffix.lower() not in {".parquet", ".csv"}]

    ordered_candidates = parquet_candidates + csv_candidates + other_candidates
    seen: set[Path] = set()

    unique_candidates: list[Path] = []
    for candidate in ordered_candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    return unique_candidates


def _resolve_safe_candidate_paths(data_dir: Path, data_file: str) -> list[Path]:
    resolved_data_dir = data_dir.resolve()
    candidate_paths = _build_candidate_paths(data_dir=data_dir, data_file=data_file)
    safe_candidate_paths: list[Path] = []

    for candidate_path in candidate_paths:
        resolved_candidate_path = candidate_path.resolve()
        if resolved_candidate_path == resolved_data_dir or resolved_data_dir not in resolved_candidate_path.parents:
            raise ValueError("Configured dataset path must stay within DATA_DIR")
        safe_candidate_paths.append(resolved_candidate_path)

    return safe_candidate_paths


def _read_dataframe(file_path: Path, n_rows: int | None = None) -> pl.DataFrame:
    suffix = file_path.suffix.lower()

    if suffix == ".parquet":
        return pl.read_parquet(file_path, n_rows=n_rows)
    if suffix == ".csv":
        return pl.read_csv(file_path, n_rows=n_rows)

    raise ValueError(f"Unsupported file format: {suffix}")


def _validate_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        raise ValueError("Loaded dataset is empty")

    missing_columns = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


def load_dataset(sample_size: int | None | object = _UNSET) -> pl.DataFrame:
    resolved_sample_size = settings.DATA_SAMPLE_SIZE if sample_size is _UNSET else sample_size

    data_dir = Path(settings.DATA_DIR)
    candidate_paths = _resolve_safe_candidate_paths(data_dir=data_dir, data_file=settings.DATA_FILE)

    existing_paths = [path for path in candidate_paths if path.exists() and path.is_file()]
    if not existing_paths:
        import logging
        import sys
        logger = logging.getLogger(__name__)
        logger.error("로컬 데이터 파일을 찾을 수 없습니다. (검색 경로: %s)", [str(p) for p in candidate_paths])
        logger.error("프로그램을 종료합니다. 'dataset_down.py'를 먼저 실행하여 데이터를 다운로드하거나 경로를 확인해주세요.")
        sys.exit(1)

    last_error: Exception | None = None

    for file_path in existing_paths:
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("로컬 파일에서 데이터를 로드합니다: %s", file_path)
            df = _read_dataframe(file_path, n_rows=resolved_sample_size)
            validated_df = _validate_dataframe(df)
            logger.info("데이터 로딩이 완료되었습니다. (로드된 행 수: %d)", validated_df.height)
            return validated_df
        except ValueError:
            raise
        except (OSError, Exception) as exc:
            last_error = exc

    raise ValueError("Failed to load dataset from available files") from last_error


def _load_huggingface_dataset(sample_size: int | None) -> pl.DataFrame:
    split = settings.HF_DATASET_SPLIT
    if sample_size:
        split = f"{split}[:{sample_size}]"

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Hugging Face에서 데이터를 로드합니다: %s (split: %s)", settings.HF_DATASET_NAME, split)
    ds = load_hf_dataset(settings.HF_DATASET_NAME, split=split)
    df = pl.from_arrow(ds.with_format("arrow")[:])
    return df
