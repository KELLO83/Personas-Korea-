import logging
import re

import pandas as pd
import polars as pl

from src.data.parser import parse_age_group, parse_district, parse_list_field

logger = logging.getLogger(__name__)

NAME_PATTERN = re.compile(r"^\s*([가-힣]{2,5})(?:\s*씨)?\s*(?:은|는)\b")
EMBEDDING_TEXT_FIELDS = (
    "persona",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "family_persona",
    "cultural_background",
    "skills_and_expertise",
    "hobbies_and_interests",
    "career_goals_and_ambitions",
    "skills_and_expertise_list",
    "hobbies_and_interests_list",
)


def preprocess(df: pl.DataFrame | pd.DataFrame, fast_mode: bool = False) -> pl.DataFrame | pd.DataFrame:
    """
    데이터프레임을 정제합니다. 텍스트 임베딩 생성은 제외하고 순수 정제/필터링에만 집중합니다.
    """
    _ = fast_mode
    input_is_pandas = isinstance(df, pd.DataFrame)
    if input_is_pandas:
        df = pl.from_pandas(df)

    if "uuid" not in df.columns:
        raise ValueError("DataFrame is missing required column: uuid")

    # 1. UUID 및 기초 필드 정규화 (Native Polars)
    # 학력(bachelors_field) 누락 방지 로직 포함
    df = df.with_columns([
        pl.col("uuid").cast(pl.String).str.strip_chars().fill_null(""),
        pl.col("bachelors_field")
         .cast(pl.String)
         .str.strip_chars()
         .replace(["", "해당없음"], None)
         .alias("bachelors_field")
    ])

    if df.select(pl.col("uuid").is_duplicated().any()).item():
        raise ValueError("UUID values must be unique")

    # 2. 리스트 필드 파싱 (스킬, 취미 등)
    if "skills_and_expertise_list" in df.columns:
        df = df.with_columns(
            pl.col("skills_and_expertise_list").map_elements(parse_list_field, return_dtype=pl.List(pl.String))
        )
    if "hobbies_and_interests_list" in df.columns:
        df = df.with_columns(
            pl.col("hobbies_and_interests_list").map_elements(parse_list_field, return_dtype=pl.List(pl.String))
        )

    # 3. 주소 및 연령대 파싱 (핵심 필터링 및 관계형 그래프 구축 용도)
    if "district" in df.columns:
        df = df.with_columns(
            pl.col("district").map_elements(parse_district, return_dtype=pl.List(pl.String)).alias("_dist_parsed")
        )
        df = df.with_columns([
            pl.col("_dist_parsed").list.get(0).alias("province_cleaned"),
            pl.col("_dist_parsed").list.get(1).alias("district_cleaned")
        ]).drop("_dist_parsed")

    if "age" in df.columns:
        df = df.with_columns(
            pl.col("age").map_elements(parse_age_group, return_dtype=pl.String).alias("age_group")
        )

    if "persona" in df.columns:
        df = df.with_columns(
            pl.col("persona").map_elements(extract_display_name, return_dtype=pl.String).alias("display_name")
        )

    text_fields = [field for field in EMBEDDING_TEXT_FIELDS if field in df.columns]
    if text_fields:
        df = df.with_columns(
            pl.struct(text_fields)
            .map_elements(lambda row: build_embedding_text(dict(row)), return_dtype=pl.String)
            .alias("embedding_text")
        )

    if input_is_pandas:
        return _to_pandas_with_python_lists(df)
    return df


def extract_display_name(persona_text: object) -> str | None:
    if not isinstance(persona_text, str):
        return None
    match = NAME_PATTERN.search(persona_text.strip())
    if not match:
        return None
    return match.group(1)


def build_embedding_text(row: dict[str, object]) -> str:
    parts: list[str] = []
    for field in EMBEDDING_TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, list):
            parts.extend(item.strip() for item in value if isinstance(item, str) and item.strip())
    return "\n".join(parts)


def _to_pandas_with_python_lists(df: pl.DataFrame) -> pd.DataFrame:
    result = df.to_pandas()
    for column in ("skills_and_expertise_list", "hobbies_and_interests_list"):
        if column in result.columns:
            result[column] = result[column].map(lambda value: value.tolist() if hasattr(value, "tolist") else value)
    return result
