from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import polars as pl
import psycopg.errors as pg_errors
from psycopg import connect, sql
from psycopg.rows import dict_row

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import settings
from src.data.loader import load_dataset
from src.data.preprocessor import preprocess
from src.logging_config import configure_logging


SCHEMA_SQL_PATH = Path(__file__).with_name("pgvector_schema.sql")
logger = logging.getLogger(__name__)
EMBEDDING_TEXT_VERSION = "persona_embedding_v1"
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
VECTOR_EXTENSION_UNAVAILABLE = (
    "pgvector extension is not available in this PostgreSQL instance. "
    "Run this script with --metadata-only first, then add pgvector extension package first."
)


@dataclass(frozen=True)
class EmbeddingRowContext:
    model_name: str
    expected_dim: int


@dataclass(frozen=True)
class LoadContext:
    table_name: str
    expected_dim: int
    batch_size: int
    metadata_only: bool
    skip_existing: bool
    source_model: str
    embedder: Any | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load persona embeddings or metadata into PostgreSQL pgvector.")
    parser.add_argument("--sample-size", type=int, default=None, help="Limit dataset rows for quick loading.")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size for DB upsert and embedding encode.")
    parser.add_argument("--table-name", default=settings.PGVECTOR_TABLE_NAME, help="Target pgvector table name.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip UUIDs already present in the table.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Load UUID and metadata only. No embedding vector is generated.",
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Only create extension/table/index. Does not load rows. Uses vector schema unless --metadata-only.",
    )
    return parser.parse_args()


def _ensure_identifier(value: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError("Invalid table name. Allowed: letters, numbers, underscore.")
    return value


def _format_vector(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.8f}" for value in values) + "]"


def _to_uuid(value: object) -> str:
    if value is None:
        raise ValueError("uuid is missing")
    value_str = str(value).strip()
    if not value_str:
        raise ValueError("uuid is empty")
    return str(UUID(value_str))


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _normalize_text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        item_text = item.strip()
        if item_text:
            result.append(item_text)
    return result


def _build_metadata(row: dict[str, Any]) -> str:
    metadata = {
        "uuid": row.get("uuid"),
        "age_group": row.get("age_group"),
        "province": row.get("province_cleaned") or row.get("province"),
        "district": row.get("district_cleaned") or row.get("district"),
        "occupation": row.get("occupation"),
        "skills": row.get("skills_and_expertise_list", []),
        "hobbies": row.get("hobbies_and_interests_list", []),
        "source": "persona_korea_dataset",
        "family_type": row.get("family_type"),
        "housing_type": row.get("housing_type"),
        "education_level": row.get("education_level"),
        "bachelors_field": row.get("bachelors_field"),
        "military_status": row.get("military_status"),
        "marital_status": row.get("marital_status"),
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
    }
    return json.dumps(metadata, ensure_ascii=False)


def _base_row_values(row: dict[str, Any]) -> tuple[Any, ...]:
    persona_text = str(row.get("persona") or "")
    return (
        _to_uuid(row.get("uuid")),
        _normalize_text(row.get("display_name")),
        _normalize_int(row.get("age")),
        _normalize_text(row.get("age_group")),
        _normalize_text(row.get("sex")),
        _normalize_text(row.get("province_cleaned")) or _normalize_text(row.get("province")),
        _normalize_text(row.get("district_cleaned")) or _normalize_text(row.get("district")),
        _normalize_text(row.get("occupation")),
        _normalize_text(row.get("marital_status")),
        _normalize_text(row.get("military_status")),
        _normalize_text(row.get("family_type")),
        _normalize_text(row.get("housing_type")),
        _normalize_text(row.get("education_level")),
        _normalize_text(row.get("bachelors_field")),
        _normalize_text_list(row.get("skills_and_expertise_list")),
        _normalize_text_list(row.get("hobbies_and_interests_list")),
        persona_text,
        str(row.get("embedding_text") or persona_text),
    )


def _metadata_schema(table_name: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    id BIGSERIAL PRIMARY KEY,
    person_uuid UUID NOT NULL UNIQUE,
    display_name TEXT,
    age INTEGER,
    age_group TEXT,
    sex TEXT,
    province TEXT,
    district TEXT,
    occupation TEXT,
    marital_status TEXT,
    military_status TEXT,
    family_type TEXT,
    housing_type TEXT,
    education_level TEXT,
    bachelors_field TEXT,
    skills TEXT[],
    hobbies TEXT[],
    persona_text TEXT NOT NULL,
    embedding_text TEXT,
    source_model TEXT,
    embedding_text_version TEXT NOT NULL DEFAULT '{EMBEDDING_TEXT_VERSION}',
    metadata JSONB NOT NULL DEFAULT '{{}}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS {table_name}_age_group_idx
ON {table_name} (age_group);

CREATE INDEX IF NOT EXISTS {table_name}_sex_idx
ON {table_name} (sex);

CREATE INDEX IF NOT EXISTS {table_name}_province_idx
ON {table_name} (province);

CREATE INDEX IF NOT EXISTS {table_name}_district_idx
ON {table_name} (district);

CREATE INDEX IF NOT EXISTS {table_name}_occupation_idx
ON {table_name} (occupation);

CREATE INDEX IF NOT EXISTS {table_name}_embedding_text_version_idx
ON {table_name} (embedding_text_version);

CREATE INDEX IF NOT EXISTS {table_name}_skills_gin_idx
ON {table_name}
USING gin (skills);

CREATE INDEX IF NOT EXISTS {table_name}_hobbies_gin_idx
ON {table_name}
USING gin (hobbies);
"""


def _prepare_rows(
    batch_rows: list[dict[str, Any]],
    embeddings: list[list[float]],
    context: EmbeddingRowContext,
) -> list[tuple[Any, ...]]:
    if len(batch_rows) != len(embeddings):
        raise ValueError("Batch size and embedding size mismatch.")

    rows: list[tuple[Any, ...]] = []
    for row, embedding in zip(batch_rows, embeddings, strict=True):
        if len(embedding) != context.expected_dim:
            raise ValueError(f"Expected embedding dimension {context.expected_dim}, got {len(embedding)}.")
        vector = _format_vector([float(v) for v in embedding])
        rows.append(
            _base_row_values(row)
            + (
                vector,
                _build_metadata(row),
                context.model_name,
                context.expected_dim,
                EMBEDDING_TEXT_VERSION,
            )
        )
    return rows


def _prepare_metadata_rows(batch_rows: list[dict[str, Any]], source_model: str) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for row in batch_rows:
        rows.append(
            _base_row_values(row)
            + (
                _build_metadata(row),
                source_model,
                EMBEDDING_TEXT_VERSION,
            )
        )
    return rows


def _ensure_schema(cur: Any, table_name: str, dimension: int, metadata_only: bool) -> None:
    if metadata_only:
        cur.execute(_metadata_schema(table_name))
        return

    template = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    ddl = template.format(table_name=table_name, dimension=dimension)
    cur.execute(ddl)


def _detect_vector_support() -> bool:
    with connect(settings.PGVECTOR_DATABASE_URI) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT name FROM pg_available_extensions WHERE name = 'vector'")
            return bool(cur.fetchone())


def _find_existing_uuids(cur: Any, table_name: str, uuids: list[str]) -> set[str]:
    if not uuids:
        return set()
    query = sql.SQL("SELECT person_uuid FROM {table} WHERE person_uuid = ANY(%s)").format(
        table=sql.Identifier(table_name)
    )
    cur.execute(query, (uuids,))
    return {str(record["person_uuid"]) for record in cur.fetchall()}


def _upsert_rows(cur: Any, table_name: str, rows: list[tuple[Any, ...]]) -> int:
    if not rows:
        return 0
    query = sql.SQL(
        """
        INSERT INTO {table} (
            person_uuid,
            display_name,
            age,
            age_group,
            sex,
            province,
            district,
            occupation,
            marital_status,
            military_status,
            family_type,
            housing_type,
            education_level,
            bachelors_field,
            skills,
            hobbies,
            persona_text,
            embedding_text,
            embedding,
            metadata,
            source_model,
            embedding_dim,
            embedding_text_version
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s::JSONB, %s, %s, %s)
        ON CONFLICT (person_uuid) DO UPDATE SET
            display_name = EXCLUDED.display_name,
            age = EXCLUDED.age,
            age_group = EXCLUDED.age_group,
            sex = EXCLUDED.sex,
            province = EXCLUDED.province,
            district = EXCLUDED.district,
            occupation = EXCLUDED.occupation,
            marital_status = EXCLUDED.marital_status,
            military_status = EXCLUDED.military_status,
            family_type = EXCLUDED.family_type,
            housing_type = EXCLUDED.housing_type,
            education_level = EXCLUDED.education_level,
            bachelors_field = EXCLUDED.bachelors_field,
            skills = EXCLUDED.skills,
            hobbies = EXCLUDED.hobbies,
            persona_text = EXCLUDED.persona_text,
            embedding_text = EXCLUDED.embedding_text,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            source_model = EXCLUDED.source_model,
            embedding_dim = EXCLUDED.embedding_dim,
            embedding_text_version = EXCLUDED.embedding_text_version,
            updated_at = now()
        """
    ).format(table=sql.Identifier(table_name))
    cur.executemany(query, rows)
    return len(rows)


def _upsert_metadata_rows(cur: Any, table_name: str, rows: list[tuple[Any, ...]]) -> int:
    if not rows:
        return 0
    query = sql.SQL(
        """
        INSERT INTO {table} (
            person_uuid,
            display_name,
            age,
            age_group,
            sex,
            province,
            district,
            occupation,
            marital_status,
            military_status,
            family_type,
            housing_type,
            education_level,
            bachelors_field,
            skills,
            hobbies,
            persona_text,
            embedding_text,
            metadata,
            source_model,
            embedding_text_version
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::JSONB, %s, %s)
        ON CONFLICT (person_uuid) DO UPDATE SET
            display_name = EXCLUDED.display_name,
            age = EXCLUDED.age,
            age_group = EXCLUDED.age_group,
            sex = EXCLUDED.sex,
            province = EXCLUDED.province,
            district = EXCLUDED.district,
            occupation = EXCLUDED.occupation,
            marital_status = EXCLUDED.marital_status,
            military_status = EXCLUDED.military_status,
            family_type = EXCLUDED.family_type,
            housing_type = EXCLUDED.housing_type,
            education_level = EXCLUDED.education_level,
            bachelors_field = EXCLUDED.bachelors_field,
            skills = EXCLUDED.skills,
            hobbies = EXCLUDED.hobbies,
            persona_text = EXCLUDED.persona_text,
            embedding_text = EXCLUDED.embedding_text,
            metadata = EXCLUDED.metadata,
            source_model = EXCLUDED.source_model,
            embedding_text_version = EXCLUDED.embedding_text_version,
            updated_at = now()
        """
    ).format(table=sql.Identifier(table_name))
    cur.executemany(query, rows)
    return len(rows)


def _load_dataset(sample_size: int | None, require_embedding_text: bool) -> pl.DataFrame:
    df = preprocess(load_dataset(sample_size=sample_size))
    if "uuid" not in df.columns:
        raise ValueError("Dataset must include uuid column.")
    if require_embedding_text and "embedding_text" not in df.columns:
        raise ValueError("Dataset must include embedding_text column from preprocessing.")
    logger.info("Loaded rows for processing: %d", len(df))
    return df


def _iter_batches(rows: list[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _create_only(table_name: str, expected_dim: int, metadata_only: bool) -> None:
    with connect(settings.PGVECTOR_DATABASE_URI) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            if metadata_only:
                _ensure_schema(cur, table_name, expected_dim, metadata_only=True)
            else:
                if not _detect_vector_support():
                    raise RuntimeError(VECTOR_EXTENSION_UNAVAILABLE)
                _ensure_schema(cur, table_name, expected_dim, metadata_only=False)
            conn.commit()


def _build_load_context(args: argparse.Namespace, table_name: str, expected_dim: int) -> LoadContext:
    batch_size = max(1, args.batch_size)
    if args.metadata_only:
        return LoadContext(
            table_name=table_name,
            expected_dim=expected_dim,
            batch_size=batch_size,
            metadata_only=True,
            skip_existing=args.skip_existing,
            source_model="metadata_only",
            embedder=None,
        )

    from src.embeddings.kure_model import KureEmbedder

    return LoadContext(
        table_name=table_name,
        expected_dim=expected_dim,
        batch_size=batch_size,
        metadata_only=False,
        skip_existing=args.skip_existing,
        source_model=settings.EMBEDDING_MODEL_NAME,
        embedder=KureEmbedder(batch_size=batch_size),
    )


def _normalize_batch_uuids(batch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in batch_rows:
        try:
            row["uuid"] = _to_uuid(row.get("uuid"))
        except ValueError:
            logger.warning("Skipping invalid uuid row.")
            continue
        normalized_rows.append(row)
    return normalized_rows


def _batch_uuids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("uuid", "")) for row in rows if row.get("uuid")]


def _skip_existing_rows(cur: Any, rows: list[dict[str, Any]], context: LoadContext) -> list[dict[str, Any]]:
    uuids = _batch_uuids(rows)
    if not context.skip_existing or not uuids:
        return rows

    existing_uuids = _find_existing_uuids(cur, context.table_name, uuids)
    if not existing_uuids:
        return rows

    logger.info("Skip existing: %d", len(existing_uuids))
    return [row for row in rows if str(row.get("uuid", "")) not in existing_uuids]


def _encode_embeddings(texts: list[str], context: LoadContext) -> list[list[float]]:
    if context.embedder is None:
        raise RuntimeError("Embedding mode requires an initialized embedder.")
    return context.embedder.encode(texts)


def _insert_batch(cur: Any, batch_rows: list[dict[str, Any]], context: LoadContext) -> int:
    rows = _skip_existing_rows(cur, _normalize_batch_uuids(batch_rows), context)
    if not rows:
        return 0

    if context.metadata_only:
        rows_for_insert = _prepare_metadata_rows(batch_rows=rows, source_model=context.source_model)
        return _upsert_metadata_rows(cur, context.table_name, rows_for_insert)

    embeddings = _encode_embeddings([str(row.get("embedding_text", "") or "") for row in rows], context)
    rows_for_insert = _prepare_rows(
        rows,
        embeddings,
        EmbeddingRowContext(model_name=context.source_model, expected_dim=context.expected_dim),
    )
    try:
        return _upsert_rows(cur, context.table_name, rows_for_insert)
    except pg_errors.FeatureNotSupported:
        raise RuntimeError(
            "Embedding insert blocked: target column requires pgvector extension, but DB lacks vector type."
        )


def main() -> None:
    configure_logging()
    args = parse_args()
    table_name = _ensure_identifier(args.table_name)
    expected_dim = settings.EMBEDDING_DIMENSION

    if args.create_only:
        _create_only(table_name, expected_dim, metadata_only=args.metadata_only)
        logger.info("create-only completed for table: %s (metadata_only=%s)", table_name, args.metadata_only)
        return

    df = _load_dataset(sample_size=args.sample_size, require_embedding_text=not args.metadata_only)
    rows = df.to_dicts()
    context = _build_load_context(args, table_name, expected_dim)

    inserted = 0
    processed = 0
    with connect(settings.PGVECTOR_DATABASE_URI) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            _ensure_schema(cur, context.table_name, context.expected_dim, metadata_only=context.metadata_only)
            logger.info("Schema ensured for table: %s", context.table_name)
            logger.info(
                "Mode: %s",
                "metadata_only" if context.metadata_only else "with_embeddings",
            )

            for batch_rows in _iter_batches(rows, context.batch_size):
                inserted_in_batch = _insert_batch(cur, batch_rows, context)
                if inserted_in_batch == 0:
                    continue
                inserted += inserted_in_batch
                processed += inserted_in_batch
                conn.commit()
                logger.info("Inserted/updated %d rows (processed %d)", inserted, processed)

    logger.info("Done. Total rows inserted/updated: %d", inserted)


if __name__ == "__main__":
    main()
