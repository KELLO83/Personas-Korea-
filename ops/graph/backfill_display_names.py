from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
import sys

import polars as pl
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_dataset
from src.data.preprocessor import preprocess
from src.graph.loader import GraphLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillArgs:
    sample_size: int | None
    batch_size: int
    overwrite: bool
    dry_run: bool


def _parse_args() -> BackfillArgs:
    parser = argparse.ArgumentParser(description="Backfill Person.display_name from raw persona text without resetting Neo4j.")
    _ = parser.add_argument("--sample-size", type=int, default=None, help="Limit raw dataset rows for a dry run or local sample.")
    _ = parser.add_argument("--batch-size", type=int, default=5000, help="Neo4j update batch size.")
    _ = parser.add_argument("--overwrite", action="store_true", help="Overwrite existing display_name values. Default updates only missing names.")
    _ = parser.add_argument("--dry-run", action="store_true", help="Compute candidate updates without writing to Neo4j.")
    namespace = parser.parse_args()
    sample_size = namespace.sample_size if isinstance(namespace.sample_size, int) else None
    batch_size = namespace.batch_size if isinstance(namespace.batch_size, int) else 5000
    return BackfillArgs(
        sample_size=sample_size,
        batch_size=batch_size,
        overwrite=bool(namespace.overwrite),
        dry_run=bool(namespace.dry_run),
    )


def _target_uuids(loader: GraphLoader, overwrite: bool) -> set[str]:
    query = """
    MATCH (p:Person)
    WHERE $overwrite OR p.display_name IS NULL OR trim(toString(p.display_name)) = ""
    RETURN p.uuid AS uuid
    """
    with loader.driver.session(database=loader.database) as session:
        return {str(record["uuid"]) for record in session.run(query, overwrite=overwrite) if record["uuid"]}


def _build_update_rows(sample_size: int | None, target_uuids: set[str]) -> list[dict[str, str]]:
    raw_df = load_dataset(sample_size=sample_size)
    processed = preprocess(raw_df, fast_mode=True)
    if not isinstance(processed, pl.DataFrame):
        processed = pl.from_pandas(processed)

    rows_df = processed.select("uuid", "display_name").filter(
        pl.col("uuid").is_in(list(target_uuids))
        & pl.col("display_name").is_not_null()
        & (pl.col("display_name").str.strip_chars() != "")
    )
    return [{"uuid": row["uuid"], "display_name": row["display_name"]} for row in rows_df.to_dicts()]


def _write_batches(loader: GraphLoader, rows: list[dict[str, str]], batch_size: int) -> int:
    query = """
    UNWIND $rows AS row
    MATCH (p:Person {uuid: row.uuid})
    SET p.display_name = row.display_name
    """
    updated = 0
    with loader.driver.session(database=loader.database) as session:
        for start in tqdm(range(0, len(rows), batch_size), desc="Backfilling display_name", unit="batch"):
            batch = rows[start : start + batch_size]
            _ = session.run(query, rows=batch)
            updated += len(batch)
    return updated


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args()
    loader = GraphLoader()
    try:
        target_uuids = _target_uuids(loader, overwrite=args.overwrite)
        logger.info("대상 Person 수: %d", len(target_uuids))
        if not target_uuids:
            return

        rows = _build_update_rows(sample_size=args.sample_size, target_uuids=target_uuids)
        logger.info("raw persona에서 추출 가능한 이름 수: %d", len(rows))
        if args.dry_run:
            preview = rows[:5]
            logger.info("dry-run preview: %s", preview)
            return

        updated = _write_batches(loader, rows=rows, batch_size=args.batch_size)
        logger.info("display_name backfill 완료: %d명", updated)
    finally:
        loader.close()


if __name__ == "__main__":
    main()
