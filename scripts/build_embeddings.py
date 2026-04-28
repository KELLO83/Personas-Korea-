import argparse
import logging

from src.data.loader import load_dataset
from src.data.preprocessor import preprocess
from src.embeddings.kure_model import KureEmbedder
from src.embeddings.vector_index import Neo4jVectorIndex, build_embedding_rows
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--skip-existing", action="store_true", help="Skip UUIDs that already have embeddings in Neo4j")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit dataset loading to the specified number of rows. Omit to load full dataset.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Deprecated: kept for compatibility. Full load is now the default when --sample-size is omitted.",
    )
    args = parser.parse_args()

    if args.full:
        sample_size = None
    elif args.sample_size is not None:
        sample_size = args.sample_size
    else:
        sample_size = None

    df = preprocess(load_dataset(sample_size=sample_size))

    index = Neo4jVectorIndex()
    try:
        index.create_index()
        if args.skip_existing:
            existing_uuids = index.get_embedded_uuids()
            logger.info("Found %d existing embedded UUIDs", len(existing_uuids))
            before_count = len(df)
            df = df[~df["uuid"].isin(existing_uuids)].copy()
            logger.info("Filtered from %d to %d rows to embed", before_count, len(df))
            if df.empty:
                logger.info("No new rows to embed. Exiting.")
                return
    finally:
        index.close()

    embedder = KureEmbedder()

    index = Neo4jVectorIndex()
    try:
        updated_count = 0
        for start in range(0, len(df), args.batch_size):
            batch_df = df.iloc[start : start + args.batch_size]
            texts = batch_df["embedding_text"].fillna("").tolist()
            embeddings = embedder.encode(texts)
            rows = build_embedding_rows(batch_df, embeddings)
            updated_count += index.set_embeddings(rows, batch_size=args.batch_size)
            logger.info("Embedded and stored %d / %d personas", min(start + len(batch_df), len(df)), len(df))
        logger.info("Stored %d persona embeddings in Neo4j", updated_count)
    finally:
        index.close()


if __name__ == "__main__":
    main()
