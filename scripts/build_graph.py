import argparse

from src.data.loader import load_dataset
from src.data.preprocessor import preprocess
from src.graph.loader import GraphLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1000)
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

    sample_size = args.sample_size if args.sample_size is not None else None
    df = preprocess(load_dataset(sample_size=sample_size))
    loader = GraphLoader()
    try:
        loader.create_schema()
        loaded_count = loader.load_personas(df, batch_size=args.batch_size)
        print(f"Loaded {loaded_count} personas into Neo4j")
    finally:
        loader.close()


if __name__ == "__main__":
    main()
