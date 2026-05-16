import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


DATASET_PATH = Path("data/raw/nemotron-personas-korea/train.parquet")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    parquet_file = pq.ParquetFile(DATASET_PATH)
    columns = parquet_file.schema.names

    print(f"File: {DATASET_PATH}")
    print(f"Rows: {parquet_file.metadata.num_rows:,}")
    print(f"Columns: {len(columns)}")
    print("\nColumn list:")
    for index, column in enumerate(columns, start=1):
        print(f"{index:02d}. {column}")

    print("\nHead:")
    df = pd.read_parquet(DATASET_PATH).head(2)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
