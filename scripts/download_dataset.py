from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


DATASET_ID = "nvidia/Nemotron-Personas-Korea"
OUTPUT_DIR = Path.cwd() / "data" / "raw"


def _save_split_parquet(split_name: str, dataset: Dataset) -> None:
    # Rename 'train' split to 'personas' to match project settings
    file_name = "personas" if split_name == "train" else split_name
    output_path = OUTPUT_DIR / f"{file_name}.parquet"
    print(f"Saving {split_name} split to {output_path}")
    
    try:
        import polars as pl
        # Hugging Face Dataset을 Pandas로 변환 후 Polars DataFrame으로 변환하여 저장
        df = dataset.to_pandas()
        pl.from_pandas(df).write_parquet(output_path)
        print(f"Polars 엔진을 사용하여 {output_path} 저장 완료")
    except ImportError:
        dataset.to_parquet(str(output_path))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_ID}...")
    ds = load_dataset(DATASET_ID)

    disk_path = OUTPUT_DIR / "hf_dataset"
    print(f"Saving Hugging Face dataset files to {disk_path}")
    ds.save_to_disk(str(disk_path))

    if isinstance(ds, DatasetDict):
        for split_name, split_dataset in ds.items():
            _save_split_parquet(split_name, split_dataset)
    else:
        _save_split_parquet("train", ds)

    print(f"Done. Dataset saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
