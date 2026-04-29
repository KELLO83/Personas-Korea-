from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


DATASET_ID = "nvidia/Nemotron-Personas-Korea"
OUTPUT_DIR = Path.cwd() / "data" / "raw" / "nemotron-personas-korea"


def _save_split_parquet(split_name: str, dataset: Dataset) -> None:
    output_path = OUTPUT_DIR / f"{split_name}.parquet"
    print(f"Saving {split_name} split to {output_path}")
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
