from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import polars as pl


KOREAN_RE = re.compile(r"[가-힣]")
REPLACEMENT_RE = re.compile(r"\ufffd")
MOJIBAKE_HINT_RE = re.compile(r"[媛怨援遺源醫諛蹂댁꾩쓣섎뒗덉쑝]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check GNN data integrity with Polars.")
    parser.add_argument("--data-dir", type=Path, default=Path("GNN_Neural_Network/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("GNN_Neural_Network/artifacts/experiments/data_integrity/polars_data_integrity.json"),
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edge_path = args.data_dir / "person_hobby_edges.csv"
    context_path = args.data_dir / "person_context.csv"

    edges = pl.read_csv(edge_path, encoding="utf8-lossy")
    contexts = pl.read_csv(context_path, encoding="utf8-lossy")

    edge_persons = edges.select("person_uuid").unique()
    context_persons = contexts.select("person_uuid").unique()
    overlap = edge_persons.join(context_persons, on="person_uuid", how="inner")

    text_columns = [
        column
        for column in contexts.columns
        if column.endswith("_text") or column in {"persona_text", "hobbies_text", "embedding_text"}
    ]
    sample = contexts.select(text_columns).head(max(1, args.sample_size))
    text_values: list[str] = []
    for column in text_columns:
        text_values.extend(value for value in sample[column].to_list() if isinstance(value, str) and value)
    joined_sample = "\n".join(text_values)

    report: dict[str, Any] = {
        "edge_path": str(edge_path),
        "context_path": str(context_path),
        "edge_rows": edges.height,
        "context_rows": contexts.height,
        "edge_person_count": edge_persons.height,
        "context_person_count": context_persons.height,
        "edge_context_intersection_count": overlap.height,
        "edge_context_intersection_ratio": overlap.height / edge_persons.height if edge_persons.height else 0.0,
        "edge_schema": {name: str(dtype) for name, dtype in zip(edges.columns, edges.dtypes, strict=False)},
        "context_schema": {name: str(dtype) for name, dtype in zip(contexts.columns, contexts.dtypes, strict=False)},
        "text_columns_checked": text_columns,
        "sample_text_count": len(text_values),
        "sample_char_count": len(joined_sample),
        "sample_korean_char_count": len(KOREAN_RE.findall(joined_sample)),
        "sample_replacement_char_count": len(REPLACEMENT_RE.findall(joined_sample)),
        "sample_mojibake_hint_char_count": len(MOJIBAKE_HINT_RE.findall(joined_sample)),
        "sample_korean_char_ratio": (
            len(KOREAN_RE.findall(joined_sample)) / len(joined_sample) if joined_sample else 0.0
        ),
        "edge_sample": edges.head(3).to_dicts(),
        "context_person_uuid_sample": contexts.select("person_uuid").head(3).to_series().to_list(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
