from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.persona_similarity.scripts.review_analysis import (  # noqa: E402
    classify_failure_modes,
    summarize_failure_taxonomy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label similar-persona manual-review rows with failure taxonomy.")
    parser.add_argument("--review-csv", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    labeled_rows: list[dict[str, str | tuple[str, ...]]] = []
    for review_path in [Path(path) for path in args.review_csv]:
        for row in _read_rows(review_path):
            modes = classify_failure_modes(row)
            labeled_rows.append({**row, "source_file": str(review_path), "failure_modes": modes})

    _write_labeled_csv(args.output_csv, labeled_rows)
    summary = summarize_failure_taxonomy(labeled_rows)
    _write_json(args.output_json, {"stage": "build_failure_taxonomy", **summary})


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _write_labeled_csv(path: Path, rows: list[dict[str, str | tuple[str, ...]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [*rows[0].keys()]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {
                key: "|".join(value) if isinstance(value, tuple) else value
                for key, value in row.items()
            }
            writer.writerow(serializable)


def _write_json(path: Path, payload: dict[str, int | dict[str, int] | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
