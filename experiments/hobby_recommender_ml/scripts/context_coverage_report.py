from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hobby_recommender_ml.hobby_recommender.config import load_config  # noqa: E402
from experiments.hobby_recommender_ml.hobby_recommender.data import (  # noqa: E402
    build_domain_tagged_persona_text,
    load_person_contexts,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report split-aligned persona context coverage.")
    parser.add_argument("--config", type=Path, default=Path("experiments/hobby_recommender_ml/configs/lightgbm_ranker.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/hobby_recommender_ml/artifacts/experiments/phase5_context_coverage/context_coverage_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = _safe_torch_load(config.paths.checkpoint)
    person_to_id = _expect_mapping(checkpoint.get("person_to_id"), "person_to_id")
    id_to_person = {person_id: person_uuid for person_uuid, person_id in person_to_id.items()}
    contexts = load_person_contexts(config.paths.person_context_csv) if config.paths.person_context_csv.exists() else {}

    report = {
        "config_path": str(args.config),
        "person_context_csv": str(config.paths.person_context_csv),
        "total_mapped_persons": len(person_to_id),
        "total_context_rows": len(contexts),
        "splits": {
            "train": _split_coverage(config.paths.train_edges, id_to_person, contexts),
            "validation": _split_coverage(config.paths.validation_edges, id_to_person, contexts),
            "test": _split_coverage(config.paths.test_edges, id_to_person, contexts),
        },
    }
    report["overall"] = _overall_coverage(report["splits"])
    save_json(args.output, report)
    print(f"Context coverage report saved: {args.output}")
    for split, summary in report["splits"].items():
        print(
            f"{split}: persons={summary['person_count']} "
            f"context={summary['context_person_count']} "
            f"domain_text={summary['domain_text_person_count']} "
            f"coverage={summary['domain_text_coverage']:.4f}"
        )


def _safe_torch_load(path: Path) -> dict[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint {path} must contain a dictionary")
    return value


def _expect_mapping(value: object, name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint missing mapping: {name}")
    return {str(key): int(raw_value) for key, raw_value in value.items()}


def _read_person_ids(path: Path) -> set[int]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return {int(row["person_id"]) for row in reader}


def _split_coverage(path: Path, id_to_person: dict[int, str], contexts: dict[str, Any]) -> dict[str, object]:
    person_ids = _read_person_ids(path)
    missing_mapping: list[int] = []
    missing_context: list[int] = []
    empty_domain_text: list[int] = []
    covered: list[int] = []
    for person_id in sorted(person_ids):
        person_uuid = id_to_person.get(person_id, "")
        if not person_uuid:
            missing_mapping.append(person_id)
            continue
        context = contexts.get(person_uuid)
        if context is None:
            missing_context.append(person_id)
            continue
        if not build_domain_tagged_persona_text(context):
            empty_domain_text.append(person_id)
            continue
        covered.append(person_id)
    total = len(person_ids)
    return {
        "path": str(path),
        "person_count": total,
        "context_person_count": total - len(missing_mapping) - len(missing_context),
        "domain_text_person_count": len(covered),
        "missing_mapping_count": len(missing_mapping),
        "missing_context_count": len(missing_context),
        "empty_domain_text_count": len(empty_domain_text),
        "domain_text_coverage": len(covered) / total if total else 0.0,
        "missing_context_person_id_sample": missing_context[:100],
        "empty_domain_text_person_id_sample": empty_domain_text[:100],
    }


def _overall_coverage(splits: object) -> dict[str, object]:
    if not isinstance(splits, dict):
        return {}
    person_count = 0
    domain_text_person_count = 0
    for summary in splits.values():
        if isinstance(summary, dict):
            person_count += int(summary.get("person_count", 0))
            domain_text_person_count += int(summary.get("domain_text_person_count", 0))
    return {
        "person_count": person_count,
        "domain_text_person_count": domain_text_person_count,
        "domain_text_coverage": domain_text_person_count / person_count if person_count else 0.0,
    }


if __name__ == "__main__":
    main()
