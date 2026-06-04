from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.experiment_analysis import (  # noqa: E402
    JsonValue,
    alias_audit_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase 6 alias/domain-text promotion caveats.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--train-status", type=Path, required=True)
    parser.add_argument("--validation-metrics", type=Path, required=True)
    parser.add_argument("--test-metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    test_metrics = _load_json(args.test_metrics) if args.test_metrics else None
    report = alias_audit_report(
        experiment_id=str(args.experiment_id),
        train_status=_load_json(args.train_status),
        validation_metrics=_load_json(args.validation_metrics),
        test_metrics=test_metrics,
    )
    _write_json(args.output, {"stage": "audit_phase6_alias_features", **report})


def _load_json(path: Path) -> dict[str, JsonValue]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
