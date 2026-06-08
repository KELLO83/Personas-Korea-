from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hobby_recommender_ml.hobby_recommender.experiment_analysis import (  # noqa: E402
    JsonValue,
    compare_metric_reports,
)


DEFAULT_METRICS = ("recall@10", "ndcg@10", "catalog_coverage@10", "novelty@10")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare existing hobby ranker metric JSON artifacts.")
    parser.add_argument("--run", action="append", required=True, help="Run as name=path")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--variant-key", default="v2_lightgbm_ranker")
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = {
        name: _extract_metric_report(_load_json(path), str(args.variant_key))
        for name, path in (_parse_run(value) for value in args.run)
    }
    metrics = tuple(args.metric) or DEFAULT_METRICS
    payload = {
        "baseline": args.baseline,
        "variant_key": args.variant_key,
        "metrics": list(metrics),
        "rows": compare_metric_reports(reports, str(args.baseline), metrics),
    }
    _write_json(args.output, payload)


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"--run must be name=path: {value}")
    name, path = value.split("=", 1)
    return name, Path(path)


def _extract_metric_report(payload: dict[str, JsonValue], variant_key: str) -> dict[str, JsonValue]:
    value = payload.get(variant_key)
    if isinstance(value, dict):
        metrics = value.get("metrics")
        if isinstance(metrics, dict):
            return {"metrics": metrics}
    metrics = payload.get("metrics")
    return {"metrics": metrics} if isinstance(metrics, dict) else payload


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
