from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.persona_similarity.scripts.review_analysis import (  # noqa: E402
    compare_experiment_metrics,
)


DEFAULT_METRICS = (
    "ndcg@10",
    "strong_reason_coverage@10",
    "low_information_dominance@10",
    "occupation_diversity@10",
    "province_diversity@10",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare structured and text-feature similar-persona metrics.")
    parser.add_argument("--run", action="append", required=True, help="Run as name=path")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = {name: _load_metric_report(path) for name, path in (_parse_run(value) for value in args.run)}
    metrics = tuple(args.metric) or DEFAULT_METRICS
    rows = compare_experiment_metrics(reports, baseline=str(args.baseline), metrics=metrics)
    _write_json(
        args.output,
        {
            "stage": "analyze_text_feature_value",
            "baseline": args.baseline,
            "metrics": list(metrics),
            "rows": rows,
        },
    )


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"--run must be name=path: {value}")
    name, path = value.split("=", 1)
    return name, Path(path)


def _load_metric_report(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict) or not isinstance(value.get("metrics"), dict):
        raise SystemExit(f"Metrics JSON must contain object field 'metrics': {path}")
    metrics = {
        key: float(metric_value)
        for key, metric_value in value["metrics"].items()
        if isinstance(metric_value, int | float)
    }
    return {"metrics": metrics}


def _write_json(path: Path, payload: dict[str, str | list[dict[str, str | float]] | list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
