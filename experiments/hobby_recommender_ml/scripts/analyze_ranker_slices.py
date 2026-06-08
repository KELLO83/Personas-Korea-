from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hobby_recommender_ml.hobby_recommender.experiment_analysis import (  # noqa: E402
    JsonValue,
    segment_gap_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-segment recall gaps from hobby ranker metrics.")
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--variant-key", default="v2_lightgbm_ranker")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = _load_json(args.metrics_path)
    metrics = _extract_metrics(payload, str(args.variant_key))
    rows = [asdict(row) for row in segment_gap_report(metrics)]
    _write_json(
        args.output,
        {
            "stage": "analyze_ranker_slices",
            "metrics_path": str(args.metrics_path),
            "variant_key": args.variant_key,
            "rows": rows,
        },
    )


def _extract_metrics(payload: dict[str, JsonValue], variant_key: str) -> dict[str, JsonValue]:
    variant = payload.get(variant_key)
    if isinstance(variant, dict):
        metrics = variant.get("metrics")
        if isinstance(metrics, dict):
            return metrics
    metrics = payload.get("metrics")
    return metrics if isinstance(metrics, dict) else payload


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
