from __future__ import annotations

"""
LEGACY / ANALYSIS-ONLY: runs two train/evaluate jobs in one invocation.

Do not use this as the default Phase 2.5 promotion path. The default path is
single-config train_ranker.py plus single-split evaluate_ranker.py, with one
explicit hyperparameter setting per invocation.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LightGBM ranker with and without source one-hot features.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("GNN_Neural_Network/artifacts"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--neg-ratio", type=int, default=4)
    parser.add_argument("--hard-ratio", type=float, default=0.8)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--ranker-val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = output_dir / "source_onehot_baseline"
    source_dir = output_dir / "source_onehot_enabled"
    baseline_eval = output_dir / "source_onehot_baseline_eval.json"
    source_eval = output_dir / "source_onehot_enabled_eval.json"

    _train_ranker(args, baseline_dir, include_source_features=False)
    _evaluate_ranker(args, baseline_dir / "ranker_model.txt", baseline_eval)

    _train_ranker(args, source_dir, include_source_features=True)
    _evaluate_ranker(args, source_dir / "ranker_model.txt", source_eval)

    baseline_payload = _read_json(baseline_eval)
    source_payload = _read_json(source_eval)
    comparison = _comparison_payload(args, baseline_payload, source_payload)

    output_path = output_dir / "source_onehot_ablation.json"
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSource one-hot ablation saved: {output_path}")


def _train_ranker(args: argparse.Namespace, output_dir: Path, include_source_features: bool) -> None:
    command = [
        sys.executable,
        str(ROOT / "GNN_Neural_Network" / "scripts" / "train_ranker.py"),
        "--config", str(args.config),
        "--output-dir", str(output_dir),
        "--neg-ratio", str(args.neg_ratio),
        "--hard-ratio", str(args.hard_ratio),
        "--num-boost-round", str(args.num_boost_round),
        "--early-stopping", str(args.early_stopping),
        "--ranker-val-ratio", str(args.ranker_val_ratio),
        "--seed", str(args.seed),
    ]
    if include_source_features:
        command.append("--include-source-features")
    label = "with source one-hot" if include_source_features else "without source one-hot"
    print(f"\nTraining ranker {label}...")
    _run(command)


def _evaluate_ranker(args: argparse.Namespace, model_path: Path, output_path: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "GNN_Neural_Network" / "scripts" / "evaluate_ranker.py"),
        "--config", str(args.config),
        "--split", args.split,
        "--model-path", str(model_path),
        "--output", str(output_path),
    ]
    print(f"\nEvaluating {model_path}...")
    _run(command)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _comparison_payload(
    args: argparse.Namespace,
    baseline_payload: dict[str, Any],
    source_payload: dict[str, Any],
) -> dict[str, Any]:
    baseline_metrics = _metrics(baseline_payload)
    source_metrics = _metrics(source_payload)
    return {
        "split": args.split,
        "neg_ratio": args.neg_ratio,
        "hard_ratio": args.hard_ratio,
        "baseline": {
            "include_source_features": False,
            "feature_columns": baseline_payload.get("feature_columns", []),
            "metrics": baseline_metrics,
        },
        "source_onehot": {
            "include_source_features": True,
            "feature_columns": source_payload.get("feature_columns", []),
            "metrics": source_metrics,
        },
        "delta_source_onehot_vs_baseline": {
            key: source_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0)
            for key in sorted(set(baseline_metrics) | set(source_metrics))
        },
        "artifacts": {
            "baseline_eval": "source_onehot_baseline_eval.json",
            "source_eval": "source_onehot_enabled_eval.json",
            "baseline_model_dir": "source_onehot_baseline",
            "source_model_dir": "source_onehot_enabled",
        },
    }


def _metrics(payload: dict[str, Any]) -> dict[str, float]:
    section = payload.get("v2_lightgbm_ranker", {})
    raw_metrics = section.get("metrics", {}) if isinstance(section, dict) else {}
    if not isinstance(raw_metrics, dict):
        return {}
    result: dict[str, float] = {}
    for key in (
        "recall@10",
        "ndcg@10",
        "hit_rate@10",
        "catalog_coverage@10",
        "novelty@10",
        "intra_list_diversity@10",
    ):
        value = raw_metrics.get(key)
        if isinstance(value, (int, float)):
            result[key] = float(value)
    return result


if __name__ == "__main__":
    main()
