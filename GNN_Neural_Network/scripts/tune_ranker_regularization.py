"""LightGBM Regularization Hyperparameter Tuning Script.

⚠️ LEGACY / ANALYSIS-ONLY ⚠️

Per PRD §2.5 execution policy, hyperparameter sweep scripts are no longer part
of the default experiment path. Use `train_ranker.py` with single-config CLI
params + `evaluate_ranker.py --split validation` instead.

This script is retained for ad-hoc exploration / historical comparison only.
Do NOT use it for promotion decisions.

Sequential greedy search over regularization parameters to reduce overfitting
on small data while maintaining or improving validation metrics.

Strategy:
1. Start with baseline (current production) params
2. Vary one param at a time, keeping others at best-so-far
3. Pick best value per param based on validation recall@10
4. Repeat until no improvement
5. Evaluate final config on test split

Usage (legacy only):
    python -m GNN_Neural_Network.scripts.tune_ranker_regularization
    python GNN_Neural_Network/scripts/tune_ranker_regularization.py
"""

from __future__ import annotations

import ast
import argparse
import csv
import random
import shlex
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.baseline import (  # noqa: E402
    build_cooccurrence_counts,
    build_popularity_counts,
    cooccurrence_candidate_provider,
    popularity_candidate_provider,
)
from GNN_Neural_Network.gnn_recommender.config import load_config  # noqa: E402
from GNN_Neural_Network.gnn_recommender.data import load_json, load_person_contexts, save_json  # noqa: E402
from GNN_Neural_Network.gnn_recommender.metrics import summarize_ranking_metrics  # noqa: E402
from GNN_Neural_Network.gnn_recommender.ranker import LightGBMRanker, build_ranker_dataset  # noqa: E402
from GNN_Neural_Network.gnn_recommender.recommend import merge_candidates_by_hobby, normalize_candidate_scores  # noqa: E402
from GNN_Neural_Network.gnn_recommender.rerank import HobbyCandidate, build_reranker_config, merge_stage1_candidates  # noqa: E402

# Promotion gate thresholds (PRD §6)
RECALL_GATE = -0.002
NDCG_GATE = 0.005

# Baseline params (current production)
BASELINE_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 15,
    "min_data_in_leaf": 50,
    "learning_rate": 0.05,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "seed": 42,
    "num_threads": -1,
}

# Tuning grid: param -> list of values to try
TUNING_GRID: dict[str, list[Any]] = {
    "num_leaves": [7, 15, 31],
    "min_data_in_leaf": [20, 50, 100],
    "max_depth": [3, 5, -1],  # -1 = no limit
    "feature_fraction": [0.7, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.7, 0.8, 1.0],
    "reg_alpha": [0.05, 0.1, 0.5, 1.0],
    "reg_lambda": [0.05, 0.1, 0.5, 1.0],
}

_TQDM_KWARGS = {
    "miniters": 200,
    "mininterval": 5.0,
    "maxinterval": 30.0,
    "dynamic_ncols": False,
    "ascii": True,
    "leave": False,
    "file": sys.stderr,
}


def _iter_with_progress(args, iterable, desc: str):
    if args.progress_mode == "off":
        return iterable
    if args.progress_mode == "auto" and not sys.stderr.isatty():
        return iterable

    try:
        total = len(iterable)  # type: ignore[arg-type]
    except Exception:
        total = None

    kwargs = dict(_TQDM_KWARGS)
    kwargs.update(
        {
            "desc": desc,
            "total": total,
            "mininterval": float(args.progress_mininterval),
            "maxinterval": float(args.progress_maxinterval),
            "miniters": int(args.progress_miniters),
        }
    )
    return tqdm(iterable, **kwargs)


def _log_policy(args: argparse.Namespace) -> dict[str, object]:
    return {
        "progress_mode": args.progress_mode,
        "progress_mininterval": float(args.progress_mininterval),
        "progress_maxinterval": float(args.progress_maxinterval),
        "progress_miniters": int(args.progress_miniters),
        "tqdm_enabled": args.progress_mode != "off",
    }


def _command_signature() -> str:
    return " ".join([Path(sys.argv[0]).name, *(shlex.quote(arg) for arg in sys.argv[1:])])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune LightGBM regularization parameters.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("GNN_Neural_Network/artifacts"))
    parser.add_argument("--neg-ratio", type=int, default=4)
    parser.add_argument("--hard-ratio", type=float, default=0.8)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--ranker-val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=3, help="Max sequential greedy rounds")
    parser.add_argument(
        "--tuning-params",
        type=str,
        default="",
        help="Comma-separated subset of tuning params to evaluate. Defaults to the full grid.",
    )
    parser.add_argument(
        "--tuning-values",
        type=str,
        default="",
        help=(
            "Optional explicit override values in the form "
            "'num_leaves=31' or 'num_leaves=15,31;reg_alpha=0.1'. "
            "Only selected --tuning-params are allowed."
        ),
    )
    parser.add_argument("--include-source-features", action="store_true")
    parser.add_argument("--experiment-id", type=str, default="", help="Optional experiment identifier")
    parser.add_argument(
        "--progress-mode",
        choices=["auto", "on", "off"],
        default="off",
        help="Progress output mode: off (default), auto (tty only), on (always).",
    )
    parser.add_argument("--progress-mininterval", type=float, default=5.0, help="Minimum seconds between progress updates")
    parser.add_argument("--progress-maxinterval", type=float, default=30.0, help="Maximum seconds between progress updates")
    parser.add_argument("--progress-miniters", type=int, default=200, help="Minimum updates between progress refresh")
    return parser.parse_args()


def _status_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "regularization_tuning.status.json"


def _write_status(
    args: argparse.Namespace,
    status: str,
    runtime_seconds: float | None = None,
    summary: dict[str, object] | None = None,
    input_config_summary: dict[str, object] | None = None,
) -> None:
    status_path = _status_path(args)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "experiment_id": args.experiment_id,
        "status": status,
        "command_signature": _command_signature(),
        "log_policy": _log_policy(args),
        "event_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_path": str(status_path),
    }
    if runtime_seconds is not None:
        payload["runtime_seconds"] = runtime_seconds
    if summary is not None:
        payload["summary"] = summary
    if input_config_summary is not None:
        payload["input_config_summary"] = input_config_summary
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_tuning_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        raise ValueError("Tuning value is empty")

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        if "." in value or "e" in lowered or "E" in value:
            try:
                return float(value)
            except ValueError:
                return value
        try:
            return int(value)
        except ValueError:
            return value
    return parsed


def _parse_tuning_override_values(raw_values: str) -> dict[str, list[Any]]:
    override_map: dict[str, list[Any]] = {}
    for token in raw_values.split(";"):
        item = token.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid tuning-values token: {item}")
        name, raw_v = [x.strip() for x in item.split("=", 1)]
        if not name:
            raise ValueError(f"Invalid tuning-values token with empty param: {item}")
        if name not in TUNING_GRID:
            raise ValueError(f"Unknown tuning param in --tuning-values: {name}")
        if not raw_v:
            raise ValueError(f"No tuning values provided for {name} in --tuning-values")

        values = [_parse_tuning_value(v) for v in raw_v.split(",") if v.strip()]
        if not values:
            raise ValueError(f"No tuning values provided for {name} in --tuning-values")

        override_map[name] = values
    return override_map


def _selected_tuning_grid(raw_names: str, raw_values: str) -> dict[str, list[Any]]:
    if not raw_names.strip():
        selected_names = list(TUNING_GRID.keys())
    else:
        selected_names = [name.strip() for name in raw_names.split(",") if name.strip()]
        unknown = sorted(set(selected_names) - set(TUNING_GRID))
        if unknown:
            raise ValueError(f"Unknown tuning params: {', '.join(unknown)}")

    selected = {name: TUNING_GRID[name] for name in selected_names}

    if raw_values.strip():
        overrides = _parse_tuning_override_values(raw_values)
        unknown_override = sorted(set(overrides) - set(selected))
        if unknown_override:
            raise ValueError(
                "Cannot override non-selected tuning params: "
                f"{', '.join(unknown_override)}"
            )
        for name, values in overrides.items():
            selected[name] = values

    return selected


# ---------------------------------------------------------------------------
# Data loading helpers (reused from train_ranker.py)
# ---------------------------------------------------------------------------

def _safe_torch_load(path: Path) -> dict[str, Any]:
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
    return {str(k): int(v) for k, v in value.items()}


def _read_indexed_edges(path: Path) -> list[tuple[int, int]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [(int(row["person_id"]), int(row["hobby_id"])) for row in reader]


def _known_from_edges(edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    known: dict[int, set[int]] = defaultdict(set)
    for pid, hid in edges:
        known[pid].add(hid)
    return dict(known)


def _normalization_method(path: Path) -> str:
    if not path.exists():
        return "rank_percentile"
    value = load_json(path)
    if not isinstance(value, dict):
        return "rank_percentile"
    return str(value.get("method", "rank_percentile"))


# ---------------------------------------------------------------------------
# Candidate pool generation (reused from evaluate_ranker.py)
# ---------------------------------------------------------------------------

def _generate_candidate_pools(
    args: argparse.Namespace,
    person_ids: list[int],
    train_edges: list[tuple[int, int]],
    train_known: dict[int, set[int]],
    candidate_k: int,
    id_to_hobby: dict[int, str],
    popularity_counts: Counter[int],
    cooccurrence_counts: dict[int, Counter[int]],
    normalization_method: str,
) -> dict[int, list[HobbyCandidate]]:
    pools: dict[int, list[HobbyCandidate]] = {}
    for person_id in _iter_with_progress(args, person_ids, "candidate pools"):
        known = train_known.get(person_id, set())
        pop = normalize_candidate_scores(
            popularity_candidate_provider(train_edges, person_id, known, candidate_k, popularity_counts=popularity_counts),
            normalization_method,
        )
        cooc = normalize_candidate_scores(
            cooccurrence_candidate_provider(train_edges, person_id, known, candidate_k, cooccurrence_counts=cooccurrence_counts),
            normalization_method,
        )
        merged = merge_candidates_by_hobby({"popularity": pop, "cooccurrence": cooc}, candidate_k)
        pools[person_id] = merge_stage1_candidates(merged, id_to_hobby)
    return pools


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _build_hobby_categories(
    id_to_hobby: dict[int, str],
    hobby_taxonomy: dict[str, object] | None,
) -> dict[int, str]:
    if hobby_taxonomy is None:
        return {}
    taxonomy_map = hobby_taxonomy.get("taxonomy", {})
    rules = hobby_taxonomy.get("rules", [])
    result: dict[int, str] = {}
    for hobby_id, hobby_name in id_to_hobby.items():
        category = ""
        if isinstance(taxonomy_map, dict):
            entry = taxonomy_map.get(hobby_name, {})
            if isinstance(entry, dict):
                category = str(entry.get("category", ""))
        if not category and isinstance(rules, list):
            for rule in rules:
                if isinstance(rule, dict) and rule.get("canonical_hobby") == hobby_name:
                    tax = rule.get("taxonomy", {})
                    if isinstance(tax, dict):
                        category = str(tax.get("category", ""))
                    break
        if category:
            result[hobby_id] = category
    return result


def _build_person_segments(
    person_ids,
    id_to_person: dict[int, str],
    contexts: dict,
) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for person_id in person_ids:
        person_uuid = id_to_person.get(person_id, "")
        ctx = contexts.get(person_uuid)
        if ctx is not None:
            result[person_id] = {
                "age_group": ctx.age_group,
                "sex": ctx.sex,
            }
    return result


def _metric_value(metrics: dict[str, object], key: str) -> float:
    value = metrics.get(key, 0.0)
    return float(value) if isinstance(value, (int, float, str)) else 0.0


# ---------------------------------------------------------------------------
# Train + evaluate a single config
# ---------------------------------------------------------------------------

def train_and_evaluate(
    args: argparse.Namespace,
    params: dict[str, Any],
    train_ds,
    val_ds,
    train_lgb,
    val_lgb,
    split_name: str,
    truth: dict[int, set[int]],
    pools: dict[int, list[HobbyCandidate]],
    all_hobby_ids: list[int],
    train_known: dict[int, set[int]],
    id_to_hobby: dict[int, str],
    id_to_person: dict[int, str],
    contexts: dict,
    hobby_profile: dict,
    reranker_config,
    hobby_categories: dict[int, str],
    person_segments: dict[int, dict[str, str]],
    num_hobbies: int,
    popularity_counts: Counter[int],
    config,
    candidate_k: int,
    max_k: int,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> dict[str, Any]:
    """Train LightGBM with given params and evaluate on the specified split."""
    ranker = LightGBMRanker(params=params)
    metadata = ranker.fit(train_lgb, val_lgb, num_boost_round, early_stopping_rounds)
    model_feature_columns = ranker.feature_columns()

    v2_rankings: dict[int, list[int]] = {}
    v1_rankings: dict[int, list[int]] = {}
    from GNN_Neural_Network.gnn_recommender.rerank import build_rerank_features, rerank_candidates

    for person_id in _iter_with_progress(args, truth, f"eval ({split_name})"):
        known = train_known.get(person_id, set())
        pool_candidates = pools.get(person_id, [])
        if not pool_candidates:
            continue

        known_names = {id_to_hobby[hid] for hid in known if hid in id_to_hobby}
        reranked = rerank_candidates(
            contexts.get(id_to_person.get(person_id, "")),
            pool_candidates,
            hobby_profile,
            known_names,
            reranker_config,
            hobby_taxonomy=None,
        )
        v1_rankings[person_id] = [c.hobby_id for c in reranked[:max_k]]

        person_uuid = id_to_person.get(person_id, "")
        person_context = contexts.get(person_uuid)

        if person_context and pool_candidates and hobby_profile:
            features_list: list[list[float]] = []
            hobby_ids_list: list[int] = []
            for candidate in pool_candidates:
                features = build_rerank_features(
                    person_context, candidate, hobby_profile, known_names, reranker_config,
                )
                features.pop("similar_person_score", None)
                features.pop("persona_text_fit", None)
                features_list.append([features.get(col, 0.0) for col in model_feature_columns])
                hobby_ids_list.append(candidate.hobby_id)

            X = np.array(features_list, dtype=np.float32)
            scores = ranker.predict(X)
            sorted_indices = np.argsort(-scores)
            v2_rankings[person_id] = [hobby_ids_list[int(i)] for i in sorted_indices[:max_k]]
        else:
            v2_rankings[person_id] = [c.hobby_id for c in pool_candidates[:max_k]]

    # Compute metrics
    v1_metrics = summarize_ranking_metrics(
        truth, v1_rankings, config.eval.top_k,
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, person_segments=person_segments,
    )
    v2_metrics = summarize_ranking_metrics(
        truth, v2_rankings, config.eval.top_k,
        num_total_items=num_hobbies, item_popularity=popularity_counts,
        hobby_categories=hobby_categories, person_segments=person_segments,
    )

    return {
        "params": params,
        "best_iteration": metadata["best_iteration"],
        "best_auc": metadata["best_score"],
        "train_auc": metadata["best_score"],
        "v1_recall@10": _metric_value(v1_metrics, "recall@10"),
        "v1_ndcg@10": _metric_value(v1_metrics, "ndcg@10"),
        "v2_recall@10": _metric_value(v2_metrics, "recall@10"),
        "v2_ndcg@10": _metric_value(v2_metrics, "ndcg@10"),
        "v2_coverage@10": _metric_value(v2_metrics, "catalog_coverage@10"),
        "v2_novelty@10": _metric_value(v2_metrics, "novelty@10"),
        "delta_recall@10": _metric_value(v2_metrics, "recall@10") - _metric_value(v1_metrics, "recall@10"),
        "delta_ndcg@10": _metric_value(v2_metrics, "ndcg@10") - _metric_value(v1_metrics, "ndcg@10"),
        "v2_metrics": v2_metrics,
        "v1_metrics": v1_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    config = load_config(args.config)
    tuning_grid = _selected_tuning_grid(args.tuning_params, args.tuning_values)
    candidate_k = config.rerank.candidate_pool_size
    if candidate_k <= 0:
        raise ValueError("candidate_pool_size must be positive")

    _write_status(
        args,
        "started",
        runtime_seconds=0.0,
        summary={
            "phase": "started",
            "config_path": str(args.config),
            "max_rounds": args.max_rounds,
        },
        input_config_summary={"candidate_pool_size": candidate_k},
    )

    # ------------------------------------------------------------------
    # 1. Load data (same as train_ranker.py)
    # ------------------------------------------------------------------
    checkpoint = _safe_torch_load(config.paths.checkpoint)
    person_to_id = _expect_mapping(checkpoint.get("person_to_id"), "person_to_id")
    hobby_to_id = _expect_mapping(checkpoint.get("hobby_to_id"), "hobby_to_id")
    id_to_hobby = {v: k for k, v in hobby_to_id.items()}
    id_to_person = {v: k for k, v in person_to_id.items()}

    train_edges = _read_indexed_edges(config.paths.train_edges)
    val_edges = _read_indexed_edges(config.paths.validation_edges)
    test_edges = _read_indexed_edges(config.paths.test_edges)
    train_known = _known_from_edges(train_edges)

    contexts = load_person_contexts(config.paths.person_context_csv) if config.paths.person_context_csv.exists() else {}
    hobby_profile = load_json(config.paths.hobby_profile) if config.paths.hobby_profile.exists() else None
    if not isinstance(hobby_profile, dict):
        raise ValueError("hobby_profile.json required")

    hobby_taxonomy = None
    # Try loading hobby_taxonomy directly
    for p in (config.paths.hobby_taxonomy, config.paths.artifact_dir / "hobby_taxonomy.json"):
        if p.exists():
            val = load_json(p)
            if isinstance(val, dict):
                hobby_taxonomy = val
                break

    normalization_method = _normalization_method(config.paths.score_normalization)
    reranker_config = build_reranker_config(config.rerank.use_text_fit, config.rerank.weights)

    popularity_counts = build_popularity_counts(train_edges)
    cooccurrence_counts = build_cooccurrence_counts(train_edges)
    all_hobby_ids = list(hobby_to_id.values())

    # ------------------------------------------------------------------
    # 2. Build ranker datasets (same split as train_ranker.py)
    # ------------------------------------------------------------------
    val_person_ids = sorted({pid for pid, _ in val_edges})
    rng = random.Random(args.seed)
    shuffled = list(val_person_ids)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1.0 - args.ranker_val_ratio)))
    ranker_train_persons = set(shuffled[:split_idx])
    ranker_val_persons = set(shuffled[split_idx:])

    ranker_train_edges = [(pid, hid) for pid, hid in val_edges if pid in ranker_train_persons]
    ranker_val_edges = [(pid, hid) for pid, hid in val_edges if pid in ranker_val_persons]

    print(f"Ranker split: {len(ranker_train_persons)} train, {len(ranker_val_persons)} val")

    train_pools = _generate_candidate_pools(
        args,
        val_person_ids, train_edges, train_known, candidate_k,
        id_to_hobby, popularity_counts, cooccurrence_counts, normalization_method,
    )

    print("Building ranker train dataset...")
    train_ds = build_ranker_dataset(
        ranker_train_edges, train_pools, all_hobby_ids, train_known,
        id_to_hobby, contexts, id_to_person, hobby_profile, reranker_config,
        neg_ratio=args.neg_ratio, hard_ratio=args.hard_ratio, seed=args.seed,
        include_source_features=args.include_source_features,
    )
    print(f"  train rows={len(train_ds.rows)}")

    print("Building ranker val dataset...")
    val_ds = build_ranker_dataset(
        ranker_val_edges, train_pools, all_hobby_ids, train_known,
        id_to_hobby, contexts, id_to_person, hobby_profile, reranker_config,
        neg_ratio=args.neg_ratio, hard_ratio=args.hard_ratio, seed=args.seed + 1,
        include_source_features=args.include_source_features,
    )
    print(f"  val rows={len(val_ds.rows)}")
    _write_status(
        args,
        "datasets_ready",
        summary={
            "phase": "datasets_ready",
            "train_rows": len(train_ds.rows),
            "val_rows": len(val_ds.rows),
            "num_val_persons": len(val_person_ids),
            "num_test_persons": len(test_edges),
        },
        input_config_summary={"candidate_pool_size": candidate_k, "top_k": max(config.eval.top_k)},
    )

    train_lgb = train_ds.to_lgb_dataset()
    val_lgb = val_ds.to_lgb_dataset(reference=train_lgb)

    # ------------------------------------------------------------------
    # 3. Sequential greedy search
    # ------------------------------------------------------------------
    best_params = dict(BASELINE_PARAMS)
    best_recall = 0.0
    best_ndcg = 0.0
    all_results: list[dict[str, Any]] = []
    tuning_log: list[dict[str, Any]] = []

    # Compute v1 baseline once (same across all configs)
    print("\n=== Phase 0: Compute v1 deterministic baseline ===")

    val_truth = _known_from_edges(val_edges)
    test_truth = _known_from_edges(test_edges)

    val_hobby_categories = _build_hobby_categories(id_to_hobby, hobby_taxonomy)
    val_person_segments_val = _build_person_segments(val_truth.keys(), id_to_person, contexts)
    val_person_segments_test = _build_person_segments(test_truth.keys(), id_to_person, contexts)
    num_hobbies = len(hobby_to_id)
    max_k = max(config.eval.top_k)

    # Generate pools for val and test splits
    print("Generating candidate pools for validation split...")
    val_pools = _generate_candidate_pools(
        args,
        sorted(val_truth.keys()), train_edges, train_known, candidate_k,
        id_to_hobby, popularity_counts, cooccurrence_counts, normalization_method,
    )
    print("Generating candidate pools for test split...")
    test_pools = _generate_candidate_pools(
        args,
        sorted(test_truth.keys()), train_edges, train_known, candidate_k,
        id_to_hobby, popularity_counts, cooccurrence_counts, normalization_method,
    )

    # Run baseline first
    print("\n=== Baseline configuration ===")
    print(f"  Params: {best_params}")
    baseline_result = train_and_evaluate(
        args, best_params, train_ds, val_ds, train_lgb, val_lgb,
        "validation", val_truth, val_pools, all_hobby_ids, train_known,
        id_to_hobby, id_to_person, contexts, hobby_profile, reranker_config,
        val_hobby_categories, val_person_segments_val, num_hobbies,
        popularity_counts, config, candidate_k, max_k,
        num_boost_round=args.num_boost_round, early_stopping_rounds=args.early_stopping,
    )
    best_recall = baseline_result["v2_recall@10"]
    best_ndcg = baseline_result["v2_ndcg@10"]
    all_results.append({"round": 0, "param_changed": "baseline", **baseline_result})
    print(f"  Baseline: val recall@10={best_recall:.4f}, ndcg@10={best_ndcg:.4f}, AUC={baseline_result['best_auc']:.4f}")
    _write_status(
        args,
        "baseline_done",
        summary={
            "phase": "baseline_done",
            "baseline_recall@10": _metric_value(baseline_result["v2_metrics"], "recall@10") if isinstance(baseline_result.get("v2_metrics"), dict) else baseline_result.get("v2_recall@10", 0.0),
            "baseline_ndcg@10": _metric_value(baseline_result["v2_metrics"], "ndcg@10") if isinstance(baseline_result.get("v2_metrics"), dict) else baseline_result.get("v2_ndcg@10", 0.0),
            "baseline_auc": baseline_result.get("best_auc", 0.0),
            "best_params": best_params,
        },
        input_config_summary={"candidate_pool_size": candidate_k, "top_k": max(config.eval.top_k)},
    )

    # Greedy tuning rounds
    for round_num in range(1, args.max_rounds + 1):
        print(f"\n=== Round {round_num} ===")
        improved = False

        for param_name, values in tuning_grid.items():
            print(f"\n--- Tuning {param_name} ---")
            round_best_value = best_params.get(param_name)
            round_best_recall = best_recall
            round_best_ndcg = best_ndcg

            for value in values:
                trial_params = dict(best_params)
                trial_params[param_name] = value
                # bagging_fraction needs bagging_freq > 0 to take effect
                if param_name == "bagging_fraction" and value < 1.0:
                    trial_params["bagging_freq"] = 5

                print(f"  Trying {param_name}={value}...", end=" ", flush=True)

                result = train_and_evaluate(
                    args, trial_params, train_ds, val_ds, train_lgb, val_lgb,
                    "validation", val_truth, val_pools, all_hobby_ids, train_known,
                    id_to_hobby, id_to_person, contexts, hobby_profile, reranker_config,
                    val_hobby_categories, val_person_segments_val, num_hobbies,
                    popularity_counts, config, candidate_k, max_k,
                    num_boost_round=args.num_boost_round, early_stopping_rounds=args.early_stopping,
                )

                recall = result["v2_recall@10"]
                ndcg = result["v2_ndcg@10"]
                auc = result["best_auc"]
                print(f"recall@10={recall:.4f} ndcg@10={ndcg:.4f} AUC={auc:.4f}")

                all_results.append({
                    "round": round_num,
                    "param_changed": param_name,
                    "value_tested": value,
                    **result,
                })

                # Select best based on recall@10 (primary metric)
                if recall > round_best_recall + 1e-6:
                    round_best_recall = recall
                    round_best_value = value
                    round_best_ndcg = ndcg

            # Check if this param improved over current best
            if round_best_recall > best_recall + 1e-6:
                print(f"  ** {param_name}={round_best_value} improved recall@10: {best_recall:.4f} -> {round_best_recall:.4f}")
                best_params[param_name] = round_best_value
                best_recall = round_best_recall
                best_ndcg = round_best_ndcg
                tuning_log.append({
                    "round": round_num,
                    "param_changed": param_name,
                    "selected_value": round_best_value,
                    "best_recall@10": best_recall,
                    "best_ndcg@10": best_ndcg,
                })
                improved = True
            else:
                print(f"  No improvement from {param_name} (best recall@10={round_best_recall:.4f} vs current={best_recall:.4f})")

        if not improved:
            print(f"\nNo improvement in round {round_num}. Stopping early.")
            _write_status(
                args,
                "round_stopped",
                summary={
                    "phase": "round_stopped",
                    "round": round_num,
                    "best_params": dict(best_params),
                    "best_recall@10": best_recall,
                    "best_ndcg@10": best_ndcg,
                    "status": "no_improvement",
                },
            )
            break
        _write_status(
            args,
            "round_done",
            summary={
                "phase": "round_done",
                "round": round_num,
                "best_params": dict(best_params),
                "best_recall@10": best_recall,
                "best_ndcg@10": best_ndcg,
            },
        )

    # ------------------------------------------------------------------
    # 4. Final evaluation: best config on validation AND test
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION: Best regularization config")
    print("=" * 60)
    _write_status(
        args,
        "final_evaluation_started",
        summary={"phase": "final_evaluation_started", "splits": ["validation", "test"], "rounds": args.max_rounds},
    )

    # Re-evaluate best config on validation
    val_result = train_and_evaluate(
        args, best_params, train_ds, val_ds, train_lgb, val_lgb,
        "validation", val_truth, val_pools, all_hobby_ids, train_known,
        id_to_hobby, id_to_person, contexts, hobby_profile, reranker_config,
        val_hobby_categories, val_person_segments_val, num_hobbies,
        popularity_counts, config, candidate_k, max_k,
        num_boost_round=args.num_boost_round, early_stopping_rounds=args.early_stopping,
    )

    # Evaluate best config on test
    test_result = train_and_evaluate(
        args, best_params, train_ds, val_ds, train_lgb, val_lgb,
        "test", test_truth, test_pools, all_hobby_ids, train_known,
        id_to_hobby, id_to_person, contexts, hobby_profile, reranker_config,
        val_hobby_categories, val_person_segments_test, num_hobbies,
        popularity_counts, config, candidate_k, max_k,
        num_boost_round=args.num_boost_round, early_stopping_rounds=args.early_stopping,
    )
    _write_status(
        args,
        "final_evaluation_done",
        summary={
            "phase": "final_evaluation_done",
            "validation_recall@10": val_result["v2_recall@10"],
            "test_recall@10": test_result["v2_recall@10"],
            "validation_ndcg@10": val_result["v2_ndcg@10"],
            "test_ndcg@10": test_result["v2_ndcg@10"],
            "best_params": dict(best_params),
        },
    )

    # Overfitting gap
    overfitting_gap = {
        "train_auc": val_result["best_auc"],
        "val_recall_vs_test_recall_diff": val_result["v2_recall@10"] - test_result["v2_recall@10"],
        "val_ndcg_vs_test_ndcg_diff": val_result["v2_ndcg@10"] - test_result["v2_ndcg@10"],
    }

    # Promotion decision
    recall_delta = test_result["delta_recall@10"]
    ndcg_delta = test_result["delta_ndcg@10"]
    recall_pass = recall_delta >= RECALL_GATE
    ndcg_pass = ndcg_delta >= NDCG_GATE
    status = "promoted" if (recall_pass and ndcg_pass) else "blocked"

    promotion_decision = {
        "status": status,
        "recall_gate": {"threshold": RECALL_GATE, "actual": recall_delta, "pass": recall_pass},
        "ndcg_gate": {"threshold": NDCG_GATE, "actual": ndcg_delta, "pass": ndcg_pass},
    }

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "baseline_params": BASELINE_PARAMS,
        "best_params": best_params,
        "log_policy": _log_policy(args),
        "command_signature": _command_signature(),
        "tuning_log": tuning_log,
        "all_results": all_results,
        "final_validation": val_result,
        "final_test": test_result,
        "overfitting_gap": overfitting_gap,
        "promotion_decision": promotion_decision,
    }

    save_json(output_dir / "regularization_tuning.json", payload)
    _write_status(
        args,
        "results_saved",
        summary={
            "phase": "results_saved",
            "artifact_path": str(output_dir / "regularization_tuning.json"),
            "summary_path": str(output_dir / "regularization_tuning_summary.md"),
            "status": status,
            "best_recall_delta@10": test_result["delta_recall@10"],
            "best_ndcg_delta@10": test_result["delta_ndcg@10"],
            "best_params": dict(best_params),
        },
        input_config_summary={"candidate_pool_size": candidate_k, "top_k": max_k},
    )

    # Human-readable summary
    summary_lines = [
        "# LightGBM Regularization Tuning Summary",
        "",
        "## Baseline Configuration",
        "",
    ]
    for k, v in BASELINE_PARAMS.items():
        summary_lines.append(f"- **{k}**: {v}")
    summary_lines.extend([
        "",
        "## Best Configuration Found",
        "",
    ])
    for k, v in best_params.items():
        changed = "**(changed)**" if v != BASELINE_PARAMS.get(k) else ""
        summary_lines.append(f"- **{k}**: {v} {changed}")
    summary_lines.extend([
        "",
        "## Results Comparison",
        "",
        "| Metric | Baseline (val) | Best (val) | Best (test) | Delta vs v1 (test) |",
        "|:---|:---|:---|:---|:---|",
        f"| recall@10 | {baseline_result['v2_recall@10']:.4f} | {val_result['v2_recall@10']:.4f} | {test_result['v2_recall@10']:.4f} | {test_result['delta_recall@10']:+.4f} |",
        f"| ndcg@10 | {baseline_result['v2_ndcg@10']:.4f} | {val_result['v2_ndcg@10']:.4f} | {test_result['v2_ndcg@10']:.4f} | {test_result['delta_ndcg@10']:+.4f} |",
        f"| AUC | {baseline_result['best_auc']:.4f} | {val_result['best_auc']:.4f} | — | — |",
        f"| best_iteration | {baseline_result['best_iteration']} | {val_result['best_iteration']} | — | — |",
        "",
        "## Overfitting Analysis",
        "",
        f"- Train AUC: {val_result['best_auc']:.4f}",
        f"- Val ↔ Test recall@10 gap: {overfitting_gap['val_recall_vs_test_recall_diff']:+.4f}",
        f"- Val ↔ Test ndcg@10 gap: {overfitting_gap['val_ndcg_vs_test_ndcg_diff']:+.4f}",
        "",
        "## Promotion Decision",
        "",
        f"- **Status**: {status}",
        f"- recall@10 gate: delta={recall_delta:+.6f} (>= {RECALL_GATE}) → {'PASS' if recall_pass else 'FAIL'}",
        f"- ndcg@10 gate: delta={ndcg_delta:+.6f} (>= {NDCG_GATE}) → {'PASS' if ndcg_pass else 'FAIL'}",
        "",
        "## All Tested Configurations",
        "",
    ])
    for r in all_results:
        p = r.get("params", {})
        changed_param = r.get("param_changed", "")
        val = r.get("value_tested", "")
        summary_lines.append(
            f"- Round {r.get('round', 0)}, {changed_param}={val}: "
            f"recall@10={r.get('v2_recall@10', 0):.4f}, ndcg@10={r.get('v2_ndcg@10', 0):.4f}, "
            f"AUC={r.get('best_auc', 0):.4f}, iter={r.get('best_iteration', 0)}"
        )

    (output_dir / "regularization_tuning_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nResults saved: {output_dir / 'regularization_tuning.json'}")
    print(f"Summary saved: {output_dir / 'regularization_tuning_summary.md'}")
    print(f"\nBest params: {best_params}")
    print(f"Best val recall@10: {val_result['v2_recall@10']:.4f}")
    print(f"Test recall@10: {test_result['v2_recall@10']:.4f}")
    print(f"Promotion: {status}")
    _write_status(
        args,
        "completed",
        runtime_seconds=time.perf_counter() - start_time,
        summary={
            "phase": "completed",
            "status": status,
            "best_recall@10": test_result["v2_recall@10"],
            "best_ndcg@10": test_result["v2_ndcg@10"],
            "recall_delta@10": test_result["delta_recall@10"],
            "ndcg_delta@10": test_result["delta_ndcg@10"],
            "result_artifacts": {
                "json": str(output_dir / "regularization_tuning.json"),
                "summary": str(output_dir / "regularization_tuning_summary.md"),
            },
        },
    )


if __name__ == "__main__":
    main()
