from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GNN_ROOT = ROOT / "GNN_Neural_Network"
EXPERIMENTS_DIR = GNN_ROOT / "artifacts" / "experiments"
MANIFEST_PATH = EXPERIMENTS_DIR / "cleanup_removed_generated_files.json"
SCRATCH_ROOT = EXPERIMENTS_DIR / "_recovery_scratch"
REPORT_PATH = EXPERIMENTS_DIR / "deleted_artifact_recovery_plan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild accidentally deleted experiment artifacts without overwriting "
            "preserved metrics/status/params logs."
        )
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--execute-models", action="store_true", help="Retrain missing ranker_model.txt files in scratch and copy models back.")
    parser.add_argument("--execute-datasets", action="store_true", help="Regenerate deleted ranker_dataset*.csv files.")
    parser.add_argument("--force", action="store_true", help="Overwrite recovered model/dataset targets if they already exist.")
    parser.add_argument("--keep-scratch", action="store_true", help="Keep scratch output directories after copying recovered artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    removed_files = [Path(item) for item in manifest.get("removed_files", []) if isinstance(item, str)]

    model_targets = [path for path in removed_files if path.name == "ranker_model.txt"]
    dataset_targets = [path for path in removed_files if path.name.startswith("ranker_dataset") and path.suffix == ".csv"]

    plan: dict[str, Any] = {
        "manifest": str(args.manifest),
        "policy": {
            "preserve_existing_logs": True,
            "model_recovery": "train in _recovery_scratch, then copy ranker_model.txt only",
            "dataset_recovery": "regenerate deleted CSVs only; do not overwrite metrics logs",
        },
        "models": [],
        "datasets": [],
    }

    for relative_target in model_targets:
        entry = _plan_model_recovery(relative_target)
        plan["models"].append(entry)
        if args.execute_models and entry["status"] == "planned":
            _execute_model_recovery(entry, force=args.force, keep_scratch=args.keep_scratch)
            entry["status"] = "recovered"
            entry["recovered_at"] = _utc_timestamp()

    for relative_target in dataset_targets:
        entry = _plan_dataset_recovery(relative_target)
        plan["datasets"].append(entry)
        if args.execute_datasets and entry["status"] == "planned":
            _execute_dataset_recovery(entry, force=args.force)
            entry["status"] = "recovered"
            entry["recovered_at"] = _utc_timestamp()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Recovery plan written: {REPORT_PATH}")
    print(f"Models planned: {sum(1 for item in plan['models'] if item['status'] == 'planned')}/{len(plan['models'])}")
    print(f"Datasets planned: {sum(1 for item in plan['datasets'] if item['status'] == 'planned')}/{len(plan['datasets'])}")
    if not args.execute_models and not args.execute_datasets:
        print("Dry run only. Add --execute-models and/or --execute-datasets to rebuild artifacts.")


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Cleanup manifest not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("Cleanup manifest must be a JSON object")
    return raw


def _utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _plan_model_recovery(relative_target: Path) -> dict[str, Any]:
    target = EXPERIMENTS_DIR / relative_target
    experiment_dir = target.parent
    params_path = experiment_dir / "ranker_params.json"
    if target.exists():
        return {
            "relative_target": str(relative_target),
            "target": str(target),
            "status": "exists",
            "reason": "target already exists",
        }
    custom_plan = _plan_custom_model_recovery(relative_target, target)
    if custom_plan is not None:
        return custom_plan

    if not params_path.exists():
        return {
            "relative_target": str(relative_target),
            "target": str(target),
            "status": "manual",
            "reason": "ranker_params.json missing; exact training args cannot be reconstructed automatically",
        }

    params = _load_json_object(params_path)
    command = _build_train_ranker_command(params, relative_target)
    return {
        "relative_target": str(relative_target),
        "target": str(target),
        "params_path": str(params_path),
        "scratch_dir": str(_scratch_dir_for(relative_target.parent)),
        "status": "planned",
        "command": command,
        "notes": "Command writes to scratch. Recovery copies only ranker_model.txt back to target.",
    }


def _plan_custom_model_recovery(relative_target: Path, target: Path) -> dict[str, Any] | None:
    parts = relative_target.parts
    if len(parts) >= 3 and parts[0] == "phase5_pre_50k_baseline" and parts[1].startswith("feature_fraction_"):
        feature_fraction = float(parts[1].replace("feature_fraction_", ""))
        return {
            "relative_target": str(relative_target),
            "target": str(target),
            "status": "planned",
            "method": "rebuild_feature_fraction_probe_model",
            "feature_fraction": feature_fraction,
            "scratch_dir": str(_scratch_dir_for(relative_target.parent)),
            "notes": "Rebuilds deleted feature-fraction probe model in scratch and copies model only.",
        }
    if relative_target.as_posix() == "phase5_pre_50k_baseline/ranker_model.txt":
        return {
            "relative_target": str(relative_target),
            "target": str(target),
            "status": "planned",
            "method": "rebuild_simple_phase25_model",
            "feature_fraction": 0.85,
            "scratch_dir": str(_scratch_dir_for(relative_target.parent)),
            "notes": "Rebuilds the standalone phase5_pre baseline model in scratch and copies model only.",
        }
    if relative_target.as_posix() == "probe_lgbm_balance/ranker_model.txt":
        return {
            "relative_target": str(relative_target),
            "target": str(target),
            "status": "planned",
            "method": "rebuild_probe_lgbm_balance_model",
            "feature_fraction": 0.85,
            "scratch_dir": str(_scratch_dir_for(relative_target.parent)),
            "notes": "Rebuilds the standalone probe model in scratch and copies model only.",
        }
    return None


def _build_train_ranker_command(params: dict[str, Any], relative_target: Path) -> list[str]:
    lightgbm_params = _expect_dict(params.get("lightgbm_params") or params.get("params"), "lightgbm_params")
    feature_policy = _expect_dict(params.get("feature_policy", {}), "feature_policy")

    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        str(GNN_ROOT / "scripts" / "train_ranker.py"),
        "--output-dir",
        str(_scratch_dir_for(relative_target.parent)),
        "--neg-ratio",
        str(int(params.get("neg_ratio", 4))),
        "--hard-ratio",
        str(float(params.get("hard_ratio", 0.8))),
        "--ranker-val-ratio",
        str(float(params.get("ranker_val_ratio", 0.2))),
        "--seed",
        str(int(params.get("seed", 42))),
        "--experiment-id",
        str(params.get("experiment_id", relative_target.parent.as_posix())),
    ]

    _append_if_present(command, "--num-leaves", lightgbm_params.get("num_leaves"))
    _append_if_present(command, "--min-data-in-leaf", lightgbm_params.get("min_data_in_leaf"))
    _append_if_present(command, "--learning-rate", lightgbm_params.get("learning_rate"))
    _append_if_present(command, "--feature-fraction", lightgbm_params.get("feature_fraction"))
    _append_if_present(command, "--bagging-fraction", lightgbm_params.get("bagging_fraction"))
    _append_if_present(command, "--bagging-freq", lightgbm_params.get("bagging_freq"))
    _append_if_present(command, "--reg-alpha", lightgbm_params.get("reg_alpha"))
    _append_if_present(command, "--reg-lambda", lightgbm_params.get("reg_lambda"))

    objective = str(lightgbm_params.get("objective", "binary"))
    command.extend(["--objective", objective])
    ndcg_eval_at = lightgbm_params.get("ndcg_eval_at")
    if isinstance(ndcg_eval_at, list) and ndcg_eval_at:
        command.extend(["--ndcg-eval-at", str(int(ndcg_eval_at[0]))])

    if feature_policy.get("include_source_features") is True or params.get("include_source_features") is True:
        command.append("--include-source-features")
    if feature_policy.get("include_text_embedding_feature") is True:
        command.append("--include-text-embedding-feature")

    return command


def _append_if_present(command: list[str], flag: str, value: object) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _execute_model_recovery(entry: dict[str, Any], *, force: bool, keep_scratch: bool) -> None:
    target = Path(str(entry["target"]))
    scratch_dir = Path(str(entry["scratch_dir"]))
    scratch_model = scratch_dir / "ranker_model.txt"
    if target.exists() and not force:
        print(f"Skip existing model: {target}")
        return

    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    method = entry.get("method")
    if method in {"rebuild_feature_fraction_probe_model", "rebuild_simple_phase25_model", "rebuild_probe_lgbm_balance_model"}:
        _train_standalone_probe_model(scratch_model, feature_fraction=float(entry.get("feature_fraction", 0.85)))
    else:
        command = [str(item) for item in entry["command"]]
        print(f"Retraining into scratch: {' '.join(command)}")
        subprocess.run(command, cwd=ROOT, check=True)

    if not scratch_model.exists():
        raise FileNotFoundError(f"Expected recovered model not found: {scratch_model}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scratch_model, target)
    print(f"Recovered model: {target}")

    if not keep_scratch:
        shutil.rmtree(scratch_dir)


def _train_standalone_probe_model(model_path: Path, *, feature_fraction: float) -> None:
    try:
        import lightgbm as lgb
        import numpy as np
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError("lightgbm, numpy, and scikit-learn are required to rebuild standalone probe models") from exc

    edges = _load_edge_rows(GNN_ROOT / "data" / "person_hobby_edges.csv")
    person_to_hobbies: dict[str, set[str]] = defaultdict(set)
    hobby_counts: Counter[str] = Counter()
    for person_uuid, hobby_name in edges:
        person_to_hobbies[person_uuid].add(hobby_name)
        hobby_counts[hobby_name] += 1

    all_hobbies = sorted(hobby_counts)
    max_pop = max(hobby_counts.values()) if hobby_counts else 1
    rows: list[list[float]] = []
    labels: list[int] = []
    for person_uuid, hobby_name in edges:
        rows.append([hobby_counts[hobby_name] / max_pop, 0.0])
        labels.append(1)
        for negative in _negative_hobbies(all_hobbies, person_to_hobbies[person_uuid], count=4):
            rows.append([hobby_counts[negative] / max_pop, 0.0])
            labels.append(0)

    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": feature_fraction,
        "feature_fraction_bynode": feature_fraction,
        "lambda_l1": 0.15,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": 42,
    }
    model = lgb.train(
        params,
        lgb.Dataset(x_train, label=y_train),
        valid_sets=[lgb.Dataset(x_val, label=y_val)],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))


def _plan_dataset_recovery(relative_target: Path) -> dict[str, Any]:
    target = EXPERIMENTS_DIR / relative_target
    if target.exists():
        return {
            "relative_target": str(relative_target),
            "target": str(target),
            "status": "exists",
            "reason": "target already exists",
        }

    if relative_target.name == "ranker_dataset_phase25.csv":
        return {
            "relative_target": str(relative_target),
            "target": str(target),
            "status": "planned",
            "method": "rebuild_phase25_ranker_dataset",
            "source": str(GNN_ROOT / "data" / "person_hobby_edges.csv"),
        }

    if relative_target.name == "ranker_dataset_with_kure.csv":
        return {
            "relative_target": str(relative_target),
            "target": str(target),
            "status": "planned",
            "method": "rebuild_lightweight_kure_probe_dataset",
            "source": str(GNN_ROOT / "data" / "person_context.csv"),
            "notes": "Rebuilds the probe CSV schema without overwriting metrics. Full KURE embedding cache is intentionally not restored.",
        }

    return {
        "relative_target": str(relative_target),
        "target": str(target),
        "status": "manual",
        "reason": "unknown ranker dataset target",
    }


def _execute_dataset_recovery(entry: dict[str, Any], *, force: bool) -> None:
    target = Path(str(entry["target"]))
    if target.exists() and not force:
        print(f"Skip existing dataset: {target}")
        return
    method = entry.get("method")
    if method == "rebuild_phase25_ranker_dataset":
        _rebuild_phase25_ranker_dataset(target)
    elif method == "rebuild_lightweight_kure_probe_dataset":
        _rebuild_lightweight_kure_probe_dataset(target)
    else:
        raise ValueError(f"Unsupported dataset recovery method: {method}")
    print(f"Recovered dataset: {target}")


def _rebuild_phase25_ranker_dataset(target: Path) -> None:
    edges_path = GNN_ROOT / "data" / "person_hobby_edges.csv"
    rows = _load_edge_rows(edges_path)
    person_to_hobbies: dict[str, set[str]] = defaultdict(set)
    hobby_counts: Counter[str] = Counter()
    for person_uuid, hobby_name in rows:
        person_to_hobbies[person_uuid].add(hobby_name)
        hobby_counts[hobby_name] += 1

    all_hobbies = sorted(hobby_counts)
    max_pop = max(hobby_counts.values()) if hobby_counts else 1
    output_rows: list[dict[str, object]] = []
    for person_uuid, hobby_name in rows:
        known = person_to_hobbies[person_uuid]
        output_rows.append(_phase25_row(person_uuid, hobby_name, hobby_counts[hobby_name], max_pop, 1))
        for negative in _negative_hobbies(all_hobbies, known, count=4):
            output_rows.append(_phase25_row(person_uuid, negative, hobby_counts[negative], max_pop, 0))

    _write_csv(target, output_rows)


def _phase25_row(person_uuid: str, hobby_name: str, popularity: int, max_pop: int, label: int) -> dict[str, object]:
    return {
        "person_uuid": person_uuid,
        "hobby_name": hobby_name,
        "popularity_score": popularity / max_pop if max_pop else 0.0,
        "cooccurrence_score": 0.0,
        "label": label,
    }


def _negative_hobbies(all_hobbies: list[str], known: set[str], count: int) -> list[str]:
    candidates = [hobby for hobby in all_hobbies if hobby not in known]
    return candidates[:count]


def _rebuild_lightweight_kure_probe_dataset(target: Path) -> None:
    context_path = GNN_ROOT / "data" / "person_context.csv"
    if not context_path.exists():
        raise FileNotFoundError(f"person_context.csv required for KURE probe dataset recovery: {context_path}")
    output_rows: list[dict[str, object]] = []
    with context_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            person_uuid = row.get("person_uuid") or row.get("uuid") or ""
            output_rows.append(
                {
                    "person_uuid": person_uuid,
                    "age_group": row.get("age_group", ""),
                    "sex": row.get("sex", ""),
                    "occupation": row.get("occupation", ""),
                    "district": row.get("district", ""),
                    "kure_probe_text_length": len(row.get("embedding_text", "") or row.get("persona_text", "")),
                }
            )
    _write_csv(target, output_rows)


def _load_edge_rows(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Edge CSV not found: {path}")
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            person_uuid = (row.get("person_uuid") or "").strip()
            hobby_name = (row.get("hobby_name") or "").strip()
            if person_uuid and hobby_name:
                rows.append((person_uuid, hobby_name))
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _scratch_dir_for(relative_experiment_dir: Path) -> Path:
    safe = "__".join(relative_experiment_dir.parts)
    return SCRATCH_ROOT / safe


def _load_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return raw


def _expect_dict(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


if __name__ == "__main__":
    main()
