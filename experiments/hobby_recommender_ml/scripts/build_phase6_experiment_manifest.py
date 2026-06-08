from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hobby_recommender_ml.hobby_recommender.phase6 import Phase6ExperimentSpec, validate_phase6_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Phase 6 experiment manifest without running training.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--changed-variable", required=True)
    parser.add_argument("--stage1-provider", default="popularity+cooccurrence")
    parser.add_argument("--stage2-recipe", default="lightgbm_num_leaves31_e5_domain")
    parser.add_argument("--candidate-text-builder", default="name_only")
    parser.add_argument("--embedding-model", default="dragonkue/multilingual-e5-small-ko-v2")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/hobby_recommender_ml/artifacts/phase6/manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = Phase6ExperimentSpec(
        experiment_id=args.experiment_id,
        changed_variable=args.changed_variable,
        stage1_provider=args.stage1_provider,
        stage2_recipe=args.stage2_recipe,
        candidate_text_builder=args.candidate_text_builder,
        embedding_model=args.embedding_model,
    )
    validate_phase6_spec(spec)
    payload = {
        "phase": "phase6_high_accuracy_hybrid_extension",
        "status": "manifest_only",
        "experiment_id": spec.experiment_id,
        "changed_variable": spec.changed_variable,
        "stage1_provider": spec.stage1_provider,
        "stage2_recipe": spec.stage2_recipe,
        "candidate_text_builder": spec.candidate_text_builder,
        "embedding_model": spec.embedding_model,
        "validation_first": spec.validation_first,
        "winner_only_test": spec.winner_only_test,
        "promotion_gate": {
            "compare_against": "e5_small_domain_stage2_default",
            "requires_recall10_and_ndcg10_improvement": True,
            "requires_no_candidate_recall50_regression": True,
            "requires_leakage_audit_pass": True,
        },
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
