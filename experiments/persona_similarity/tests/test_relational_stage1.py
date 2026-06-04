from __future__ import annotations

import pytest

from experiments.persona_similarity.scripts.relational_stage1 import (
    RelationalStage1Spec,
    build_experiment_manifest,
    relational_stage1_promotion_passed,
    validate_relational_stage1_spec,
)


def test_relational_stage1_manifest_keeps_controls_fixed() -> None:
    manifest = build_experiment_manifest(RelationalStage1Spec(experiment_id="hgt_probe", provider="hgt"))

    assert manifest["baseline_stage1"] == "fastrp_knn_topk50"
    assert manifest["fixed_controls"]["top_k"] == 50
    assert manifest["fixed_controls"]["text_feature_policy"] == "stage2_pair_cosine_only"
    assert "ENJOYS_HOBBY" in manifest["graph_schema"]["relationship_types"]


def test_relational_stage1_spec_rejects_unfixed_reranker() -> None:
    spec = RelationalStage1Spec(
        experiment_id="bad",
        provider="rgcn",
        reranker_recipe="changed_reranker",
    )

    with pytest.raises(ValueError, match="reranker"):
        validate_relational_stage1_spec(spec)


def test_relational_stage1_promotion_gate_reports_failures() -> None:
    passed, failures = relational_stage1_promotion_passed(
        baseline={
            "candidate_recall@50": 0.8,
            "ndcg@5": 0.5,
            "ndcg@10": 0.5,
            "explanation_coverage": 0.9,
            "diversity": 0.7,
            "refresh_cost_seconds": 10.0,
        },
        candidate={
            "candidate_recall@50": 0.79,
            "ndcg@5": 0.51,
            "ndcg@10": 0.49,
            "explanation_coverage": 0.9,
            "diversity": 0.7,
            "refresh_cost_seconds": 40.0,
        },
    )

    assert not passed
    assert "candidate_recall@50_regressed" in failures
    assert "ndcg@10_regressed" in failures
    assert "refresh_cost_too_high" in failures
