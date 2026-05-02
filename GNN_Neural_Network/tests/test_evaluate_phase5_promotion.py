from __future__ import annotations

from GNN_Neural_Network.scripts.evaluate_ranker import (
    _phase5_diversity_probe_decision,
    _phase5_promotion_decision,
)


def _base_payload() -> dict[str, float]:
    return {
        "recall@10": 0.0,
        "ndcg@10": 0.0,
        "catalog_coverage@10": 0.0,
        "novelty@10": 4.0,
        "intra_list_diversity@10": 0.0,
    }


def test_phase5_requires_two_metrics_with_min_gains() -> None:
    payload = _base_payload()
    payload.update(
        {
            "catalog_coverage@10": 0.026,
            "novelty@10": 0.050,
            "intra_list_diversity@10": 0.021,
        },
    )
    result = _phase5_promotion_decision(
        split="validation",
        delta_v2_vs_baseline=payload,
        candidate_recall_delta=0.0,
        v2_fallback_count=0,
        mmr_embedding_meta={"cache_enabled": True},
        baseline_path="baseline",
    )
    assert result["status"] == "eligible_for_test"
    assert result["gates"]["novelty@10"]["pass"] is False
    assert result["criteria"]["diversity"]["improved_metric_names"] == [
        "catalog_coverage@10",
        "intra_list_diversity@10",
    ]


def test_phase5_diversity_probe_pass_with_two_metrics() -> None:
    payload = _base_payload()
    payload.update(
        {
            "catalog_coverage@10": 0.03,
            "novelty@10": 0.11,
            "intra_list_diversity@10": 0.021,
            "recall@10": -0.001,
            "ndcg@10": -0.001,
        },
    )
    result = _phase5_diversity_probe_decision(
        split="validation",
        delta_v2_vs_baseline=payload,
        candidate_recall_delta=0.0,
        v2_fallback_count=0,
        mmr_embedding_meta={"cache_enabled": True},
        baseline_path="baseline",
    )
    assert result["status"] == "passed"
    assert result["diversity"]["improved_metric_count"] == 3


def test_phase5_diversity_probe_requires_additional_review() -> None:
    payload = _base_payload()
    payload.update(
        {
            "catalog_coverage@10": 0.03,
            "novelty@10": 0.11,
            "intra_list_diversity@10": 0.03,
            "recall@10": -0.007,
            "ndcg@10": -0.007,
        },
    )
    result = _phase5_diversity_probe_decision(
        split="validation",
        delta_v2_vs_baseline=payload,
        candidate_recall_delta=0.0,
        v2_fallback_count=0,
        mmr_embedding_meta={"cache_enabled": True},
        baseline_path="baseline",
    )
    assert result["status"] == "requires_additional_review"
    assert result["accuracy"]["requires_additional_review"] is True


def test_phase5_blocked_when_candidate_recall_drift_is_not_allowed() -> None:
    payload = _base_payload()
    payload.update(
        {
            "catalog_coverage@10": 0.04,
            "novelty@10": 0.11,
            "intra_list_diversity@10": 0.03,
        },
    )
    result = _phase5_promotion_decision(
        split="validation",
        delta_v2_vs_baseline=payload,
        candidate_recall_delta=1e-3,
        v2_fallback_count=0,
        mmr_embedding_meta={"cache_enabled": True},
        baseline_path="baseline",
    )
    assert result["status"] == "blocked"
    assert result["gates"]["candidate_recall@50"]["pass"] is False


def test_phase5_promotes_when_all_gates_pass() -> None:
    payload = _base_payload()
    payload.update(
        {
            "recall@10": 0.0,
            "ndcg@10": 0.0,
            "catalog_coverage@10": 0.03,
            "novelty@10": 0.11,
            "intra_list_diversity@10": 0.03,
        },
    )
    result = _phase5_promotion_decision(
        split="validation",
        delta_v2_vs_baseline=payload,
        candidate_recall_delta=0.0,
        v2_fallback_count=0,
        mmr_embedding_meta={"cache_enabled": True},
        baseline_path="baseline",
    )
    assert result["status"] == "eligible_for_test"
    assert result["criteria"]["diversity"]["improved_metrics"] == 3
