from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.metrics import (
    intra_list_diversity_at_k,
    oracle_recall_at_k,
    per_segment_metrics,
    summarize_ranking_metrics,
)


class TestIntraListDiversity:
    def test_all_same_category(self) -> None:
        recs = {1: [10, 11, 12], 2: [10, 11, 12]}
        cats = {10: "sports", 11: "sports", 12: "sports"}
        assert intra_list_diversity_at_k(recs, cats, 3) == pytest.approx(0.0)

    def test_all_different_category(self) -> None:
        recs = {1: [10, 11, 12]}
        cats = {10: "sports", 11: "music", 12: "cooking"}
        assert intra_list_diversity_at_k(recs, cats, 3) == pytest.approx(1.0)

    def test_mixed_categories(self) -> None:
        recs = {1: [10, 11, 12]}
        cats = {10: "sports", 11: "sports", 12: "music"}
        result = intra_list_diversity_at_k(recs, cats, 3)
        assert 0.0 < result < 1.0
        assert result == pytest.approx(2.0 / 3.0)

    def test_single_recommendation(self) -> None:
        recs = {1: [10]}
        cats = {10: "sports"}
        assert intra_list_diversity_at_k(recs, cats, 1) == pytest.approx(0.0)

    def test_empty_recommendations(self) -> None:
        assert intra_list_diversity_at_k({}, {}, 5) == pytest.approx(0.0)

    def test_missing_category(self) -> None:
        recs = {1: [10, 11, 12]}
        cats = {10: "sports"}
        result = intra_list_diversity_at_k(recs, cats, 3)
        assert result == pytest.approx(2.0 / 3.0)

    def test_k_limits_list(self) -> None:
        recs = {1: [10, 11, 12, 13]}
        cats = {10: "sports", 11: "sports", 12: "music", 13: "music"}
        result_k2 = intra_list_diversity_at_k(recs, cats, 2)
        assert result_k2 == pytest.approx(0.0)


class TestOracleRecall:
    def test_perfect_pool(self) -> None:
        truth = {1: {10, 11}, 2: {12}}
        pool = {1: [10, 11, 20, 21, 22], 2: [12, 20, 21, 22, 23]}
        assert oracle_recall_at_k(truth, pool, 10) == pytest.approx(1.0)

    def test_partial_pool(self) -> None:
        truth = {1: {10, 11, 12}}
        pool = {1: [10, 20, 21, 22, 23]}
        assert oracle_recall_at_k(truth, pool, 10) == pytest.approx(1.0 / 3.0)

    def test_empty_truth(self) -> None:
        truth: dict[int, set[int]] = {1: set()}
        pool = {1: [10, 11]}
        assert oracle_recall_at_k(truth, pool, 10) == pytest.approx(0.0)

    def test_no_persons(self) -> None:
        assert oracle_recall_at_k({}, {}, 10) == pytest.approx(0.0)

    def test_k_limits_pool(self) -> None:
        truth = {1: {10, 11, 12}}
        pool = {1: [20, 21, 10, 11, 12]}
        result_k2 = oracle_recall_at_k(truth, pool, 2)
        assert result_k2 <= 1.0


class TestPerSegmentMetrics:
    def test_basic_segments(self) -> None:
        truth = {1: {10}, 2: {10}, 3: {11}}
        recs = {1: [10, 20], 2: [20, 30], 3: [11, 20]}
        segments = {
            1: {"age_group": "20대", "sex": "남성"},
            2: {"age_group": "20대", "sex": "여성"},
            3: {"age_group": "30대", "sex": "남성"},
        }
        result = per_segment_metrics(truth, recs, segments, 2)
        assert "age_group" in result
        assert "sex" in result
        assert "recall_gap" in result
        assert result["recall_gap"]["age_group"] >= 0.0
        assert result["recall_gap"]["sex"] >= 0.0

    def test_single_group(self) -> None:
        truth = {1: {10}, 2: {11}}
        recs = {1: [10], 2: [11]}
        segments = {
            1: {"age_group": "20대", "sex": "남성"},
            2: {"age_group": "20대", "sex": "남성"},
        }
        result = per_segment_metrics(truth, recs, segments, 1)
        assert result["recall_gap"]["age_group"] == pytest.approx(0.0)
        assert result["recall_gap"]["sex"] == pytest.approx(0.0)

    def test_empty_segments(self) -> None:
        result = per_segment_metrics({}, {}, {}, 5)
        assert "recall_gap" in result


class TestSummarizeIntegration:
    def _base_data(self) -> tuple:
        truth = {1: {10, 11}, 2: {12}}
        recs = {1: [10, 20, 11], 2: [12, 30, 40]}
        return truth, recs

    def test_backward_compatible(self) -> None:
        truth, recs = self._base_data()
        result = summarize_ranking_metrics(truth, recs, (5,), num_total_items=50)
        assert "recall@5" in result
        assert "intra_list_diversity@5" not in result
        assert "oracle_recall@5" not in result

    def test_with_new_params(self) -> None:
        truth, recs = self._base_data()
        cats = {10: "a", 11: "b", 12: "a", 20: "c", 30: "a", 40: "b"}
        pool = {1: [10, 11, 20, 30, 40], 2: [12, 30, 40, 50, 60]}
        segments = {
            1: {"age_group": "20대", "sex": "남성"},
            2: {"age_group": "30대", "sex": "여성"},
        }
        result = summarize_ranking_metrics(
            truth, recs, (5,),
            num_total_items=50,
            hobby_categories=cats,
            candidate_pool_by_person=pool,
            person_segments=segments,
        )
        assert "recall@5" in result
        assert "intra_list_diversity@5" in result
        assert "oracle_recall@5" in result
        assert "per_segment" in result
