from __future__ import annotations

import numpy as np
import pytest

from GNN_Neural_Network.gnn_recommender.ranker_explain import (
    FALLBACK_REASON,
    REASON_TEMPLATES,
    batch_generate_reasons,
    generate_reason,
    validate_reason_batch,
)


class TestGenerateReason:
    def test_empty_contributions_return_fallback(self) -> None:
        contributions = np.array([0.0, 0.0, 0.0])
        feat = np.array([1.0, 2.0, 3.0])
        names = ["a", "b", "c"]
        reason = generate_reason(contributions, feat, names, top_k=3)
        assert reason == FALLBACK_REASON

    def test_picks_top_positive_contributors(self) -> None:
        contributions = np.array([0.5, -0.2, 0.8, 0.1])
        feat = np.array([1.0, 2.0, 3.0, 4.0])
        names = ["lightgcn_score", "cooccurrence_score", "age_group_fit", "popularity_prior"]
        reason = generate_reason(contributions, feat, names, top_k=2)
        assert "연령대" in reason or "그래프" in reason
        assert "비슷한 사용자" not in reason

    def test_ignores_nan(self) -> None:
        contributions = np.array([np.nan, 0.5, np.inf])
        feat = np.array([1.0, 2.0, 3.0])
        names = ["a", "b", "c"]
        reason = generate_reason(contributions, feat, names, top_k=3)
        assert isinstance(reason, str)

    def test_known_feature_names_have_templates(self) -> None:
        for name in REASON_TEMPLATES:
            contributions = np.array([1.0 if n == name else 0.0 for n in REASON_TEMPLATES])
            feat = np.ones(len(REASON_TEMPLATES))
            names = list(REASON_TEMPLATES)
            reason = generate_reason(contributions, feat, names, top_k=1)
            assert reason != FALLBACK_REASON
            assert REASON_TEMPLATES[name] in reason


class TestValidateReasonBatch:
    def test_all_non_empty_passes(self) -> None:
        recs = [{"reason": "연령대 특성과 잘 맞습니다"}, {"reason": "기존 취미와 함께 나타나는 경향이 있습니다"}]
        result = validate_reason_batch(recs, None, None, None)
        assert result["pass"] is True
        assert result["meaningful_rate"] == 1.0

    def test_empty_reasons_fails(self) -> None:
        recs = [{"reason": ""}, {"reason": ""}]
        result = validate_reason_batch(recs, None, None, None)
        assert result["pass"] is False
        assert result["meaningful_rate"] == 0.0

    def test_masked_hobby_detected(self) -> None:
        recs = [{"reason": "취미 [MASK]가 좋습니다"}]
        result = validate_reason_batch(recs, None, None, None)
        assert result["has_masked_hobby"] is True
        assert result["pass"] is False

    def test_nan_in_reason_detected(self) -> None:
        recs = [{"reason": "score is NaN"}]
        result = validate_reason_batch(recs, None, None, None)
        assert result["has_nan_in_reason"] is True
        assert result["pass"] is False


class TestBatchGenerateReasons:
    def test_requires_trained_model(self) -> None:
        with pytest.raises(ValueError, match="not trained"):
            batch_generate_reasons(None, np.zeros((1, 3)), ["a", "b", "c"])
