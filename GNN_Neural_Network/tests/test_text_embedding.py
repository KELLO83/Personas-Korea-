from __future__ import annotations

import pytest

from GNN_Neural_Network.gnn_recommender.text_embedding import (
    compute_text_embedding_similarity,
    mask_holdout_hobbies,
    post_mask_leakage_audit,
)


class TestMaskHoldoutHobbies:
    def test_basic_mask(self):
        text = "저는 산책, 등산을 좋아합니다."
        result = mask_holdout_hobbies(text, {"산책"}, mask_token="[MASK]")
        assert "산책" not in result
        assert "[MASK]" in result
        assert "등산" in result

    def test_word_boundary_respects_prefix_suffix(self):
        text = "산책로가 예쁩니다. 산책, 등산을 좋아합니다."
        result = mask_holdout_hobbies(text, {"산책"}, mask_token="[MASK]")
        assert "산책로" in result
        assert "[MASK]" in result

    def test_alias_map_masks_aliases(self):
        text = "저는 걷기, 산책, 등산을 좋아합니다."
        alias_map = {"산책": ["걷기"]}
        result = mask_holdout_hobbies(text, {"산책"}, alias_map=alias_map, mask_token="[MASK]")
        assert "산책" not in result
        assert "걷기" not in result
        assert result.count("[MASK]") == 2

    def test_empty_holdout_returns_unchanged(self):
        text = "저는 등산을 좋아합니다."
        result = mask_holdout_hobbies(text, set())
        assert result == text

    def test_empty_text_returns_empty(self):
        result = mask_holdout_hobbies("", {"산책"})
        assert result == ""


class TestPostMaskLeakageAudit:
    def test_no_leakage_passes(self):
        masked = "저는 [MASK]와 등산을 좋아합니다."
        assert post_mask_leakage_audit(masked, {"산책"}) is True

    def test_residual_leakage_fails(self):
        masked = "저는 산책과 등산을 좋아합니다."
        assert post_mask_leakage_audit(masked, {"산책"}) is False

    def test_alias_leakage_detected(self):
        masked = "저는 걷기를 좋아합니다."
        alias_map = {"산책": ["걷기"]}
        assert post_mask_leakage_audit(masked, {"산책"}, alias_map=alias_map) is False

    def test_empty_holdout_passes(self):
        assert post_mask_leakage_audit("아무 텍스트", set()) is True


class TestComputeTextEmbeddingSimilarity:
    def test_empty_inputs_return_zero(self):
        assert compute_text_embedding_similarity("", "hobby") == 0.0
        assert compute_text_embedding_similarity("text", "") == 0.0

    def test_returns_float_in_range(self):
        sim = compute_text_embedding_similarity("저는 산책을 좋아합니다.", "산책")
        assert isinstance(sim, float)
        assert 0.0 <= sim <= 1.0
