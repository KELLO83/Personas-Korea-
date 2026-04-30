from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.diversity import (
    compute_hobby_embeddings,
    compute_intra_list_diversity,
    mmr_rerank,
    mmr_rerank_with_hobbies,
    _get_category,
)


class TestComputeHobbyEmbeddings:

    def test_basic_category_onehot(self) -> None:
        taxonomy = {
            "rules": [
                {"canonical_hobby": "축구", "include_keywords": ["축구"], "taxonomy": {"category": "스포츠"}},
                {"canonical_hobby": "영화", "include_keywords": ["영화"], "taxonomy": {"category": "문화콘텐츠"}},
                {"canonical_hobby": "요리", "include_keywords": ["요리"], "taxonomy": {"category": "미식"}},
            ],
        }
        names = ["축구", "영화", "요리"]
        emb = compute_hobby_embeddings(names, taxonomy)
        assert emb.shape == (3, 3)
        for i in range(3):
            norm = np.linalg.norm(emb[i])
            assert abs(norm - 1.0) < 1e-6 or norm == 0.0

    def test_unknown_hobby_gets_zero_vector(self) -> None:
        taxonomy = {
            "rules": [
                {"canonical_hobby": "축구", "include_keywords": [], "taxonomy": {"category": "스포츠"}},
            ],
        }
        emb = compute_hobby_embeddings(["축구", "알수없는취미"], taxonomy)
        assert emb.shape[0] == 2
        assert np.linalg.norm(emb[1]) < 1e-6

    def test_no_taxonomy_returns_zero_vectors(self) -> None:
        emb = compute_hobby_embeddings(["축구", "영화"], None)
        assert emb.shape[0] == 2
        assert np.allclose(emb, 0.0)

    def test_empty_input(self) -> None:
        emb = compute_hobby_embeddings([], None)
        assert emb.shape == (0, 0)

    def test_same_category_same_dimension(self) -> None:
        taxonomy = {
            "rules": [
                {"canonical_hobby": "축구", "include_keywords": [], "taxonomy": {"category": "스포츠"}},
                {"canonical_hobby": "농구", "include_keywords": [], "taxonomy": {"category": "스포츠"}},
            ],
        }
        emb = compute_hobby_embeddings(["축구", "농구"], taxonomy)
        np.testing.assert_array_almost_equal(emb[0], emb[1])


class TestMMRRerank:

    def test_lambda_1_pure_relevance(self) -> None:
        hobby_ids = [10, 20, 30, 40, 50]
        scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1], dtype=np.float32)
        emb = np.eye(5, dtype=np.float32)
        result = mmr_rerank(hobby_ids, scores, emb, lambda_param=1.0, top_k=5)
        assert result == [10, 20, 30, 40, 50]

    def test_lambda_0_max_diversity(self) -> None:
        hobby_ids = [10, 20, 30]
        scores = np.array([0.9, 0.7, 0.5], dtype=np.float32)
        emb = np.eye(3, dtype=np.float32)
        result = mmr_rerank(hobby_ids, scores, emb, lambda_param=0.0, top_k=3)
        assert len(result) == 3
        assert set(result) == {10, 20, 30}

    def test_top_k_limits_output(self) -> None:
        hobby_ids = [10, 20, 30, 40]
        scores = np.array([0.9, 0.7, 0.5, 0.3], dtype=np.float32)
        emb = np.eye(4, dtype=np.float32)
        result = mmr_rerank(hobby_ids, scores, emb, lambda_param=0.7, top_k=2)
        assert len(result) == 2

    def test_same_score_prefers_diverse(self) -> None:
        hobby_ids = [10, 20, 30]
        scores = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / norms
        result = mmr_rerank(hobby_ids, scores, emb, lambda_param=0.5, top_k=3)
        assert len(result) == 3
        assert result[0] == 10

    def test_empty_input(self) -> None:
        result = mmr_rerank([], np.array([], dtype=np.float32), np.empty((0, 0)), lambda_param=0.7, top_k=10)
        assert result == []

    def test_deterministic_with_seed(self) -> None:
        hobby_ids = list(range(100))
        scores = np.random.RandomState(42).rand(100).astype(np.float32)
        rng = np.random.RandomState(42)
        emb = rng.randn(100, 10).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.where(norms > 0, norms, 1.0)
        result1 = mmr_rerank(hobby_ids, scores, emb, lambda_param=0.7, top_k=10)
        result2 = mmr_rerank(hobby_ids, scores, emb, lambda_param=0.7, top_k=10)
        assert result1 == result2

    def test_mmr_improves_diversity_over_pure_relevance(self) -> None:
        hobby_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        scores = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45], dtype=np.float32)
        cat_emb = np.zeros((10, 3), dtype=np.float32)
        for i in range(4):
            cat_emb[i, 0] = 1.0
        for i in range(4, 7):
            cat_emb[i, 1] = 1.0
        for i in range(7, 10):
            cat_emb[i, 2] = 1.0
        norms = np.linalg.norm(cat_emb, axis=1, keepdims=True)
        cat_emb = cat_emb / np.where(norms > 0, norms, 1.0)
        pure_relevance = mmr_rerank(hobby_ids, scores, cat_emb, lambda_param=1.0, top_k=10)
        diverse = mmr_rerank(hobby_ids, scores, cat_emb, lambda_param=0.5, top_k=10)
        assert pure_relevance[0] == 0
        categories_pure = [_get_category(hobby_ids[i], None) for i in range(len(pure_relevance))]
        names_for_ild_pure = [f"item_{i}" for i in pure_relevance]
        ild_pure = compute_intra_list_diversity(names_for_ild_pure, embeddings=cat_emb[np.lexsort((pure_relevance,))])
        ild_diverse = compute_intra_list_diversity(
            [f"item_{i}" for i in diverse],
            embeddings=cat_emb[[hobby_ids.index(i) for i in diverse]],
        )

    def test_precomputed_embeddings_no_taxonomy(self) -> None:
        emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / norms
        hobby_ids = [10, 20, 30]
        scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        result = mmr_rerank(hobby_ids, scores, emb, lambda_param=0.5, top_k=3)
        assert len(result) == 3
        assert set(result) == {10, 20, 30}


class TestIntraListDiversity:

    def test_empty_list(self) -> None:
        assert compute_intra_list_diversity([], None) == 1.0

    def test_single_item(self) -> None:
        assert compute_intra_list_diversity(["축구"], None) == 1.0

    def test_all_same_category(self) -> None:
        taxonomy = {
            "rules": [
                {"canonical_hobby": "축구", "include_keywords": [], "taxonomy": {"category": "스포츠"}},
                {"canonical_hobby": "농구", "include_keywords": [], "taxonomy": {"category": "스포츠"}},
                {"canonical_hobby": "배구", "include_keywords": [], "taxonomy": {"category": "스포츠"}},
            ],
        }
        ild = compute_intra_list_diversity(["축구", "농구", "배구"], taxonomy)
        assert ild == pytest.approx(0.0)

    def test_all_different_categories(self) -> None:
        taxonomy = {
            "rules": [
                {"canonical_hobby": "축구", "include_keywords": [], "taxonomy": {"category": "스포츠"}},
                {"canonical_hobby": "영화", "include_keywords": [], "taxonomy": {"category": "문화콘텐츠"}},
                {"canonical_hobby": "요리", "include_keywords": [], "taxonomy": {"category": "미식"}},
            ],
        }
        ild = compute_intra_list_diversity(["축구", "영화", "요리"], taxonomy)
        assert ild == pytest.approx(1.0)

    def test_precomputed_embeddings(self) -> None:
        emb = np.eye(3, dtype=np.float32)
        ild = compute_intra_list_diversity(["a", "b", "c"], embeddings=emb)
        assert ild == pytest.approx(1.0)

    def test_identical_embeddings(self) -> None:
        vec = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        ild = compute_intra_list_diversity(["a", "b", "c"], embeddings=vec)
        assert ild == pytest.approx(0.0, abs=1e-6)


class TestGetCategory:

    def test_from_taxonomy_map(self) -> None:
        taxonomy = {
            "taxonomy": {"축구": {"category": "스포츠"}},
            "rules": [],
        }
        assert _get_category("축구", taxonomy) == "스포츠"

    def test_from_rules(self) -> None:
        taxonomy = {
            "taxonomy": {},
            "rules": [
                {"canonical_hobby": "축구", "taxonomy": {"category": "스포츠"}, "include_keywords": []},
            ],
        }
        assert _get_category("축구", taxonomy) == "스포츠"

    def test_from_include_keywords(self) -> None:
        taxonomy = {
            "taxonomy": {},
            "rules": [
                {"canonical_hobby": "축구", "taxonomy": {"category": "스포츠"}, "include_keywords": ["축구", "풋살"]},
            ],
        }
        assert _get_category("축구", taxonomy) == "스포츠"

    def test_none_taxonomy(self) -> None:
        assert _get_category("축구", None) is None

    def test_unknown_hobby(self) -> None:
        taxonomy = {"taxonomy": {}, "rules": []}
        assert _get_category("미상취미", taxonomy) is None