from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.ranker import (
    RANKER_CATEGORICAL_FEATURES,
    RANKER_FEATURE_COLUMNS,
    RANKER_FEATURE_COLUMNS_WITH_SOURCE,
    LightGBMRanker,
    RankerDataset,
    RankerRow,
    build_ranker_dataset,
    load_or_build_candidate_pool,
    get_candidate_pool_cache_key,
    get_ranker_categorical_features,
    sample_negatives,
)
from GNN_Neural_Network.gnn_recommender.baseline import build_cooccurrence_counts, build_popularity_counts
from GNN_Neural_Network.gnn_recommender.rerank import (
    HobbyCandidate,
    RerankerConfig,
    RerankerWeights,
    build_reranker_config,
)
from GNN_Neural_Network.gnn_recommender.data import PersonContext

import random


class TestRankerFeatureSchema:
    def test_excluded_features_not_in_columns(self) -> None:
        assert "similar_person_score" not in RANKER_FEATURE_COLUMNS
        assert "persona_text_fit" not in RANKER_FEATURE_COLUMNS

    def test_is_cold_start_is_categorical(self) -> None:
        assert "is_cold_start" in RANKER_CATEGORICAL_FEATURES

    def test_feature_count(self) -> None:
        assert len(RANKER_FEATURE_COLUMNS) == 13

    def test_feature_count_with_source(self) -> None:
        assert len(RANKER_FEATURE_COLUMNS_WITH_SOURCE) == 16

    def test_source_features_are_categorical_when_enabled(self) -> None:
        categorical = get_ranker_categorical_features(RANKER_FEATURE_COLUMNS_WITH_SOURCE)
        assert "is_cold_start" in categorical
        assert "source_is_popularity" in categorical
        assert "source_is_cooccurrence" in categorical


class TestSampleNegatives:
    def test_basic_ratio(self) -> None:
        positives = {10, 11}
        pool = [20, 21, 22, 23, 24, 25]
        all_ids = list(range(100))
        known = {10, 11}
        result = sample_negatives(1, positives, pool, all_ids, known, neg_ratio=4, hard_ratio=0.8, rng=random.Random(42))
        assert len(result) == 8

    def test_no_positives_returns_empty(self) -> None:
        result = sample_negatives(1, set(), [20, 21], list(range(50)), set(), neg_ratio=4, rng=random.Random(0))
        assert result == []

    def test_negatives_exclude_positives_and_known(self) -> None:
        positives = {10}
        known = {10, 11}
        pool = [10, 11, 20, 21, 22]
        all_ids = [10, 11, 20, 21, 22, 30, 31, 32]
        result = sample_negatives(1, positives, pool, all_ids, known, neg_ratio=4, hard_ratio=1.0, rng=random.Random(42))
        for hid in result:
            assert hid not in positives
            assert hid not in known

    def test_hard_easy_split(self) -> None:
        positives = {10}
        known = {10}
        pool = [20, 21, 22, 23, 24]
        all_ids = list(range(100))
        rng = random.Random(42)
        result = sample_negatives(1, positives, pool, all_ids, known, neg_ratio=4, hard_ratio=0.5, rng=rng)
        pool_set = set(pool) - positives - known
        hard_count = sum(1 for h in result if h in pool_set)
        assert hard_count >= 1

    def test_insufficient_hard_fills_with_easy(self) -> None:
        positives = {10}
        known = {10}
        pool = [20]
        all_ids = list(range(100))
        result = sample_negatives(1, positives, pool, all_ids, known, neg_ratio=4, hard_ratio=1.0, rng=random.Random(0))
        assert len(result) >= 1
        assert 20 in result


class TestRankerDataset:
    def _make_rows(self, n_pos: int = 3, n_neg: int = 12) -> list[RankerRow]:
        rows: list[RankerRow] = []
        for i in range(n_pos):
            rows.append(RankerRow(person_id=1, hobby_id=100 + i, label=1, features={col: float(i) for col in RANKER_FEATURE_COLUMNS}))
        for i in range(n_neg):
            rows.append(RankerRow(person_id=1, hobby_id=200 + i, label=0, features={col: float(i) * 0.1 for col in RANKER_FEATURE_COLUMNS}))
        return rows

    def test_to_numpy_shapes(self) -> None:
        ds = RankerDataset(rows=self._make_rows())
        X, y = ds.to_numpy()
        assert X.shape == (15, 13)
        assert y.shape == (15,)
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_to_numpy_labels(self) -> None:
        ds = RankerDataset(rows=self._make_rows(n_pos=2, n_neg=3))
        _, y = ds.to_numpy()
        assert sum(y) == 2.0
        assert len(y) == 5

    def test_to_lgb_dataset(self) -> None:
        ds = RankerDataset(rows=self._make_rows())
        lgb_ds = ds.to_lgb_dataset()
        lgb_ds.construct()
        assert lgb_ds.num_data() == 15
        assert lgb_ds.num_feature() == 13

    def test_empty_dataset(self) -> None:
        ds = RankerDataset(rows=[])
        X, y = ds.to_numpy()
        assert X.shape == (0, 13)
        assert y.shape == (0,)

    def test_dataset_with_source_feature_columns(self) -> None:
        rows = [RankerRow(person_id=1, hobby_id=1, label=1, features={col: 1.0 for col in RANKER_FEATURE_COLUMNS_WITH_SOURCE})]
        ds = RankerDataset(rows=rows, feature_columns=RANKER_FEATURE_COLUMNS_WITH_SOURCE)
        X, y = ds.to_numpy()
        assert X.shape == (1, 16)
        assert y.shape == (1,)


class TestBuildRankerDataset:
    def _make_fixtures(self) -> tuple:
        split_edges = [(0, 100), (0, 101), (1, 102)]
        candidate_pool: dict[int, list[HobbyCandidate]] = {
            0: [
                HobbyCandidate(hobby_id=100, hobby_name="축구", source_scores={"popularity": 0.8}, raw_source_scores={}, reason_features={}),
                HobbyCandidate(hobby_id=103, hobby_name="농구", source_scores={"popularity": 0.5}, raw_source_scores={}, reason_features={}),
                HobbyCandidate(hobby_id=104, hobby_name="수영", source_scores={"popularity": 0.3}, raw_source_scores={}, reason_features={}),
            ],
            1: [
                HobbyCandidate(hobby_id=102, hobby_name="요리", source_scores={"cooccurrence": 0.6}, raw_source_scores={}, reason_features={}),
                HobbyCandidate(hobby_id=105, hobby_name="독서", source_scores={"popularity": 0.4}, raw_source_scores={}, reason_features={}),
            ],
        }
        all_hobby_ids = [100, 101, 102, 103, 104, 105, 106, 107]
        known_by_person: dict[int, set[int]] = {0: {50, 51}, 1: {52}}
        id_to_hobby = {100: "축구", 101: "배구", 102: "요리", 103: "농구", 104: "수영", 105: "독서", 106: "등산", 107: "낚시"}
        contexts = {
            "uuid_0": PersonContext(
                person_uuid="uuid_0", age="25", age_group="20대", sex="남성",
                occupation="학생", district="강남구", province="서울",
                family_type="1인 가구", housing_type="아파트", education_level="대학교",
                persona_text="", professional_text="", sports_text="", arts_text="",
                travel_text="", culinary_text="", family_text="",
                hobbies_text="축구,배구", skills_text="", career_goals="", embedding_text="",
            ),
            "uuid_1": PersonContext(
                person_uuid="uuid_1", age="35", age_group="30대", sex="여성",
                occupation="회사원", district="해운대구", province="부산",
                family_type="부부", housing_type="아파트", education_level="대학교",
                persona_text="", professional_text="", sports_text="", arts_text="",
                travel_text="", culinary_text="", family_text="",
                hobbies_text="요리", skills_text="", career_goals="", embedding_text="",
            ),
        }
        id_to_person = {0: "uuid_0", 1: "uuid_1"}
        hobby_profile: dict[str, object] = {
            "hobbies": {
                "축구": {"count": 10, "age_groups": {"20대": 5}, "occupations": {"학생": 3}, "regions": {"서울": 4}},
                "배구": {"count": 5, "age_groups": {"20대": 2}, "occupations": {"학생": 1}, "regions": {"서울": 2}},
                "요리": {"count": 8, "age_groups": {"30대": 4}, "occupations": {"회사원": 3}, "regions": {"부산": 3}},
                "농구": {"count": 6, "age_groups": {"20대": 3}, "occupations": {"학생": 2}, "regions": {"서울": 3}},
                "수영": {"count": 4, "age_groups": {"20대": 1}, "occupations": {"학생": 1}, "regions": {"서울": 1}},
                "독서": {"count": 7, "age_groups": {"30대": 3}, "occupations": {"회사원": 2}, "regions": {"부산": 2}},
                "등산": {"count": 3, "age_groups": {}, "occupations": {}, "regions": {}},
                "낚시": {"count": 2, "age_groups": {}, "occupations": {}, "regions": {}},
            },
            "total_persons": 100,
        }
        reranker_config = build_reranker_config(False, None)
        return split_edges, candidate_pool, all_hobby_ids, known_by_person, id_to_hobby, contexts, id_to_person, hobby_profile, reranker_config

    def test_builds_positive_and_negative_rows(self) -> None:
        args = self._make_fixtures()
        ds = build_ranker_dataset(*args, neg_ratio=2, hard_ratio=0.5, seed=42)
        pos = [r for r in ds.rows if r.label == 1]
        neg = [r for r in ds.rows if r.label == 0]
        assert len(pos) >= 1
        assert len(neg) >= 1
        assert len(neg) >= len(pos)

    def test_excludes_leakage_features(self) -> None:
        args = self._make_fixtures()
        ds = build_ranker_dataset(*args, neg_ratio=2, seed=42)
        for row in ds.rows:
            assert "similar_person_score" not in row.features
            assert "persona_text_fit" not in row.features

    def test_all_features_present(self) -> None:
        args = self._make_fixtures()
        ds = build_ranker_dataset(*args, neg_ratio=2, seed=42)
        for row in ds.rows:
            for col in RANKER_FEATURE_COLUMNS:
                assert col in row.features, f"Missing feature: {col}"

    def test_deterministic_with_same_seed(self) -> None:
        args = self._make_fixtures()
        ds1 = build_ranker_dataset(*args, neg_ratio=2, seed=99)
        ds2 = build_ranker_dataset(*args, neg_ratio=2, seed=99)
        assert len(ds1.rows) == len(ds2.rows)
        for r1, r2 in zip(ds1.rows, ds2.rows):
            assert r1.hobby_id == r2.hobby_id
            assert r1.label == r2.label

    def test_skips_persons_without_context(self) -> None:
        args = list(self._make_fixtures())
        args[5] = {}
        ds = build_ranker_dataset(*args, neg_ratio=2, seed=42)
        assert len(ds.rows) == 0

    def test_source_features_added_when_enabled(self) -> None:
        args = self._make_fixtures()
        ds = build_ranker_dataset(*args, neg_ratio=1, seed=42, include_source_features=True)
        assert ds.feature_columns == RANKER_FEATURE_COLUMNS_WITH_SOURCE
        row_by_hobby = {row.hobby_id: row for row in ds.rows}
        assert row_by_hobby[100].features["source_is_popularity"] == 1.0
        assert row_by_hobby[100].features["source_is_cooccurrence"] == 0.0
        assert row_by_hobby[100].features["source_count"] == 1.0
        assert row_by_hobby[102].features["source_is_popularity"] == 0.0
        assert row_by_hobby[102].features["source_is_cooccurrence"] == 1.0
        assert row_by_hobby[102].features["source_count"] == 1.0


class TestLightGBMRanker:
    def test_predict_before_fit_raises(self) -> None:
        ranker = LightGBMRanker()
        with pytest.raises(ValueError, match="not trained"):
            ranker.predict(np.zeros((5, 13), dtype=np.float32))

    def test_save_before_fit_raises(self, tmp_path: Path) -> None:
        ranker = LightGBMRanker()
        with pytest.raises(ValueError, match="not trained"):
            ranker.save(tmp_path / "model.txt")

    def test_feature_importance_before_fit(self) -> None:
        ranker = LightGBMRanker()
        assert ranker.feature_importance() == {}

    def test_default_params(self) -> None:
        ranker = LightGBMRanker()
        assert ranker.params["objective"] == "binary"
        assert ranker.params["metric"] == "auc"
        assert ranker.params["num_leaves"] == 15

    def test_custom_params_override(self) -> None:
        ranker = LightGBMRanker(params={"num_leaves": 31, "learning_rate": 0.1})
        assert ranker.params["num_leaves"] == 31
        assert ranker.params["learning_rate"] == 0.1
        assert ranker.params["objective"] == "binary"

    def test_fit_predict_save_load(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(42)
        n_train, n_val, n_feat = 200, 50, 13
        X_train = rng.randn(n_train, n_feat).astype(np.float32)
        y_train = (rng.rand(n_train) > 0.5).astype(np.float32)
        X_val = rng.randn(n_val, n_feat).astype(np.float32)
        y_val = (rng.rand(n_val) > 0.5).astype(np.float32)

        import lightgbm as lgb
        train_ds = lgb.Dataset(X_train, label=y_train, feature_name=RANKER_FEATURE_COLUMNS, free_raw_data=False)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds, free_raw_data=False)

        ranker = LightGBMRanker()
        metadata = ranker.fit(train_ds, val_ds, num_boost_round=10, early_stopping_rounds=5)

        assert "best_iteration" in metadata
        assert "best_score" in metadata
        assert "feature_importance" in metadata
        assert metadata["best_iteration"] >= 1

        preds = ranker.predict(X_val)
        assert preds.shape == (n_val,)
        assert all(0.0 <= p <= 1.0 for p in preds)

        model_path = tmp_path / "ranker_model.txt"
        ranker.save(model_path)
        assert model_path.exists()

        loaded = LightGBMRanker.load(model_path)
        loaded_preds = loaded.predict(X_val)
        np.testing.assert_array_almost_equal(preds, loaded_preds, decimal=6)


class TestCandidatePoolCache:
    def test_cache_hit_and_miss_when_loading_candidate_pool(self, tmp_path: Path) -> None:
        person_ids = [1]
        train_edges = [(1, 101), (2, 102), (2, 103)]
        train_known = {1: {101}, 2: {102, 103}}
        candidate_k = 5
        id_to_hobby = {101: "축구", 102: "농구", 103: "요리"}

        popularity_counts = build_popularity_counts(train_edges)
        cooccurrence_counts = build_cooccurrence_counts(train_edges)
        pool_dir = tmp_path / "cache_root"
        pool_dir.mkdir(parents=True)

        first = load_or_build_candidate_pool(
            person_ids=person_ids,
            train_edges=train_edges,
            train_known=train_known,
            candidate_k=candidate_k,
            id_to_hobby=id_to_hobby,
            popularity_counts=popularity_counts,
            cooccurrence_counts=cooccurrence_counts,
            normalization_method="rank_percentile",
            cache_dir=pool_dir,
            label="validation_internal_ranker_split",
        )
        assert 1 in first
        assert len(first[1]) >= 1
        cache_files = list((pool_dir / "cache").glob("pool_*.json"))
        assert len(cache_files) == 1

        second = load_or_build_candidate_pool(
            person_ids=person_ids,
            train_edges=train_edges,
            train_known=train_known,
            candidate_k=candidate_k,
            id_to_hobby=id_to_hobby,
            popularity_counts=popularity_counts,
            cooccurrence_counts=cooccurrence_counts,
            normalization_method="rank_percentile",
            cache_dir=pool_dir,
            label="validation_internal_ranker_split",
        )
        assert [candidate.hobby_id for candidate in second[1]] == [candidate.hobby_id for candidate in first[1]]
        assert second[1][0].raw_source_scores == first[1][0].raw_source_scores

        assert first[1][0].raw_source_scores is not None

    def test_invalid_candidate_pool_cache_is_rebuilt(self, tmp_path: Path) -> None:
        person_ids = [1]
        train_edges = [(1, 101), (2, 102)]
        train_known = {1: {101}, 2: {102}}
        candidate_k = 5
        id_to_hobby = {101: "축구", 102: "농구"}

        popularity_counts = build_popularity_counts(train_edges)
        cooccurrence_counts = build_cooccurrence_counts(train_edges)
        pool_dir = tmp_path / "cache_root"
        pool_dir.mkdir(parents=True)

        cache_key = get_candidate_pool_cache_key(
            person_ids=person_ids,
            train_edges=train_edges,
            id_to_hobby=id_to_hobby,
            candidate_k=candidate_k,
            normalization_method="rank_percentile",
            label="validation_internal_ranker_split",
        )
        (pool_dir / "cache").mkdir(parents=True)
        corrupted_cache = pool_dir / "cache" / f"{cache_key}.json"
        corrupted_cache.write_text('{"bad_person": {}}', encoding="utf-8")

        loaded = load_or_build_candidate_pool(
            person_ids=person_ids,
            train_edges=train_edges,
            train_known=train_known,
            candidate_k=candidate_k,
            id_to_hobby=id_to_hobby,
            popularity_counts=popularity_counts,
            cooccurrence_counts=cooccurrence_counts,
            normalization_method="rank_percentile",
            cache_dir=pool_dir,
            label="validation_internal_ranker_split",
        )
        assert 1 in loaded
        assert len(loaded[1]) >= 1
