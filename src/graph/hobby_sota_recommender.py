from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MODEL_VERSION = "hobby-e5-domain-lightgbm-2026-05-17"
MODEL_NAME = "dragonkue/multilingual-e5-small-ko-v2"
TEXT_PREPROCESSING_VERSION = "domain_tagged_masked_v1"


@dataclass(frozen=True)
class _SotaAssets:
    person_to_id: dict[str, int]
    id_to_hobby: dict[int, str]
    train_edges: list[tuple[int, int]]
    train_known: dict[int, set[int]]
    popularity_counts: Counter[int]
    cooccurrence_counts: dict[int, Counter[int]]
    normalization_method: str
    contexts: dict[str, Any]
    hobby_profile: dict[str, object]
    reranker_config: Any
    ranker: Any
    feature_columns: list[str]
    person_embedding_cache: Any
    hobby_embedding_cache: Any


class HobbySotaRecommendationService:
    """Read-only adapter for the promoted hobby_recommender_ml hobby recommender."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self._assets: _SotaAssets | None = None

    @property
    def artifact_path(self) -> Path:
        return (
            self.repo_root
            / "experiments/hobby_recommender_ml"
            / "artifacts"
            / "experiments"
            / "phase5_c_text_embedding"
            / "e5_domain_features_validation_thread18"
            / "ranker_model.txt"
        )

    def is_available(self) -> bool:
        return self.artifact_path.exists()

    def recommend(
        self,
        uuid: str,
        *,
        top_n: int = 5,
        exclude_hobby_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        assets = self._load_assets()
        person_id = assets.person_to_id.get(uuid)
        if person_id is None:
            raise ValueError(f"Persona UUID is not present in hobby recommender mapping: {uuid}")

        candidates = self._build_candidates(assets, person_id)
        excluded = {self._normalize_name(name) for name in (exclude_hobby_names or set()) if name}
        candidates = [
            candidate
            for candidate in candidates
            if self._normalize_name(str(getattr(candidate, "hobby_name", ""))) not in excluded
        ]
        if not candidates:
            return []

        context = assets.contexts.get(uuid)
        if context is None:
            from experiments.hobby_recommender_ml.hobby_recommender.data import empty_person_context

            context = empty_person_context(uuid)

        known_names = {
            assets.id_to_hobby[hobby_id]
            for hobby_id in assets.train_known.get(person_id, set())
            if hobby_id in assets.id_to_hobby
        }
        known_names.update(exclude_hobby_names or set())

        feature_rows = self._build_feature_rows(assets, context, candidates, known_names)
        if feature_rows.size == 0:
            return []

        scores = assets.ranker.predict(feature_rows)
        ranked = sorted(
            zip(candidates, scores, strict=False),
            key=lambda item: (-float(item[1]), int(getattr(item[0], "hobby_id", 0))),
        )
        return [
            self._format_item(candidate, float(score), rank=index)
            for index, (candidate, score) in enumerate(ranked[:top_n], start=1)
        ]

    def _load_assets(self) -> _SotaAssets:
        if self._assets is not None:
            return self._assets

        from experiments.hobby_recommender_ml.hobby_recommender.baseline import build_cooccurrence_counts, build_popularity_counts
        from experiments.hobby_recommender_ml.hobby_recommender.config import load_config
        from experiments.hobby_recommender_ml.hobby_recommender.data import load_json, load_person_contexts
        from experiments.hobby_recommender_ml.hobby_recommender.embedding_cache import HobbyEmbeddingCache, PersonEmbeddingCache
        from experiments.hobby_recommender_ml.hobby_recommender.ranker import LightGBMRanker
        from experiments.hobby_recommender_ml.hobby_recommender.rerank import build_reranker_config

        config_path = self.repo_root / "experiments/hobby_recommender_ml" / "configs" / "kure_text_optin_ranker.yaml"
        config = load_config(config_path)
        model_path = self.artifact_path
        if not model_path.exists():
            raise FileNotFoundError(f"Promoted hobby ranker artifact not found: {model_path}")

        person_to_id = {str(key): int(value) for key, value in load_json(config.paths.person_mapping).items()}
        hobby_to_id = {str(key): int(value) for key, value in load_json(config.paths.hobby_mapping).items()}
        id_to_hobby = {value: key for key, value in hobby_to_id.items()}
        train_edges = _read_indexed_edges(config.paths.train_edges)
        train_known = _known_from_edges(train_edges)
        popularity_counts = build_popularity_counts(train_edges)
        cooccurrence_counts = build_cooccurrence_counts(train_edges)
        normalization_method = _normalization_method(config.paths.score_normalization)
        contexts = load_person_contexts(config.paths.person_context_csv)
        hobby_profile = load_json(config.paths.hobby_profile)
        if not isinstance(hobby_profile, dict):
            raise ValueError("hobby_profile.json must contain an object")
        reranker_config = build_reranker_config(config.rerank.use_text_fit, config.rerank.weights)
        ranker = LightGBMRanker.load(model_path)
        feature_columns = ranker.feature_columns()
        cache_dir = config.paths.artifact_dir / "text_embedding_cache"
        person_embedding_cache = PersonEmbeddingCache(
            cache_dir,
            model_name=MODEL_NAME,
            preprocessing_version=TEXT_PREPROCESSING_VERSION,
            batch_size=8,
        )
        hobby_embedding_cache = HobbyEmbeddingCache(
            cache_dir,
            model_name=MODEL_NAME,
            preprocessing_version=TEXT_PREPROCESSING_VERSION,
            batch_size=8,
        )
        self._assets = _SotaAssets(
            person_to_id=person_to_id,
            id_to_hobby=id_to_hobby,
            train_edges=train_edges,
            train_known=train_known,
            popularity_counts=popularity_counts,
            cooccurrence_counts=cooccurrence_counts,
            normalization_method=normalization_method,
            contexts=contexts,
            hobby_profile=hobby_profile,
            reranker_config=reranker_config,
            ranker=ranker,
            feature_columns=feature_columns,
            person_embedding_cache=person_embedding_cache,
            hobby_embedding_cache=hobby_embedding_cache,
        )
        return self._assets

    def _build_candidates(self, assets: _SotaAssets, person_id: int) -> list[Any]:
        from experiments.hobby_recommender_ml.hobby_recommender.ranker import load_or_build_candidate_pool

        pools = load_or_build_candidate_pool(
            person_ids=[person_id],
            train_edges=assets.train_edges,
            train_known=assets.train_known,
            candidate_k=50,
            id_to_hobby=assets.id_to_hobby,
            popularity_counts=assets.popularity_counts,
            cooccurrence_counts=assets.cooccurrence_counts,
            normalization_method=assets.normalization_method,
            cache_dir=None,
            label="serving",
            disable_progress=True,
            stage1_providers=("popularity", "cooccurrence"),
        )
        return pools.get(person_id, [])

    def _build_feature_rows(
        self,
        assets: _SotaAssets,
        context: Any,
        candidates: list[Any],
        known_hobby_names: set[str],
    ) -> np.ndarray:
        from experiments.hobby_recommender_ml.hobby_recommender.data import build_domain_persona_texts, build_domain_tagged_persona_text
        from experiments.hobby_recommender_ml.hobby_recommender.rerank import build_rerank_features

        person_text = build_domain_tagged_persona_text(context)
        person_domain_texts = build_domain_persona_texts(context)
        person_vector = self._normalized_embedding(assets.person_embedding_cache, person_text)
        domain_vectors = {
            domain: vector
            for domain, text in person_domain_texts.items()
            if (vector := self._normalized_embedding(assets.person_embedding_cache, text)) is not None
        }

        rows: list[list[float]] = []
        for candidate in candidates:
            hobby_name = str(getattr(candidate, "hobby_name", "") or "")
            hobby_vector = self._normalized_embedding(assets.hobby_embedding_cache, hobby_name)
            text_similarity = (
                float(np.dot(person_vector, hobby_vector))
                if person_vector is not None and hobby_vector is not None
                else 0.0
            )
            features = build_rerank_features(
                context,
                candidate,
                assets.hobby_profile,
                known_hobby_names,
                assets.reranker_config,
                text_embedding_similarity=max(0.0, min(1.0, text_similarity)),
            )
            features.update(self._domain_similarity_features(domain_vectors, hobby_vector))
            rows.append([float(features.get(column, 0.0)) for column in assets.feature_columns])

        if not rows:
            return np.empty((0, len(assets.feature_columns)), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def _domain_similarity_features(domain_vectors: dict[str, np.ndarray], hobby_vector: np.ndarray | None) -> dict[str, float]:
        if hobby_vector is None:
            return {}
        mapping = {
            "professional": "e5_professional_similarity",
            "sports": "e5_sports_similarity",
            "arts": "e5_arts_similarity",
            "travel": "e5_travel_similarity",
            "food": "e5_food_similarity",
            "family": "e5_family_similarity",
        }
        return {
            feature: max(0.0, min(1.0, float(np.dot(vector, hobby_vector))))
            for domain, vector in domain_vectors.items()
            if (feature := mapping.get(domain))
        }

    @staticmethod
    def _normalized_embedding(cache: Any, text: str) -> np.ndarray | None:
        text = str(text or "").strip()
        if not text:
            return None
        vector = cache.get(text)
        if vector is None:
            vector = cache.encode(text)
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(array))
        if norm <= 0.0:
            return None
        return array / norm

    @staticmethod
    def _format_item(candidate: Any, score: float, rank: int) -> dict[str, Any]:
        source_scores = {
            str(key): float(value)
            for key, value in dict(getattr(candidate, "source_scores", {}) or {}).items()
        }
        sources = sorted({"lightgbm_ranker", "e5_domain_similarity", *source_scores.keys()})
        reason = "SOTA 취미 추천 모델이 Stage1 후보군 안에서 이 취미를 상위로 재정렬했습니다."
        return {
            "item_name": str(getattr(candidate, "hobby_name", "") or ""),
            "reason": reason,
            "reason_score": round(score, 6),
            "similar_users_count": 0,
            "supporting_personas": [],
            "score": round(score, 6),
            "rank": rank,
            "reason_cards": [
                {
                    "type": "model_rank",
                    "title": "SOTA 랭킹 모델",
                    "detail": "E5-small-ko-v2 domain-specific feature와 LightGBM Stage2 랭커가 상위로 정렬한 취미입니다.",
                    "strength": round(max(0.0, min(1.0, score)), 6),
                },
                {
                    "type": "cooccurrence",
                    "title": "Stage1 후보 근거",
                    "detail": "popularity와 cooccurrence 기반 후보풀에서 선택된 취미입니다.",
                    "strength": round(max(source_scores.values()) if source_scores else 0.0, 6),
                },
            ],
            "already_known": False,
            "sources": sources,
            "score_source": "promoted",
            "model_version": MODEL_VERSION,
            "graph_snapshot_id": None,
            "fallback_used": False,
            "fallback_reason": "",
        }

    @staticmethod
    def _normalize_name(value: str) -> str:
        return " ".join(str(value or "").strip().lower().split())


def _read_indexed_edges(path: Path) -> list[tuple[int, int]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [(int(row["person_id"]), int(row["hobby_id"])) for row in reader]


def _known_from_edges(edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    known: dict[int, set[int]] = defaultdict(set)
    for person_id, hobby_id in edges:
        known[person_id].add(hobby_id)
    return dict(known)


def _normalization_method(path: Path) -> str:
    if not path.exists():
        return "rank_percentile"
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        return "rank_percentile"
    return str(value.get("method", "rank_percentile"))
