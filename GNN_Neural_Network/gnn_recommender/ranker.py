from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
from tqdm import tqdm

from .data import PersonContext
from .rerank import (
    HobbyCandidate, RerankerConfig, build_rerank_features, merge_stage1_candidates,
)


# --- Ranker Row Schema ---
# PRD §4.2: person_id, candidate_hobby_id, split, label, + 14 features
# Features to INCLUDE (from build_rerank_features):
#   lightgcn_score, cooccurrence_score, segment_popularity_score,
#   known_hobby_compatibility, age_group_fit, occupation_fit, region_fit,
#   popularity_prior, mismatch_penalty, popularity_penalty, novelty_bonus,
#   category_diversity_reward, is_cold_start
# Features to EXCLUDE: similar_person_score (always 0), persona_text_fit (leakage)
# Optional (Phase 4): text_embedding_similarity (default 0.0)

RANKER_BASE_FEATURE_COLUMNS: list[str] = [
    "lightgcn_score", "cooccurrence_score", "segment_popularity_score",
    "known_hobby_compatibility", "age_group_fit", "occupation_fit", "region_fit",
    "popularity_prior", "mismatch_penalty", "popularity_penalty", "novelty_bonus",
    "category_diversity_reward", "is_cold_start",
]

RANKER_TEXT_FEATURE_COLUMNS: list[str] = [
    "text_embedding_similarity",
]

RANKER_SOURCE_FEATURE_COLUMNS: list[str] = [
    "source_is_popularity",
    "source_is_cooccurrence",
    "source_count",
]

RANKER_FEATURE_COLUMNS: list[str] = list(RANKER_BASE_FEATURE_COLUMNS)
RANKER_FEATURE_COLUMNS_WITH_TEXT: list[str] = list(RANKER_BASE_FEATURE_COLUMNS) + list(RANKER_TEXT_FEATURE_COLUMNS)
RANKER_FEATURE_COLUMNS_WITH_SOURCE: list[str] = list(RANKER_BASE_FEATURE_COLUMNS) + list(RANKER_SOURCE_FEATURE_COLUMNS)
RANKER_FEATURE_COLUMNS_WITH_SOURCE_AND_TEXT: list[str] = list(RANKER_BASE_FEATURE_COLUMNS) + list(RANKER_SOURCE_FEATURE_COLUMNS) + list(RANKER_TEXT_FEATURE_COLUMNS)

RANKER_CATEGORICAL_FEATURES: list[str] = ["is_cold_start"]
RANKER_CATEGORICAL_FEATURES_WITH_SOURCE: list[str] = [
    "is_cold_start",
    "source_is_popularity",
    "source_is_cooccurrence",
]


def get_ranker_feature_columns(
    include_source_features: bool = False,
    include_text_embedding_feature: bool = False,
) -> list[str]:
    if include_source_features and include_text_embedding_feature:
        return list(RANKER_FEATURE_COLUMNS_WITH_SOURCE_AND_TEXT)
    if include_source_features:
        return list(RANKER_FEATURE_COLUMNS_WITH_SOURCE)
    if include_text_embedding_feature:
        return list(RANKER_FEATURE_COLUMNS_WITH_TEXT)
    return list(RANKER_FEATURE_COLUMNS)


def get_ranker_categorical_features(feature_columns: list[str] | None = None) -> list[str]:
    columns = feature_columns or RANKER_FEATURE_COLUMNS
    base = RANKER_CATEGORICAL_FEATURES_WITH_SOURCE if any(col in columns for col in RANKER_SOURCE_FEATURE_COLUMNS) else RANKER_CATEGORICAL_FEATURES
    return [col for col in base if col in columns]


@dataclass
class RankerRow:
    person_id: int
    hobby_id: int
    label: int  # 1=positive, 0=negative
    features: dict[str, float]


@dataclass
class RankerDataset:
    rows: list[RankerRow]
    feature_columns: list[str] = field(default_factory=lambda: list(RANKER_FEATURE_COLUMNS))
    
    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (X, y) numpy arrays for LightGBM."""
        if not self.rows:
            return np.empty((0, len(self.feature_columns)), dtype=np.float32), np.empty((0,), dtype=np.float32)
        X = np.array([[row.features.get(col, 0.0) for col in self.feature_columns] for row in self.rows], dtype=np.float32)
        y = np.array([row.label for row in self.rows], dtype=np.float32)
        return X, y

    def to_lgb_dataset(self, reference: lgb.Dataset | None = None) -> lgb.Dataset:
        X, y = self.to_numpy()
        categorical_features = get_ranker_categorical_features(self.feature_columns)
        cat_indices = [self.feature_columns.index(c) for c in categorical_features if c in self.feature_columns]
        return lgb.Dataset(
            X, label=y,
            feature_name=self.feature_columns,
            categorical_feature=cat_indices if cat_indices else "auto",
            reference=reference,
            free_raw_data=False,
        )


def sample_negatives(
    person_id: int,
    positive_hobby_ids: set[int],
    candidate_pool: list[int],
    all_hobby_ids: list[int],
    known_hobby_ids: set[int],
    neg_ratio: int = 4,
    hard_ratio: float = 0.8,
    rng: random.Random | None = None,
) -> list[int]:
    """
    Mixed Negative Sampling (MNS): Hard+Easy 4:1 ratio.
    - Hard negatives: from candidate_pool, not in positive_hobby_ids and not in known_hobby_ids
    - Easy negatives: random from all_hobby_ids, not in positive_hobby_ids and not in known_hobby_ids
    - hard_ratio=0.8 means 80% hard, 20% easy within the neg_ratio * len(positives) total
    """
    if rng is None:
        rng = random.Random()
        
    _ = person_id  # Unused but kept for signature compatibility
        
    num_positives = len(positive_hobby_ids)
    total_negatives = neg_ratio * num_positives
    if total_negatives == 0:
        return []
        
    num_hard = int(total_negatives * hard_ratio)
    
    # Hard negatives: from candidate_pool, not in positive_hobby_ids and not in known_hobby_ids
    hard_candidates = [h for h in candidate_pool if h not in positive_hobby_ids and h not in known_hobby_ids]
    
    # Easy negatives: from all_hobby_ids, not in positive_hobby_ids, not in known_hobby_ids, not in candidate_pool
    candidate_pool_set = set(candidate_pool)
    easy_candidates = [h for h in all_hobby_ids if h not in positive_hobby_ids and h not in known_hobby_ids and h not in candidate_pool_set]
    
    sampled_hard = []
    if hard_candidates:
        sampled_hard = rng.sample(hard_candidates, min(num_hard, len(hard_candidates)))
        
    # If not enough hard candidates, fill remaining with easy
    remaining_easy_needed = total_negatives - len(sampled_hard)
    
    sampled_easy = []
    if easy_candidates and remaining_easy_needed > 0:
        sampled_easy = rng.sample(easy_candidates, min(remaining_easy_needed, len(easy_candidates)))
        
    return sampled_hard + sampled_easy


def build_ranker_dataset(
    split_edges: list[tuple[int, int]],
    candidate_pools: dict[int, list[HobbyCandidate]],
    all_hobby_ids: list[int],
    known_by_person: dict[int, set[int]],
    id_to_hobby: dict[int, str],
    contexts: dict[str, PersonContext],
    id_to_person: dict[int, str],
    hobby_profile: dict[str, object],
    reranker_config: RerankerConfig,
    neg_ratio: int = 4,
    hard_ratio: float = 0.8,
    seed: int = 42,
    include_source_features: bool = False,
    include_text_embedding_feature: bool = False,
) -> RankerDataset:
    rng = random.Random(seed)

    positives_by_person: dict[int, set[int]] = {}
    for person_id, hobby_id in split_edges:
        positives_by_person.setdefault(person_id, set()).add(hobby_id)

    rows: list[RankerRow] = []
    feature_columns = get_ranker_feature_columns(
        include_source_features=include_source_features,
        include_text_embedding_feature=include_text_embedding_feature,
    )

    for person_id, positive_hobby_ids in positives_by_person.items():
        person_uuid = id_to_person.get(person_id)
        if not person_uuid:
            continue
        context = contexts.get(person_uuid)
        if not context:
            continue

        known_hobby_ids = known_by_person.get(person_id, set())
        known_hobby_names = {id_to_hobby[h] for h in known_hobby_ids if h in id_to_hobby}

        pool_candidates = candidate_pools.get(person_id, [])
        pool_hobby_ids = [c.hobby_id for c in pool_candidates]
        pool_lookup: dict[int, HobbyCandidate] = {c.hobby_id: c for c in pool_candidates}

        negatives = sample_negatives(
            person_id=person_id,
            positive_hobby_ids=positive_hobby_ids,
            candidate_pool=pool_hobby_ids,
            all_hobby_ids=all_hobby_ids,
            known_hobby_ids=known_hobby_ids,
            neg_ratio=neg_ratio,
            hard_ratio=hard_ratio,
            rng=rng,
        )

        def _make_candidate(hid: int) -> HobbyCandidate:
            if hid in pool_lookup:
                return pool_lookup[hid]
            return HobbyCandidate(
                hobby_id=hid,
                hobby_name=id_to_hobby.get(hid, ""),
                source_scores={},
                raw_source_scores={},
                reason_features={},
            )

        person_context: PersonContext = context

        def _build_row(hid: int, label: int) -> RankerRow:
            candidate = _make_candidate(hid)
            features = build_rerank_features(person_context, candidate, hobby_profile, known_hobby_names, reranker_config)
            features.pop("similar_person_score", None)
            features.pop("persona_text_fit", None)
            return RankerRow(person_id=person_id, hobby_id=hid, label=label, features=features)

        for hobby_id in positive_hobby_ids:
            rows.append(_build_row(hobby_id, 1))

        for hobby_id in negatives:
            rows.append(_build_row(hobby_id, 0))

    return RankerDataset(rows=rows, feature_columns=feature_columns)


class LightGBMRanker:
    DEFAULT_PARAMS: dict[str, Any] = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 15,
        "min_data_in_leaf": 50,
        "learning_rate": 0.05,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "seed": 42,
        "num_threads": -1,
    }

    def __init__(self, params: dict[str, Any] | None = None):
        self.params: dict[str, Any] = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: lgb.Booster | None = None
        self.best_iteration: int = 0
        self.best_score: float = 0.0

    def fit(
        self,
        train_dataset: lgb.Dataset,
        val_dataset: lgb.Dataset,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
    ) -> dict[str, Any]:
        """Train and return training metadata."""
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        ]
        
        self.model = lgb.train(
            params=self.params,
            train_set=train_dataset,
            num_boost_round=num_boost_round,
            valid_sets=[train_dataset, val_dataset],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
        
        self.best_iteration = self.model.best_iteration
        self.best_score = self.model.best_score["val"]["auc"]
        
        return {
            "params": self.params,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "feature_importance": self.feature_importance(),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probability scores."""
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        result = self.model.predict(X, num_iteration=self.best_iteration)
        return cast(np.ndarray, result)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = self.model.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> LightGBMRanker:
        ranker = cls()
        ranker.model = lgb.Booster(model_file=str(path))
        ranker.best_iteration = ranker.model.best_iteration
        return ranker

    def feature_importance(self) -> dict[str, float]:
        """Return feature importance as {feature_name: importance}."""
        if self.model is None:
            return {}
        importance = self.model.feature_importance(importance_type="gain")
        names = self.model.feature_name()
        return {name: float(imp) for name, imp in zip(names, importance, strict=False)}

    def feature_columns(self) -> list[str]:
        if self.model is None:
            return list(RANKER_FEATURE_COLUMNS)
        return [str(name) for name in self.model.feature_name()]


def _pool_cache_key(
    person_ids: list[int],
    train_edges: list[tuple[int, int]],
    id_to_hobby: dict[int, str],
    candidate_k: int,
    normalization_method: str,
    label: str,
) -> str:
    pid_hash = hashlib.md5(str(sorted(person_ids)).encode()).hexdigest()[:8]
    edge_hash = _hash_indexed_edges(train_edges)
    hobby_hash = _hash_id_mapping(id_to_hobby)
    providers = "popularity-cooccurrence"
    return f"pool_{label}_{providers}_k{candidate_k}_{normalization_method}_e{edge_hash}_h{hobby_hash}_p{pid_hash}"


def get_candidate_pool_cache_key(
    person_ids: list[int],
    train_edges: list[tuple[int, int]],
    id_to_hobby: dict[int, str],
    candidate_k: int,
    normalization_method: str,
    label: str,
) -> str:
    return _pool_cache_key(
        person_ids=person_ids,
        train_edges=train_edges,
        id_to_hobby=id_to_hobby,
        candidate_k=candidate_k,
        normalization_method=normalization_method,
        label=label,
    )


def _hash_indexed_edges(edges: list[tuple[int, int]]) -> str:
    hasher = hashlib.md5()
    for person_id, hobby_id in sorted(edges):
        hasher.update(f"{person_id}:{hobby_id};".encode("utf-8"))
    return hasher.hexdigest()[:12]


def _hash_id_mapping(id_to_hobby: dict[int, str]) -> str:
    payload = json.dumps(sorted(id_to_hobby.items()), ensure_ascii=False, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def _coerce_score_dict(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, raw_value in value.items():
        try:
            result[str(key)] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return result


def load_or_build_candidate_pool(
    person_ids: list[int],
    train_edges: list[tuple[int, int]],
    train_known: dict[int, set[int]],
    candidate_k: int,
    id_to_hobby: dict[int, str],
    popularity_counts: Counter[int],
    cooccurrence_counts: dict[int, Counter[int]],
    normalization_method: str,
    cache_dir: Path | None = None,
    label: str = "validation",
    disable_progress: bool = False,
) -> dict[int, list[HobbyCandidate]]:
    from .baseline import (
        cooccurrence_candidate_provider,
        popularity_candidate_provider,
    )
    from .recommend import merge_candidates_by_hobby, normalize_candidate_scores

    cache_key = _pool_cache_key(person_ids, train_edges, id_to_hobby, candidate_k, normalization_method, label)

    if cache_dir is not None:
        cache_path = cache_dir / "cache" / f"{cache_key}.json"
        if cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Candidate pool cache read failed, rebuilding: {cache_path} ({exc})")
            else:
                if isinstance(raw, dict):
                    try:
                        pools: dict[int, list[HobbyCandidate]] = {}
                        for pid_str, entries in raw.items():
                            pid = int(pid_str)
                            if not isinstance(entries, list):
                                raise TypeError(f"Candidate entries for person {pid} are not a list")
                            pools[pid] = [
                                HobbyCandidate(
                                    hobby_id=int(e[0]),
                                    hobby_name=id_to_hobby.get(int(e[0]), ""),
                                    source_scores=_coerce_score_dict(e[1] if len(e) >= 2 else None),
                                    raw_source_scores=_coerce_score_dict(e[2] if len(e) > 2 else None),
                                    reason_features={},
                                )
                                for e in entries
                                if isinstance(e, list)
                                and len(e) >= 2
                            ]
                        print(f"Loaded candidate pool from cache: {cache_path}")
                        return pools
                    except (TypeError, ValueError, KeyError) as exc:
                        print(f"Candidate pool cache format invalid, rebuilding: {cache_path} ({exc})")
                else:
                    print(f"Candidate pool cache format invalid, rebuilding: {cache_path}")

    pools = {}
    for person_id in tqdm(person_ids, desc=f"candidate pools ({label})", disable=disable_progress):
        known = train_known.get(person_id, set())
        pop = normalize_candidate_scores(
            popularity_candidate_provider(train_edges, person_id, known, candidate_k, popularity_counts=popularity_counts),
            normalization_method,
        )
        cooc = normalize_candidate_scores(
            cooccurrence_candidate_provider(train_edges, person_id, known, candidate_k, cooccurrence_counts=cooccurrence_counts),
            normalization_method,
        )
        merged = merge_candidates_by_hobby({"popularity": pop, "cooccurrence": cooc}, candidate_k)
        pools[person_id] = merge_stage1_candidates(merged, id_to_hobby)

    if cache_dir is not None:
        cache_path = cache_dir / "cache" / f"{cache_key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        serializable: dict[str, list[list[Any]]] = {}
        for pid, candidates in pools.items():
            serializable[str(pid)] = [
                [c.hobby_id, dict(c.source_scores), dict(c.raw_source_scores)] for c in candidates
            ]
        cache_path.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
        print(f"Candidate pool cached: {cache_path}")

    return pools
