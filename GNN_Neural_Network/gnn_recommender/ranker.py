from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

import lightgbm as lgb
import numpy as np
import torch
from tqdm import tqdm

from .data import PersonContext, empty_person_context
from .rerank import (
    HobbyCandidate, RerankerConfig, build_rerank_features, merge_stage1_candidates,
)

_ranker_worker_all_hobby_ids: list[int] = []
_ranker_worker_known_by_person: dict[int, set[int]] = {}
_ranker_worker_id_to_hobby: dict[int, str] = {}
_ranker_worker_contexts: dict[str, PersonContext] = {}
_ranker_worker_id_to_person: dict[int, str] = {}
_ranker_worker_hobby_profile: dict[str, object] = {}
_ranker_worker_reranker_config: RerankerConfig | None = None
_ranker_worker_neg_ratio = 4
_ranker_worker_hard_ratio = 0.8
_ranker_worker_include_text_embedding_feature = False
_ranker_worker_text_similarity_lookup: dict[int, dict[int, float]] = {}
_ranker_worker_include_domain_text_embedding_features = False
_ranker_worker_domain_similarity_lookup: dict[int, dict[int, dict[str, float]]] = {}
_ranker_worker_include_text_rank_margin_features = False
_ranker_worker_text_rank_margin_lookup: dict[int, dict[int, dict[str, float]]] = {}


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

RANKER_DOMAIN_TEXT_FEATURE_COLUMNS: list[str] = [
    "e5_professional_similarity",
    "e5_sports_similarity",
    "e5_arts_similarity",
    "e5_travel_similarity",
    "e5_food_similarity",
    "e5_family_similarity",
]

RANKER_TEXT_RANK_MARGIN_FEATURE_COLUMNS: list[str] = [
    "e5_similarity_rank",
    "e5_similarity_percentile",
    "e5_similarity_gap_to_top",
    "e5_similarity_gap_to_mean",
]

RANKER_PHASE6_CROSS_FEATURE_COLUMNS: list[str] = [
    "age_group_region_cross_fit",
    "occupation_region_cross_fit",
    "demographic_text_cross_fit",
]

RANKER_SOURCE_FEATURE_COLUMNS: list[str] = [
    "source_is_popularity",
    "source_is_cooccurrence",
    "source_count",
]

RANKER_FEATURE_COLUMNS: list[str] = list(RANKER_BASE_FEATURE_COLUMNS)
RANKER_FEATURE_COLUMNS_WITH_TEXT: list[str] = list(RANKER_BASE_FEATURE_COLUMNS) + list(RANKER_TEXT_FEATURE_COLUMNS)
RANKER_FEATURE_COLUMNS_WITH_TEXT_AND_DOMAIN: list[str] = (
    list(RANKER_BASE_FEATURE_COLUMNS)
    + list(RANKER_TEXT_FEATURE_COLUMNS)
    + list(RANKER_DOMAIN_TEXT_FEATURE_COLUMNS)
)
RANKER_FEATURE_COLUMNS_WITH_TEXT_DOMAIN_AND_RANK_MARGIN: list[str] = (
    list(RANKER_BASE_FEATURE_COLUMNS)
    + list(RANKER_TEXT_FEATURE_COLUMNS)
    + list(RANKER_DOMAIN_TEXT_FEATURE_COLUMNS)
    + list(RANKER_TEXT_RANK_MARGIN_FEATURE_COLUMNS)
)
RANKER_FEATURE_COLUMNS_WITH_SOURCE: list[str] = list(RANKER_BASE_FEATURE_COLUMNS) + list(RANKER_SOURCE_FEATURE_COLUMNS)
RANKER_FEATURE_COLUMNS_WITH_SOURCE_AND_TEXT: list[str] = list(RANKER_BASE_FEATURE_COLUMNS) + list(RANKER_SOURCE_FEATURE_COLUMNS) + list(RANKER_TEXT_FEATURE_COLUMNS)
RANKER_FEATURE_COLUMNS_WITH_SOURCE_TEXT_AND_DOMAIN: list[str] = (
    list(RANKER_BASE_FEATURE_COLUMNS)
    + list(RANKER_SOURCE_FEATURE_COLUMNS)
    + list(RANKER_TEXT_FEATURE_COLUMNS)
    + list(RANKER_DOMAIN_TEXT_FEATURE_COLUMNS)
)
RANKER_FEATURE_COLUMNS_WITH_SOURCE_TEXT_DOMAIN_AND_RANK_MARGIN: list[str] = (
    list(RANKER_BASE_FEATURE_COLUMNS)
    + list(RANKER_SOURCE_FEATURE_COLUMNS)
    + list(RANKER_TEXT_FEATURE_COLUMNS)
    + list(RANKER_DOMAIN_TEXT_FEATURE_COLUMNS)
    + list(RANKER_TEXT_RANK_MARGIN_FEATURE_COLUMNS)
)

RANKER_CATEGORICAL_FEATURES: list[str] = ["is_cold_start"]
RANKER_CATEGORICAL_FEATURES_WITH_SOURCE: list[str] = [
    "is_cold_start",
    "source_is_popularity",
    "source_is_cooccurrence",
]


def get_ranker_feature_columns(
    include_source_features: bool = False,
    include_text_embedding_feature: bool = False,
    include_domain_text_embedding_features: bool = False,
    include_text_rank_margin_features: bool = False,
    include_phase6_cross_features: bool = False,
) -> list[str]:
    columns = list(RANKER_BASE_FEATURE_COLUMNS)
    if include_source_features:
        columns.extend(RANKER_SOURCE_FEATURE_COLUMNS)
    if include_text_embedding_feature or include_domain_text_embedding_features or include_text_rank_margin_features:
        columns.extend(RANKER_TEXT_FEATURE_COLUMNS)
    if include_domain_text_embedding_features:
        columns.extend(RANKER_DOMAIN_TEXT_FEATURE_COLUMNS)
    if include_text_rank_margin_features:
        columns.extend(RANKER_TEXT_RANK_MARGIN_FEATURE_COLUMNS)
    if include_phase6_cross_features:
        columns.extend(RANKER_PHASE6_CROSS_FEATURE_COLUMNS)
    return columns



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

    def _ordered_rows(self) -> list[RankerRow]:
        return sorted(self.rows, key=lambda row: row.person_id)

    @staticmethod
    def _group_sizes_by_person(rows: list[RankerRow]) -> list[int]:
        if not rows:
            return []

        groups: list[int] = []
        current_person = rows[0].person_id
        count = 0
        for row in rows:
            if row.person_id != current_person:
                groups.append(count)
                current_person = row.person_id
                count = 1
            else:
                count += 1
        groups.append(count)
        return groups

    def to_numpy(self, rows: list[RankerRow] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (X, y) numpy arrays for LightGBM."""
        source_rows = self.rows if rows is None else rows
        if not source_rows:
            return np.empty((0, len(self.feature_columns)), dtype=np.float32), np.empty((0,), dtype=np.float32)
        X = np.array(
            [[row.features.get(col, 0.0) for col in self.feature_columns] for row in source_rows],
            dtype=np.float32,
        )
        y = np.array([row.label for row in source_rows], dtype=np.float32)
        return X, y

    def to_lgb_dataset(
        self,
        reference: lgb.Dataset | None = None,
        *,
        group_by_person: bool = False,
    ) -> lgb.Dataset:
        rows = self._ordered_rows() if group_by_person else self.rows
        X, y = self.to_numpy(rows=rows)
        categorical_features = get_ranker_categorical_features(self.feature_columns)
        cat_indices = [self.feature_columns.index(c) for c in categorical_features if c in self.feature_columns]
        group = self._group_sizes_by_person(rows) if group_by_person else None
        return lgb.Dataset(
            X, label=y,
            feature_name=self.feature_columns,
            group=group,
            categorical_feature=cat_indices if cat_indices else "auto",
            reference=reference,
            free_raw_data=False,
        )

    def person_group_sizes(self) -> list[int]:
        return self._group_sizes_by_person(self._ordered_rows())


class LambdaRankLoss:
    """Minimal LambdaRank surrogate objective helper for pairwise ranking experiments."""

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = float(max(sigma, 1e-6))

    def __call__(
        self,
        scores: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        group_sizes: list[int],
    ) -> torch.Tensor:
        score_tensor = torch.as_tensor(scores, dtype=torch.float32).flatten()
        label_tensor = torch.as_tensor(labels, dtype=torch.float32).flatten()
        total_loss = torch.tensor(0.0, dtype=torch.float32)

        if score_tensor.numel() == 0 or len(group_sizes) == 0:
            return total_loss

        start = 0
        for size in group_sizes:
            if size <= 0:
                continue
            end = start + size
            batch_scores = score_tensor[start:end]
            batch_labels = label_tensor[start:end]
            if batch_scores.numel() < 2:
                start = end
                continue

            score_diff = batch_scores[:, None] - batch_scores[None, :]
            label_diff = batch_labels[:, None] - batch_labels[None, :]
            pair_mask = label_diff != 0
            if not bool(pair_mask.any()):
                start = end
                continue

            pair_sign = torch.sign(label_diff)
            pair_loss = torch.log1p(torch.exp(-self.sigma * pair_sign * score_diff))
            total_loss = total_loss + pair_loss[pair_mask].mean()
            start = end

        return total_loss


def create_lambda_rank_dataset(
    dataset: RankerDataset,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return LightGBM LambdaRank-compatible arrays and group sizes."""
    ordered_rows = dataset._ordered_rows()
    X, y = dataset.to_numpy(rows=ordered_rows)
    group_sizes = dataset._group_sizes_by_person(ordered_rows)
    return X, y, group_sizes


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
    include_domain_text_embedding_features: bool = False,
    include_text_rank_margin_features: bool = False,
    include_phase6_cross_features: bool = False,
    text_similarity_fn: Callable[[int, HobbyCandidate], float] | None = None,
    text_similarity_lookup: dict[int, dict[int, float]] | None = None,
    domain_similarity_lookup: dict[int, dict[int, dict[str, float]]] | None = None,
    text_rank_margin_lookup: dict[int, dict[int, dict[str, float]]] | None = None,
    parallel_workers: int | None = None,
    parallel_backend: str = "thread",
    thread_workers: int | None = None,
    show_progress: bool = False,
    progress_desc: str = "ranker rows",
) -> RankerDataset:
    rng = random.Random(seed)

    positives_by_person: dict[int, set[int]] = {}
    for person_id, hobby_id in split_edges:
        positives_by_person.setdefault(person_id, set()).add(hobby_id)

    rows: list[RankerRow] = []
    feature_columns = get_ranker_feature_columns(
        include_source_features=include_source_features,
        include_text_embedding_feature=include_text_embedding_feature,
        include_domain_text_embedding_features=include_domain_text_embedding_features,
        include_text_rank_margin_features=include_text_rank_margin_features,
        include_phase6_cross_features=include_phase6_cross_features,
    )

    worker_count = parallel_workers if parallel_workers is not None else thread_workers
    workers = max(1, int(worker_count or 1))
    backend = parallel_backend.strip().lower()
    if backend == "auto":
        backend = "thread"
    if backend not in {"thread", "serial"}:
        raise ValueError(f"Unsupported ranker dataset parallel_backend: {parallel_backend}")
    if backend == "serial":
        workers = 1

    if workers > 1 and text_similarity_fn is None:
        payloads = [
            (
                person_id,
                positive_hobby_ids,
                candidate_pools.get(person_id, []),
                int(rng.randrange(0, 2**31 - 1)),
            )
            for person_id, positive_hobby_ids in positives_by_person.items()
        ]
        chunksize = max(1, min(64, len(payloads) // (workers * 4) if workers else 1))
        with ThreadPoolExecutor(
            max_workers=workers,
            initializer=_init_ranker_dataset_worker,
            initargs=(
                all_hobby_ids,
                known_by_person,
                id_to_hobby,
                contexts,
                id_to_person,
                hobby_profile,
                reranker_config,
                neg_ratio,
                hard_ratio,
                include_text_embedding_feature,
                text_similarity_lookup or {},
                include_domain_text_embedding_features,
                domain_similarity_lookup or {},
                include_text_rank_margin_features,
                text_rank_margin_lookup or build_text_rank_margin_lookup(candidate_pools, text_similarity_lookup or {}),
            ),
        ) as executor:
            iterator = executor.map(_build_ranker_rows_for_person_worker, payloads, chunksize=chunksize)
            if show_progress:
                iterator = tqdm(iterator, total=len(payloads), desc=progress_desc, dynamic_ncols=False)
            for person_rows in iterator:
                rows.extend(person_rows)
    else:
        serial_items = positives_by_person.items()
        if show_progress:
            serial_items = tqdm(serial_items, total=len(positives_by_person), desc=progress_desc, dynamic_ncols=False)
        for person_id, positive_hobby_ids in serial_items:
            rows.extend(
                _build_ranker_rows_for_person(
                    person_id=person_id,
                    positive_hobby_ids=positive_hobby_ids,
                    pool_candidates=candidate_pools.get(person_id, []),
                    rng=random.Random(rng.randrange(0, 2**31 - 1)),
                    all_hobby_ids=all_hobby_ids,
                    known_by_person=known_by_person,
                    id_to_hobby=id_to_hobby,
                    contexts=contexts,
                    id_to_person=id_to_person,
                    hobby_profile=hobby_profile,
                    reranker_config=reranker_config,
                    neg_ratio=neg_ratio,
                    hard_ratio=hard_ratio,
                    include_text_embedding_feature=include_text_embedding_feature,
                    text_similarity_fn=text_similarity_fn,
                    text_similarity_lookup=text_similarity_lookup or {},
                    include_domain_text_embedding_features=include_domain_text_embedding_features,
                    domain_similarity_lookup=domain_similarity_lookup or {},
                    include_text_rank_margin_features=include_text_rank_margin_features,
                    text_rank_margin_lookup=text_rank_margin_lookup or build_text_rank_margin_lookup(candidate_pools, text_similarity_lookup or {}),
                ),
            )

    return RankerDataset(rows=rows, feature_columns=feature_columns)


def _init_ranker_dataset_worker(
    all_hobby_ids: list[int],
    known_by_person: dict[int, set[int]],
    id_to_hobby: dict[int, str],
    contexts: dict[str, PersonContext],
    id_to_person: dict[int, str],
    hobby_profile: dict[str, object],
    reranker_config: RerankerConfig,
    neg_ratio: int,
    hard_ratio: float,
    include_text_embedding_feature: bool,
    text_similarity_lookup: dict[int, dict[int, float]],
    include_domain_text_embedding_features: bool,
    domain_similarity_lookup: dict[int, dict[int, dict[str, float]]],
    include_text_rank_margin_features: bool,
    text_rank_margin_lookup: dict[int, dict[int, dict[str, float]]],
) -> None:
    global _ranker_worker_all_hobby_ids
    global _ranker_worker_known_by_person
    global _ranker_worker_id_to_hobby
    global _ranker_worker_contexts
    global _ranker_worker_id_to_person
    global _ranker_worker_hobby_profile
    global _ranker_worker_reranker_config
    global _ranker_worker_neg_ratio
    global _ranker_worker_hard_ratio
    global _ranker_worker_include_text_embedding_feature
    global _ranker_worker_text_similarity_lookup
    global _ranker_worker_include_domain_text_embedding_features
    global _ranker_worker_domain_similarity_lookup
    global _ranker_worker_include_text_rank_margin_features
    global _ranker_worker_text_rank_margin_lookup
    _ranker_worker_all_hobby_ids = all_hobby_ids
    _ranker_worker_known_by_person = known_by_person
    _ranker_worker_id_to_hobby = id_to_hobby
    _ranker_worker_contexts = contexts
    _ranker_worker_id_to_person = id_to_person
    _ranker_worker_hobby_profile = hobby_profile
    _ranker_worker_reranker_config = reranker_config
    _ranker_worker_neg_ratio = neg_ratio
    _ranker_worker_hard_ratio = hard_ratio
    _ranker_worker_include_text_embedding_feature = include_text_embedding_feature
    _ranker_worker_text_similarity_lookup = text_similarity_lookup
    _ranker_worker_include_domain_text_embedding_features = include_domain_text_embedding_features
    _ranker_worker_domain_similarity_lookup = domain_similarity_lookup
    _ranker_worker_include_text_rank_margin_features = include_text_rank_margin_features
    _ranker_worker_text_rank_margin_lookup = text_rank_margin_lookup


def _build_ranker_rows_for_person_worker(
    payload: tuple[int, set[int], list[HobbyCandidate], int],
) -> list[RankerRow]:
    if _ranker_worker_reranker_config is None:
        raise RuntimeError("ranker dataset worker was not initialized")
    person_id, positive_hobby_ids, pool_candidates, seed = payload
    return _build_ranker_rows_for_person(
        person_id=person_id,
        positive_hobby_ids=positive_hobby_ids,
        pool_candidates=pool_candidates,
        rng=random.Random(seed),
        all_hobby_ids=_ranker_worker_all_hobby_ids,
        known_by_person=_ranker_worker_known_by_person,
        id_to_hobby=_ranker_worker_id_to_hobby,
        contexts=_ranker_worker_contexts,
        id_to_person=_ranker_worker_id_to_person,
        hobby_profile=_ranker_worker_hobby_profile,
        reranker_config=_ranker_worker_reranker_config,
        neg_ratio=_ranker_worker_neg_ratio,
        hard_ratio=_ranker_worker_hard_ratio,
        include_text_embedding_feature=_ranker_worker_include_text_embedding_feature,
        text_similarity_fn=None,
        text_similarity_lookup=_ranker_worker_text_similarity_lookup,
        include_domain_text_embedding_features=_ranker_worker_include_domain_text_embedding_features,
        domain_similarity_lookup=_ranker_worker_domain_similarity_lookup,
        include_text_rank_margin_features=_ranker_worker_include_text_rank_margin_features,
        text_rank_margin_lookup=_ranker_worker_text_rank_margin_lookup,
    )


def _build_ranker_rows_for_person(
    *,
    person_id: int,
    positive_hobby_ids: set[int],
    pool_candidates: list[HobbyCandidate],
    rng: random.Random,
    all_hobby_ids: list[int],
    known_by_person: dict[int, set[int]],
    id_to_hobby: dict[int, str],
    contexts: dict[str, PersonContext],
    id_to_person: dict[int, str],
    hobby_profile: dict[str, object],
    reranker_config: RerankerConfig,
    neg_ratio: int,
    hard_ratio: float,
    include_text_embedding_feature: bool,
    text_similarity_fn: Callable[[int, HobbyCandidate], float] | None,
    text_similarity_lookup: dict[int, dict[int, float]],
    include_domain_text_embedding_features: bool,
    domain_similarity_lookup: dict[int, dict[int, dict[str, float]]],
    include_text_rank_margin_features: bool,
    text_rank_margin_lookup: dict[int, dict[int, dict[str, float]]],
) -> list[RankerRow]:
    person_uuid = id_to_person.get(person_id)
    if not person_uuid:
        return []
    context = contexts.get(person_uuid) or empty_person_context(person_uuid)
    known_hobby_ids = known_by_person.get(person_id, set())
    known_hobby_names = {id_to_hobby[h] for h in known_hobby_ids if h in id_to_hobby}
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

    def _build_row(hid: int, label: int) -> RankerRow:
        candidate = _make_candidate(hid)
        text_embedding_similarity = 0.0
        if include_text_embedding_feature:
            if text_similarity_lookup:
                text_embedding_similarity = float(text_similarity_lookup.get(person_id, {}).get(candidate.hobby_id, 0.0))
            elif text_similarity_fn is not None:
                try:
                    text_embedding_similarity = float(text_similarity_fn(person_id, candidate))
                except Exception:
                    text_embedding_similarity = 0.0
        domain_similarities: dict[str, float] = {}
        if include_domain_text_embedding_features:
            domain_similarities = domain_similarity_lookup.get(person_id, {}).get(candidate.hobby_id, {})
        text_rank_margin_features: dict[str, float] = {}
        if include_text_rank_margin_features:
            text_rank_margin_features = text_rank_margin_lookup.get(person_id, {}).get(candidate.hobby_id, {})
        features = build_rerank_features(
            context,
            candidate,
            hobby_profile,
            known_hobby_names,
            reranker_config,
            text_embedding_similarity=text_embedding_similarity,
        )
        if domain_similarities:
            features.update(domain_similarities)
        if text_rank_margin_features:
            features.update(text_rank_margin_features)
        features.pop("similar_person_score", None)
        features.pop("persona_text_fit", None)
        return RankerRow(person_id=person_id, hobby_id=hid, label=label, features=features)

    return [
        *(_build_row(hobby_id, 1) for hobby_id in positive_hobby_ids),
        *(_build_row(hobby_id, 0) for hobby_id in negatives),
    ]


def build_text_rank_margin_lookup(
    candidate_pools: dict[int, list[HobbyCandidate]],
    text_similarity_lookup: dict[int, dict[int, float]],
) -> dict[int, dict[int, dict[str, float]]]:
    output: dict[int, dict[int, dict[str, float]]] = {}
    for person_id, candidates in candidate_pools.items():
        scores_by_hobby = text_similarity_lookup.get(person_id, {})
        if not candidates or not scores_by_hobby:
            continue
        scored = [
            (candidate.hobby_id, float(scores_by_hobby.get(candidate.hobby_id, 0.0)))
            for candidate in candidates
        ]
        if not scored:
            continue
        sorted_scored = sorted(scored, key=lambda item: (-item[1], item[0]))
        top_score = sorted_scored[0][1]
        mean_score = float(sum(score for _, score in scored) / len(scored))
        denom = max(len(sorted_scored) - 1, 1)
        person_features: dict[int, dict[str, float]] = {}
        for zero_index, (hobby_id, score) in enumerate(sorted_scored):
            rank = float(zero_index + 1)
            person_features[hobby_id] = {
                "e5_similarity_rank": rank,
                "e5_similarity_percentile": 1.0 - (float(zero_index) / float(denom)),
                "e5_similarity_gap_to_top": float(top_score - score),
                "e5_similarity_gap_to_mean": float(score - mean_score),
            }
        output[person_id] = person_features
    return output


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
        "num_threads": 18,
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
        evals_result: dict[str, Any] = {}

        train_kwargs: dict[str, Any] = {
            "params": self.params,
            "train_set": train_dataset,
            "num_boost_round": num_boost_round,
            "valid_sets": [train_dataset, val_dataset],
            "valid_names": ["train", "val"],
            "callbacks": callbacks,
        }

        try:
            train_kwargs["evals_result"] = evals_result
            self.model = lgb.train(**train_kwargs)
        except TypeError:
            # Older LightGBM versions used in this environment may not support evals_result kwarg.
            train_kwargs.pop("evals_result", None)
            self.model = lgb.train(**train_kwargs)
        
        self.best_iteration = self.model.best_iteration
        val_metrics = self.model.best_score.get("val", {}) if isinstance(self.model.best_score, dict) else {}
        if isinstance(val_metrics, dict) and val_metrics:
            best_metric_key = next(iter(val_metrics.keys()))
            self.best_score = float(val_metrics[best_metric_key])
        else:
            self.best_score = 0.0
            best_metric_key = self.params.get("metric", "auc")
        
        return {
            "params": self.params,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "best_metric": best_metric_key,
            "train_metrics": evals_result,
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
    providers: tuple[str, ...] = ("popularity", "cooccurrence"),
    provider_cache_fingerprint: str = "",
) -> str:
    pid_hash = hashlib.md5(str(sorted(person_ids)).encode()).hexdigest()[:8]
    edge_hash = _hash_indexed_edges(train_edges)
    hobby_hash = _hash_id_mapping(id_to_hobby)
    providers_key = "-".join(providers)
    provider_suffix = f"_s{provider_cache_fingerprint}" if provider_cache_fingerprint else ""
    return f"pool_{label}_{providers_key}_k{candidate_k}_{normalization_method}_e{edge_hash}_h{hobby_hash}_p{pid_hash}{provider_suffix}"


def get_candidate_pool_cache_key(
    person_ids: list[int],
    train_edges: list[tuple[int, int]],
    id_to_hobby: dict[int, str],
    candidate_k: int,
    normalization_method: str,
    label: str,
    providers: tuple[str, ...] = ("popularity", "cooccurrence"),
    provider_cache_fingerprint: str = "",
) -> str:
    return _pool_cache_key(
        person_ids=person_ids,
        train_edges=train_edges,
        id_to_hobby=id_to_hobby,
        candidate_k=candidate_k,
        normalization_method=normalization_method,
        label=label,
        providers=providers,
        provider_cache_fingerprint=provider_cache_fingerprint,
    )


def _l2_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


def build_kure_semantic_candidate_scores(
    person_text_by_id: dict[int, str],
    person_embedding_cache: Any,
    hobby_embedding_cache: Any,
    id_to_hobby: dict[int, str],
    train_known: dict[int, set[int]],
    top_k: int,
    *,
    score_batch_size: int = 128,
    show_progress_bar: bool = False,
    progress_desc: str = "KURE Stage1 semantic scoring",
) -> tuple[dict[int, dict[int, float]], dict[str, object]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    person_ids = [person_id for person_id, text in person_text_by_id.items() if text]
    if not person_ids:
        return {}, {
            "provider": "kure_semantic",
            "person_count": 0,
            "hobby_count": len(id_to_hobby),
            "top_k": top_k,
            "score_batch_size": max(1, score_batch_size),
            "enabled": False,
            "reason": "no eligible person text",
        }

    person_texts = [person_text_by_id[person_id] for person_id in person_ids]
    person_embeddings_by_text = person_embedding_cache.encode_batch(
        person_texts,
        show_progress_bar=show_progress_bar,
        progress_desc=f"{progress_desc} personas",
    )
    hobby_ids = sorted(id_to_hobby)
    hobby_names = [id_to_hobby[hobby_id] for hobby_id in hobby_ids]
    hobby_embeddings_by_name = hobby_embedding_cache.encode_batch(
        hobby_names,
        show_progress_bar=show_progress_bar,
        progress_desc=f"{progress_desc} hobbies",
    )

    hobby_vectors: list[np.ndarray] = []
    retained_hobby_ids: list[int] = []
    for hobby_id, hobby_name in zip(hobby_ids, hobby_names, strict=False):
        vector = hobby_embeddings_by_name.get(hobby_name)
        if vector is None:
            continue
        hobby_vectors.append(np.asarray(vector, dtype=np.float32))
        retained_hobby_ids.append(hobby_id)
    if not hobby_vectors:
        return {}, {
            "provider": "kure_semantic",
            "person_count": len(person_ids),
            "hobby_count": 0,
            "top_k": top_k,
            "score_batch_size": max(1, score_batch_size),
            "enabled": False,
            "reason": "no hobby embeddings",
        }

    hobby_matrix = _l2_normalize_matrix(np.vstack([vector.reshape(1, -1) for vector in hobby_vectors]))
    batch_size = max(1, int(score_batch_size))
    scores_by_person: dict[int, dict[int, float]] = {}
    iterator = range(0, len(person_ids), batch_size)
    if show_progress_bar:
        iterator = tqdm(
            iterator,
            desc=progress_desc,
            unit="batch",
            dynamic_ncols=False,
            leave=True,
            mininterval=1.0,
            maxinterval=10.0,
        )

    for start in iterator:
        batch_person_ids = person_ids[start:start + batch_size]
        batch_vectors: list[np.ndarray] = []
        retained_person_ids: list[int] = []
        for person_id in batch_person_ids:
            vector = person_embeddings_by_text.get(person_text_by_id[person_id])
            if vector is None:
                continue
            batch_vectors.append(np.asarray(vector, dtype=np.float32))
            retained_person_ids.append(person_id)
        if not batch_vectors:
            continue
        person_matrix = _l2_normalize_matrix(np.vstack([vector.reshape(1, -1) for vector in batch_vectors]))
        score_matrix = person_matrix @ hobby_matrix.T
        for row_index, person_id in enumerate(retained_person_ids):
            row = score_matrix[row_index].astype(np.float32, copy=True)
            for known_hobby_id in train_known.get(person_id, set()):
                try:
                    known_index = retained_hobby_ids.index(known_hobby_id)
                except ValueError:
                    continue
                row[known_index] = -np.inf
            finite_count = int(np.isfinite(row).sum())
            if finite_count <= 0:
                continue
            k = min(top_k, finite_count)
            if k >= len(row):
                candidate_indices = np.argsort(-row)
            else:
                candidate_indices = np.argpartition(-row, k - 1)[:k]
                candidate_indices = candidate_indices[np.argsort(-row[candidate_indices])]
            person_scores: dict[int, float] = {}
            for idx in candidate_indices[:k]:
                score = float(row[int(idx)])
                if np.isfinite(score):
                    person_scores[retained_hobby_ids[int(idx)]] = score
            scores_by_person[person_id] = person_scores

    fingerprint_payload = {
        "provider": "kure_semantic",
        "model_name": getattr(hobby_embedding_cache, "model_name", ""),
        "model_revision": getattr(hobby_embedding_cache, "model_revision", ""),
        "preprocessing_version": getattr(hobby_embedding_cache, "preprocessing_version", ""),
        "top_k": top_k,
        "person_count": len(scores_by_person),
        "hobby_count": len(retained_hobby_ids),
    }
    fingerprint = hashlib.md5(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()[:12]
    return scores_by_person, {
        **fingerprint_payload,
        "enabled": True,
        "score_batch_size": batch_size,
        "fingerprint": fingerprint,
        "candidate_pair_count": sum(len(values) for values in scores_by_person.values()),
    }


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
    stage1_providers: tuple[str, ...] = ("popularity", "cooccurrence"),
    semantic_scores_by_person: dict[int, dict[int, float]] | None = None,
    extra_scores_by_provider: dict[str, dict[int, dict[int, float]]] | None = None,
    provider_quotas: dict[str, int] | None = None,
    fill_order: tuple[str, ...] | None = None,
    backfill_order: tuple[str, ...] = ("cooccurrence", "popularity"),
    provider_cache_fingerprint: str = "",
) -> dict[int, list[HobbyCandidate]]:
    from .baseline import (
        cooccurrence_candidate_provider,
        kure_semantic_candidate_provider,
        popularity_candidate_provider,
    )
    from .recommend import Candidate, merge_candidates_by_hobby, normalize_candidate_scores

    provider_set = set(stage1_providers)
    extra_scores_by_provider = extra_scores_by_provider or {}
    score_backed_providers = set(extra_scores_by_provider)
    unknown_providers = provider_set - {"popularity", "cooccurrence", "kure_semantic"} - score_backed_providers
    if unknown_providers:
        unknown = ", ".join(sorted(unknown_providers))
        raise ValueError(f"Unsupported Stage1 providers: {unknown}")
    if "kure_semantic" in provider_set and semantic_scores_by_person is None:
        raise ValueError("semantic_scores_by_person is required when kure_semantic Stage1 provider is enabled")

    cache_key = _pool_cache_key(
        person_ids,
        train_edges,
        id_to_hobby,
        candidate_k,
        normalization_method,
        label,
        providers=stage1_providers,
        provider_cache_fingerprint=provider_cache_fingerprint,
    )

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

    pools: dict[int, list[HobbyCandidate]] = {}
    for person_id in tqdm(person_ids, desc=f"candidate pools ({label})", disable=disable_progress):
        known = train_known.get(person_id, set())
        provider_candidates: dict[str, list[Any]] = {}
        if "popularity" in provider_set:
            provider_candidates["popularity"] = normalize_candidate_scores(
                popularity_candidate_provider(
                    train_edges,
                    person_id,
                    known,
                    candidate_k,
                    popularity_counts=popularity_counts,
                ),
                normalization_method,
            )
        if "cooccurrence" in provider_set:
            provider_candidates["cooccurrence"] = normalize_candidate_scores(
                cooccurrence_candidate_provider(
                    train_edges,
                    person_id,
                    known,
                    candidate_k,
                    cooccurrence_counts=cooccurrence_counts,
                ),
                normalization_method,
            )
        if "kure_semantic" in provider_set:
            provider_candidates["kure_semantic"] = normalize_candidate_scores(
                kure_semantic_candidate_provider(
                    person_id,
                    known,
                    candidate_k,
                    semantic_scores_by_person or {},
                ),
                normalization_method,
            )
        for provider, scores_by_person in extra_scores_by_provider.items():
            if provider not in provider_set:
                continue
            provider_candidates[provider] = normalize_candidate_scores(
                _score_dict_candidate_provider(
                    provider,
                    person_id,
                    known,
                    candidate_k,
                    scores_by_person,
                ),
                normalization_method,
            )
        ordered_provider_candidates = {
            provider: provider_candidates[provider]
            for provider in stage1_providers
            if provider in provider_candidates
        }
        if provider_quotas:
            merged = _merge_candidates_by_quota(
                ordered_provider_candidates,
                candidate_k,
                provider_quotas,
                fill_order or stage1_providers,
                backfill_order,
            )
        else:
            merged = merge_candidates_by_hobby(ordered_provider_candidates, candidate_k)
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


def _score_dict_candidate_provider(
    provider: str,
    person_id: int,
    known_hobbies: set[int],
    top_k: int,
    scores_by_person: dict[int, dict[int, float]],
) -> list[Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    candidates: list[Any] = []
    ranked = sorted(scores_by_person.get(person_id, {}).items(), key=lambda item: (-float(item[1]), item[0]))
    from .recommend import Candidate

    for rank, (hobby_id, score) in enumerate(ranked, start=1):
        if hobby_id in known_hobbies:
            continue
        candidates.append(
            Candidate(
                hobby_id=hobby_id,
                provider=provider,
                raw_score=float(score),
                rank=rank,
                reason_features={f"{provider}_score": float(score), "person_id": person_id},
                source_scores={provider: float(score)},
            )
        )
        if len(candidates) >= top_k:
            break
    return candidates


def _merge_candidates_by_quota(
    provider_candidates: dict[str, list[Any]],
    top_k: int,
    provider_quotas: dict[str, int],
    fill_order: tuple[str, ...],
    backfill_order: tuple[str, ...],
) -> list[Any]:
    from .recommend import Candidate

    by_hobby: dict[int, list[Any]] = {}
    for candidates in provider_candidates.values():
        for candidate in candidates:
            by_hobby.setdefault(candidate.hobby_id, []).append(candidate)

    selected: list[Any] = []
    selected_ids: set[int] = set()

    def add_from(provider: str, limit: int) -> None:
        if limit <= 0:
            return
        added = 0
        for candidate in provider_candidates.get(provider, []):
            if candidate.hobby_id in selected_ids:
                continue
            selected.append(_merge_candidate_sources(candidate, by_hobby))
            selected_ids.add(candidate.hobby_id)
            added += 1
            if added >= limit or len(selected) >= top_k:
                break

    for provider in fill_order:
        add_from(provider, int(provider_quotas.get(provider, 0)))
        if len(selected) >= top_k:
            break

    while len(selected) < top_k:
        before = len(selected)
        for provider in backfill_order:
            add_from(provider, top_k - len(selected))
            if len(selected) >= top_k:
                break
        if len(selected) == before:
            break

    return selected[:top_k]


def _merge_candidate_sources(candidate: Any, by_hobby: dict[int, list[Any]]) -> Any:
    from .recommend import Candidate

    candidates = by_hobby.get(candidate.hobby_id, [candidate])
    source_scores = {str(item.provider): float(item.score) for item in candidates}
    raw_source_scores = {f"{item.provider}_raw": float(item.raw_score) for item in candidates}
    reason_features = {str(item.provider): item.reason_features or {} for item in candidates}
    reason_features["raw_source_scores"] = raw_source_scores
    return Candidate(
        hobby_id=candidate.hobby_id,
        provider=candidate.provider,
        raw_score=float(candidate.raw_score),
        normalized_score=candidate.normalized_score,
        rank=candidate.rank,
        reason_features=reason_features,
        source_scores=source_scores,
    )
