from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from tqdm import tqdm

from .model import LightGCN


@dataclass(frozen=True)
class Recommendation:
    hobby_id: int
    score: float


@dataclass(frozen=True)
class Candidate:
    hobby_id: int
    provider: str
    raw_score: float
    normalized_score: float | None = None
    rank: int | None = None
    reason_features: dict[str, object] | None = None
    source_scores: dict[str, float] | None = None

    @property
    def score(self) -> float:
        return self.normalized_score if self.normalized_score is not None else self.raw_score


def recommend_hobbies_for_person(
    model: LightGCN,
    adjacency: Tensor,
    person_id: int,
    known_hobbies: set[int],
    top_k: int,
    chunk_size: int,
    device: torch.device,
) -> list[Recommendation]:
    person_embeddings, hobby_embeddings = compute_lightgcn_embeddings(model, adjacency)
    candidates = lightgcn_candidate_provider(
        model=model,
        adjacency=adjacency,
        person_id=person_id,
        known_hobbies=known_hobbies,
        top_k=top_k,
        chunk_size=chunk_size,
        device=device,
        person_embeddings=person_embeddings,
        hobby_embeddings=hobby_embeddings,
    )
    return [Recommendation(hobby_id=candidate.hobby_id, score=candidate.raw_score) for candidate in candidates]


def compute_lightgcn_embeddings(model: LightGCN, adjacency: Tensor) -> tuple[Tensor, Tensor]:
    model.eval()
    with torch.no_grad():
        return model.all_embeddings(adjacency)


def lightgcn_candidate_provider(
    model: LightGCN,
    adjacency: Tensor,
    person_id: int,
    known_hobbies: set[int],
    top_k: int,
    chunk_size: int,
    device: torch.device,
    person_embeddings: Tensor | None = None,
    hobby_embeddings: Tensor | None = None,
) -> list[Candidate]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if person_embeddings is None or hobby_embeddings is None:
        person_embeddings, hobby_embeddings = compute_lightgcn_embeddings(model, adjacency)
    with torch.no_grad():
        person_embedding = person_embeddings[person_id]
        scores = torch.matmul(hobby_embeddings, person_embedding)
        if known_hobbies:
            known_tensor = torch.tensor(sorted(known_hobbies), dtype=torch.long, device=device)
            valid_known = known_tensor[known_tensor < scores.shape[0]]
            if valid_known.numel() > 0:
                scores = scores.clone()
                scores[valid_known] = float("-inf")
        available = int((scores != float("-inf")).sum().item())
        select_k = min(top_k, available)
        if select_k <= 0:
            return []
        top_scores, top_indices = torch.topk(scores, k=select_k)
        ranked_pairs = sorted(zip(top_scores.tolist(), top_indices.tolist(), strict=False), key=lambda item: (-float(item[0]), int(item[1])))
    return [
        Candidate(
            hobby_id=int(hobby_id),
            provider="lightgcn",
            raw_score=float(score),
            rank=rank,
            reason_features={"source": "lightgcn_dot_product"},
            source_scores={"lightgcn": float(score)},
        )
        for rank, (score, hobby_id) in enumerate(ranked_pairs, start=1)
    ]


def normalize_candidate_scores(candidates: list[Candidate], method: str = "rank_percentile") -> list[Candidate]:
    if not candidates:
        return []
    if method == "rank_percentile":
        total = len(candidates)
        ranked = sorted(candidates, key=lambda item: (-item.raw_score, item.hobby_id))
        if total == 1:
            return [_replace_candidate_score(ranked[0], normalized_score=1.0, rank=1)]
        return [
            _replace_candidate_score(candidate, normalized_score=1.0 - ((rank - 1) / (total - 1)), rank=rank)
            for rank, candidate in enumerate(ranked, start=1)
        ]
    if method == "min_max":
        scores = [candidate.raw_score for candidate in candidates]
        minimum = min(scores)
        maximum = max(scores)
        ranked = sorted(candidates, key=lambda item: (-item.raw_score, item.hobby_id))
        if maximum == minimum:
            return [_replace_candidate_score(candidate, normalized_score=1.0, rank=rank) for rank, candidate in enumerate(ranked, start=1)]
        return [
            _replace_candidate_score(candidate, normalized_score=(candidate.raw_score - minimum) / (maximum - minimum), rank=rank)
            for rank, candidate in enumerate(ranked, start=1)
        ]
    raise ValueError("score normalization method must be 'rank_percentile' or 'min_max'")


def merge_candidates_by_hobby(provider_candidates: dict[str, list[Candidate]], top_k: int) -> list[Candidate]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    grouped: dict[int, list[Candidate]] = {}
    provider_order = {provider: index for index, provider in enumerate(provider_candidates)}
    for candidates in provider_candidates.values():
        for candidate in candidates:
            grouped.setdefault(candidate.hobby_id, []).append(candidate)
    merged = [_merge_candidate_group(candidates, provider_order) for candidates in grouped.values()]
    return sorted(merged, key=lambda item: (-item.score, provider_order.get(item.provider, 999), item.hobby_id))[:top_k]


def _merge_candidate_group(candidates: list[Candidate], provider_order: dict[str, int]) -> Candidate:
    best = sorted(candidates, key=lambda item: (-item.score, provider_order.get(item.provider, 999), item.hobby_id))[0]
    source_scores = {candidate.provider: candidate.score for candidate in candidates}
    raw_source_scores = {f"{candidate.provider}_raw": candidate.raw_score for candidate in candidates}
    reason_features: dict[str, object] = {candidate.provider: candidate.reason_features or {} for candidate in candidates}
    reason_features["raw_source_scores"] = raw_source_scores
    return Candidate(
        hobby_id=best.hobby_id,
        provider=best.provider,
        raw_score=best.raw_score,
        normalized_score=best.normalized_score,
        rank=best.rank,
        reason_features=reason_features,
        source_scores=source_scores,
    )


def build_provider_contribution_artifact(
    provider_candidates: dict[str, list[Candidate]],
    selected_candidates: list[Candidate],
    requested_top_k: int,
    fallback_type: str,
) -> dict[str, object]:
    selected_by_provider: dict[str, int] = {}
    for candidate in selected_candidates:
        selected_by_provider[candidate.provider] = selected_by_provider.get(candidate.provider, 0) + 1
    return {
        "requested_top_k": requested_top_k,
        "fallback_type": fallback_type,
        "providers": {
            provider: {
                "raw_candidate_count": len(candidates),
                "selected_count": selected_by_provider.get(provider, 0),
                "top_candidates": [candidate_to_dict(candidate) for candidate in candidates[: min(10, len(candidates))]],
            }
            for provider, candidates in provider_candidates.items()
        },
        "selected_candidates": [candidate_to_dict(candidate) for candidate in selected_candidates],
    }


def candidate_to_dict(candidate: Candidate) -> dict[str, object]:
    return {
        "hobby_id": candidate.hobby_id,
        "provider": candidate.provider,
        "raw_score": candidate.raw_score,
        "normalized_score": candidate.normalized_score,
        "rank": candidate.rank,
        "reason_features": candidate.reason_features or {},
        "source_scores": candidate.source_scores or {candidate.provider: candidate.score},
    }


def _replace_candidate_score(candidate: Candidate, *, normalized_score: float, rank: int) -> Candidate:
    reason_features = dict(candidate.reason_features or {})
    reason_features.setdefault("raw_score", candidate.raw_score)
    return Candidate(
        hobby_id=candidate.hobby_id,
        provider=candidate.provider,
        raw_score=candidate.raw_score,
        normalized_score=normalized_score,
        rank=rank,
        reason_features=reason_features,
        source_scores={candidate.provider: normalized_score},
    )


def batch_recommend_hobbies(
    model: LightGCN,
    adjacency: Tensor,
    person_ids: list[int],
    known_by_person: dict[int, set[int]],
    top_k: int,
    chunk_size: int,
    device: torch.device,
    show_progress: bool = False,
) -> dict[int, list[int]]:
    recommendations: dict[int, list[int]] = {}
    person_embeddings, hobby_embeddings = compute_lightgcn_embeddings(model, adjacency)
    batch_size = max(1, chunk_size)
    iterator = range(0, len(person_ids), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=(len(person_ids) + batch_size - 1) // batch_size, desc="batch recommend")
    for start in iterator:
        batch_person_ids = person_ids[start : start + batch_size]
        if not batch_person_ids:
            continue
        person_tensor = torch.tensor(batch_person_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            score_matrix = torch.matmul(person_embeddings[person_tensor], hobby_embeddings.transpose(0, 1))
        for row_index, person_id in enumerate(batch_person_ids):
            scores = score_matrix[row_index].clone()
            known_hobbies = known_by_person.get(person_id, set())
            if known_hobbies:
                known_tensor = torch.tensor(sorted(known_hobbies), dtype=torch.long, device=device)
                valid_known = known_tensor[known_tensor < scores.shape[0]]
                if valid_known.numel() > 0:
                    scores[valid_known] = float("-inf")
            available = int((scores != float("-inf")).sum().item())
            select_k = min(top_k, available)
            if select_k <= 0:
                recommendations[person_id] = []
                continue
            top_scores, top_indices = torch.topk(scores, k=select_k)
            ranked_pairs = sorted(zip(top_scores.tolist(), top_indices.tolist(), strict=False), key=lambda item: (-float(item[0]), int(item[1])))
            recommendations[person_id] = [int(hobby_id) for _, hobby_id in ranked_pairs]
    return recommendations
