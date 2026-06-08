from __future__ import annotations

from typing import Any

import numpy as np


REASON_TEMPLATES: dict[str, str] = {
    "lightgcn_score": "그래프 기반 추천 점수가 높습니다",
    "cooccurrence_score": "비슷한 사용자들이 함께 선호하는 취미입니다",
    "segment_popularity_score": "비슷한 세그먼트에서 인기가 높습니다",
    "known_hobby_compatibility": "기존 취미와 함께 나타나는 경향이 있습니다",
    "age_group_fit": "연령대 특성과 잘 맞습니다",
    "occupation_fit": "직업군 특성과 잘 맞습니다",
    "region_fit": "거주 지역 특성과 잘 맞습니다",
    "popularity_prior": "전체적으로 선호도가 높은 취미입니다",
    "mismatch_penalty": "부정적 조정 요인이 적습니다",
    "popularity_penalty": "과도한 인기 편향이 조정되었습니다",
    "novelty_bonus": "새로운 취미로 추천 다양성을 보완합니다",
    "category_diversity_reward": "취미 카테고리 다양성을 보완합니다",
    "is_cold_start": "정보가 적은 사용자에게 안정적인 추천 신호입니다",
    "source_is_popularity": "인기도 기반 후보 신호가 반영되었습니다",
    "source_is_cooccurrence": "공동 선호 기반 후보 신호가 반영되었습니다",
    "source_count": "여러 후보 생성 경로에서 함께 추천되었습니다",
    "text_embedding_similarity": "페르소나와 취미 설명의 의미 유사도가 높습니다",
}

FALLBACK_REASON = "주요 추천 요인을 확인할 수 없습니다."
EMPTY_REASON = "추천 이유를 생성할 수 없습니다."


def compute_feature_contributions(
    ranker_model: Any,
    X: np.ndarray,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Compute LightGBM feature contributions for a ranker model."""
    model = getattr(ranker_model, "model", ranker_model)
    if model is None:
        raise ValueError("Ranker model is not trained.")

    if feature_names is None:
        feature_names = [str(name) for name in model.feature_name()]

    contributions = np.asarray(model.predict(X, pred_contrib=True))
    if contributions.ndim == 2 and contributions.shape[1] == len(feature_names) + 1:
        contributions = contributions[:, : len(feature_names)]
    return contributions


def generate_reason(
    contributions: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    top_k: int = 3,
) -> str:
    """Generate a Korean reason string from positive feature contributions."""
    contributions = np.asarray(contributions).flatten()
    feature_values = np.asarray(feature_values).flatten()

    if len(contributions) != len(feature_names):
        feature_names = feature_names[: len(contributions)]

    valid_mask = np.isfinite(contributions) & np.isfinite(feature_values)
    if not valid_mask.any():
        return EMPTY_REASON

    sorted_indices = np.argsort(-np.abs(contributions))

    reasons: list[str] = []
    seen: set[str] = set()
    for idx in sorted_indices:
        if len(reasons) >= top_k:
            break
        if not valid_mask[idx] or float(contributions[idx]) <= 0:
            continue
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        if name in seen:
            continue
        seen.add(name)
        reasons.append(REASON_TEMPLATES.get(name, f"{name} 특성이 긍정적으로 작용했습니다"))

    if not reasons:
        return FALLBACK_REASON
    return "; ".join(reasons) + "."


def validate_reason_batch(
    recommendations: list[dict[str, Any]],
    contributions: np.ndarray | None,
    feature_matrix: np.ndarray | None,
    feature_names: list[str] | None,
) -> dict[str, Any]:
    """Validate that reasons are generated correctly for a batch."""
    total = len(recommendations)
    if total == 0:
        return {"total": 0, "non_empty_rate": 0.0, "has_nan": False, "has_masked_hobby": False, "pass": False}

    meaningful = 0
    has_nan = False
    has_masked_hobby = False
    fallback_reasons = {FALLBACK_REASON, EMPTY_REASON}

    for rec in recommendations:
        reason = rec.get("reason", "")
        is_meaningful = bool(reason) and isinstance(reason, str) and reason not in fallback_reasons
        if is_meaningful:
            meaningful += 1
        if isinstance(reason, str) and "NaN" in reason:
            has_nan = True
        if isinstance(reason, str) and "[MASK]" in reason:
            has_masked_hobby = True

    meaningful_rate = meaningful / total if total else 0.0
    contributions_ok = True
    if contributions is not None:
        contributions_ok = bool(np.isfinite(contributions).all())

    return {
        "total": total,
        "meaningful": meaningful,
        "meaningful_rate": meaningful_rate,
        "has_nan_in_reason": has_nan,
        "has_masked_hobby": has_masked_hobby,
        "contributions_finite": contributions_ok,
        "pass": meaningful_rate >= 0.9 and not has_nan and not has_masked_hobby and contributions_ok,
    }


def batch_generate_reasons(
    ranker_model: Any,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    top_k: int = 3,
) -> tuple[np.ndarray, list[str]]:
    """Generate LightGBM feature contributions and reason strings."""
    contributions = compute_feature_contributions(ranker_model, X, feature_names)
    if feature_names is None:
        model = getattr(ranker_model, "model", ranker_model)
        feature_names = [str(name) for name in model.feature_name()] if model else []

    reasons = [
        generate_reason(contributions[i], X[i], feature_names, top_k=top_k)
        for i in range(X.shape[0])
    ]
    return contributions, reasons
