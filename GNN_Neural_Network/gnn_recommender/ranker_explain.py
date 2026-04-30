from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


# Templates for Korean reason generation (PRD §4.4)
# These describe model factors, not causal reasons.
REASON_TEMPLATES: dict[str, str] = {
    "lightgcn_score": "그래프 기반 추천 점수가 높습니다",
    "cooccurrence_score": "비슷한 사용자들이 함께 선호하는 취미입니다",
    "segment_popularity_score": "당신의 세그먼트에서 인기가 높습니다",
    "known_hobby_compatibility": "기존 취미와 자주 함께 등장합니다",
    "age_group_fit": "당신의 연령대에서 인기가 높습니다",
    "occupation_fit": "당신의 직업군에서 선호도가 높습니다",
    "region_fit": "당신의 거주 지역에서 접근하기 좋습니다",
    "popularity_prior": "전체적으로 인기 있는 취미입니다",
    "mismatch_penalty": "인구통계학적 적합도가 낮습니다",
    "popularity_penalty": "인기가 너무 높아 다양성을 위해 조정되었습니다",
    "novelty_bonus": "흔하지 않은 새로운 취미입니다",
    "category_diversity_reward": "카테고리 다양성을 위해 선택되었습니다",
    "is_cold_start": "취미 정보가 적은 사용자를 위한 추천입니다",
    "source_is_popularity": "인기도 기반 후보입니다",
    "source_is_cooccurrence": "공동선호 기반 후보입니다",
    "source_count": "여러 소스에서 추천된 후보입니다",
    "text_embedding_similarity": "페르소나 텍스트와 유사도가 높습니다",
}


def compute_shap_values(
    ranker_model: Any,
    X: np.ndarray,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Compute SHAP values for a LightGBM ranker model.

    Args:
        ranker_model: Trained LightGBM Booster or LightGBMRanker instance.
        X: Feature matrix (n_samples, n_features).
        feature_names: Optional feature name list. If None, uses model.feature_name().

    Returns:
        SHAP values array (n_samples, n_features).
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required for explanation. Install: pip install shap>=0.45.0") from exc

    # Unwrap LightGBMRanker if needed
    model = getattr(ranker_model, "model", ranker_model)
    if model is None:
        raise ValueError("Ranker model is not trained.")

    if feature_names is None:
        feature_names = [str(name) for name in model.feature_name()]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values may be a list of two arrays (class 0, class 1)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]  # Use positive class SHAP values

    return np.asarray(shap_values)


def generate_reason(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    top_k: int = 3,
) -> str:
    """Generate a Korean reason string from SHAP values.

    Args:
        shap_values: SHAP values for a single sample (n_features,).
        feature_values: Feature values for a single sample (n_features,).
        feature_names: Feature names corresponding to the values.
        top_k: Number of top features to include in the reason.

    Returns:
        A Korean string describing the top positive factors.
    """
    shap_values = np.asarray(shap_values).flatten()
    feature_values = np.asarray(feature_values).flatten()

    if len(shap_values) != len(feature_names):
        feature_names = feature_names[: len(shap_values)]

    # Filter out NaN/Inf and very small contributions
    valid_mask = np.isfinite(shap_values) & np.isfinite(feature_values)
    if not valid_mask.any():
        return "추천 이유를 생성할 수 없습니다."

    # Sort by absolute SHAP value descending, then pick top positive contributors
    abs_shap = np.abs(shap_values)
    sorted_indices = np.argsort(-abs_shap)

    reasons: list[str] = []
    seen: set[str] = set()

    for idx in sorted_indices:
        if len(reasons) >= top_k:
            break
        if not valid_mask[idx]:
            continue
        shap_val = float(shap_values[idx])
        if shap_val <= 0:
            continue  # Only positive contributors for reason generation
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        if name in seen:
            continue
        seen.add(name)
        template = REASON_TEMPLATES.get(name, f"{name} 특성이 긍정적으로 작용했습니다")
        reasons.append(template)

    if not reasons:
        return "주요 추천 요인을 확인할 수 없습니다."

    return "; ".join(reasons) + "."


def validate_reason_batch(
    recommendations: list[dict[str, Any]],
    shap_values: np.ndarray | None,
    feature_matrix: np.ndarray | None,
    feature_names: list[str] | None,
) -> dict[str, Any]:
    """Validate that reasons are generated correctly for a batch.

    Args:
        recommendations: List of recommendation dicts (should contain 'reason' key after generation).
        shap_values: SHAP values array (n_samples, n_features) or None.
        feature_matrix: Feature values array (n_samples, n_features) or None.
        feature_names: Feature names list or None.

    Returns:
        Validation report dict.
    """
    total = len(recommendations)
    if total == 0:
        return {"total": 0, "non_empty_rate": 0.0, "has_nan": False, "has_masked_hobby": False, "pass": False}

    _FALLBACK_REASONS = {
        "추천 이유를 생성할 수 없습니다.",
        "주요 추천 요인을 확인할 수 없습니다.",
    }

    meaningful = 0
    has_nan = False
    has_masked_hobby = False

    for rec in recommendations:
        reason = rec.get("reason", "")
        is_meaningful = (
            bool(reason)
            and isinstance(reason, str)
            and reason not in _FALLBACK_REASONS
        )
        if is_meaningful:
            meaningful += 1
        if isinstance(reason, str) and "NaN" in reason:
            has_nan = True
        if isinstance(reason, str) and "[MASK]" in reason:
            has_masked_hobby = True

    meaningful_rate = meaningful / total if total else 0.0
    pass_threshold = meaningful_rate >= 0.9

    # Additional checks on SHAP values
    shap_ok = True
    if shap_values is not None:
        shap_ok = np.isfinite(shap_values).all()

    return {
        "total": total,
        "meaningful": meaningful,
        "meaningful_rate": meaningful_rate,
        "has_nan_in_reason": has_nan,
        "has_masked_hobby": has_masked_hobby,
        "shap_finite": shap_ok,
        "pass": pass_threshold and not has_nan and not has_masked_hobby and shap_ok,
    }


def batch_generate_reasons(
    ranker_model: Any,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    top_k: int = 3,
) -> tuple[np.ndarray, list[str]]:
    """Generate SHAP values and reasons for a batch of samples.

    Args:
        ranker_model: Trained LightGBM model.
        X: Feature matrix (n_samples, n_features).
        feature_names: Optional feature names.
        top_k: Number of top features per reason.

    Returns:
        Tuple of (shap_values_array, list_of_reason_strings).
    """
    shap_values = compute_shap_values(ranker_model, X, feature_names)
    if feature_names is None:
        model = getattr(ranker_model, "model", ranker_model)
        feature_names = [str(name) for name in model.feature_name()] if model else []

    reasons: list[str] = []
    for i in range(X.shape[0]):
        reason = generate_reason(shap_values[i], X[i], feature_names, top_k=top_k)
        reasons.append(reason)

    return shap_values, reasons
