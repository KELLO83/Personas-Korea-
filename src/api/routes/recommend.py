from typing import Any

from fastapi import APIRouter, Query

from ...graph.recommendation import (
    RecommendationService,
    VALID_CENTRALITY_METRICS,
    VALID_RECOMMENDATION_CATEGORIES,
)
from ..exceptions import BadRequestException, NotFoundException, ServiceUnavailableException
from ..schemas import RecommendItem, RecommendationModelInfo, RecommendationStatusResponse, RecommendResponse

router = APIRouter(prefix="/api", tags=["recommendations"])

HOBBY_MODEL_INFO = RecommendationModelInfo(
    target="hobby",
    status="under_development",
    score_source="fallback",
    model_version=None,
    graph_snapshot_id=None,
    fallback_used=True,
    fallback_reason="hobby_recommender_model_and_weights_not_promoted",
    message="취미 추천 모델은 아직 모델 선택과 가중치가 확정되지 않은 개발 중 상태입니다. 현재 API는 Neo4j graph/rule 기반 fallback 추천을 제공합니다.",
)
PERSONA_SIMILARITY_MODEL_INFO = RecommendationModelInfo(
    target="persona_similarity",
    status="under_development",
    score_source="fallback",
    model_version=None,
    graph_snapshot_id=None,
    fallback_used=True,
    fallback_reason="persona_similarity_model_and_weights_not_promoted",
    message="유사 페르소나 추천 모델은 아직 모델 선택과 가중치가 확정되지 않은 개발 중 상태입니다. 현재 API는 SIMILAR_TO와 post-hoc reason 기반 fallback을 사용합니다.",
)


def _model_info_for_category(category: str) -> RecommendationModelInfo:
    if category == "hobby":
        return HOBBY_MODEL_INFO
    return RecommendationModelInfo(
        target=category,
        status="fallback_only",
        score_source="fallback",
        model_version=None,
        graph_snapshot_id=None,
        fallback_used=True,
        fallback_reason=f"{category}_recommendation_model_not_defined",
        message=f"{category} 추천은 아직 별도 모델 승격 대상이 아닙니다. 현재 API는 Neo4j graph/rule 기반 fallback 추천을 제공합니다.",
    )


def get_recommendation_service() -> RecommendationService:
    return RecommendationService()


@router.get("/recommend/{uuid}", response_model=RecommendResponse)
def recommend(
    uuid: str,
    category: str = Query(
        default="hobby",
        description="추천 카테고리: hobby, skill, occupation, district",
    ),
    top_n: int = Query(
        default=5,
        ge=1,
        le=20,
        description="반환할 추천 항목 개수. 1~20 범위",
    ),
    influence_metric: str | None = Query(
        default=None,
        description="중심성 가중치: pagerank, betweenness, degree 중 선택",
    ),
) -> RecommendResponse:
    if category not in VALID_RECOMMENDATION_CATEGORIES:
        raise BadRequestException(
            f"유효하지 않은 추천 카테고리입니다: {category}. "
            f"유효한 값: {', '.join(sorted(VALID_RECOMMENDATION_CATEGORIES))}"
        )
    if influence_metric is not None and influence_metric not in VALID_CENTRALITY_METRICS:
        raise BadRequestException(
            f"유효하지 않은 영향력 지표입니다: {influence_metric}. "
            f"유효한 값: {', '.join(sorted(VALID_CENTRALITY_METRICS))}"
        )

    service = get_recommendation_service()
    try:
        if not service.persona_exists(uuid):
            raise NotFoundException("해당 UUID의 페르소나를 찾을 수 없습니다.")
        if not service.has_similarity_data(uuid):
            raise ServiceUnavailableException("유사도 매칭 데이터가 없습니다. 관리자에게 KNN 파이프라인 실행을 요청하세요.")
        recommendations = service.recommend(
            uuid=uuid,
            category=category,
            top_n=top_n,
            influence_metric=influence_metric,
        )
    finally:
        service.close()

    return RecommendResponse(
        uuid=uuid,
        category=category,
        recommendations=[RecommendItem(**_with_fallback_contract(item, rank=index)) for index, item in enumerate(recommendations, start=1)],
        model_status=_model_info_for_category(category),
    )


@router.get("/recommendation/status", response_model=RecommendationStatusResponse)
def recommendation_status() -> RecommendationStatusResponse:
    return RecommendationStatusResponse(
        hobby_recommender=HOBBY_MODEL_INFO,
        persona_similarity_recommender=PERSONA_SIMILARITY_MODEL_INFO,
        product_policy=(
            "Recommendation model experiments remain in their experiment folders. "
            "Root APIs expose fallback behavior and adapter metadata until a model is promoted."
        ),
    )


def _with_fallback_contract(item: dict[str, Any], rank: int) -> dict[str, Any]:
    normalized = dict(item)
    normalized.setdefault("rank", rank)
    normalized.setdefault("score", normalized.get("reason_score", 0.0))
    normalized.setdefault("already_known", False)
    normalized.setdefault("sources", ["similar_person", "graph_frequency"])
    normalized.setdefault("score_source", "fallback")
    normalized.setdefault("model_version", None)
    normalized.setdefault("graph_snapshot_id", None)
    normalized.setdefault("fallback_used", True)
    normalized.setdefault("fallback_reason", "recommendation_model_under_development")
    if not normalized.get("reason_cards"):
        normalized["reason_cards"] = [
            {
                "type": "similar_person",
                "title": "유사 페르소나 기반",
                "detail": str(normalized.get("reason") or ""),
                "strength": float(normalized.get("reason_score") or 0.0),
            }
        ]
    return normalized
