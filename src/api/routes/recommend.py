from fastapi import APIRouter, Query

from ...graph.recommendation import RecommendationService, VALID_RECOMMENDATION_CATEGORIES
from ..exceptions import BadRequestException, NotFoundException, ServiceUnavailableException
from ..schemas import RecommendItem, RecommendResponse

router = APIRouter(prefix="/api", tags=["recommendations"])


def get_recommendation_service() -> RecommendationService:
    return RecommendationService()


@router.get("/recommend/{uuid}", response_model=RecommendResponse)
def recommend(
    uuid: str,
    category: str = Query(default="hobby"),
    top_n: int = Query(default=5, ge=1, le=20),
) -> RecommendResponse:
    if category not in VALID_RECOMMENDATION_CATEGORIES:
        raise BadRequestException(
            f"유효하지 않은 추천 카테고리입니다: {category}. "
            f"유효한 값: {', '.join(sorted(VALID_RECOMMENDATION_CATEGORIES))}"
        )

    service = get_recommendation_service()
    try:
        if not service.persona_exists(uuid):
            raise NotFoundException("해당 UUID의 페르소나를 찾을 수 없습니다.")
        if not service.has_similarity_data(uuid):
            raise ServiceUnavailableException("유사도 매칭 데이터가 없습니다. 관리자에게 KNN 파이프라인 실행을 요청하세요.")
        recommendations = service.recommend(uuid=uuid, category=category, top_n=top_n)
    finally:
        service.close()

    return RecommendResponse(
        uuid=uuid,
        category=category,
        recommendations=[RecommendItem(**item) for item in recommendations],
    )
