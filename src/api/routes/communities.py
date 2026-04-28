from fastapi import APIRouter, Query

from src.api.exceptions import BadRequestException

from src.api.schemas import CommunityResponse
from src.gds.communities import CommunityService

router = APIRouter(prefix="/api", tags=["communities"])


def get_community_service() -> CommunityService:
    return CommunityService()


@router.get("/communities", response_model=CommunityResponse)
def communities(
    algorithm: str = Query(default="leiden"),
    min_size: int = Query(default=10, ge=1),
) -> CommunityResponse:
    if algorithm != "leiden":
        raise BadRequestException("현재는 leiden 알고리즘만 지원합니다.")
    service = get_community_service()
    try:
        return CommunityResponse(communities=service.summarize_communities(min_size=min_size))
    finally:
        service.close()
