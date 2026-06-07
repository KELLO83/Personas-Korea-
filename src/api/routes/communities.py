from fastapi import APIRouter, Query

from src.api.exceptions import BadRequestException, NotFoundException

from src.api.schemas import CommunityProfileResponse, CommunityResponse
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


@router.get("/communities/{community_id}", response_model=CommunityProfileResponse)
def community_profile(community_id: int) -> CommunityProfileResponse:
    service = get_community_service()
    try:
        row = service.community_profile(community_id)
    finally:
        service.close()
    if row is None or int(row.get("size") or 0) == 0:
        raise NotFoundException("해당 community_id의 커뮤니티를 찾을 수 없습니다.")
    label = str(row.get("label") or f"Community {community_id}")
    summary = (
        f"{label}: {row.get('size', 0)}명, "
        f"주요 지역 {', '.join(item['label'] for item in row.get('top_provinces', [])[:3]) or '-'}, "
        f"주요 취미 {', '.join(item['label'] for item in row.get('top_hobbies', [])[:3]) or '-'}"
    )
    return CommunityProfileResponse(**row, summary=summary)
