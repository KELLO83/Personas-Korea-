from fastapi import APIRouter

from src.api.schemas import InsightRequest, InsightResponse
from src.rag.router import InsightRouter, get_insight_router as get_shared_insight_router

router = APIRouter(prefix="/api", tags=["insight"])


def get_insight_router() -> InsightRouter:
    return get_shared_insight_router()


@router.post("/insight", response_model=InsightResponse)
def insight(request: InsightRequest) -> InsightResponse:
    result = get_insight_router().ask(request.question)
    return InsightResponse(**result)
