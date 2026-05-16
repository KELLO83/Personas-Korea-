from fastapi import APIRouter

from src.api.schemas import InsightRequest, InsightResponse
from src.rag.router import InsightRouter, get_insight_router as get_shared_insight_router
from src.rag.tracing import trace_request, trace_span

router = APIRouter(prefix="/api", tags=["insight"])


def get_insight_router() -> InsightRouter:
    return get_shared_insight_router()


@router.post("/insight", response_model=InsightResponse)
def insight(request: InsightRequest) -> InsightResponse:
    with trace_request("insight", question=request.question):
        with trace_span("insight_router.ask"):
            result = get_insight_router().ask(request.question)
    return InsightResponse(**result)
