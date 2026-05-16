from dataclasses import asdict

from fastapi import APIRouter, Query

from src.api.exceptions import NotFoundException, ServiceUnavailableException
from src.api.schemas import RagTraceListResponse, RagTraceRecord
from src.config import settings
from src.rag.tracing import trace_sink, tracing_enabled

router = APIRouter(prefix="/api/admin/rag", tags=["rag-observability"])


@router.get("/traces", response_model=RagTraceListResponse)
def list_rag_traces(limit: int = Query(default=50, ge=1, le=200)) -> RagTraceListResponse:
    _ensure_admin_trace_api_enabled()
    return RagTraceListResponse(
        tracing_enabled=tracing_enabled(),
        traces=[RagTraceRecord(**asdict(record)) for record in trace_sink.list_traces(limit)],
    )


@router.get("/traces/{trace_id}", response_model=RagTraceRecord)
def get_rag_trace(trace_id: str) -> RagTraceRecord:
    _ensure_admin_trace_api_enabled()
    record = trace_sink.get_trace(trace_id)
    if record is None:
        raise NotFoundException("해당 trace를 찾을 수 없습니다.")
    return RagTraceRecord(**asdict(record))


def _ensure_admin_trace_api_enabled() -> None:
    if not settings.RAG_TRACE_ADMIN_ENABLED:
        raise ServiceUnavailableException("RAG trace admin API is disabled.")
