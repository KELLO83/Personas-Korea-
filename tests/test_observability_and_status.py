from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import settings
import pytest

from src.rag.tracing import NoopTraceSink, TraceRecord, TraceSpan, set_trace_sink, trace_request, trace_sink, trace_span


def test_recommendation_status_marks_models_under_development() -> None:
    client = TestClient(create_app())

    response = client.get("/api/recommendation/status")

    assert response.status_code == 200
    body = response.json()
    assert body["hobby_recommender"]["status"] == "under_development"
    assert body["hobby_recommender"]["fallback_used"] is True
    assert body["persona_similarity_recommender"]["status"] == "under_development"
    assert body["persona_similarity_recommender"]["fallback_used"] is True


def test_rag_trace_admin_api_lists_in_memory_traces(monkeypatch) -> None:
    trace_sink.clear()
    monkeypatch.setattr(settings, "RAG_TRACING_ENABLED", True)
    monkeypatch.setattr(settings, "RAG_TRACE_ADMIN_ENABLED", True)
    monkeypatch.setattr(settings, "RAG_TRACE_STORE_RAW_INPUT", True)

    with trace_request("test", session_id="s1", question="질문"):
        with trace_span("unit_span", {"token": "secret", "count": 1}):
            pass

    client = TestClient(create_app())
    response = client.get("/api/admin/rag/traces?limit=1")

    assert response.status_code == 200
    body = response.json()
    assert body["tracing_enabled"] is True
    assert body["traces"][0]["route"] == "test"
    assert body["traces"][0]["spans"][0]["name"] == "unit_span"
    assert body["traces"][0]["spans"][0]["metadata"]["token"] == "[REDACTED]"

    trace_id = body["traces"][0]["trace_id"]
    detail = client.get(f"/api/admin/rag/traces/{trace_id}")
    assert detail.status_code == 200
    assert detail.json()["trace_id"] == trace_id
    trace_sink.clear()


def test_rag_trace_admin_api_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr(settings, "RAG_TRACE_ADMIN_ENABLED", False)

    client = TestClient(create_app())
    response = client.get("/api/admin/rag/traces")

    assert response.status_code == 503


def test_trace_sink_failure_does_not_break_user_response(monkeypatch) -> None:
    class FailingSink(NoopTraceSink):
        def create_trace(self, route: str, session_id: str | None, question: str | None) -> TraceRecord:
            raise RuntimeError("sink down")

    monkeypatch.setattr(settings, "RAG_TRACING_ENABLED", True)
    set_trace_sink(FailingSink())
    try:
        with trace_request("chat", session_id="s1", question="hello") as trace_id:
            assert trace_id is None
    finally:
        set_trace_sink(trace_sink)


def test_trace_span_failure_does_not_hide_application_exception(monkeypatch) -> None:
    class FailingSpanSink(NoopTraceSink):
        def create_trace(self, route: str, session_id: str | None, question: str | None) -> TraceRecord:
            return TraceRecord(trace_id="trace-1", route=route, session_id=session_id, question=None)

        def add_span(self, trace_id: str, span: TraceSpan) -> None:
            raise RuntimeError("span write failed")

    monkeypatch.setattr(settings, "RAG_TRACING_ENABLED", True)
    set_trace_sink(FailingSpanSink())
    try:
        with pytest.raises(ValueError, match="app failed"):
            with trace_request("chat", session_id="s1", question="hello"):
                with trace_span("unit"):
                    raise ValueError("app failed")
    finally:
        set_trace_sink(trace_sink)
