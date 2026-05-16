from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
import re
from time import perf_counter
from typing import Any, Iterator, Protocol
from uuid import uuid4

from src.config import settings

SENSITIVE_KEYS = ("api_key", "access_token", "authorization", "neo4j_password", "password", "token")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_PATTERN = re.compile(r"\b(?:01[016789]-?\d{3,4}-?\d{4}|\d{2,3}-\d{3,4}-\d{4})\b")
_current_trace_id: ContextVar[str | None] = ContextVar("current_rag_trace_id", default=None)
logger = logging.getLogger(__name__)


@dataclass
class TraceSpan:
    name: str
    status: str = "ok"
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class TraceRecord:
    trace_id: str
    route: str
    session_id: str | None
    question: str | None
    status: str = "running"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    latency_ms: float = 0.0
    spans: list[TraceSpan] = field(default_factory=list)
    response_preview: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class TraceSink(Protocol):
    def create_trace(self, route: str, session_id: str | None, question: str | None) -> TraceRecord: ...
    def add_span(self, trace_id: str, span: TraceSpan) -> None: ...
    def finish_trace(
        self,
        trace_id: str,
        *,
        status: str,
        latency_ms: float,
        response: str | None = None,
        error: Exception | None = None,
    ) -> None: ...
    def list_traces(self, limit: int = 50) -> list[TraceRecord]: ...
    def get_trace(self, trace_id: str) -> TraceRecord | None: ...
    def clear(self) -> None: ...


class NoopTraceSink:
    def create_trace(self, route: str, session_id: str | None, question: str | None) -> TraceRecord:
        return TraceRecord(trace_id="", route=route, session_id=session_id, question=None, status="disabled")

    def add_span(self, trace_id: str, span: TraceSpan) -> None:
        return None

    def finish_trace(
        self,
        trace_id: str,
        *,
        status: str,
        latency_ms: float,
        response: str | None = None,
        error: Exception | None = None,
    ) -> None:
        return None

    def list_traces(self, limit: int = 50) -> list[TraceRecord]:
        return []

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        return None

    def clear(self) -> None:
        return None


class InMemoryTraceSink:
    def __init__(self, max_events: int) -> None:
        self.max_events = max(max_events, 1)
        self._records: list[TraceRecord] = []

    def create_trace(self, route: str, session_id: str | None, question: str | None) -> TraceRecord:
        record = TraceRecord(
            trace_id=str(uuid4()),
            route=route,
            session_id=session_id,
            question=question if settings.RAG_TRACE_STORE_RAW_INPUT else None,
        )
        self._records.insert(0, record)
        del self._records[self.max_events :]
        return record

    def add_span(self, trace_id: str, span: TraceSpan) -> None:
        record = self.get_trace(trace_id)
        if record:
            record.spans.append(span)

    def finish_trace(
        self,
        trace_id: str,
        *,
        status: str,
        latency_ms: float,
        response: str | None = None,
        error: Exception | None = None,
    ) -> None:
        record = self.get_trace(trace_id)
        if not record:
            return
        record.status = status
        record.latency_ms = round(latency_ms, 2)
        if response and settings.RAG_TRACE_STORE_RESPONSE:
            record.response_preview = response[:500]
        if error:
            record.error_type = type(error).__name__
            record.error_message = str(error)[:500]

    def list_traces(self, limit: int = 50) -> list[TraceRecord]:
        return self._records[: max(min(limit, 200), 1)]

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        return next((record for record in self._records if record.trace_id == trace_id), None)

    def clear(self) -> None:
        self._records.clear()


trace_sink: TraceSink = InMemoryTraceSink(settings.RAG_TRACE_MAX_EVENTS)


def tracing_enabled() -> bool:
    return bool(settings.RAG_TRACING_ENABLED)


def current_trace_id() -> str | None:
    return _current_trace_id.get()


def set_trace_sink(sink: TraceSink) -> None:
    global trace_sink
    trace_sink = sink


@contextmanager
def trace_request(route: str, *, session_id: str | None = None, question: str | None = None) -> Iterator[str | None]:
    if not tracing_enabled():
        yield None
        return

    try:
        record = trace_sink.create_trace(route=route, session_id=session_id, question=_redact_value(question))
    except Exception as exc:
        logger.warning("RAG trace creation failed: %s", exc)
        yield None
        return

    token = _current_trace_id.set(record.trace_id)
    start = perf_counter()
    try:
        yield record.trace_id
    except Exception as exc:
        _safe_finish_trace(
            record.trace_id,
            status="error",
            latency_ms=(perf_counter() - start) * 1000,
            error=exc,
        )
        raise
    finally:
        current_record = _safe_get_trace(record.trace_id)
        if current_record and current_record.status == "running":
            _safe_finish_trace(record.trace_id, status="ok", latency_ms=(perf_counter() - start) * 1000)
        _current_trace_id.reset(token)


@contextmanager
def trace_span(name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
    trace_id = current_trace_id()
    if not trace_id:
        yield
        return

    start = perf_counter()
    try:
        yield
    except Exception as exc:
        _safe_add_span(
            trace_id,
            TraceSpan(
                name=name,
                status="error",
                latency_ms=round((perf_counter() - start) * 1000, 2),
                metadata=_redact_metadata(metadata or {}),
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
            ),
        )
        raise
    else:
        _safe_add_span(
            trace_id,
            TraceSpan(
                name=name,
                latency_ms=round((perf_counter() - start) * 1000, 2),
                metadata=_redact_metadata(metadata or {}),
            ),
        )


def finish_current_trace(*, response: str | None = None) -> None:
    trace_id = current_trace_id()
    if not trace_id:
        return
    record = _safe_get_trace(trace_id)
    if record:
        _safe_finish_trace(trace_id, status="ok", latency_ms=record.latency_ms, response=response)


def _safe_add_span(trace_id: str, span: TraceSpan) -> None:
    try:
        trace_sink.add_span(trace_id, span)
    except Exception as exc:
        logger.warning("RAG trace span write failed: %s", exc)


def _safe_finish_trace(
    trace_id: str,
    *,
    status: str,
    latency_ms: float,
    response: str | None = None,
    error: Exception | None = None,
) -> None:
    try:
        trace_sink.finish_trace(trace_id, status=status, latency_ms=latency_ms, response=response, error=error)
    except Exception as exc:
        logger.warning("RAG trace finish failed: %s", exc)


def _safe_get_trace(trace_id: str) -> TraceRecord | None:
    try:
        return trace_sink.get_trace(trace_id)
    except Exception as exc:
        logger.warning("RAG trace read failed: %s", exc)
        return None


def _redact_metadata(value: dict[str, Any]) -> dict[str, Any]:
    return {key: _redact_value(item, key=key) for key, item in value.items()}


def _redact_value(value: Any, *, key: str | None = None) -> Any:
    if key and any(sensitive in key.lower() for sensitive in SENSITIVE_KEYS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return _redact_metadata(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value[:20]]
    if isinstance(value, str):
        text = value
        for marker in ("api_key", "access_token", "password", "token"):
            if marker in text.lower():
                return "[REDACTED]"
        text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
        text = PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
        return text[:1000]
    return value
