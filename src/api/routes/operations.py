from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from fastapi import APIRouter
from neo4j import GraphDatabase

from src.api.schemas import (
    OperationsHealthCheck,
    OperationsHealthResponse,
    OperationsReadinessMetric,
    OperationsReadinessResponse,
    OperationsWarning,
    OperationsWarningsResponse,
)
from src.config import settings

router = APIRouter(prefix="/api/operations", tags=["operations"])

HEALTH_QUERY = """
MATCH (p:Person)
WITH count(p) AS total_personas
MATCH ()-[relationship]->()
RETURN total_personas, count(relationship) AS total_relationships
"""

READINESS_QUERY = """
MATCH (p:Person)
WITH count(p) AS total_personas
CALL () {
    MATCH (p:Person)
    WHERE EXISTS { MATCH (p)-[:SIMILAR_TO]->(:Person) }
    RETURN count(p) AS similar_ready
}
CALL () {
    MATCH (p:Person)
    WHERE p.community_id IS NOT NULL
    RETURN count(p) AS community_ready
}
CALL () {
    MATCH (p:Person)
    WHERE EXISTS { MATCH (p)-[:ENJOYS_HOBBY|LIKES]->(:Hobby) }
    RETURN count(p) AS hobby_ready
}
CALL () {
    MATCH (p:Person)
    WHERE EXISTS { MATCH (p)-->(:Skill) }
    RETURN count(p) AS skill_ready
}
RETURN total_personas, similar_ready, community_ready, hobby_ready, skill_ready
"""

SCHEMA_QUERY = """
CALL db.relationshipTypes() YIELD relationshipType
WITH collect(relationshipType) AS relationship_types
CALL db.propertyKeys() YIELD propertyKey
RETURN relationship_types, collect(propertyKey) AS property_keys
"""

def get_neo4j_driver():  # noqa: ANN201
    return GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _ratio(value: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return round(value / total, 6)


def _readiness_metric(name: str, value: int, total: int, threshold: float, detail: str) -> OperationsReadinessMetric:
    ratio = _ratio(value, total)
    ready = ratio >= threshold
    return OperationsReadinessMetric(
        name=name,
        ready=ready,
        value=float(value),
        total=float(total),
        ratio=ratio,
        status="ready" if ready else "warning",
        detail=detail,
    )


def _warning(code: str, severity: str, title: str, detail: str, action: str) -> OperationsWarning:
    return OperationsWarning(code=code, severity=severity, title=title, detail=detail, action=action)


@router.get("/health", response_model=OperationsHealthResponse)
def operations_health() -> OperationsHealthResponse:
    started_at = perf_counter()
    api_check = OperationsHealthCheck(name="fastapi", status="ok", detail="FastAPI route is responding")
    try:
        driver = get_neo4j_driver()
        try:
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                record = session.run(HEALTH_QUERY).single()
        finally:
            driver.close()
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        return OperationsHealthResponse(
            status="ok",
            generated_at=_utc_now(),
            api=api_check,
            neo4j=OperationsHealthCheck(name="neo4j", status="ok", latency_ms=latency_ms, detail=settings.NEO4J_URI),
            total_personas=int(record["total_personas"]) if record else 0,
            total_relationships=int(record["total_relationships"]) if record else 0,
        )
    except Exception as exc:
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        return OperationsHealthResponse(
            status="degraded",
            generated_at=_utc_now(),
            api=api_check,
            neo4j=OperationsHealthCheck(name="neo4j", status="error", latency_ms=latency_ms, detail=str(exc)[:300]),
        )


@router.get("/readiness", response_model=OperationsReadinessResponse)
def operations_readiness() -> OperationsReadinessResponse:
    try:
        driver = get_neo4j_driver()
        try:
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                record = session.run(READINESS_QUERY).single()
        finally:
            driver.close()
    except Exception as exc:
        return OperationsReadinessResponse(
            status="degraded",
            generated_at=_utc_now(),
            metrics=[
                OperationsReadinessMetric(
                    name="neo4j_readiness",
                    ready=False,
                    value=0,
                    total=1,
                    ratio=0,
                    status="error",
                    detail=str(exc)[:300],
                )
            ],
        )

    row = dict(record) if record else {}
    total = int(row.get("total_personas") or 0)
    metrics = [
        _readiness_metric("SIMILAR_TO coverage", int(row.get("similar_ready") or 0), total, 0.9, "유사 페르소나/추천 후보 데이터"),
        _readiness_metric("community_id coverage", int(row.get("community_ready") or 0), total, 0.9, "Community Profile, Guild 후보 그룹화"),
        _readiness_metric("hobby edge coverage", int(row.get("hobby_ready") or 0), total, 0.75, "취미 추천과 Guild 공통 취미 근거"),
        _readiness_metric("skill edge coverage", int(row.get("skill_ready") or 0), total, 0.25, "스킬 기반 설명/추천 근거"),
    ]
    status = "ready" if all(metric.ready for metric in metrics[:3]) else "warning"
    return OperationsReadinessResponse(status=status, generated_at=_utc_now(), metrics=metrics)


@router.get("/warnings", response_model=OperationsWarningsResponse)
def operations_warnings() -> OperationsWarningsResponse:
    warnings: list[OperationsWarning] = []
    try:
        schema = _load_warning_context()
        relationship_types = set(schema.get("relationship_types", []))
        property_keys = set(schema.get("property_keys", []))
        if "HAS_SKILL" not in relationship_types:
            warnings.append(
                _warning(
                    "schema.skill_relationship_alias",
                    "warning",
                    "HAS_SKILL relationship is absent",
                    "현재 DB는 Skill 연결이 있지만 HAS_SKILL 타입이 없어 일부 legacy query가 Neo4j warning을 냅니다.",
                    "Skill 조회 쿼리는 (p)-->(:Skill) 또는 실제 관계 타입으로 통일하세요.",
                )
            )
        missing_centrality = sorted({"pagerank", "degree"} - property_keys)
        if missing_centrality:
            warnings.append(
                _warning(
                    "graph.centrality_not_written",
                    "info",
                    "Centrality properties are not available",
                    f"누락 속성: {', '.join(missing_centrality)}",
                    "PageRank/Degree 배치를 실행한 뒤 운영 대시보드 leader/centrality 표시를 활성화하세요.",
                )
            )
    except Exception as exc:
        warnings.append(
            _warning(
                "operations.warning_context_failed",
                "error",
                "Warning context query failed",
                str(exc)[:300],
                "Neo4j 컨테이너와 인증 정보를 확인하세요.",
            )
        )
    if not settings.RAG_TRACE_ADMIN_ENABLED:
        warnings.append(
            _warning(
                "rag.trace_admin_disabled",
                "info",
                "RAG trace admin is disabled",
                "RAG trace list API가 기본 비활성화 상태입니다.",
                "로컬/관리자 검수 시 RAG_TRACE_ADMIN_ENABLED=true로 켜세요.",
            )
        )
    if not settings.NVIDIA_API_KEY:
        warnings.append(
            _warning(
                "rag.llm_api_key_missing",
                "warning",
                "NVIDIA API key is missing",
                "LLM 기반 chat/segment analysis는 403 또는 빈 분석으로 degrade될 수 있습니다.",
                ".env에 NVIDIA_API_KEY를 설정하거나 deterministic summary fallback만 사용하세요.",
            )
        )
    status = "ok" if not any(item.severity in {"warning", "error"} for item in warnings) else "warning"
    return OperationsWarningsResponse(status=status, generated_at=_utc_now(), warnings=warnings)


def _load_warning_context() -> dict[str, Any]:
    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            schema_record = session.run(SCHEMA_QUERY).single()
    finally:
        driver.close()
    return dict(schema_record) if schema_record else {"relationship_types": [], "property_keys": []}
