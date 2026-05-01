from fastapi import APIRouter
from neo4j import GraphDatabase
from typing import Any, LiteralString, cast

from src.api.schemas import GraphMigrationStep, GraphQualityCheck, GraphQualityDistributionItem, GraphQualityResponse
from src.config import settings

router = APIRouter(prefix="/api", tags=["graph-quality"])


QUALITY_CHECKS = {
    "country": {
        "query": """
        MATCH (p:Person)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(:Province)-[:IN_COUNTRY]->(c:Country)
        RETURN c.name AS label, count(DISTINCT p) AS count
        ORDER BY count DESC
        """,
        "issue": "Country 값이 단일 국가로 수렴하면 그래프 노드로서 정보량이 없습니다.",
        "recommendation": "cardinality가 1이면 Country 노드를 제거하고 District→Province 직접 연결만 유지합니다.",
        "single_action": "remove_node",
        "skew_action": "review",
    },
    "military_status": {
        "query": """
        MATCH (p:Person)-[:HAS_MILITARY_STATUS]->(m:MilitaryStatus)
        RETURN m.name AS label, count(DISTINCT p) AS count
        ORDER BY count DESC
        """,
        "issue": "MilitaryStatus 분포가 한 값에 치우치면 필터/시각화 정보량이 낮습니다.",
        "recommendation": "최빈값 비율이 과도하게 높으면 기본 필터와 시각화에서 숨깁니다.",
        "single_action": "hide_filter",
        "skew_action": "hide_filter",
    },
    "bachelors_field": {
        "query": """
        MATCH (p:Person)-[:HAS_BACHELORS_FIELD]->(b:BachelorsField)
        RETURN b.name AS label, count(DISTINCT p) AS count
        ORDER BY count DESC
        """,
        "issue": "bachelors_field의 '해당없음' 비율이 높으면 추천/필터 신호로 약합니다.",
        "recommendation": "노드는 유지하되 기본 UI 중요도를 낮추고 상세 프로필 중심으로 노출합니다.",
        "single_action": "lower_priority",
        "skew_action": "lower_priority",
    },
}

COUNTRY_MIGRATION_PLAN = [
    GraphMigrationStep(
        name="District to Province direct relation validation",
        cypher="MATCH (d:District)-[:IN_PROVINCE]->(p:Province) RETURN count(*) AS district_province_edges",
        validation="district_province_edges가 District 노드 수와 일치해야 합니다.",
    ),
    GraphMigrationStep(
        name="Country dependency scan",
        cypher="MATCH ()-[r]->(c:Country) RETURN type(r) AS relation_type, count(*) AS count ORDER BY count DESC",
        validation="IN_COUNTRY 외 관계가 있으면 제거 전 API/쿼리 영향 범위를 재검토합니다.",
    ),
    GraphMigrationStep(
        name="Country node removal candidate",
        cypher="MATCH (:Province)-[r:IN_COUNTRY]->(:Country) DELETE r WITH 1 AS _ MATCH (c:Country) DETACH DELETE c",
        validation="MATCH (c:Country) RETURN count(c) AS country_count 결과가 0이어야 합니다.",
    ),
]


def get_neo4j_driver():  # noqa: ANN201
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def _build_check(name: str, rows: list[dict[str, Any]]) -> GraphQualityCheck:
    total_count = sum(int(row.get("count", 0)) for row in rows)
    distribution = [
        GraphQualityDistributionItem(
            label=str(row.get("label", "")),
            count=int(row.get("count", 0)),
            ratio=round((int(row.get("count", 0)) / total_count) if total_count > 0 else 0.0, 4),
        )
        for row in rows
        if row.get("label") is not None
    ]
    config = QUALITY_CHECKS[name]
    dominant_ratio = distribution[0].ratio if distribution else 0.0
    if len(distribution) <= 1 and total_count > 0:
        action = str(config["single_action"])
        severity = "high"
    elif dominant_ratio >= 0.9:
        action = str(config["skew_action"])
        severity = "medium"
    elif dominant_ratio >= 0.75:
        action = "review"
        severity = "low"
    else:
        action = "keep"
        severity = "none"

    return GraphQualityCheck(
        name=name,
        cardinality=len(distribution),
        total_count=total_count,
        issue=str(config["issue"]),
        recommendation=str(config["recommendation"]),
        action=action,
        severity=severity,
        dominant_ratio=dominant_ratio,
        distribution=distribution,
    )


@router.get("/graph-quality", response_model=GraphQualityResponse)
def graph_quality() -> GraphQualityResponse:
    checks: list[GraphQualityCheck] = []
    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            for name, config in QUALITY_CHECKS.items():
                query_text = config["query"]
                rows = [dict(record) for record in session.run(cast(LiteralString, query_text))]
                checks.append(_build_check(name, rows))
    finally:
        driver.close()

    return GraphQualityResponse(checks=checks, migration_plan=COUNTRY_MIGRATION_PLAN)
