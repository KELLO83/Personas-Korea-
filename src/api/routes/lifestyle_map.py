import re
from collections import Counter
from typing import Any, LiteralString, cast

from fastapi import APIRouter, Query
from neo4j import GraphDatabase

from src.api.exceptions import BadRequestException
from src.api.schemas import LifestyleMapEdge, LifestyleMapResponse
from src.config import settings

router = APIRouter(prefix="/api", tags=["lifestyle-map"])


PERSONA_FIELDS = {
    "professional_persona": "p.professional_persona",
    "sports_persona": "p.sports_persona",
    "arts_persona": "p.arts_persona",
    "travel_persona": "p.travel_persona",
    "culinary_persona": "p.culinary_persona",
    "family_persona": "p.family_persona",
    "persona": "p.persona",
}
KEYWORD_POLICY = "candidate_keywords를 명시하면 해당 키워드만 평가하고, 없으면 target_field 텍스트에서 2글자 이상 토큰을 추출한 뒤 min_keyword_count 이상만 사용합니다."
SEGMENT_POLICY = "세그먼트 필터는 province, age_group, sex를 지원하며 모든 조건은 AND로 결합합니다."
VISUALIZATION_POLICY = "프론트에서는 overlap_count 기준 상위 항목과 conditional_ratio를 함께 표시하고, 낮은 빈도 키워드는 min_keyword_count로 제외합니다."

KOREAN_STOPWORDS = {
    "그리고",
    "하지만",
    "또한",
    "위해",
    "통해",
    "정도",
    "관심",
    "활동",
    "생활",
    "사람",
    "현재",
    "주로",
    "대한",
    "있음",
    "있다",
}


def get_neo4j_driver():  # noqa: ANN201
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _extract_keywords(texts: list[str], top_k: int, min_count: int) -> list[str]:
    token_counter: Counter[str] = Counter()
    for text in texts:
        for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", text):
            lowered = token.lower()
            if lowered in KOREAN_STOPWORDS:
                continue
            token_counter[token] += 1
    return [token for token, count in token_counter.most_common(top_k) if count >= min_count]


@router.get("/lifestyle-map", response_model=LifestyleMapResponse)
def lifestyle_map(
    source_field: str = Query(default="sports_persona"),
    target_field: str = Query(default="culinary_persona"),
    source_keyword: str = Query(..., min_length=1),
    candidate_keywords: str | None = Query(default=None, description="comma-separated target keywords"),
    auto_keyword_top_k: int = Query(default=10, ge=3, le=30),
    min_keyword_count: int = Query(default=2, ge=1, le=20),
    province: str | None = Query(default=None),
    age_group: str | None = Query(default=None),
    sex: str | None = Query(default=None),
) -> LifestyleMapResponse:
    if source_field not in PERSONA_FIELDS or target_field not in PERSONA_FIELDS:
        raise BadRequestException("source_field/target_field 값이 유효하지 않습니다.")
    if source_field == target_field:
        raise BadRequestException("source_field와 target_field는 서로 달라야 합니다.")

    keywords: list[str] = []
    if candidate_keywords is not None:
        keywords = [k.strip() for k in candidate_keywords.split(",") if k.strip()]

    province_value = _normalize(province)
    age_group_value = _normalize(age_group)
    sex_value = _normalize(sex)

    filters: dict[str, str] = {}
    where_parts: list[str] = [f"{PERSONA_FIELDS[source_field]} CONTAINS $source_keyword"]
    params: dict[str, Any] = {"source_keyword": source_keyword.strip()}
    match_parts = ["MATCH (p:Person)"]

    if province_value:
        match_parts.append("MATCH (p)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(prov:Province)")
        where_parts.append("prov.name = $province")
        params["province"] = province_value
        filters["province"] = province_value
    if age_group_value:
        where_parts.append("p.age_group = $age_group")
        params["age_group"] = age_group_value
        filters["age_group"] = age_group_value
    if sex_value:
        where_parts.append("p.sex = $sex")
        params["sex"] = sex_value
        filters["sex"] = sex_value

    where_clause = "WHERE " + " AND ".join(where_parts)
    source_count_query = "\n".join(match_parts + [where_clause, "RETURN count(DISTINCT p) AS source_count"])

    target_text_query = "\n".join(
        match_parts
        + [
            where_clause,
            f"RETURN {PERSONA_FIELDS[target_field]} AS target_text",
            "LIMIT 2000",
        ]
    )

    edge_query = "\n".join(
        match_parts
        + [
            where_clause,
            "WITH DISTINCT p, $candidate_keywords AS keywords",
            "UNWIND keywords AS kw",
            f"WITH p, kw, ({PERSONA_FIELDS[target_field]} CONTAINS kw) AS has_kw",
            "WITH kw, count(DISTINCT p) AS support_count, "
            "sum(CASE WHEN has_kw THEN 1 ELSE 0 END) AS overlap_count",
            "WHERE overlap_count > 0",
            "RETURN kw AS target_keyword, overlap_count, support_count",
            "ORDER BY overlap_count DESC, support_count DESC",
        ]
    )

    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            source_count_record = session.run(cast(LiteralString, source_count_query), parameters=params).single()
            source_count = int(source_count_record["source_count"]) if source_count_record else 0
            if not keywords:
                target_rows = [dict(record) for record in session.run(cast(LiteralString, target_text_query), parameters=params)]
                target_texts = [str(row.get("target_text", "")) for row in target_rows if row.get("target_text")]
                keywords = _extract_keywords(
                    target_texts,
                    top_k=auto_keyword_top_k,
                    min_count=min_keyword_count,
                )
            if not keywords:
                raise BadRequestException("candidate_keywords를 생성할 수 없습니다. 직접 지정해 주세요.")
            params["candidate_keywords"] = keywords
            edge_rows = [dict(record) for record in session.run(cast(LiteralString, edge_query), parameters=params)]
    finally:
        driver.close()

    edges: list[LifestyleMapEdge] = []
    for row in edge_rows:
        overlap_count = int(row.get("overlap_count", 0))
        support_count = int(row.get("support_count", 0))
        ratio = (overlap_count / source_count) if source_count > 0 else 0.0
        edges.append(
            LifestyleMapEdge(
                source_field=source_field,
                target_field=target_field,
                source_keyword=source_keyword.strip(),
                target_keyword=str(row.get("target_keyword", "")),
                overlap_count=overlap_count,
                target_support_count=support_count,
                conditional_ratio=round(ratio, 4),
            )
        )

    return LifestyleMapResponse(
        filters=filters,
        source_field=source_field,
        target_field=target_field,
        source_keyword=source_keyword.strip(),
        matched_source_count=source_count,
        available_fields=list(PERSONA_FIELDS.keys()),
        keyword_policy=KEYWORD_POLICY,
        segment_policy=SEGMENT_POLICY,
        visualization_policy=VISUALIZATION_POLICY,
        edges=edges,
    )
