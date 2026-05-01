from fastapi import APIRouter, Query
from neo4j import GraphDatabase
from typing import Any, LiteralString, cast

from src.api.exceptions import BadRequestException
from src.api.schemas import CareerTransitionItem, CareerTransitionResponse
from src.config import settings

router = APIRouter(prefix="/api", tags=["career-transition"])
MAPPING_POLICY = "occupation은 CONTAINS 매칭, career_goals_and_ambitions는 comma/semicolon/pipe/slash/newline 구분자 기반 분리, skills는 HAS_SKILL 관계를 사용합니다. 인접 직업은 공통 Hobby를 공유하는 다른 Person의 Occupation으로 계산합니다."
ANALYSIS_SCOPE = "이 화면은 직업 기준 목표/스킬/인접 직업 비교용이며 개인별 추천은 하지 않습니다. 목표/스킬 기반 추천은 별도 추천 API 또는 후속 UI에서 다룹니다."


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


def _to_items(rows: list[dict[str, Any]], total: int) -> list[CareerTransitionItem]:
    items: list[CareerTransitionItem] = []
    for row in rows:
        name = row.get("name")
        if not name:
            continue
        count = int(row.get("count", 0))
        ratio = (count / total) if total > 0 else 0.0
        items.append(CareerTransitionItem(name=str(name), count=count, ratio=round(ratio, 4)))
    return items


@router.get("/career-transition-map", response_model=CareerTransitionResponse)
def career_transition_map(
    occupation: str = Query(..., min_length=1),
    province: str | None = Query(default=None),
    age_group: str | None = Query(default=None),
    sex: str | None = Query(default=None),
    top_k: int = Query(default=10, ge=3, le=30),
    compare_by: str = Query(default="age_group"),
) -> CareerTransitionResponse:
    occupation_value = _normalize(occupation)
    if not occupation_value:
        raise BadRequestException("occupation 값이 비어 있습니다.")

    province_value = _normalize(province)
    age_group_value = _normalize(age_group)
    sex_value = _normalize(sex)

    matches = ["MATCH (p:Person)-[:WORKS_AS]->(occ:Occupation)"]
    where_clauses = ["occ.name CONTAINS $occupation"]
    params: dict[str, Any] = {"occupation": occupation_value, "top_k": top_k}
    filters: dict[str, str] = {"occupation": occupation_value}

    if province_value:
        matches.append("MATCH (p)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(prov:Province)")
        where_clauses.append("prov.name = $province")
        params["province"] = province_value
        filters["province"] = province_value
    if age_group_value:
        where_clauses.append("p.age_group = $age_group")
        params["age_group"] = age_group_value
        filters["age_group"] = age_group_value
    if sex_value:
        where_clauses.append("p.sex = $sex")
        params["sex"] = sex_value
        filters["sex"] = sex_value

    where_clause = "WHERE " + " AND ".join(where_clauses)
    base = "\n".join(matches + [where_clause])

    compare_fields = {
        "age_group": "p.age_group",
        "sex": "p.sex",
        "province": "prov.name",
    }
    if compare_by not in compare_fields:
        raise BadRequestException("compare_by는 age_group, sex, province 중 하나여야 합니다.")

    compare_expr = compare_fields[compare_by]

    count_query = "\n".join([base, "RETURN count(DISTINCT p) AS matched_count"])
    goals_query = "\n".join(
        [
            base,
            "WITH p, replace(replace(replace(replace(coalesce(p.career_goals_and_ambitions, ''), ';', ','), '|', ','), '/', ','), '\\n', ',') AS goals_text",
            "WITH p, split(goals_text, ',') AS goals",
            "UNWIND goals AS goal",
            "WITH trim(goal) AS g",
            "WHERE g <> '' AND size(g) >= 2",
            "RETURN g AS name, count(*) AS count",
            "ORDER BY count DESC",
            "LIMIT $top_k",
        ]
    )
    skills_query = "\n".join(
        [
            base,
            "MATCH (p)-[:HAS_SKILL]->(s:Skill)",
            "RETURN s.name AS name, count(DISTINCT p) AS count",
            "ORDER BY count DESC",
            "LIMIT $top_k",
        ]
    )
    neighbor_occupation_query = "\n".join(
        [
            base,
            "MATCH (p)-[:ENJOYS_HOBBY]->(h:Hobby)<-[:ENJOYS_HOBBY]-(n:Person)-[:WORKS_AS]->(nocc:Occupation)",
            "WHERE n.uuid <> p.uuid AND nocc.name <> occ.name",
            "RETURN nocc.name AS name, count(DISTINCT n) AS count",
            "ORDER BY count DESC",
            "LIMIT $top_k",
        ]
    )
    segment_query_parts = matches.copy()
    if compare_by == "province" and not province_value:
        segment_query_parts.append("MATCH (p)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(prov:Province)")
    segment_query = "\n".join(
        segment_query_parts
        + [
            where_clause,
            f"WITH {compare_expr} AS seg",
            "WHERE seg IS NOT NULL AND seg <> ''",
            "RETURN seg AS name, count(*) AS count",
            "ORDER BY count DESC",
            "LIMIT $top_k",
        ]
    )

    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            count_record = session.run(cast(LiteralString, count_query), parameters=params).single()
            matched_count = int(count_record["matched_count"]) if count_record else 0
            goal_rows = [dict(r) for r in session.run(cast(LiteralString, goals_query), parameters=params)]
            skill_rows = [dict(r) for r in session.run(cast(LiteralString, skills_query), parameters=params)]
            neighbor_occ_rows = [dict(r) for r in session.run(cast(LiteralString, neighbor_occupation_query), parameters=params)]
            segment_rows = [dict(r) for r in session.run(cast(LiteralString, segment_query), parameters=params)]
    finally:
        driver.close()

    return CareerTransitionResponse(
        filters=filters,
        matched_count=matched_count,
        top_goals=_to_items(goal_rows, matched_count),
        top_skills=_to_items(skill_rows, matched_count),
        top_neighbor_occupations=_to_items(neighbor_occ_rows, matched_count),
        segment_distribution=_to_items(segment_rows, matched_count),
        mapping_policy=MAPPING_POLICY,
        top_k_limit=30,
        analysis_scope=ANALYSIS_SCOPE,
    )
