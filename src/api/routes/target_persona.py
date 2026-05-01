from fastapi import APIRouter, Query
from neo4j import GraphDatabase
from typing import Any, LiteralString, cast

from src.api.exceptions import BadRequestException
from src.api.schemas import TargetPersonaResponse, TargetPersonaSample
from src.config import settings

router = APIRouter(prefix="/api", tags=["target-persona"])

TARGET_PERSONA_GUARDRAILS = [
    "반드시 sample_personas와 representative_hobbies/skills에 근거해 작성합니다.",
    "근거에 없는 직업, 지역, 취미, 기술을 새로 만들지 않습니다.",
    "개별 페르소나를 실제 인물처럼 단정하지 않고 집합의 대표 경향으로 표현합니다.",
    "불확실한 내용은 추정이라고 명시합니다.",
]
INPUT_POLICY = "지원 필터는 age_group, sex, province, district, occupation, hobby, skill, semantic_query입니다. 최소 1개 필터가 필요하며 top_k는 1~10명, semantic_top_k는 5~200명으로 제한합니다."


def get_neo4j_driver():  # noqa: ANN201
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _build_filter_query_parts(
    age_group: str | None,
    sex: str | None,
    province: str | None,
    district: str | None,
    occupation: str | None,
    hobby: str | None,
    skill: str | None,
) -> tuple[list[str], list[str], dict[str, object], dict[str, str]]:
    matches: list[str] = ["MATCH (p:Person)"]
    where_clauses: list[str] = []
    params: dict[str, object] = {}
    filters: dict[str, str] = {}

    age_group_value = _normalize_text(age_group)
    sex_value = _normalize_text(sex)
    province_value = _normalize_text(province)
    district_value = _normalize_text(district)
    occupation_value = _normalize_text(occupation)
    hobby_value = _normalize_text(hobby)
    skill_value = _normalize_text(skill)

    if age_group_value:
        where_clauses.append("p.age_group = $age_group")
        params["age_group"] = age_group_value
        filters["age_group"] = age_group_value
    if sex_value:
        where_clauses.append("p.sex = $sex")
        params["sex"] = sex_value
        filters["sex"] = sex_value
    if province_value:
        matches.append("MATCH (p)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(prov:Province)")
        where_clauses.append("prov.name = $province")
        params["province"] = province_value
        filters["province"] = province_value
    if district_value:
        matches.append("MATCH (p)-[:LIVES_IN]->(d:District)")
        where_clauses.append("(d.name = $district OR d.key = $district)")
        params["district"] = district_value
        filters["district"] = district_value
    if occupation_value:
        matches.append("MATCH (p)-[:WORKS_AS]->(occ:Occupation)")
        where_clauses.append("occ.name CONTAINS $occupation")
        params["occupation"] = occupation_value
        filters["occupation"] = occupation_value
    if hobby_value:
        matches.append("MATCH (p)-[:ENJOYS_HOBBY]->(h:Hobby)")
        where_clauses.append("h.name CONTAINS $hobby")
        params["hobby"] = hobby_value
        filters["hobby"] = hobby_value
    if skill_value:
        matches.append("MATCH (p)-[:HAS_SKILL]->(s:Skill)")
        where_clauses.append("s.name CONTAINS $skill")
        params["skill"] = skill_value
        filters["skill"] = skill_value

    return matches, where_clauses, params, filters


def _build_representative_persona_text(
    filters: dict[str, str],
    sample_personas: list[TargetPersonaSample],
    hobbies: list[str],
    skills: list[str],
) -> str:
    if not sample_personas:
        return "조건에 맞는 페르소나가 없어 대표 프로필을 생성할 수 없습니다."

    filter_summary = ", ".join(f"{k}={v}" for k, v in filters.items())
    profile_parts: list[str] = []

    if filter_summary:
        profile_parts.append(f"조건({filter_summary})에 맞는 페르소나 집합의 대표 프로필입니다.")
    else:
        profile_parts.append("선택된 표본 페르소나 집합의 대표 프로필입니다.")

    first = sample_personas[0]
    if first.persona:
        profile_parts.append(f"핵심 성향: {first.persona}")
    if hobbies:
        profile_parts.append(f"주요 취미: {', '.join(hobbies[:5])}")
    if skills:
        profile_parts.append(f"주요 스킬: {', '.join(skills[:5])}")

    return " ".join(profile_parts)


def _build_synthesis_prompt(
    filters: dict[str, str],
    sample_personas: list[TargetPersonaSample],
    hobbies: list[str],
    skills: list[str],
) -> str:
    samples = [
        {
            "uuid": sample.uuid,
            "age": sample.age,
            "sex": sample.sex,
            "province": sample.province,
            "district": sample.district,
            "occupation": sample.occupation,
            "persona": sample.persona,
        }
        for sample in sample_personas
    ]
    return (
        "조건 기반 대표 페르소나를 한국어 3~5문장으로 합성하세요. "
        f"필터={filters}. 대표 취미={hobbies}. 대표 스킬={skills}. "
        f"근거 샘플={samples}. 금지사항={TARGET_PERSONA_GUARDRAILS}"
    )


def _synthesize_with_llm(prompt: str) -> str | None:
    try:
        from src.rag.llm import create_llm

        response = create_llm(temperature=0.2).invoke(prompt)
    except Exception:
        return None

    content = getattr(response, "content", None)
    if isinstance(content, str):
        stripped = content.strip()
        return stripped or None
    return None


def _collect_semantic_persona_uuids(query_text: str, top_k: int) -> list[str]:
    try:
        from src.embeddings.kure_model import KureEmbedder
        from src.embeddings.vector_index import Neo4jVectorIndex
    except Exception:
        return []

    try:
        embedder = KureEmbedder()
        vector_index = Neo4jVectorIndex()
    except Exception:
        return []

    try:
        embedding = embedder.encode([query_text])[0]
        hits = vector_index.search(embedding, top_k=top_k)
    except Exception:
        vector_index.close()
        return []

    uuids: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        uuid_value = hit.get("uuid")
        if not uuid_value:
            continue
        uuid = str(uuid_value)
        if uuid in seen:
            continue
        seen.add(uuid)
        uuids.append(uuid)

    vector_index.close()
    return uuids


def _as_int(value: int | float | str | bool | None) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


@router.get("/target-persona", response_model=TargetPersonaResponse)
def target_persona(
    age_group: str | None = Query(default=None),
    sex: str | None = Query(default=None),
    province: str | None = Query(default=None),
    district: str | None = Query(default=None),
    occupation: str | None = Query(default=None),
    hobby: str | None = Query(default=None),
    skill: str | None = Query(default=None),
    top_k: int = Query(default=5, ge=1, le=10),
    semantic_query: str | None = Query(default=None, description="KURE semantic query"),
    semantic_top_k: int = Query(default=50, ge=5, le=200),
    use_llm: bool = Query(default=False),
) -> TargetPersonaResponse:
    matches, where_clauses, params, filters = _build_filter_query_parts(
        age_group=age_group,
        sex=sex,
        province=province,
        district=district,
        occupation=occupation,
        hobby=hobby,
        skill=skill,
    )

    if not filters:
        raise BadRequestException("최소 1개 이상의 필터를 입력해야 합니다.")

    semantic_query_value = _normalize_text(semantic_query)
    semantic_persona_uuids: list[str] = []
    if semantic_query_value:
        semantic_persona_uuids = _collect_semantic_persona_uuids(
            query_text=semantic_query_value,
            top_k=semantic_top_k,
        )
        if semantic_persona_uuids:
            filters["semantic_query"] = semantic_query_value

    if semantic_persona_uuids:
        where_clauses.append("p.uuid IN $semantic_persona_uuids")
        params["semantic_persona_uuids"] = semantic_persona_uuids

    where_part = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    base_match = "\n".join(matches)

    count_query = "\n".join([
        base_match,
        where_part,
        "RETURN count(DISTINCT p) AS matched_count",
    ])

    sample_query = "\n".join([
        base_match,
        where_part,
        "WITH DISTINCT p",
        "OPTIONAL MATCH (p)-[:LIVES_IN]->(d:District)-[:IN_PROVINCE]->(prov:Province)",
        "OPTIONAL MATCH (p)-[:WORKS_AS]->(occ:Occupation)",
        "RETURN p.uuid AS uuid, p.display_name AS display_name, p.age AS age, p.sex AS sex, "
        "p.persona AS persona, prov.name AS province, d.name AS district, occ.name AS occupation",
        "ORDER BY p.age ASC",
        "LIMIT $top_k",
    ])

    hobbies_query = "\n".join([
        base_match,
        where_part,
        "WITH DISTINCT p",
        "MATCH (p)-[:ENJOYS_HOBBY]->(h:Hobby)",
        "RETURN h.name AS name, count(*) AS cnt",
        "ORDER BY cnt DESC",
        "LIMIT 5",
    ])

    skills_query = "\n".join([
        base_match,
        where_part,
        "WITH DISTINCT p",
        "MATCH (p)-[:HAS_SKILL]->(s:Skill)",
        "RETURN s.name AS name, count(*) AS cnt",
        "ORDER BY cnt DESC",
        "LIMIT 5",
    ])

    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            query_params: dict[str, Any] = {**params, "top_k": top_k}
            count_record = session.run(cast(LiteralString, count_query), parameters=query_params).single()
            matched_count = _as_int(count_record["matched_count"]) if count_record else 0

            sample_rows = [dict(record) for record in session.run(cast(LiteralString, sample_query), parameters=query_params)]
            hobby_rows = [dict(record) for record in session.run(cast(LiteralString, hobbies_query), parameters=query_params)]
            skill_rows = [dict(record) for record in session.run(cast(LiteralString, skills_query), parameters=query_params)]
    finally:
        driver.close()

    sample_personas = [TargetPersonaSample(**row) for row in sample_rows]
    representative_hobbies = [str(row["name"]) for row in hobby_rows if row.get("name")]
    representative_skills = [str(row["name"]) for row in skill_rows if row.get("name")]
    representative_persona = _build_representative_persona_text(
        filters=filters,
        sample_personas=sample_personas,
        hobbies=representative_hobbies,
        skills=representative_skills,
    )
    synthesis_prompt = _build_synthesis_prompt(
        filters=filters,
        sample_personas=sample_personas,
        hobbies=representative_hobbies,
        skills=representative_skills,
    )
    generation_method = "deterministic"
    if use_llm and sample_personas:
        llm_text = _synthesize_with_llm(synthesis_prompt)
        if llm_text:
            representative_persona = llm_text
            generation_method = "llm"

    return TargetPersonaResponse(
        filters=filters,
        matched_count=matched_count,
        sample_size=len(sample_personas),
        representative_persona=representative_persona,
        representative_hobbies=representative_hobbies,
        representative_skills=representative_skills,
        sample_personas=sample_personas,
        evidence_uuids=[sample.uuid for sample in sample_personas],
        generation_method=generation_method,
        synthesis_prompt=synthesis_prompt,
        guardrails=TARGET_PERSONA_GUARDRAILS,
        input_policy=INPUT_POLICY,
    )
