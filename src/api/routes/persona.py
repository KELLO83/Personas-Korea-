from neo4j import GraphDatabase

from fastapi import APIRouter

from src.api.exceptions import NotFoundException
from src.api.schemas import (
    CommunityInfo,
    Demographics,
    GraphStats,
    Location,
    PersonaProfileResponse,
    Personas,
    SimilarityExplanationResponse,
    SimilarityReason,
    SimilarPreview,
)
from src.config import settings
from src.graph.persona_queries import GRAPH_STATS_QUERY, PROFILE_QUERY, SIMILAR_PREVIEW_QUERY

router = APIRouter(prefix="/api", tags=["persona"])


SIMILARITY_EXPLANATION_QUERY = """
MATCH (source:Person {uuid: $source_uuid})
MATCH (target:Person {uuid: $target_uuid})
OPTIONAL MATCH (source)-[sim:SIMILAR_TO]->(target)
OPTIONAL MATCH (target)-[sim_reverse:SIMILAR_TO]->(source)
OPTIONAL MATCH (source)-[:LIVES_IN]->(source_district:District)-[:IN_PROVINCE]->(source_province:Province)
OPTIONAL MATCH (target)-[:LIVES_IN]->(target_district:District)-[:IN_PROVINCE]->(target_province:Province)
OPTIONAL MATCH (source)-[:WORKS_AS]->(source_occupation:Occupation)
OPTIONAL MATCH (target)-[:WORKS_AS]->(target_occupation:Occupation)
OPTIONAL MATCH (source)-[:EDUCATED_AT]->(source_education:EducationLevel)
OPTIONAL MATCH (target)-[:EDUCATED_AT]->(target_education:EducationLevel)
OPTIONAL MATCH (source)-[:MAJORED_IN]->(source_field:Field)
OPTIONAL MATCH (target)-[:MAJORED_IN]->(target_field:Field)
OPTIONAL MATCH (source)-[:MARITAL_STATUS]->(source_marital:MaritalStatus)
OPTIONAL MATCH (target)-[:MARITAL_STATUS]->(target_marital:MaritalStatus)
OPTIONAL MATCH (source)-[:LIVES_WITH]->(source_family:FamilyType)
OPTIONAL MATCH (target)-[:LIVES_WITH]->(target_family:FamilyType)
OPTIONAL MATCH (source)-[:LIVES_IN_HOUSING]->(source_housing:HousingType)
OPTIONAL MATCH (target)-[:LIVES_IN_HOUSING]->(target_housing:HousingType)
CALL (source, target) {
    OPTIONAL MATCH (source)-[:ENJOYS_HOBBY|LIKES]->(shared_hobby:Hobby)<-[:ENJOYS_HOBBY|LIKES]-(target)
    RETURN collect(DISTINCT shared_hobby.name) AS shared_hobbies
}
CALL (source, target) {
    OPTIONAL MATCH (source)-->(shared_skill:Skill)<--(target)
    RETURN collect(DISTINCT shared_skill.name) AS shared_skills
}
RETURN
    source.uuid AS source_uuid,
    target.uuid AS target_uuid,
    coalesce(sim.score, sim_reverse.score) AS similarity_score,
    source.age_group AS source_age_group,
    target.age_group AS target_age_group,
    source.sex AS source_sex,
    target.sex AS target_sex,
    source_province.name AS source_province,
    target_province.name AS target_province,
    source_district.name AS source_district,
    target_district.name AS target_district,
    source_occupation.name AS source_occupation,
    target_occupation.name AS target_occupation,
    source_education.name AS source_education,
    target_education.name AS target_education,
    source_field.name AS source_field,
    target_field.name AS target_field,
    source_marital.name AS source_marital,
    target_marital.name AS target_marital,
    source_family.name AS source_family,
    target_family.name AS target_family,
    source_housing.name AS source_housing,
    target_housing.name AS target_housing,
    source.community_id AS source_community_id,
    target.community_id AS target_community_id,
    [hobby IN shared_hobbies WHERE hobby IS NOT NULL] AS shared_hobbies,
    [skill IN shared_skills WHERE skill IS NOT NULL] AS shared_skills
"""

EXPLANATION_FEATURES: tuple[tuple[str, str, str, str, float], ...] = (
    ("province", "지역", "source_province", "target_province", 0.15),
    ("district", "시군구", "source_district", "target_district", 0.12),
    ("occupation", "직업", "source_occupation", "target_occupation", 0.18),
    ("age_group", "연령대", "source_age_group", "target_age_group", 0.08),
    ("sex", "성별", "source_sex", "target_sex", 0.04),
    ("education", "학력", "source_education", "target_education", 0.10),
    ("field", "전공", "source_field", "target_field", 0.08),
    ("marital_status", "혼인", "source_marital", "target_marital", 0.05),
    ("family_type", "가구", "source_family", "target_family", 0.07),
    ("housing_type", "주거", "source_housing", "target_housing", 0.07),
    ("community", "커뮤니티", "source_community_id", "target_community_id", 0.16),
)

HOBBY_WEIGHT = 0.22
SKILL_WEIGHT = 0.12


def get_neo4j_driver():  # noqa: ANN201
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def _clean_values(values: list[object]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized and normalized not in seen:
            cleaned.append(normalized)
            seen.add(normalized)
    return cleaned


def _add_reason(
    reasons: list[dict[str, object]],
    feature: str,
    label: str,
    value: object,
    raw_score: float,
) -> None:
    if value is None or value == "":
        return
    reasons.append(
        {
            "feature": feature,
            "label": label,
            "value": str(value),
            "raw_score": raw_score,
        }
    )


def _build_explanation(record: dict[str, object]) -> SimilarityExplanationResponse:
    raw_reasons: list[dict[str, object]] = []

    for feature, label, source_key, target_key, weight in EXPLANATION_FEATURES:
        source_value = record.get(source_key)
        target_value = record.get(target_key)
        if source_value is not None and source_value == target_value:
            _add_reason(raw_reasons, feature, label, source_value, weight)

    shared_hobbies = _clean_values(record.get("shared_hobbies", []))  # type: ignore[arg-type]
    shared_skills = _clean_values(record.get("shared_skills", []))  # type: ignore[arg-type]

    hobby_score = HOBBY_WEIGHT * min(len(shared_hobbies), 5) / 5
    skill_score = SKILL_WEIGHT * min(len(shared_skills), 5) / 5
    for hobby in shared_hobbies[:5]:
        _add_reason(raw_reasons, "shared_hobby", "공유 취미", hobby, hobby_score / max(1, min(len(shared_hobbies), 5)))
    for skill in shared_skills[:5]:
        _add_reason(raw_reasons, "shared_skill", "공유 스킬", skill, skill_score / max(1, min(len(shared_skills), 5)))

    total_score = sum(float(reason["raw_score"]) for reason in raw_reasons)
    top_reasons = [
        SimilarityReason(
            feature=str(reason["feature"]),
            label=str(reason["label"]),
            value=str(reason["value"]),
            raw_score=round(float(reason["raw_score"]), 6),
            contribution=round(float(reason["raw_score"]) / total_score, 6) if total_score > 0 else 0.0,
        )
        for reason in sorted(raw_reasons, key=lambda item: float(item["raw_score"]), reverse=True)
    ]

    note = (
        "FastRP/KNN으로 선택된 유사 후보에 대해, 현재 API에서 조회 가능한 공통 속성을 post-hoc 방식으로 정규화한 설명입니다."
        if top_reasons
        else "FastRP/KNN 유사 후보이지만 현재 API에서 직접 일치하는 설명 속성을 찾지 못했습니다."
    )

    return SimilarityExplanationResponse(
        source_uuid=str(record["source_uuid"]),
        target_uuid=str(record["target_uuid"]),
        similarity_score=record.get("similarity_score"),  # type: ignore[arg-type]
        top_reasons=top_reasons,
        shared_hobbies=shared_hobbies,
        shared_skills=shared_skills,
        note=note,
    )


@router.get("/persona/{uuid}", response_model=PersonaProfileResponse)
def persona_profile(uuid: str) -> PersonaProfileResponse:
    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            profile_record = session.run(PROFILE_QUERY, uuid=uuid).single()
            if not profile_record:
                raise NotFoundException("해당 UUID의 페르소나를 찾을 수 없습니다.")

            similar_records = [
                dict(record) for record in session.run(SIMILAR_PREVIEW_QUERY, uuid=uuid)
            ]
            stats_record = session.run(GRAPH_STATS_QUERY, uuid=uuid).single()
    finally:
        driver.close()

    p = dict(profile_record["p"])

    demographics = Demographics(
        age=p.get("age"),
        age_group=p.get("age_group"),
        sex=p.get("sex"),
        marital_status=profile_record["marital_status"],
        military_status=profile_record["military_status"],
        family_type=profile_record["family_type"],
        housing_type=profile_record["housing_type"],
        education_level=profile_record["education_level"],
        bachelors_field=profile_record["bachelors_field"],
    )

    location = Location(
        country=profile_record["country_name"],
        province=profile_record["province_name"],
        district=profile_record["district_name"],
    )

    personas = Personas(
        summary=p.get("persona"),
        professional=p.get("professional_persona"),
        sports=p.get("sports_persona"),
        arts=p.get("arts_persona"),
        travel=p.get("travel_persona"),
        culinary=p.get("culinary_persona"),
        family=p.get("family_persona"),
    )

    community = CommunityInfo(
        community_id=p.get("community_id"),
        label=p.get("community_label"),
    )

    similar_preview = [
        SimilarPreview(
            uuid=rec["uuid"],
            display_name=rec.get("display_name"),
            age=rec.get("age"),
            similarity=rec.get("similarity"),
            shared_hobbies=rec.get("shared_hobbies", []),
        )
        for rec in similar_records
    ]

    graph_stats = GraphStats(
        total_connections=stats_record["total_connections"] if stats_record else 0,
        hobby_count=stats_record["hobby_count"] if stats_record else 0,
        skill_count=stats_record["skill_count"] if stats_record else 0,
    )

    return PersonaProfileResponse(
        uuid=uuid,
        display_name=p.get("display_name"),
        demographics=demographics,
        location=location,
        occupation=profile_record["occupation_name"],
        personas=personas,
        cultural_background=p.get("cultural_background"),
        career_goals=p.get("career_goals_and_ambitions"),
        skills=profile_record["skills"],
        hobbies=profile_record["hobbies"],
        community=community,
        similar_preview=similar_preview,
        graph_stats=graph_stats,
    )


@router.get(
    "/persona/{source_uuid}/similar/{target_uuid}/explanation",
    response_model=SimilarityExplanationResponse,
)
def similarity_explanation(source_uuid: str, target_uuid: str) -> SimilarityExplanationResponse:
    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            record = session.run(
                SIMILARITY_EXPLANATION_QUERY,
                source_uuid=source_uuid,
                target_uuid=target_uuid,
            ).single()
            if not record:
                raise NotFoundException("source 또는 target 페르소나를 찾을 수 없습니다.")
            return _build_explanation(dict(record))
    finally:
        driver.close()
