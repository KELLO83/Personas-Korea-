from neo4j import GraphDatabase

from fastapi import APIRouter, Query

from src.api.exceptions import BadRequestException, NotFoundException
from src.api.schemas import (
    CommunityInfo,
    Demographics,
    GraphStats,
    GuildMember,
    LifeTrackResponse,
    LifeTrackRoleModel,
    LifeTrackTimelineItem,
    Location,
    PersonaProfileResponse,
    PersonaGuild,
    PersonaGuildResponse,
    Personas,
    RankedItem,
    SimilarDiversePersona,
    SimilarDiverseResponse,
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
MAX_GUILD_MEMBERS = 6
VALID_DIVERSITY_AXES = {"mixed", "occupation", "location", "community", "demographic"}

GUILD_CANDIDATE_QUERY = """
MATCH (source:Person {uuid: $uuid})
OPTIONAL MATCH (source)-[:LIVES_IN]->(source_district:District)-[:IN_PROVINCE]->(source_province:Province)
OPTIONAL MATCH (source)-[:WORKS_AS]->(source_occupation:Occupation)
CALL (source) {
    OPTIONAL MATCH (source)-[:ENJOYS_HOBBY|LIKES]->(source_hobby:Hobby)
    RETURN collect(DISTINCT source_hobby.name) AS source_hobbies
}
CALL (source) {
    OPTIONAL MATCH (source)-->(source_skill:Skill)
    RETURN collect(DISTINCT source_skill.name) AS source_skills
}
MATCH (source)-[similarity:SIMILAR_TO]->(candidate:Person)
OPTIONAL MATCH (candidate)-[:LIVES_IN]->(candidate_district:District)-[:IN_PROVINCE]->(candidate_province:Province)
OPTIONAL MATCH (candidate)-[:WORKS_AS]->(candidate_occupation:Occupation)
CALL (source, candidate) {
    OPTIONAL MATCH (source)-[:ENJOYS_HOBBY|LIKES]->(shared_hobby:Hobby)<-[:ENJOYS_HOBBY|LIKES]-(candidate)
    RETURN collect(DISTINCT shared_hobby.name) AS shared_hobbies
}
CALL (source, candidate) {
    OPTIONAL MATCH (source)-->(shared_skill:Skill)<--(candidate)
    RETURN collect(DISTINCT shared_skill.name) AS shared_skills
}
WITH source,
     candidate,
     source_province,
     source_district,
     source_occupation,
     candidate_province,
     candidate_district,
     candidate_occupation,
     coalesce(similarity.score, 0.0) AS similarity_score,
     [item IN shared_hobbies WHERE item IS NOT NULL] AS shared_hobbies,
     [item IN shared_skills WHERE item IS NOT NULL] AS shared_skills
WITH source,
     candidate,
     source_province,
     source_district,
     source_occupation,
     candidate_province,
     candidate_district,
     candidate_occupation,
     similarity_score,
     shared_hobbies,
     shared_skills,
     CASE WHEN source.community_id IS NOT NULL AND source.community_id = candidate.community_id THEN 1.0 ELSE 0.0 END AS same_community,
     CASE WHEN source_district.name IS NOT NULL AND source_district.name = candidate_district.name THEN 1.0 ELSE 0.0 END AS same_district,
     CASE WHEN source_province.name IS NOT NULL AND source_province.name = candidate_province.name THEN 1.0 ELSE 0.0 END AS same_province
RETURN
    source.community_id AS source_community_id,
    candidate.uuid AS uuid,
    candidate.display_name AS display_name,
    candidate.age AS age,
    candidate_occupation.name AS occupation,
    candidate_province.name AS province,
    candidate_district.name AS district,
    null AS pagerank,
    null AS degree,
    candidate.community_id AS community_id,
    similarity_score,
    shared_hobbies,
    shared_skills,
    same_community,
    same_district,
    same_province,
    (
        similarity_score * 0.45
        + same_community * 0.12
        + same_district * 0.15
        + same_province * 0.08
        + toFloat(size(shared_hobbies)) * 0.04
        + toFloat(size(shared_skills)) * 0.03
    ) AS guild_score
ORDER BY guild_score DESC
LIMIT $candidate_limit
"""

SIMILAR_DIVERSE_QUERY = """
MATCH (source:Person {uuid: $uuid})-[similarity:SIMILAR_TO]->(candidate:Person)
OPTIONAL MATCH (source)-[:LIVES_IN]->(source_district:District)-[:IN_PROVINCE]->(source_province:Province)
OPTIONAL MATCH (candidate)-[:LIVES_IN]->(candidate_district:District)-[:IN_PROVINCE]->(candidate_province:Province)
OPTIONAL MATCH (source)-[:WORKS_AS]->(source_occupation:Occupation)
OPTIONAL MATCH (candidate)-[:WORKS_AS]->(candidate_occupation:Occupation)
OPTIONAL MATCH (source)-[:EDUCATED_AT]->(source_education:EducationLevel)
OPTIONAL MATCH (candidate)-[:EDUCATED_AT]->(candidate_education:EducationLevel)
CALL (source, candidate) {
    OPTIONAL MATCH (source)-[:ENJOYS_HOBBY|LIKES]->(shared_hobby:Hobby)<-[:ENJOYS_HOBBY|LIKES]-(candidate)
    RETURN collect(DISTINCT shared_hobby.name) AS shared_hobbies
}
CALL (source, candidate) {
    OPTIONAL MATCH (source)-->(shared_skill:Skill)<--(candidate)
    RETURN collect(DISTINCT shared_skill.name) AS shared_skills
}
RETURN
    candidate.uuid AS uuid,
    candidate.display_name AS display_name,
    candidate.age AS age,
    candidate.age_group AS age_group,
    candidate.sex AS sex,
    candidate_occupation.name AS occupation,
    candidate_province.name AS province,
    candidate_district.name AS district,
    candidate.community_id AS community_id,
    coalesce(similarity.score, 0.0) AS similarity,
    source.age_group AS source_age_group,
    source.sex AS source_sex,
    source.community_id AS source_community_id,
    source_occupation.name AS source_occupation,
    source_province.name AS source_province,
    source_district.name AS source_district,
    source_education.name AS source_education,
    candidate_education.name AS candidate_education,
    [item IN shared_hobbies WHERE item IS NOT NULL] AS shared_hobbies,
    [item IN shared_skills WHERE item IS NOT NULL] AS shared_skills
ORDER BY similarity DESC
LIMIT $candidate_limit
"""

LIFE_TRACK_SOURCE_QUERY = """
MATCH (source:Person {uuid: $uuid})
RETURN source.uuid AS uuid, source.age AS age, source.age_group AS age_group, source.community_id AS community_id
"""

LIFE_TRACK_QUERY = """
MATCH (source:Person {uuid: $uuid})
MATCH (source)-[similarity:SIMILAR_TO]->(target:Person)
WHERE target.age >= $target_age_min AND target.age <= $target_age_max
OPTIONAL MATCH (source)-[:WORKS_AS]->(source_occupation:Occupation)
OPTIONAL MATCH (target)-[:WORKS_AS]->(target_occupation:Occupation)
OPTIONAL MATCH (target)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(target_province:Province)
CALL (source, target) {
    OPTIONAL MATCH (source)-[:ENJOYS_HOBBY|LIKES]->(shared_hobby:Hobby)<-[:ENJOYS_HOBBY|LIKES]-(target)
    RETURN collect(DISTINCT shared_hobby.name) AS shared_hobbies
}
CALL (source, target) {
    OPTIONAL MATCH (source)-->(shared_skill:Skill)<--(target)
    RETURN collect(DISTINCT shared_skill.name) AS shared_skills
}
CALL (target) {
    OPTIONAL MATCH (target)-[:ENJOYS_HOBBY|LIKES]->(target_hobby:Hobby)
    RETURN collect(DISTINCT target_hobby.name)[0..8] AS target_hobbies
}
CALL (target) {
    OPTIONAL MATCH (target)-->(target_skill:Skill)
    RETURN collect(DISTINCT target_skill.name)[0..8] AS target_skills
}
RETURN
    target.uuid AS uuid,
    target.display_name AS display_name,
    target.age AS age,
    target.age_group AS age_group,
    target_occupation.name AS occupation,
    target_province.name AS province,
    coalesce(similarity.score, 0.0) AS similarity,
    source_occupation.name AS source_occupation,
    [item IN shared_hobbies WHERE item IS NOT NULL] AS shared_hobbies,
    [item IN shared_skills WHERE item IS NOT NULL] AS shared_skills,
    [item IN target_hobbies WHERE item IS NOT NULL] AS target_hobbies,
    [item IN target_skills WHERE item IS NOT NULL] AS target_skills
ORDER BY similarity DESC
LIMIT $candidate_limit
"""


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


def _float_value(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _unique_strings(values: object, limit: int | None = None) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned = _clean_values(values)
    return cleaned if limit is None else cleaned[:limit]


def _build_member(record: dict[str, object], *, is_leader: bool) -> GuildMember:
    return GuildMember(
        uuid=str(record["uuid"]),
        display_name=record.get("display_name"),  # type: ignore[arg-type]
        age=record.get("age"),  # type: ignore[arg-type]
        occupation=record.get("occupation"),  # type: ignore[arg-type]
        province=record.get("province"),  # type: ignore[arg-type]
        district=record.get("district"),  # type: ignore[arg-type]
        pagerank=record.get("pagerank"),  # type: ignore[arg-type]
        degree=record.get("degree"),  # type: ignore[arg-type]
        is_leader=is_leader,
        score=round(_float_value(record.get("guild_score")), 4),
    )


def _build_guild(source_uuid: str, title: str, records: list[dict[str, object]]) -> PersonaGuild | None:
    if not records:
        return None
    ranked_records = sorted(
        records[:MAX_GUILD_MEMBERS],
        key=lambda item: (_float_value(item.get("pagerank")), _float_value(item.get("degree")), _float_value(item.get("guild_score"))),
        reverse=True,
    )
    leader_uuid = str(ranked_records[0]["uuid"])
    members = [_build_member(record, is_leader=str(record["uuid"]) == leader_uuid) for record in records[:MAX_GUILD_MEMBERS]]
    shared_hobbies = _most_common_values(records, "shared_hobbies", 8)
    shared_skills = _most_common_values(records, "shared_skills", 8)
    top_occupations = _most_common_scalar_values(records, "occupation", 5)
    evidence_parts = [
        f"공유 취미 {len(shared_hobbies)}개" if shared_hobbies else "",
        f"공유 스킬 {len(shared_skills)}개" if shared_skills else "",
        f"대표 직업 {', '.join(top_occupations[:3])}" if top_occupations else "",
    ]
    return PersonaGuild(
        guild_id=f"{source_uuid}-{title}".replace(" ", "-")[:96],
        title=title,
        score=round(sum(_float_value(record.get("guild_score")) for record in records[:MAX_GUILD_MEMBERS]), 4),
        reason=" · ".join(part for part in evidence_parts if part) or "SIMILAR_TO 기반 graph/rule 후보",
        shared_hobbies=shared_hobbies,
        shared_skills=shared_skills,
        top_occupations=top_occupations,
        members=members,
    )


def _most_common_values(records: list[dict[str, object]], key: str, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for record in records:
        for value in _unique_strings(record.get(key)):
            counts[value] = counts.get(value, 0) + 1
    return [value for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _most_common_scalar_values(records: list[dict[str, object]], key: str, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            counts[value] = counts.get(value, 0) + 1
    return [value for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _group_guild_candidates(source_uuid: str, records: list[dict[str, object]]) -> list[PersonaGuild]:
    guild_specs = [
        ("커뮤니티 기반 소모임", [record for record in records if _float_value(record.get("same_community")) > 0]),
        ("동네/지역 기반 소모임", [record for record in records if _float_value(record.get("same_district")) > 0 or _float_value(record.get("same_province")) > 0]),
        ("취미/스킬 기반 소모임", [record for record in records if record.get("shared_hobbies") or record.get("shared_skills")]),
    ]
    guilds = [_build_guild(source_uuid, title, group) for title, group in guild_specs]
    unique_guilds: list[PersonaGuild] = []
    seen_member_sets: set[tuple[str, ...]] = set()
    for guild in guilds:
        if guild is None:
            continue
        member_key = tuple(member.uuid for member in guild.members)
        if member_key and member_key not in seen_member_sets:
            unique_guilds.append(guild)
            seen_member_sets.add(member_key)
    return sorted(unique_guilds, key=lambda guild: guild.score, reverse=True)


def _axis_weight(axis: str, feature: str) -> float:
    if axis == "occupation" and feature == "occupation":
        return 0.34
    if axis == "location" and feature in {"province", "district"}:
        return 0.24
    if axis == "community" and feature == "community":
        return 0.34
    if axis == "demographic" and feature in {"age_group", "sex", "education"}:
        return 0.22
    return 0.12


def _diversity_reasons(record: dict[str, object], axis: str) -> tuple[float, list[str]]:
    checks = (
        ("occupation", "다른 직업", record.get("source_occupation"), record.get("occupation")),
        ("province", "다른 광역지역", record.get("source_province"), record.get("province")),
        ("district", "다른 시군구", record.get("source_district"), record.get("district")),
        ("age_group", "다른 연령대", record.get("source_age_group"), record.get("age_group")),
        ("sex", "다른 성별", record.get("source_sex"), record.get("sex")),
        ("education", "다른 학력", record.get("source_education"), record.get("candidate_education")),
        ("community", "다른 커뮤니티", record.get("source_community_id"), record.get("community_id")),
    )
    score = 0.0
    reasons: list[str] = []
    for feature, label, source_value, target_value in checks:
        if source_value is None or target_value is None or source_value == target_value:
            continue
        score += _axis_weight(axis, feature)
        reasons.append(f"{label}: {source_value} -> {target_value}")
    return min(score, 1.0), reasons[:6]


def _build_similar_diverse_item(record: dict[str, object], axis: str) -> SimilarDiversePersona:
    diversity_score, reasons = _diversity_reasons(record, axis)
    similarity = _float_value(record.get("similarity"))
    shared_score = min(len(_unique_strings(record.get("shared_hobbies"))) * 0.03 + len(_unique_strings(record.get("shared_skills"))) * 0.02, 0.16)
    final_score = similarity * 0.58 + diversity_score * 0.30 + shared_score
    return SimilarDiversePersona(
        uuid=str(record["uuid"]),
        display_name=record.get("display_name"),  # type: ignore[arg-type]
        age=record.get("age"),  # type: ignore[arg-type]
        sex=record.get("sex"),  # type: ignore[arg-type]
        occupation=record.get("occupation"),  # type: ignore[arg-type]
        province=record.get("province"),  # type: ignore[arg-type]
        district=record.get("district"),  # type: ignore[arg-type]
        similarity=round(similarity, 4),
        diversity_score=round(diversity_score, 4),
        final_score=round(final_score, 4),
        contrast_reasons=reasons,
        shared_hobbies=_unique_strings(record.get("shared_hobbies"), 8),
        shared_skills=_unique_strings(record.get("shared_skills"), 8),
    )


def _build_role_model(record: dict[str, object]) -> LifeTrackRoleModel:
    different_attributes = []
    if record.get("source_occupation") and record.get("source_occupation") != record.get("occupation"):
        different_attributes.append(f"직업 변화 관찰: {record.get('source_occupation')} -> {record.get('occupation')}")
    return LifeTrackRoleModel(
        uuid=str(record["uuid"]),
        display_name=record.get("display_name"),  # type: ignore[arg-type]
        age=record.get("age"),  # type: ignore[arg-type]
        age_group=record.get("age_group"),  # type: ignore[arg-type]
        occupation=record.get("occupation"),  # type: ignore[arg-type]
        province=record.get("province"),  # type: ignore[arg-type]
        similarity=round(_float_value(record.get("similarity")), 4),
        shared_hobbies=_unique_strings(record.get("shared_hobbies"), 8),
        shared_skills=_unique_strings(record.get("shared_skills"), 8),
        different_attributes=different_attributes,
    )


def _ranked_items_from_records(records: list[dict[str, object]], key: str, limit: int) -> list[RankedItem]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(key)
        if isinstance(value, list):
            for item in _unique_strings(value):
                counts[item] = counts.get(item, 0) + 1
        elif isinstance(value, str) and value.strip():
            counts[value] = counts.get(value, 0) + 1
    return [RankedItem(label=value, count=count) for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _build_life_track_timeline(records: list[dict[str, object]]) -> list[LifeTrackTimelineItem]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        age_band = str(record.get("age_group") or "미분류")
        grouped.setdefault(age_band, []).append(record)
    timeline: list[LifeTrackTimelineItem] = []
    for age_band, group in sorted(grouped.items()):
        timeline.append(
            LifeTrackTimelineItem(
                age_band=age_band,
                evidence_count=len(group),
                representative_occupations=[item.label for item in _ranked_items_from_records(group, "occupation", 5)],
                representative_skills=[item.label for item in _ranked_items_from_records(group, "target_skills", 6)],
                representative_hobbies=[item.label for item in _ranked_items_from_records(group, "target_hobbies", 6)],
            )
        )
    return timeline

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


@router.get("/persona/{uuid}/guilds", response_model=PersonaGuildResponse)
def persona_guilds(uuid: str) -> PersonaGuildResponse:
    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            records = [
                dict(record)
                for record in session.run(
                    GUILD_CANDIDATE_QUERY,
                    uuid=uuid,
                    candidate_limit=36,
                )
            ]
            if not records:
                exists = session.run("MATCH (p:Person {uuid: $uuid}) RETURN p.uuid AS uuid", uuid=uuid).single()
                if not exists:
                    raise NotFoundException("해당 UUID의 페르소나를 찾을 수 없습니다.")
    finally:
        driver.close()

    return PersonaGuildResponse(
        source_uuid=uuid,
        source_community_id=records[0].get("source_community_id") if records else None,  # type: ignore[arg-type]
        scoring_policy="SIMILAR_TO score + same community/location + shared hobby/skill + centrality fallback",
        guilds=_group_guild_candidates(uuid, records),
    )


@router.get("/persona/{uuid}/similar-diverse", response_model=SimilarDiverseResponse)
def similar_diverse_personas(
    uuid: str,
    diversity_axis: str = Query(default="mixed"),
    top_k: int = Query(default=10, ge=1, le=30),
) -> SimilarDiverseResponse:
    if diversity_axis not in VALID_DIVERSITY_AXES:
        allowed = ", ".join(sorted(VALID_DIVERSITY_AXES))
        raise BadRequestException(f"지원하지 않는 diversity_axis입니다: {diversity_axis}. 유효한 값: {allowed}")
    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            records = [
                dict(record)
                for record in session.run(
                    SIMILAR_DIVERSE_QUERY,
                    uuid=uuid,
                    candidate_limit=max(top_k * 8, 40),
                )
            ]
            if not records:
                exists = session.run("MATCH (p:Person {uuid: $uuid}) RETURN p.uuid AS uuid", uuid=uuid).single()
                if not exists:
                    raise NotFoundException("해당 UUID의 페르소나를 찾을 수 없습니다.")
    finally:
        driver.close()

    ranked = sorted(
        (_build_similar_diverse_item(record, diversity_axis) for record in records),
        key=lambda item: item.final_score,
        reverse=True,
    )
    return SimilarDiverseResponse(
        source_uuid=uuid,
        diversity_axis=diversity_axis,
        scoring_policy="final = similarity * 0.58 + diversity * 0.30 + shared hobby/skill support",
        results=ranked[:top_k],
    )


@router.get("/persona/{uuid}/life-track", response_model=LifeTrackResponse)
def persona_life_track(
    uuid: str,
    target_age_min: int | None = Query(default=None, ge=1, le=100),
    target_age_max: int = Query(default=39, ge=1, le=100),
    top_k: int = Query(default=8, ge=1, le=30),
) -> LifeTrackResponse:
    source_age: int | None = None
    resolved_age_min = target_age_min or 30
    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            source_record = session.run(LIFE_TRACK_SOURCE_QUERY, uuid=uuid).single()
            if not source_record:
                raise NotFoundException("해당 UUID의 페르소나를 찾을 수 없습니다.")
            source_age = source_record["age"]
            resolved_age_min = target_age_min or (int(source_age) + 5 if source_age else 30)
            records = [
                dict(record)
                for record in session.run(
                    LIFE_TRACK_QUERY,
                    uuid=uuid,
                    target_age_min=resolved_age_min,
                    target_age_max=target_age_max,
                    candidate_limit=max(top_k * 4, 24),
                )
            ]
    finally:
        driver.close()

    role_models = [_build_role_model(record) for record in records[:top_k]]
    return LifeTrackResponse(
        source_uuid=uuid,
        source_age=source_age,
        cohort_definition={
            "target_age_min": resolved_age_min,
            "target_age_max": target_age_max,
            "candidate_source": "SIMILAR_TO older cohort",
            "fallback_used": False,
        },
        role_models=role_models,
        timeline=_build_life_track_timeline(records),
        transitions={
            "occupations": _ranked_items_from_records(records, "occupation", 8),
            "skills": _ranked_items_from_records(records, "target_skills", 8),
            "hobbies": _ranked_items_from_records(records, "target_hobbies", 8),
        },
        interpretation_policy="미래 예측이 아니라 유사 older cohort에서 관찰된 직업/스킬/취미 패턴 탐색입니다.",
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
