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
    SimilarPreview,
)
from src.config import settings
from src.graph.persona_queries import GRAPH_STATS_QUERY, PROFILE_QUERY, SIMILAR_PREVIEW_QUERY

router = APIRouter(prefix="/api", tags=["persona"])


def get_neo4j_driver():  # noqa: ANN201
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
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
