from fastapi import APIRouter, Query
from neo4j import GraphDatabase

from src.api.exceptions import BadRequestException
from src.api.schemas import (
    DimensionStatsResponse,
    DistributionItem,
    RankedItem,
    StatsResponse,
)
from src.config import settings
from src.graph.stats_queries import (
    AGE_DISTRIBUTION_QUERY,
    EDUCATION_DISTRIBUTION_QUERY,
    MARITAL_DISTRIBUTION_QUERY,
    PROVINCE_DISTRIBUTION_QUERY,
    SEX_DISTRIBUTION_QUERY,
    TOP_HOBBIES_QUERY,
    TOP_OCCUPATIONS_QUERY,
    TOP_SKILLS_QUERY,
    TOTAL_COUNT_QUERY,
    VALID_DIMENSIONS,
    build_dimension_query,
)

router = APIRouter(prefix="/api", tags=["stats"])


def get_neo4j_driver():  # noqa: ANN201
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def _to_distribution(records: list[dict[str, object]], key: str, total: int) -> list[DistributionItem]:
    items: list[DistributionItem] = []
    for rec in records:
        count = int(rec["count"])  # type: ignore[arg-type]
        ratio = count / total if total > 0 else 0.0
        items.append(DistributionItem(label=str(rec[key]), count=count, ratio=round(ratio, 4)))
    return items


def _to_ranked(records: list[dict[str, object]], key: str) -> list[RankedItem]:
    return [RankedItem(label=str(rec[key]), count=int(rec["count"])) for rec in records]  # type: ignore[arg-type]


@router.get("/stats", response_model=StatsResponse)
def stats_overview() -> StatsResponse:
    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            total_record = session.run(TOTAL_COUNT_QUERY).single()
            total = int(total_record["total"]) if total_record else 0  # type: ignore[arg-type]

            age_records = [dict(r) for r in session.run(AGE_DISTRIBUTION_QUERY)]
            sex_records = [dict(r) for r in session.run(SEX_DISTRIBUTION_QUERY)]
            province_records = [dict(r) for r in session.run(PROVINCE_DISTRIBUTION_QUERY)]
            education_records = [dict(r) for r in session.run(EDUCATION_DISTRIBUTION_QUERY)]
            marital_records = [dict(r) for r in session.run(MARITAL_DISTRIBUTION_QUERY)]
            occupation_records = [dict(r) for r in session.run(TOP_OCCUPATIONS_QUERY, limit=20)]
            hobby_records = [dict(r) for r in session.run(TOP_HOBBIES_QUERY, limit=20)]
            skill_records = [dict(r) for r in session.run(TOP_SKILLS_QUERY, limit=20)]
    finally:
        driver.close()

    return StatsResponse(
        total_personas=total,
        age_distribution=_to_distribution(age_records, "age_group", total),
        sex_distribution=_to_distribution(sex_records, "sex", total),
        province_distribution=_to_distribution(province_records, "province", total),
        top_occupations=_to_ranked(occupation_records, "occupation"),
        top_hobbies=_to_ranked(hobby_records, "hobby"),
        top_skills=_to_ranked(skill_records, "skill"),
        education_distribution=_to_distribution(education_records, "education_level", total),
        marital_distribution=_to_distribution(marital_records, "marital_status", total),
    )


@router.get("/stats/{dimension}", response_model=DimensionStatsResponse)
def stats_dimension(
    dimension: str,
    province: str | None = None,
    age_group: str | None = None,
    sex: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
) -> DimensionStatsResponse:
    if dimension not in VALID_DIMENSIONS:
        raise BadRequestException(
            f"유효하지 않은 차원입니다: {dimension}. "
            f"유효한 값: {', '.join(sorted(VALID_DIMENSIONS))}"
        )

    query, params = build_dimension_query(dimension, province=province, age_group=age_group, sex=sex)
    params["limit"] = limit

    filters_applied: dict[str, str] = {}
    if province is not None:
        filters_applied["province"] = province
    if age_group is not None:
        filters_applied["age_group"] = age_group
    if sex is not None:
        filters_applied["sex"] = sex

    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            records = [dict(r) for r in session.run(query, **params)]
    finally:
        driver.close()

    filtered_count = sum(int(r["count"]) for r in records)  # type: ignore[arg-type]
    distribution = _to_distribution(records, "label", filtered_count)

    return DimensionStatsResponse(
        dimension=dimension,
        filters_applied=filters_applied,
        filtered_count=filtered_count,
        distribution=distribution,
    )
