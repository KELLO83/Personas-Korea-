from fastapi import APIRouter
from neo4j import GraphDatabase

from src.api.exceptions import BadRequestException
from src.api.schemas import (
    CompareDistributionItem,
    DimensionComparison,
    SegmentCompareRequest,
    SegmentCompareResponse,
    SegmentFilter,
    SegmentSummary,
)
from src.config import settings
from src.graph.stats_queries import VALID_DIMENSIONS, _DIMENSION_CONFIG
from src.rag.compare_chain import CompareChain

router = APIRouter(prefix="/api", tags=["compare"])


def get_neo4j_driver():  # noqa: ANN201
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def _build_segment_query(dimension: str, filters: SegmentFilter, top_k: int) -> tuple[str, dict[str, object]]:
    config = _DIMENSION_CONFIG[dimension]
    params: dict[str, object] = {}
    parts = ["MATCH (p:Person)"]
    
    if config["match"] is not None:
        parts.append(f"MATCH {config['match']}")
    
    if filters.province or filters.district:
        parts.append("MATCH (p)-[:LIVES_IN]->(filt_d:District)-[:IN_PROVINCE]->(filt_prov:Province)")
    if filters.hobby:
        parts.append("MATCH (p)-[:ENJOYS_HOBBY]->(filt_h:Hobby)")
    if filters.skill:
        parts.append("MATCH (p)-[:HAS_SKILL]->(filt_s:Skill)")
    if filters.education_level:
        parts.append("MATCH (p)-[:EDUCATED_AT]->(filt_edu:EducationLevel)")
        
    where_clauses = []
    if filters.province:
        where_clauses.append("filt_prov.name = $province")
        params["province"] = filters.province
    if filters.district:
        where_clauses.append("filt_d.name = $district")
        params["district"] = filters.district
    if filters.age_group:
        where_clauses.append("p.age_group = $age_group")
        params["age_group"] = filters.age_group
    if filters.sex:
        where_clauses.append("p.sex = $sex")
        params["sex"] = filters.sex
    if filters.hobby:
        where_clauses.append("filt_h.name = $hobby")
        params["hobby"] = filters.hobby
    if filters.skill:
        where_clauses.append("filt_s.name = $skill")
        params["skill"] = filters.skill
    if filters.education_level:
        where_clauses.append("filt_edu.name = $education_level")
        params["education_level"] = filters.education_level
        
    return_field = config["return_field"]
    if dimension in ("age", "sex") and return_field is not None:
        where_clauses.append(f"{return_field} IS NOT NULL")
        
    if where_clauses:
        parts.append("WHERE " + " AND ".join(where_clauses))
        
    parts.append(f"RETURN {return_field} AS label, count(p) AS count")
    parts.append("ORDER BY count DESC LIMIT $limit")
    params["limit"] = top_k
    return "\n".join(parts), params


def _build_count_query(filters: SegmentFilter) -> tuple[str, dict[str, object]]:
    params: dict[str, object] = {}
    parts = ["MATCH (p:Person)"]
    
    if filters.province or filters.district:
        parts.append("MATCH (p)-[:LIVES_IN]->(filt_d:District)-[:IN_PROVINCE]->(filt_prov:Province)")
    if filters.hobby:
        parts.append("MATCH (p)-[:ENJOYS_HOBBY]->(filt_h:Hobby)")
    if filters.skill:
        parts.append("MATCH (p)-[:HAS_SKILL]->(filt_s:Skill)")
    if filters.education_level:
        parts.append("MATCH (p)-[:EDUCATED_AT]->(filt_edu:EducationLevel)")
        
    where_clauses = []
    if filters.province:
        where_clauses.append("filt_prov.name = $province")
        params["province"] = filters.province
    if filters.district:
        where_clauses.append("filt_d.name = $district")
        params["district"] = filters.district
    if filters.age_group:
        where_clauses.append("p.age_group = $age_group")
        params["age_group"] = filters.age_group
    if filters.sex:
        where_clauses.append("p.sex = $sex")
        params["sex"] = filters.sex
    if filters.hobby:
        where_clauses.append("filt_h.name = $hobby")
        params["hobby"] = filters.hobby
    if filters.skill:
        where_clauses.append("filt_s.name = $skill")
        params["skill"] = filters.skill
    if filters.education_level:
        where_clauses.append("filt_edu.name = $education_level")
        params["education_level"] = filters.education_level
        
    if where_clauses:
        parts.append("WHERE " + " AND ".join(where_clauses))
        
    parts.append("RETURN count(DISTINCT p) AS total")
    return "\n".join(parts), params


@router.post("/compare/segments", response_model=SegmentCompareResponse)
def compare_segments(request: SegmentCompareRequest) -> SegmentCompareResponse:
    for dim in request.dimensions:
        if dim not in VALID_DIMENSIONS:
            raise BadRequestException(f"Invalid dimension: {dim}. Valid options are {', '.join(sorted(VALID_DIMENSIONS))}")

    driver = get_neo4j_driver()
    
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            count_q_a, params_a = _build_count_query(request.segment_a.filters)
            count_res_a = session.run(count_q_a, **params_a).single()
            count_a = int(count_res_a["total"]) if count_res_a else 0
            
            count_q_b, params_b = _build_count_query(request.segment_b.filters)
            count_res_b = session.run(count_q_b, **params_b).single()
            count_b = int(count_res_b["total"]) if count_res_b else 0

            comparisons: dict[str, DimensionComparison] = {}
            for dim in request.dimensions:
                dim_q_a, dim_params_a = _build_segment_query(dim, request.segment_a.filters, request.top_k)
                dim_records_a = [dict(r) for r in session.run(dim_q_a, **dim_params_a)]
                
                dim_q_b, dim_params_b = _build_segment_query(dim, request.segment_b.filters, request.top_k)
                dim_records_b = [dict(r) for r in session.run(dim_q_b, **dim_params_b)]
                
                filtered_count_a = sum(int(r["count"]) for r in dim_records_a)
                filtered_count_b = sum(int(r["count"]) for r in dim_records_b)

                dist_a: list[CompareDistributionItem] = []
                names_a = set()
                for rec in dim_records_a:
                    c = int(rec["count"])
                    r = c / filtered_count_a if filtered_count_a > 0 else 0.0
                    name = str(rec["label"])
                    names_a.add(name)
                    dist_a.append(CompareDistributionItem(name=name, count=c, ratio=r))

                dist_b: list[CompareDistributionItem] = []
                names_b = set()
                for rec in dim_records_b:
                    c = int(rec["count"])
                    r = c / filtered_count_b if filtered_count_b > 0 else 0.0
                    name = str(rec["label"])
                    names_b.add(name)
                    dist_b.append(CompareDistributionItem(name=name, count=c, ratio=r))

                comparisons[dim] = DimensionComparison(
                    segment_a=dist_a,
                    segment_b=dist_b,
                    common=list(names_a & names_b),
                    only_a=list(names_a - names_b),
                    only_b=list(names_b - names_a)
                )
    finally:
        driver.close()

    ai_analysis = ""
    if request.dimensions and count_a > 0 and count_b > 0:
        first_dim = request.dimensions[0]
        comp = comparisons[first_dim]
        
        dist_a_dict = [{"name": item.name, "ratio": item.ratio} for item in comp.segment_a]
        dist_b_dict = [{"name": item.name, "ratio": item.ratio} for item in comp.segment_b]
        
        chain = CompareChain()
        ai_analysis = chain.analyze(
            dimension=first_dim,
            label_a=request.segment_a.label,
            count_a=count_a,
            dist_a=dist_a_dict,
            label_b=request.segment_b.label,
            count_b=count_b,
            dist_b=dist_b_dict
        )

    return SegmentCompareResponse(
        segment_a=SegmentSummary(label=request.segment_a.label, count=count_a),
        segment_b=SegmentSummary(label=request.segment_b.label, count=count_b),
        comparisons=comparisons,
        ai_analysis=ai_analysis
    )
