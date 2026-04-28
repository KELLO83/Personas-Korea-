import math

from fastapi import APIRouter, Query
from neo4j import GraphDatabase

from src.api.exceptions import BadRequestException
from src.api.schemas import SearchResponse, SearchResult
from src.config import settings
from src.graph.search_queries import (
    VALID_SORT_FIELDS,
    VALID_SORT_ORDERS,
    build_search_query,
)

router = APIRouter(prefix="/api", tags=["search"])


def get_neo4j_driver():  # noqa: ANN201
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items if items else None


def _normalize_text_filter(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_sex(value: str | None) -> str | None:
    normalized = _normalize_text_filter(value)
    if normalized is None:
        return None
    return {"여성": "여자", "남성": "남자"}.get(normalized, normalized)
def _normalize_occupation(value: str | None) -> list[str] | None:
    """Normalize occupation input as a generic token list.

    공통 규칙만 적용해 모든 직업명을 동일하게 처리한다.
    """
    normalized = _normalize_text_filter(value)
    if normalized is None:
        return None

    csv_terms = _parse_csv(normalized)
    return csv_terms if csv_terms else [normalized]


def _collect_semantic_persona_uuids(
    query_terms: list[str],
    top_k_per_term: int = 40,
    min_score: float = 0.34,
) -> list[str]:
    cleaned_terms = [term for term in query_terms if _normalize_text_filter(term)]
    if not cleaned_terms:
        return []

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

    semantic_uuids: list[str] = []
    seen: set[str] = set()

    try:
        embeddings = embedder.encode(cleaned_terms)
        for embedding in embeddings:
            for hit in vector_index.search(embedding, top_k=top_k_per_term):
                score = hit.get("score")
                if score is None:
                    continue
                try:
                    if float(score) < min_score:
                        continue
                except (TypeError, ValueError):
                    continue

                uuid_value = hit.get("uuid")
                if uuid_value is None:
                    continue

                uuid = str(uuid_value)
                if uuid in seen:
                    continue
                seen.add(uuid)
                semantic_uuids.append(uuid)
    finally:
        vector_index.close()

    return semantic_uuids


@router.get("/search", response_model=SearchResponse)
def search(
    province: str | None = Query(default=None, description="시도 (comma-separated)"),
    district: str | None = Query(default=None, description="시군구 (comma-separated)"),
    age_min: int | None = Query(default=None, ge=0, le=150),
    age_max: int | None = Query(default=None, ge=0, le=150),
    age_group: str | None = Query(default=None, description="연령대 (comma-separated)"),
    sex: str | None = Query(default=None),
    occupation: str | None = Query(default=None),
    education_level: str | None = Query(default=None, description="학력 (comma-separated)"),
    hobby: str | None = Query(default=None, description="취미 (comma-separated)"),
    skill: str | None = Query(default=None, description="스킬 (comma-separated)"),
    keyword: str | None = Query(default=None),
    sort_by: str = Query(default="age"),
    sort_order: str = Query(default="asc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> SearchResponse:
    if sort_by not in VALID_SORT_FIELDS:
        raise BadRequestException(
            f"유효하지 않은 정렬 필드입니다: {sort_by}. "
            f"유효한 값: {', '.join(sorted(VALID_SORT_FIELDS))}"
        )
    if sort_order not in VALID_SORT_ORDERS:
        raise BadRequestException(
            f"유효하지 않은 정렬 순서입니다: {sort_order}. "
            f"유효한 값: {', '.join(sorted(VALID_SORT_ORDERS))}"
        )

    normalized_keyword = _normalize_text_filter(keyword)
    occupation_terms = _normalize_occupation(occupation)
    query_terms_for_fallback = list(occupation_terms or [])
    if normalized_keyword is not None:
        query_terms_for_fallback.append(normalized_keyword)

    data_query, count_query, params = build_search_query(
        province=_parse_csv(province),
        district=_parse_csv(district),
        age_min=age_min,
        age_max=age_max,
        age_group=_parse_csv(age_group),
        sex=_normalize_sex(sex),
        occupation=occupation_terms,
        education_level=_parse_csv(education_level),
        hobby=_parse_csv(hobby),
        skill=_parse_csv(skill),
        keyword=normalized_keyword,
        sort_by=sort_by,
        sort_order=sort_order,
        page=page,
        page_size=page_size,
    )

    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            count_record = session.run(count_query, **params).single()
            total_count = int(count_record["total_count"]) if count_record else 0  # type: ignore[arg-type]

            if total_count == 0 and query_terms_for_fallback:
                semantic_persona_uuids = _collect_semantic_persona_uuids(query_terms_for_fallback)
                if semantic_persona_uuids:
                    semantic_data_query, semantic_count_query, semantic_params = build_search_query(
                        province=_parse_csv(province),
                        district=_parse_csv(district),
                        age_min=age_min,
                        age_max=age_max,
                        age_group=_parse_csv(age_group),
                        sex=_normalize_sex(sex),
                        occupation=None,
                        education_level=_parse_csv(education_level),
                        hobby=_parse_csv(hobby),
                        skill=_parse_csv(skill),
                        keyword=None,
                        semantic_persona_uuids=semantic_persona_uuids,
                        sort_by=sort_by,
                        sort_order=sort_order,
                        page=page,
                        page_size=page_size,
                    )

                    semantic_count_record = session.run(semantic_count_query, **semantic_params).single()
                    semantic_count = int(semantic_count_record["total_count"]) if semantic_count_record else 0  # type: ignore[arg-type]

                    if semantic_count > 0:
                        count_query = semantic_count_query
                        data_query = semantic_data_query
                        params = semantic_params
                        total_count = semantic_count

            result_records: list[dict[str, object]] = []
            if total_count > 0:
                result_records = [dict(r) for r in session.run(data_query, **params)]
    finally:
        driver.close()

    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 0
    results = [SearchResult(**rec) for rec in result_records]

    return SearchResponse(
        total_count=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        results=results,
    )
