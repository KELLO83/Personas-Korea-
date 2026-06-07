from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from fastapi import APIRouter, Query

from ...graph.recommendation import (
    RecommendationService,
    VALID_CENTRALITY_METRICS,
    VALID_RECOMMENDATION_CATEGORIES,
)
from ..exceptions import BadRequestException, NotFoundException, ServiceUnavailableException
from ..schemas import (
    RankedItem,
    RecommendItem,
    RecommendationModelInfo,
    RecommendationQualityMetric,
    RecommendationQualityResponse,
    RecommendationQualityTarget,
    RecommendationStatusResponse,
    RecommendResponse,
)

router = APIRouter(prefix="/api", tags=["recommendations"])

HOBBY_MODEL_INFO = RecommendationModelInfo(
    target="hobby",
    status="under_development",
    score_source="fallback",
    model_version=None,
    graph_snapshot_id=None,
    fallback_used=True,
    fallback_reason="hobby_recommender_model_and_weights_not_promoted",
    message="취미 추천 모델은 아직 모델 선택과 가중치가 확정되지 않은 개발 중 상태입니다. 현재 API는 Neo4j graph/rule 기반 fallback 추천을 제공합니다.",
)
PERSONA_SIMILARITY_MODEL_INFO = RecommendationModelInfo(
    target="persona_similarity",
    status="under_development",
    score_source="fallback",
    model_version=None,
    graph_snapshot_id=None,
    fallback_used=True,
    fallback_reason="persona_similarity_model_and_weights_not_promoted",
    message="유사 페르소나 추천 모델은 아직 모델 선택과 가중치가 확정되지 않은 개발 중 상태입니다. 현재 API는 SIMILAR_TO와 post-hoc reason 기반 fallback을 사용합니다.",
)


def _model_info_for_category(category: str) -> RecommendationModelInfo:
    if category == "hobby":
        return HOBBY_MODEL_INFO
    return RecommendationModelInfo(
        target=category,
        status="fallback_only",
        score_source="fallback",
        model_version=None,
        graph_snapshot_id=None,
        fallback_used=True,
        fallback_reason=f"{category}_recommendation_model_not_defined",
        message=f"{category} 추천은 아직 별도 모델 승격 대상이 아닙니다. 현재 API는 Neo4j graph/rule 기반 fallback 추천을 제공합니다.",
    )


def get_recommendation_service() -> RecommendationService:
    return RecommendationService()


@router.get("/recommend/{uuid}", response_model=RecommendResponse)
def recommend(
    uuid: str,
    category: str = Query(
        default="hobby",
        description="추천 카테고리: hobby, skill, occupation, district",
    ),
    top_n: int = Query(
        default=5,
        ge=1,
        le=20,
        description="반환할 추천 항목 개수. 1~20 범위",
    ),
    influence_metric: str | None = Query(
        default=None,
        description="중심성 가중치: pagerank, betweenness, degree 중 선택",
    ),
) -> RecommendResponse:
    if category not in VALID_RECOMMENDATION_CATEGORIES:
        raise BadRequestException(
            f"유효하지 않은 추천 카테고리입니다: {category}. "
            f"유효한 값: {', '.join(sorted(VALID_RECOMMENDATION_CATEGORIES))}"
        )
    if influence_metric is not None and influence_metric not in VALID_CENTRALITY_METRICS:
        raise BadRequestException(
            f"유효하지 않은 영향력 지표입니다: {influence_metric}. "
            f"유효한 값: {', '.join(sorted(VALID_CENTRALITY_METRICS))}"
        )

    service = get_recommendation_service()
    try:
        if not service.persona_exists(uuid):
            raise NotFoundException("해당 UUID의 페르소나를 찾을 수 없습니다.")
        if not service.has_similarity_data(uuid):
            raise ServiceUnavailableException("유사도 매칭 데이터가 없습니다. 관리자에게 KNN 파이프라인 실행을 요청하세요.")
        recommendations = service.recommend(
            uuid=uuid,
            category=category,
            top_n=top_n,
            influence_metric=influence_metric,
        )
    finally:
        service.close()

    return RecommendResponse(
        uuid=uuid,
        category=category,
        recommendations=[RecommendItem(**_with_fallback_contract(item, rank=index)) for index, item in enumerate(recommendations, start=1)],
        model_status=_model_info_for_category(category),
    )


@router.get("/recommendation/status", response_model=RecommendationStatusResponse)
def recommendation_status() -> RecommendationStatusResponse:
    return RecommendationStatusResponse(
        hobby_recommender=HOBBY_MODEL_INFO,
        persona_similarity_recommender=PERSONA_SIMILARITY_MODEL_INFO,
        product_policy=(
            "Recommendation model experiments remain in their experiment folders. "
            "Root APIs expose fallback behavior and adapter metadata until a model is promoted."
        ),
    )


@router.get("/recommendation/quality", response_model=RecommendationQualityResponse)
def recommendation_quality(
    target: str | None = Query(default=None, description="hobby 또는 persona_similarity. 비우면 둘 다 반환합니다."),
) -> RecommendationQualityResponse:
    targets = ["hobby", "persona_similarity"] if target is None else [target]
    invalid_targets = sorted(set(targets) - {"hobby", "persona_similarity"})
    if invalid_targets:
        raise BadRequestException(f"지원하지 않는 추천 품질 target입니다: {', '.join(invalid_targets)}")

    service = get_recommendation_service()
    try:
        snapshots = [_build_quality_target(item, service) for item in targets]
    finally:
        service.close()

    return RecommendationQualityResponse(
        targets=snapshots,
        dashboard_policy=(
            "운영 상태 탭의 개발자용 품질 대시보드입니다. 현재 값은 승격 모델이 아니라 Neo4j graph/rule fallback 후보에서 계산됩니다."
        ),
    )


def _with_fallback_contract(item: dict[str, Any], rank: int) -> dict[str, Any]:
    normalized = dict(item)
    normalized.setdefault("rank", rank)
    normalized.setdefault("score", normalized.get("reason_score", 0.0))
    normalized.setdefault("already_known", False)
    normalized.setdefault("sources", ["similar_person", "graph_frequency"])
    normalized.setdefault("score_source", "fallback")
    normalized.setdefault("model_version", None)
    normalized.setdefault("graph_snapshot_id", None)
    normalized.setdefault("fallback_used", True)
    normalized.setdefault("fallback_reason", "recommendation_model_under_development")
    if not normalized.get("reason_cards"):
        normalized["reason_cards"] = [
            {
                "type": "similar_person",
                "title": "유사 페르소나 기반",
                "detail": str(normalized.get("reason") or ""),
                "strength": float(normalized.get("reason_score") or 0.0),
            }
        ]
    return normalized


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _metric(name: str, value: float, description: str, *, unit: str = "ratio", warn_below: float | None = None) -> RecommendationQualityMetric:
    status = "warning" if warn_below is not None and value < warn_below else "ok"
    return RecommendationQualityMetric(
        name=name,
        value=round(value, 6),
        unit=unit,
        status=status,
        description=description,
    )


def _build_quality_target(target: str, service: RecommendationService) -> RecommendationQualityTarget:
    started_at = perf_counter()
    row = service.quality_snapshot(target)
    latency_ms = (perf_counter() - started_at) * 1000
    sample_size = int(row.get("sample_size") or 0)
    catalog_size = int(row.get("catalog_size") or 0)
    recommendation_count = int(row.get("recommendation_count") or 0)
    unique_target_count = int(row.get("unique_target_count") or 0)
    max_frequency = int(row.get("max_frequency") or 0)
    weak_count = int(row.get("weak_count") or 0)
    top_targets = _top_quality_targets(row.get("targets"))
    metrics = [
        _metric("coverage", _ratio(unique_target_count, catalog_size), "추천 후보가 전체 카탈로그를 얼마나 넓게 쓰는지", warn_below=0.05),
        _metric("diversity", _ratio(unique_target_count, recommendation_count), "상위 추천 슬롯 중 서로 다른 target 비율", warn_below=0.2),
        _metric("hub_target_rate", _ratio(max_frequency, recommendation_count), "가장 많이 반복된 target의 슬롯 점유율"),
        _metric("weak_only_rate", _ratio(weak_count, recommendation_count), "약한 근거만 가진 추천 슬롯 비율"),
        _metric("explanation_coverage", 1.0 if recommendation_count else 0.0, "fallback reason card를 붙일 수 있는 추천 비율", warn_below=0.95),
        _metric("query_latency_ms", latency_ms, "품질 스냅샷 집계 쿼리 시간", unit="ms"),
    ]
    if target == "persona_similarity":
        metrics.extend(
            [
                _metric("occupation_diversity", _ratio(float(row.get("occupation_diversity") or 0), recommendation_count), "추천 persona 직업 다양성"),
                _metric("province_diversity", _ratio(float(row.get("province_diversity") or 0), recommendation_count), "추천 persona 지역 다양성"),
            ]
        )
    warnings = []
    if sample_size == 0:
        warnings.append("SIMILAR_TO 기반 추천 후보가 없어 품질 지표를 계산하지 못했습니다.")
    if catalog_size == 0:
        warnings.append("카탈로그 크기가 0입니다. 그래프 적재 상태를 확인하세요.")
    return RecommendationQualityTarget(
        target=target,
        score_source="fallback",
        sample_size=sample_size,
        catalog_size=catalog_size,
        metrics=metrics,
        top_targets=top_targets,
        warnings=warnings,
        generated_at=datetime.now(UTC).isoformat(),
    )


def _top_quality_targets(value: object) -> list[RankedItem]:
    if not isinstance(value, list):
        return []
    rows: list[RankedItem] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if not label:
            continue
        rows.append(RankedItem(label=str(label), count=int(item.get("count") or 0)))
    return sorted(rows, key=lambda item: (-item.count, item.label))[:8]
