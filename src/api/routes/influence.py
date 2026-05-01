from fastapi import APIRouter, Query

from ...gds.centrality import CentralityService, SimulationTimeoutError, VALID_CENTRALITY_METRICS
from ..exceptions import BadRequestException, ServiceUnavailableException, UnprocessableRequestException
from ..schemas import InfluenceResponse, InfluenceTopItem, RemovalSimulationRequest, RemovalSimulationResponse

router = APIRouter(prefix="/api", tags=["influence"])


def get_centrality_service() -> CentralityService:
    return CentralityService()


@router.get("/influence/top", response_model=InfluenceResponse)
def influence_top(
    metric: str = Query(default="pagerank"),
    limit: int = Query(default=10, ge=1, le=100),
    community_id: int | None = Query(default=None),
) -> InfluenceResponse:
    if metric not in VALID_CENTRALITY_METRICS:
        raise BadRequestException(
            f"유효하지 않은 중심성 지표입니다: {metric}. "
            f"유효한 값: {', '.join(sorted(VALID_CENTRALITY_METRICS))}"
        )

    service = get_centrality_service()
    try:
        status = service.read_status()
        if not service.has_scores(metric):
            raise ServiceUnavailableException("중심성 점수가 아직 준비되지 않았습니다.")
        rows = service.find_top(metric=metric, limit=limit, community_id=community_id)
    finally:
        service.close()

    status_text = status.get("status") if status else None
    last_updated_at = str(status.get("last_success_at")) if status and status.get("last_success_at") else None
    run_id = str(status.get("run_id")) if status and status.get("run_id") else None
    stale_warning = status_text not in {None, "success"}
    return InfluenceResponse(
        metric=metric,
        last_updated_at=last_updated_at,
        run_id=run_id,
        stale_warning=stale_warning,
        results=[InfluenceTopItem(**row) for row in rows],
    )


@router.post("/influence/simulate-removal", response_model=RemovalSimulationResponse)
def simulate_removal(request: RemovalSimulationRequest) -> RemovalSimulationResponse:
    if len(set(request.target_uuids)) != len(request.target_uuids):
        raise BadRequestException("중복된 UUID는 제거 시뮬레이션에 사용할 수 없습니다.")

    service = get_centrality_service()
    try:
        try:
            result = service.simulate_removal(request.target_uuids, max_depth=request.max_depth)
        except SimulationTimeoutError as exc:
            raise UnprocessableRequestException("시뮬레이션 계산이 10초 제한을 초과했습니다.") from exc
    finally:
        service.close()
    return RemovalSimulationResponse(**result)
