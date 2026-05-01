from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes import influence
from src.gds.centrality import SimulationTimeoutError


class FakeCentralityService:
    def __init__(self, has_scores: bool = True, status: str = "success") -> None:
        self._has_scores = has_scores
        self._status = status
        self.ensure_projection_called = False

    def read_status(self) -> dict[str, object]:
        return {
            "last_success_at": "2026-04-28T02:00:00+00:00",
            "run_id": "run-20260428020000",
            "status": self._status,
        }

    def ensure_projection(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.ensure_projection_called = True
        return {"graphName": "persona_graph", "already_exists": True}

    def has_scores(self, metric: str) -> bool:
        return self._has_scores

    def find_top(self, metric: str, limit: int, community_id: int | None = None) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = [
            {
                "uuid": "u1",
                "display_name": "핵심 인물",
                "score": 0.9,
                "rank": 1,
                "community_id": community_id or 3,
            }
        ]
        return rows[:limit]

    def simulate_removal(
        self,
        target_uuids: list[str],
        max_depth: int,
        timeout_seconds: float = 10.0,
    ) -> dict[str, object]:
        return {
            "path_found": True,
            "original_connectivity": 1.0,
            "current_connectivity": 0.6,
            "fragmentation_increase": 0.4,
            "affected_communities": [3],
        }

    def close(self) -> None:
        return None


class TimeoutCentralityService(FakeCentralityService):
    def simulate_removal(
        self,
        target_uuids: list[str],
        max_depth: int,
        timeout_seconds: float = 10.0,
    ) -> dict[str, object]:
        raise SimulationTimeoutError("timed out")


def test_influence_top(monkeypatch) -> None:
    service = FakeCentralityService()
    monkeypatch.setattr(influence, "get_centrality_service", lambda: service)
    client = TestClient(create_app())

    response = client.get("/api/influence/top?metric=pagerank&limit=10")

    assert response.status_code == 200
    body = response.json()
    assert body["metric"] == "pagerank"
    assert body["last_updated_at"] == "2026-04-28T02:00:00+00:00"
    assert body["run_id"] == "run-20260428020000"
    assert body["stale_warning"] is False
    assert body["results"][0]["uuid"] == "u1"
    assert body["results"][0]["rank"] == 1
    assert service.ensure_projection_called is False


def test_influence_top_reports_stale_warning_for_failed_status(monkeypatch) -> None:
    monkeypatch.setattr(influence, "get_centrality_service", lambda: FakeCentralityService(status="failed"))
    client = TestClient(create_app())

    response = client.get("/api/influence/top?metric=pagerank&limit=10")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "run-20260428020000"
    assert body["stale_warning"] is True


def test_influence_top_does_not_recompute_projection(monkeypatch) -> None:
    service = FakeCentralityService()
    monkeypatch.setattr(influence, "get_centrality_service", lambda: service)
    client = TestClient(create_app())

    response = client.get("/api/influence/top?metric=degree&limit=5")

    assert response.status_code == 200
    assert service.ensure_projection_called is False


def test_influence_top_rejects_invalid_metric(monkeypatch) -> None:
    monkeypatch.setattr(influence, "get_centrality_service", lambda: FakeCentralityService())
    client = TestClient(create_app())

    response = client.get("/api/influence/top?metric=bad")

    assert response.status_code == 400
    assert "pagerank" in response.json()["error"]


def test_influence_top_requires_precomputed_scores(monkeypatch) -> None:
    monkeypatch.setattr(influence, "get_centrality_service", lambda: FakeCentralityService(has_scores=False))
    client = TestClient(create_app())

    response = client.get("/api/influence/top?metric=degree")

    assert response.status_code == 503
    assert "중심성" in response.json()["error"]


def test_simulate_removal(monkeypatch) -> None:
    monkeypatch.setattr(influence, "get_centrality_service", lambda: FakeCentralityService())
    client = TestClient(create_app())

    response = client.post(
        "/api/influence/simulate-removal",
        json={"target_uuids": ["u1"], "max_depth": 3},
    )

    assert response.status_code == 200
    assert response.json()["fragmentation_increase"] == 0.4


def test_simulate_removal_rejects_duplicate_uuid(monkeypatch) -> None:
    monkeypatch.setattr(influence, "get_centrality_service", lambda: FakeCentralityService())
    client = TestClient(create_app())

    response = client.post(
        "/api/influence/simulate-removal",
        json={"target_uuids": ["u1", "u1"], "max_depth": 3},
    )

    assert response.status_code == 400


def test_simulate_removal_returns_422_on_timeout(monkeypatch) -> None:
    monkeypatch.setattr(influence, "get_centrality_service", lambda: TimeoutCentralityService())
    client = TestClient(create_app())

    response = client.post(
        "/api/influence/simulate-removal",
        json={"target_uuids": ["u1"], "max_depth": 3},
    )

    assert response.status_code == 422
    assert "10초" in response.json()["error"]
