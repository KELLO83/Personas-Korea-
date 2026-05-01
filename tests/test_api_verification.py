from time import perf_counter
from typing import cast

from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.rag.chat_graph import ChatGraph


def test_openapi_includes_phase_15_17_and_18_endpoints() -> None:
    client = TestClient(create_app())

    docs_response = client.get("/docs")
    assert docs_response.status_code == 200
    assert "swagger" in docs_response.text.lower()

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    schema = openapi.json()
    paths = set(schema.get("paths", {}).keys())

    assert "/api/influence/top" in paths
    assert "/api/influence/simulate-removal" in paths
    assert "/api/recommend/{uuid}" in paths
    assert "/api/chat" in paths
    assert "/api/target-persona" in paths
    assert "/api/lifestyle-map" in paths
    assert "/api/career-transition-map" in paths
    assert "/api/graph-quality" in paths


def test_chat_search_and_stats_responses_meet_response_budget(monkeypatch: MonkeyPatch) -> None:
    graph = ChatGraph()

    def fake_run_search(filters):
        return (
            "서울 검색 결과",
            [{"type": "search", "filters": dict(filters), "total_count": 1}],
            [{"uuid": "u1", "display_name": "샘플", "age": 30}],
        )

    def fake_run_stats(filters, requested_dimension=None):
        return (
            "서울 취미 분포",
            [{"type": "stats", "dimension": requested_dimension or "hobby", "filters": dict(filters)}],
            [{"label": "독서", "count": 3}],
        )

    monkeypatch.setattr(graph, "_run_search", fake_run_search)
    monkeypatch.setattr(graph, "_run_stats", fake_run_stats)
    monkeypatch.setattr(graph, "_synthesize_response", lambda **kwargs: "통합 응답")

    start_search = perf_counter()
    search_result = graph.invoke("session-budget", "서울 보여줘")
    assert search_result["response"] == "통합 응답"
    assert search_result["sources"][0]["type"] == "search"
    assert (perf_counter() - start_search) < 5.0

    start_stats = perf_counter()
    stats_result = graph.invoke("session-budget", "서울 취미는?")
    assert stats_result["response"] == "통합 응답"
    assert stats_result["sources"][0]["type"] == "stats"
    assert (perf_counter() - start_stats) < 5.0


def test_chat_general_responses_meet_response_budget(monkeypatch: MonkeyPatch) -> None:
    graph = ChatGraph()

    def fake_run_general(message: str, filters: dict[str, object]) -> tuple[str, list[dict[str, str]], list[dict[str, object]]]:
        return ("일반 응답", [{"type": "general"}], [])

    monkeypatch.setattr(graph, "_run_general", fake_run_general)

    start = perf_counter()
    result = graph.invoke("session-budget-general", "그건 뭐지?")
    assert result["response"] == "일반 응답"
    assert result["sources"][0]["type"] == "general"
    assert (perf_counter() - start) < 10.0


def test_recommend_api_response_meets_budget(monkeypatch: MonkeyPatch) -> None:
    from src.api.routes import recommend

    class FakeService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.seen: list[tuple[str, str]] = []

        def persona_exists(self, uuid: str) -> bool:
            return True

        def has_similarity_data(self, uuid: str) -> bool:
            return True

        def recommend(
            self,
            uuid: str,
            category: str,
            top_n: int,
            *,
            influence_metric: str | None = None,
        ) -> list[dict[str, object]]:
            self.seen.append((uuid, category))
            return cast(
                list[dict[str, object]],
                [
                {
                    "item_name": "클라이밍",
                    "reason": "reason",
                    "reason_score": 0.5,
                    "similar_users_count": 5,
                }
                ][:top_n],
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(recommend, "get_recommendation_service", lambda: FakeService())

    client = TestClient(create_app())
    start = perf_counter()
    response = client.get("/api/recommend/u1?category=hobby&top_n=1")
    elapsed = perf_counter() - start

    assert response.status_code == 200
    assert response.json()["category"] == "hobby"
    assert elapsed < 0.5
