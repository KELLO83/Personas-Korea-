from fastapi.testclient import TestClient
import pytest
from typing import cast

from src.api.main import create_app
from src.api.routes import recommend
from src.graph.recommendation import RecommendationService, _format_recommendation


class FakeRecommendationService:
    def __init__(self, exists: bool = True, has_similarity: bool = True) -> None:
        self._exists = exists
        self._has_similarity = has_similarity
        self.ensure_projection_called = False

    def persona_exists(self, uuid: str) -> bool:
        return self._exists

    def has_similarity_data(self, uuid: str) -> bool:
        return self._has_similarity

    def recommend(
        self,
        uuid: str,
        category: str,
        top_n: int,
        *,
        influence_metric: str | None = None,
    ) -> list[dict[str, object]]:
        return [
            cast(dict[str, object], {
                "item_name": "클라이밍",
                "reason": "당신과 유사한 10명 중 70%가 '클라이밍'을(를) 취미로 가지고 있습니다.",
                "reason_score": 0.7,
                "similar_users_count": 10,
            })
        ][:top_n]

    def close(self) -> None:
        return None

    def ensure_projection(self, *args: object, **kwargs: object) -> None:
        self.ensure_projection_called = True
        return None


def test_format_recommendation_uses_template() -> None:
    item = _format_recommendation(
        {
            "item_name": "Python",
            "similar_users_count": 8,
            "reason_score": 0.5,
            "supporting_personas": [{"uuid": "u1", "display_name": "홍길동", "similarity": 0.91}],
        },
        "skill",
    )

    assert item["reason_score"] == 0.5
    assert item["similar_users_count"] == 8
    assert "Python" in item["reason"]
    assert "스킬" in item["reason"]
    assert item["supporting_personas"][0]["uuid"] == "u1"


def test_recommend_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(recommend, "get_recommendation_service", lambda: FakeRecommendationService())
    client = TestClient(create_app())

    response = client.get("/api/recommend/u1?category=hobby&top_n=1")

    assert response.status_code == 200
    body = response.json()
    assert body["uuid"] == "u1"
    assert body["category"] == "hobby"
    assert body["recommendations"][0]["item_name"] == "클라이밍"


def test_recommend_endpoint_rejects_invalid_category(monkeypatch) -> None:
    monkeypatch.setattr(recommend, "get_recommendation_service", lambda: FakeRecommendationService())
    client = TestClient(create_app())

    response = client.get("/api/recommend/u1?category=bad")

    assert response.status_code == 400
    assert "hobby" in response.json()["error"]


def test_recommend_endpoint_returns_404_for_missing_persona(monkeypatch) -> None:
    monkeypatch.setattr(recommend, "get_recommendation_service", lambda: FakeRecommendationService(exists=False))
    client = TestClient(create_app())

    response = client.get("/api/recommend/missing?category=hobby")

    assert response.status_code == 404


def test_recommend_endpoint_requires_similarity_data(monkeypatch) -> None:
    monkeypatch.setattr(
        recommend,
        "get_recommendation_service",
        lambda: FakeRecommendationService(has_similarity=False),
    )
    client = TestClient(create_app())

    response = client.get("/api/recommend/u1?category=hobby")

    assert response.status_code == 503
    assert "유사도" in response.json()["error"]


def test_recommend_endpoint_does_not_recompute_similarity_projection(monkeypatch) -> None:
    service = FakeRecommendationService()
    monkeypatch.setattr(recommend, "get_recommendation_service", lambda: service)
    client = TestClient(create_app())

    response = client.get("/api/recommend/u1?category=hobby&top_n=1")

    assert response.status_code == 200
    assert service.ensure_projection_called is False


def test_recommend_endpoint_accepts_influence_metric(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    class FakeRecommendationServiceWithMetric:
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
            captured["influence_metric"] = influence_metric
            return [
                cast(dict[str, object], {
                    "item_name": "클라이밍",
                    "reason": "당신과 유사한 10명 중 70%가 '클라이밍'을(를) 취미로 가지고 있습니다.",
                    "reason_score": 0.7,
                    "similar_users_count": 10,
                })
            ][:top_n]

        def close(self) -> None:
            return None

    monkeypatch.setattr(recommend, "get_recommendation_service", lambda: FakeRecommendationServiceWithMetric())
    client = TestClient(create_app())

    response = client.get("/api/recommend/u1?category=hobby&top_n=1&influence_metric=pagerank")

    assert response.status_code == 200
    assert captured["influence_metric"] == "pagerank"


def test_recommend_endpoint_rejects_invalid_influence_metric(monkeypatch) -> None:
    monkeypatch.setattr(recommend, "get_recommendation_service", lambda: FakeRecommendationService())
    client = TestClient(create_app())

    response = client.get("/api/recommend/u1?category=hobby&influence_metric=invalid")

    assert response.status_code == 400
    assert "영향력" in response.json()["error"]


def test_recommend_service_passes_influence_metric_to_query(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    class FakeResult:
        def __iter__(self):
            return iter(())

    class FakeSession:
        def run(self, query, **params):
            captured["score_property"] = params.get("score_property")
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

    class FakeDriver:
        def session(self, database=None):
            return FakeSession()

        def close(self):
            return None

    monkeypatch.setattr(
        "src.graph.recommendation.GraphDatabase.driver",
        lambda *args, **kwargs: FakeDriver(),
    )
    service = RecommendationService(uri="bolt://localhost:7687", user="neo4j", password="test")
    service.recommend("uuid-1", category="hobby", top_n=3, influence_metric="degree")
    service.close()

    assert captured["score_property"] == "degree"


def test_recommend_service_rejects_invalid_influence_metric(monkeypatch) -> None:
    class FakeDriver:
        def close(self):
            return None

    monkeypatch.setattr(
        "src.graph.recommendation.GraphDatabase.driver",
        lambda *args, **kwargs: FakeDriver(),
    )

    service = RecommendationService(uri="bolt://localhost:7687", user="neo4j", password="test")
    with pytest.raises(ValueError, match="Invalid influence metric"):
        service.recommend("uuid-1", category="hobby", top_n=3, influence_metric="invalid")
    service.close()
