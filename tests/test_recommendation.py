from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes import recommend
from src.graph.recommendation import _format_recommendation


class FakeRecommendationService:
    def __init__(self, exists: bool = True, has_similarity: bool = True) -> None:
        self._exists = exists
        self._has_similarity = has_similarity

    def persona_exists(self, uuid: str) -> bool:
        return self._exists

    def has_similarity_data(self, uuid: str) -> bool:
        return self._has_similarity

    def recommend(self, uuid: str, category: str, top_n: int) -> list[dict[str, object]]:
        return [
            {
                "item_name": "클라이밍",
                "reason": "당신과 유사한 10명 중 70%가 '클라이밍'을(를) 취미로 가지고 있습니다.",
                "reason_score": 0.7,
                "similar_users_count": 10,
            }
        ][:top_n]

    def close(self) -> None:
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
