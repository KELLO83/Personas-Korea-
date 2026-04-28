from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes import communities, insight, path, similar


class FakeInsightRouter:
    def ask(self, question: str) -> dict[str, object]:
        return {"answer": f"answer: {question}", "sources": [], "query_type": "cypher"}


class FakeSimilarityService:
    def get_query_persona(self, uuid: str) -> dict[str, object] | None:
        return {"uuid": uuid, "display_name": "기준", "age": 71, "district_name": "서초구", "occupation_name": "회계 사무원"}

    def find_similar_personas(self, uuid: str, top_k: int) -> list[dict[str, object]]:
        return [{"uuid": "target", "display_name": "대상", "persona": "요약", "age": 30, "sex": "여자", "similarity": 0.9}]

    def find_text_similar_personas(self, uuid: str, top_k: int) -> list[dict[str, object]]:
        return [{"uuid": "target", "display_name": "대상", "persona": "요약", "age": 30, "sex": "여자", "similarity": 0.85}]

    def find_shared_traits(self, source_uuid: str, target_uuid: str) -> list[dict[str, object]]:
        return [{"type": "Hobby", "name": "고궁 산책"}]

    def find_shared_hobbies(self, source_uuid: str, target_uuid: str) -> list[str]:
        return ["고궁 산책"]

    def close(self) -> None:
        return None


class FakeSimilarityServiceNotFound(FakeSimilarityService):
    def get_query_persona(self, uuid: str) -> dict[str, object] | None:
        return None


class FakeSimilarityServiceNoSimilarity(FakeSimilarityService):
    def find_similar_personas(self, uuid: str, top_k: int) -> list[dict[str, object]]:
        return []

    def find_text_similar_personas(self, uuid: str, top_k: int) -> list[dict[str, object]]:
        return []


class FakeCommunityService:
    def summarize_communities(self, min_size: int) -> list[dict[str, object]]:
        return [{"community_id": 1, "size": min_size, "label": "서울 고궁 산책", "top_traits": {"province": "서울", "hobbies": ["고궁 산책"]}}]

    def close(self) -> None:
        return None


def test_insight_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(insight, "get_insight_router", lambda: FakeInsightRouter())
    client = TestClient(create_app())

    response = client.post("/api/insight", json={"question": "공통 취미는?"})

    assert response.status_code == 200
    assert response.json()["query_type"] == "cypher"


def test_similar_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(similar, "get_similarity_service", lambda: FakeSimilarityService())
    client = TestClient(create_app())

    response = client.post("/api/similar/source", json={"top_k": 1})

    assert response.status_code == 200
    assert response.json()["query_persona"]["name_summary"] == "기준, 71세, 서초구, 회계 사무원"
    assert response.json()["similar_personas"][0]["shared_traits"][0]["name"] == "고궁 산책"


def test_similar_endpoint_returns_404_for_missing_persona(monkeypatch) -> None:
    monkeypatch.setattr(similar, "get_similarity_service", lambda: FakeSimilarityServiceNotFound())
    client = TestClient(create_app())

    response = client.post("/api/similar/missing", json={"top_k": 1})

    assert response.status_code == 404
    assert response.json()["error"] == "Persona not found"


def test_similar_endpoint_returns_503_when_similarity_data_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(similar, "get_similarity_service", lambda: FakeSimilarityServiceNoSimilarity())
    client = TestClient(create_app())

    response = client.post("/api/similar/source", json={"top_k": 1})

    assert response.status_code == 503
    assert response.json()["error"] == "Similarity data is not available"


def test_communities_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(communities, "get_community_service", lambda: FakeCommunityService())
    client = TestClient(create_app())

    response = client.get("/api/communities?algorithm=leiden&min_size=3")

    assert response.status_code == 200
    assert response.json()["communities"][0]["size"] == 3
    assert response.json()["communities"][0]["top_traits"]["province"] == "서울"


def test_communities_endpoint_rejects_unsupported_algorithm(monkeypatch) -> None:
    monkeypatch.setattr(communities, "get_community_service", lambda: FakeCommunityService())
    client = TestClient(create_app())

    response = client.get("/api/communities?algorithm=louvain&min_size=3")

    assert response.status_code == 400
    assert "leiden" in response.json()["error"]


def test_path_format_helpers() -> None:
    items = path._format_path([{"name": "A", "labels": ["Person"]}, {"name": "B", "labels": ["Person"]}], ["ENJOYS_HOBBY"])
    summary = path._summarize_shared_nodes([{"type": "Hobby", "name": "고궁 산책"}])

    assert items[0].get("edge") == "ENJOYS_HOBBY"
    assert "고궁 산책" in summary
