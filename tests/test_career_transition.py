from typing import Any

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes import career_transition


class FakeNeo4jRecord:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> object:
        return self._data[key]


class FakeNeo4jResult:
    def __init__(self, records: list[dict[str, Any]] | None = None, single_data: dict[str, Any] | None = None) -> None:
        self._records = records or []
        self._single_data = single_data

    def single(self) -> FakeNeo4jRecord | None:
        if self._single_data is None:
            return None
        return FakeNeo4jRecord(self._single_data)

    def __iter__(self):
        return iter(self._records)


class FakeNeo4jSession:
    def run(self, query: str, **kwargs: object) -> FakeNeo4jResult:
        if "matched_count" in query:
            return FakeNeo4jResult(single_data={"matched_count": 10})
        if "goals_text" in query:
            return FakeNeo4jResult(records=[{"name": "데이터 분석", "count": 4}])
        if "MATCH (p)-[:HAS_SKILL]->(s:Skill)" in query:
            return FakeNeo4jResult(records=[{"name": "Python", "count": 5}])
        if "nocc:Occupation" in query:
            return FakeNeo4jResult(records=[{"name": "데이터 사이언티스트", "count": 3}])
        if "RETURN seg AS name" in query:
            return FakeNeo4jResult(records=[{"name": "30대", "count": 6}])
        return FakeNeo4jResult(records=[])

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeNeo4jDriver:
    def session(self, **kwargs: object) -> FakeNeo4jSession:
        return FakeNeo4jSession()

    def close(self) -> None:
        return None


def _make_driver() -> FakeNeo4jDriver:
    return FakeNeo4jDriver()


def test_career_transition_map_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(career_transition, "get_neo4j_driver", _make_driver)
    client = TestClient(create_app())

    response = client.get("/api/career-transition-map?occupation=개발자&top_k=5")

    assert response.status_code == 200
    body = response.json()
    assert body["matched_count"] == 10
    assert body["top_goals"][0]["name"] == "데이터 분석"
    assert body["top_skills"][0]["name"] == "Python"
    assert body["top_neighbor_occupations"][0]["name"] == "데이터 사이언티스트"
    assert body["segment_distribution"][0]["name"] == "30대"
    assert "career_goals_and_ambitions" in body["mapping_policy"]
    assert body["top_k_limit"] == 30
    assert "개인별 추천은 하지 않습니다" in body["analysis_scope"]


def test_career_transition_map_requires_occupation(monkeypatch) -> None:
    monkeypatch.setattr(career_transition, "get_neo4j_driver", _make_driver)
    client = TestClient(create_app())

    response = client.get("/api/career-transition-map?occupation=")

    assert response.status_code in (400, 422)


def test_career_transition_map_rejects_invalid_compare_by(monkeypatch) -> None:
    monkeypatch.setattr(career_transition, "get_neo4j_driver", _make_driver)
    client = TestClient(create_app())

    response = client.get("/api/career-transition-map?occupation=개발자&compare_by=bad")

    assert response.status_code == 400
