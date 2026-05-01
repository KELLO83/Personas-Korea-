from typing import Any

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes import lifestyle_map


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
        if "source_count" in query:
            return FakeNeo4jResult(single_data={"source_count": 20})
        if "RETURN p.culinary_persona AS target_text" in query:
            return FakeNeo4jResult(records=[
                {"target_text": "요리와 맛집 탐방을 즐기고 한식 레시피를 자주 시도함"},
                {"target_text": "주말마다 베이킹과 홈카페 활동을 즐김"},
            ])
        if "target_keyword" in query:
            return FakeNeo4jResult(records=[
                {"target_keyword": "요리", "overlap_count": 12, "support_count": 20},
                {"target_keyword": "맛집", "overlap_count": 8, "support_count": 20},
            ])
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


def test_lifestyle_map_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(lifestyle_map, "get_neo4j_driver", _make_driver)
    client = TestClient(create_app())

    response = client.get(
        "/api/lifestyle-map?source_field=sports_persona&target_field=culinary_persona"
        "&source_keyword=러닝&candidate_keywords=요리,맛집&province=서울"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["matched_source_count"] == 20
    assert body["edges"][0]["target_keyword"] == "요리"
    assert body["edges"][0]["conditional_ratio"] == 0.6
    assert "sports_persona" in body["available_fields"]
    assert "min_keyword_count" in body["keyword_policy"]
    assert "province" in body["segment_policy"]
    assert "conditional_ratio" in body["visualization_policy"]


def test_lifestyle_map_rejects_invalid_field(monkeypatch) -> None:
    monkeypatch.setattr(lifestyle_map, "get_neo4j_driver", _make_driver)
    client = TestClient(create_app())

    response = client.get(
        "/api/lifestyle-map?source_field=bad_field&target_field=culinary_persona"
        "&source_keyword=러닝&candidate_keywords=요리"
    )

    assert response.status_code == 400


def test_lifestyle_map_auto_keywords(monkeypatch) -> None:
    monkeypatch.setattr(lifestyle_map, "get_neo4j_driver", _make_driver)
    client = TestClient(create_app())

    response = client.get(
        "/api/lifestyle-map?source_field=sports_persona&target_field=culinary_persona&source_keyword=러닝&min_keyword_count=1"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["matched_source_count"] == 20
    assert len(body["edges"]) >= 1
