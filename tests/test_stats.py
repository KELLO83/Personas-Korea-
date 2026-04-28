from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes import stats


def _create_test_app() -> FastAPI:
    app = create_app()
    app.include_router(stats.router)
    return app


class FakeNeo4jRecord:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def get(self, key: str, default: object = None) -> object:
        return self._data.get(key, default)


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
        if "count(p) AS total" in query:
            return FakeNeo4jResult(single_data={"total": 1000})

        if "AS label" in query:
            return FakeNeo4jResult(records=[
                {"label": "등산", "count": 80},
                {"label": "독서", "count": 60},
            ])

        if "age_group" in query and "AS age_group" in query:
            return FakeNeo4jResult(records=[
                {"age_group": "30대", "count": 300},
                {"age_group": "20대", "count": 200},
            ])

        if "p.sex" in query and "AS sex" in query:
            return FakeNeo4jResult(records=[
                {"sex": "남자", "count": 520},
                {"sex": "여자", "count": 480},
            ])

        if "Province" in query and "AS province" in query:
            return FakeNeo4jResult(records=[
                {"province": "서울", "count": 400},
            ])

        if "EducationLevel" in query:
            return FakeNeo4jResult(records=[
                {"education_level": "대학교(4년)", "count": 350},
            ])

        if "MaritalStatus" in query:
            return FakeNeo4jResult(records=[
                {"marital_status": "기혼", "count": 600},
            ])

        if "Occupation" in query:
            return FakeNeo4jResult(records=[
                {"occupation": "사무원", "count": 100},
            ])

        if "Hobby" in query:
            return FakeNeo4jResult(records=[
                {"hobby": "등산", "count": 150},
            ])

        if "Skill" in query:
            return FakeNeo4jResult(records=[
                {"skill": "Python", "count": 120},
            ])

        return FakeNeo4jResult(records=[])

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        pass


class FakeNeo4jDriver:
    def __init__(self, session: FakeNeo4jSession) -> None:
        self._session = session

    def session(self, **kwargs: object) -> FakeNeo4jSession:
        return self._session

    def close(self) -> None:
        pass


def _make_driver() -> FakeNeo4jDriver:
    return FakeNeo4jDriver(FakeNeo4jSession())


def test_stats_overview(monkeypatch) -> None:
    monkeypatch.setattr(stats, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/stats")

    assert response.status_code == 200
    body = response.json()
    assert body["total_personas"] == 1000

    assert len(body["age_distribution"]) == 2
    assert len(body["sex_distribution"]) == 2
    assert len(body["province_distribution"]) == 1
    assert len(body["education_distribution"]) == 1
    assert len(body["marital_distribution"]) == 1
    assert len(body["top_occupations"]) == 1
    assert len(body["top_hobbies"]) == 1
    assert len(body["top_skills"]) == 1

    age_30 = body["age_distribution"][0]
    assert age_30["label"] == "30대"
    assert age_30["count"] == 300
    assert age_30["ratio"] == 0.3

    sex_m = body["sex_distribution"][0]
    assert sex_m["label"] == "남자"
    assert sex_m["count"] == 520
    assert sex_m["ratio"] == 0.52


def test_stats_dimension_hobby(monkeypatch) -> None:
    monkeypatch.setattr(stats, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/stats/hobby")

    assert response.status_code == 200
    body = response.json()
    assert body["dimension"] == "hobby"
    assert len(body["distribution"]) == 2
    assert body["distribution"][0]["label"] == "등산"
    assert body["distribution"][0]["count"] == 80
    assert body["filtered_count"] == 140


def test_stats_dimension_with_filters(monkeypatch) -> None:
    monkeypatch.setattr(stats, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/stats/hobby?province=서울&age_group=30대")

    assert response.status_code == 200
    body = response.json()
    assert body["dimension"] == "hobby"
    assert body["filters_applied"]["province"] == "서울"
    assert body["filters_applied"]["age_group"] == "30대"
    assert len(body["distribution"]) >= 1


def test_stats_invalid_dimension(monkeypatch) -> None:
    monkeypatch.setattr(stats, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/stats/invalid_dim")

    assert response.status_code == 400
    assert "유효하지 않은" in response.json()["error"]


def test_stats_dimension_limit(monkeypatch) -> None:
    monkeypatch.setattr(stats, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/stats/occupation?limit=5")

    assert response.status_code == 200
    body = response.json()
    assert len(body["distribution"]) <= 5
