from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes import persona


def _create_test_app() -> FastAPI:
    app = create_app()
    app.include_router(persona.router)
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
    def __init__(self, profile_data: dict[str, Any] | None = None, similar_data: list[dict[str, Any]] | None = None, stats_data: dict[str, Any] | None = None) -> None:
        self._profile_data = profile_data
        self._similar_data = similar_data or []
        self._stats_data = stats_data

    def run(self, query: str, **kwargs: object) -> FakeNeo4jResult:
        if "SIMILAR_TO" in query:
            return FakeNeo4jResult(records=self._similar_data)
        if "count(DISTINCT r)" in query:
            return FakeNeo4jResult(single_data=self._stats_data)
        return FakeNeo4jResult(single_data=self._profile_data)

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


FAKE_PERSON_NODE = {
    "uuid": "test-uuid-001",
    "display_name": "김철수",
    "age": 35,
    "age_group": "30대",
    "sex": "남자",
    "persona": "활발한 직장인",
    "professional_persona": "IT 전문가",
    "sports_persona": "축구 애호가",
    "arts_persona": "미술 감상가",
    "travel_persona": "해외여행 마니아",
    "culinary_persona": "한식 전문가",
    "family_persona": "가정적인 아버지",
    "cultural_background": "서울 출신 도시 문화",
    "career_goals_and_ambitions": "CTO가 되고 싶다",
    "community_id": 42,
    "community_label": "서울 IT 직장인",
}

FAKE_PROFILE_DATA = {
    "p": FAKE_PERSON_NODE,
    "district_name": "강남구",
    "district_key": "서울_강남구",
    "province_name": "서울",
    "country_name": "대한민국",
    "occupation_name": "소프트웨어 개발자",
    "education_level": "대학교(4년)",
    "bachelors_field": "컴퓨터공학",
    "marital_status": "기혼",
    "military_status": "군필",
    "family_type": "부부+자녀",
    "housing_type": "아파트",
    "skills": ["Python", "데이터 분석"],
    "hobbies": ["축구", "독서"],
}

FAKE_SIMILAR_DATA = [
    {
        "uuid": "similar-uuid-001",
        "display_name": "이영희",
        "age": 33,
        "similarity": 0.92,
        "shared_hobbies": ["축구"],
    },
]

FAKE_STATS_DATA = {
    "total_connections": 15,
    "hobby_count": 2,
    "skill_count": 2,
}


def _make_driver(
    profile_data: dict[str, Any] | None = None,
    similar_data: list[dict[str, Any]] | None = None,
    stats_data: dict[str, Any] | None = None,
) -> FakeNeo4jDriver:
    session = FakeNeo4jSession(
        profile_data=profile_data,
        similar_data=similar_data,
        stats_data=stats_data,
    )
    return FakeNeo4jDriver(session)


def test_persona_profile_found(monkeypatch) -> None:
    driver = _make_driver(
        profile_data=FAKE_PROFILE_DATA,
        similar_data=FAKE_SIMILAR_DATA,
        stats_data=FAKE_STATS_DATA,
    )
    monkeypatch.setattr(persona, "get_neo4j_driver", lambda: driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/persona/test-uuid-001")

    assert response.status_code == 200
    body = response.json()
    assert body["uuid"] == "test-uuid-001"
    assert body["display_name"] == "김철수"
    assert body["demographics"]["age"] == 35
    assert body["demographics"]["sex"] == "남자"
    assert body["demographics"]["marital_status"] == "기혼"
    assert body["demographics"]["military_status"] == "군필"
    assert body["demographics"]["family_type"] == "부부+자녀"
    assert body["demographics"]["housing_type"] == "아파트"
    assert body["demographics"]["education_level"] == "대학교(4년)"
    assert body["demographics"]["bachelors_field"] == "컴퓨터공학"
    assert body["location"]["country"] == "대한민국"
    assert body["location"]["province"] == "서울"
    assert body["location"]["district"] == "강남구"
    assert body["occupation"] == "소프트웨어 개발자"
    assert body["personas"]["summary"] == "활발한 직장인"
    assert body["personas"]["professional"] == "IT 전문가"
    assert body["personas"]["sports"] == "축구 애호가"
    assert body["personas"]["arts"] == "미술 감상가"
    assert body["personas"]["travel"] == "해외여행 마니아"
    assert body["personas"]["culinary"] == "한식 전문가"
    assert body["personas"]["family"] == "가정적인 아버지"
    assert body["cultural_background"] == "서울 출신 도시 문화"
    assert body["career_goals"] == "CTO가 되고 싶다"
    assert body["skills"] == ["Python", "데이터 분석"]
    assert body["hobbies"] == ["축구", "독서"]
    assert body["community"]["community_id"] == 42
    assert body["similar_preview"][0]["uuid"] == "similar-uuid-001"
    assert body["similar_preview"][0]["similarity"] == 0.92
    assert body["similar_preview"][0]["shared_hobbies"] == ["축구"]
    assert body["graph_stats"]["total_connections"] == 15
    assert body["graph_stats"]["hobby_count"] == 2
    assert body["graph_stats"]["skill_count"] == 2


def test_persona_profile_not_found(monkeypatch) -> None:
    driver = _make_driver(profile_data=None)
    monkeypatch.setattr(persona, "get_neo4j_driver", lambda: driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/persona/unknown-uuid")

    assert response.status_code == 404
    assert "해당 UUID" in response.json()["error"]


def test_persona_profile_empty_skills_hobbies(monkeypatch) -> None:
    profile_data = {
        **FAKE_PROFILE_DATA,
        "skills": [],
        "hobbies": [],
    }
    driver = _make_driver(
        profile_data=profile_data,
        similar_data=FAKE_SIMILAR_DATA,
        stats_data=FAKE_STATS_DATA,
    )
    monkeypatch.setattr(persona, "get_neo4j_driver", lambda: driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/persona/test-uuid-001")

    assert response.status_code == 200
    body = response.json()
    assert body["skills"] == []
    assert body["hobbies"] == []


def test_persona_profile_no_similar(monkeypatch) -> None:
    driver = _make_driver(
        profile_data=FAKE_PROFILE_DATA,
        similar_data=[],
        stats_data=FAKE_STATS_DATA,
    )
    monkeypatch.setattr(persona, "get_neo4j_driver", lambda: driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/persona/test-uuid-001")

    assert response.status_code == 200
    body = response.json()
    assert body["similar_preview"] == []
