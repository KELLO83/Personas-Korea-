from typing import Any

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes import target_persona


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
            return FakeNeo4jResult(single_data={"matched_count": 3})

        if "RETURN p.uuid AS uuid" in query:
            return FakeNeo4jResult(records=[
                {
                    "uuid": "u1",
                    "display_name": "홍길동",
                    "age": 34,
                    "sex": "남자",
                    "persona": "활동적이고 사교적인 성향",
                    "province": "서울",
                    "district": "서초구",
                    "occupation": "개발자",
                },
                {
                    "uuid": "u2",
                    "display_name": "김영희",
                    "age": 33,
                    "sex": "남자",
                    "persona": "새로운 기술 학습을 즐김",
                    "province": "서울",
                    "district": "강남구",
                    "occupation": "기획자",
                },
            ])

        if "ENJOYS_HOBBY" in query:
            return FakeNeo4jResult(records=[{"name": "러닝", "cnt": 10}, {"name": "독서", "cnt": 8}])

        if "HAS_SKILL" in query:
            return FakeNeo4jResult(records=[{"name": "Python", "cnt": 9}, {"name": "SQL", "cnt": 7}])

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


def test_target_persona_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(target_persona, "get_neo4j_driver", _make_driver)
    client = TestClient(create_app())

    response = client.get("/api/target-persona?age_group=30대&province=서울&top_k=2")

    assert response.status_code == 200
    body = response.json()
    assert body["matched_count"] == 3
    assert body["sample_size"] == 2
    assert body["filters"]["age_group"] == "30대"
    assert body["representative_hobbies"][0] == "러닝"
    assert body["representative_skills"][0] == "Python"
    assert "대표 프로필" in body["representative_persona"]
    assert body["evidence_uuids"] == ["u1", "u2"]
    assert body["generation_method"] == "deterministic"
    assert "금지사항" in body["synthesis_prompt"]
    assert len(body["guardrails"]) >= 1
    assert "age_group" in body["input_policy"]


def test_target_persona_requires_filter(monkeypatch) -> None:
    monkeypatch.setattr(target_persona, "get_neo4j_driver", _make_driver)
    client = TestClient(create_app())

    response = client.get("/api/target-persona")

    assert response.status_code == 400
    assert "최소 1개" in response.json()["error"]


def test_target_persona_semantic_filter(monkeypatch) -> None:
    monkeypatch.setattr(target_persona, "get_neo4j_driver", _make_driver)
    monkeypatch.setattr(target_persona, "_collect_semantic_persona_uuids", lambda query_text, top_k: ["u1", "u2"])  # noqa: ARG005
    client = TestClient(create_app())

    response = client.get("/api/target-persona?age_group=30대&semantic_query=데이터 분석가")

    assert response.status_code == 200
    body = response.json()
    assert body["filters"]["semantic_query"] == "데이터 분석가"


def test_target_persona_llm_synthesis(monkeypatch) -> None:
    monkeypatch.setattr(target_persona, "get_neo4j_driver", _make_driver)
    monkeypatch.setattr(target_persona, "_synthesize_with_llm", lambda prompt: "LLM 대표 페르소나")  # noqa: ARG005
    client = TestClient(create_app())

    response = client.get("/api/target-persona?age_group=30대&use_llm=true")

    assert response.status_code == 200
    body = response.json()
    assert body["generation_method"] == "llm"
    assert body["representative_persona"] == "LLM 대표 페르소나"
