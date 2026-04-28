from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes import graph_viz


def _create_test_app() -> FastAPI:
    app = create_app()
    app.include_router(graph_viz.router)
    return app


FAKE_DEPTH1_RECORDS = [
    {
        "center_uuid": "test-uuid",
        "center_label": "김철수",
        "rel_type": "ENJOYS_HOBBY",
        "node_labels": ["Hobby"],
        "n_uuid": None,
        "n_name": "등산",
        "n_display_name": None,
        "n_age": None,
        "n_sex": None,
        "n_persona": None,
        "n_key": None,
        "n_province": None,
    },
    {
        "center_uuid": "test-uuid",
        "center_label": "김철수",
        "rel_type": "LIVES_IN",
        "node_labels": ["District"],
        "n_uuid": None,
        "n_name": "강남구",
        "n_display_name": None,
        "n_age": None,
        "n_sex": None,
        "n_persona": None,
        "n_key": "서울_강남구",
        "n_province": "서울",
    },
]

FAKE_DEPTH2_RECORDS = [
    {
        "entity_labels": ["Hobby"],
        "entity_name": "등산",
        "rel1_type": "ENJOYS_HOBBY",
        "rel2_type": "ENJOYS_HOBBY",
        "other_uuid": "other-uuid-001",
        "other_display_name": "이영희",
        "other_age": 33,
        "other_sex": "여자",
    },
]

FAKE_DEPTH3_RECORDS = [
    {
        "other_uuid": "other-uuid-001",
        "other_display_name": "이영희",
        "next_entity_labels": ["Skill"],
        "next_entity_name": "데이터 분석",
        "next_entity_key": None,
        "rel3_type": "HAS_SKILL",
    },
]


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

    def __iter__(self):  # noqa: ANN204
        return iter(self._records)


class FakeNeo4jSession:
    def run(self, query: str, **kwargs: object) -> FakeNeo4jResult:
        if "p.uuid AS uuid LIMIT 1" in query:
            return FakeNeo4jResult(single_data={"uuid": "test-uuid"})
        if "SIMILAR_TO" in query:
            return FakeNeo4jResult(records=FAKE_DEPTH1_RECORDS)
        if "next_entity" in query:
            return FakeNeo4jResult(records=FAKE_DEPTH3_RECORDS)
        if "entity" in query:
            return FakeNeo4jResult(records=FAKE_DEPTH2_RECORDS)
        return FakeNeo4jResult(records=[])

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *args: object) -> None:
        pass


class FakeNeo4jSessionNotFound:
    def run(self, query: str, **kwargs: object) -> FakeNeo4jResult:
        if "p.uuid AS uuid LIMIT 1" in query:
            return FakeNeo4jResult(single_data=None)
        return FakeNeo4jResult(records=[])

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *args: object) -> None:
        pass


class FakeNeo4jDriver:
    def __init__(self, session: FakeNeo4jSession | FakeNeo4jSessionNotFound) -> None:
        self._session = session

    def session(self, **kwargs: object) -> FakeNeo4jSession | FakeNeo4jSessionNotFound:
        return self._session

    def close(self) -> None:
        pass


def _make_driver() -> FakeNeo4jDriver:
    return FakeNeo4jDriver(FakeNeo4jSession())


def _make_not_found_driver() -> FakeNeo4jDriver:
    return FakeNeo4jDriver(FakeNeo4jSessionNotFound())


def test_subgraph_depth1(monkeypatch) -> None:
    monkeypatch.setattr(graph_viz, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/graph/subgraph/test-uuid")

    assert response.status_code == 200
    body = response.json()
    assert body["center_uuid"] == "test-uuid"
    assert body["center_label"] == "김철수"
    assert body["node_count"] == 3
    assert body["edge_count"] == 2

    node_ids = [n["id"] for n in body["nodes"]]
    assert "person_test-uuid" in node_ids
    assert "hobby_등산" in node_ids
    assert "district_서울_강남구" in node_ids

    node_types = {n["id"]: n["type"] for n in body["nodes"]}
    assert node_types["person_test-uuid"] == "Person"
    assert node_types["hobby_등산"] == "Hobby"
    assert node_types["district_서울_강남구"] == "District"


def test_subgraph_depth2(monkeypatch) -> None:
    monkeypatch.setattr(graph_viz, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/graph/subgraph/test-uuid?depth=2")

    assert response.status_code == 200
    body = response.json()
    assert body["center_uuid"] == "test-uuid"

    node_ids = [n["id"] for n in body["nodes"]]
    assert "person_other-uuid-001" in node_ids

    other_node = next(n for n in body["nodes"] if n["id"] == "person_other-uuid-001")
    assert other_node["type"] == "Person"
    assert other_node["label"] == "이영희"
    assert other_node["properties"]["age"] == 33
    assert other_node["properties"]["sex"] == "여자"


def test_subgraph_depth3(monkeypatch) -> None:
    monkeypatch.setattr(graph_viz, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/graph/subgraph/test-uuid?depth=3")

    assert response.status_code == 200
    body = response.json()
    node_ids = [n["id"] for n in body["nodes"]]
    assert "person_other-uuid-001" in node_ids
    assert "skill_데이터 분석" in node_ids

    edge_pairs = {(e["source"], e["target"], e["type"]) for e in body["edges"]}
    assert ("person_other-uuid-001", "skill_데이터 분석", "HAS_SKILL") in edge_pairs


def test_subgraph_include_similar(monkeypatch) -> None:
    monkeypatch.setattr(graph_viz, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/graph/subgraph/test-uuid?include_similar=true")

    assert response.status_code == 200
    body = response.json()
    assert body["center_uuid"] == "test-uuid"
    assert body["node_count"] >= 1


def test_subgraph_not_found(monkeypatch) -> None:
    monkeypatch.setattr(graph_viz, "get_neo4j_driver", _make_not_found_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/graph/subgraph/unknown")

    assert response.status_code == 404
    assert "찾을 수 없습니다" in response.json()["error"]


def test_subgraph_max_nodes(monkeypatch) -> None:
    monkeypatch.setattr(graph_viz, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/graph/subgraph/test-uuid?max_nodes=2")

    assert response.status_code == 200
    body = response.json()
    assert body["node_count"] <= 2

    node_ids = [n["id"] for n in body["nodes"]]
    assert "person_test-uuid" in node_ids
