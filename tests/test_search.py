from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import exceptions as api_exceptions
from src.api.routes import search


def _create_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(search.router)
    api_exceptions.add_exception_handlers(app)
    return app


FAKE_SEARCH_RESULTS = [
    {
        "uuid": "search-uuid-001",
        "display_name": "김철수",
        "age": 35,
        "sex": "남자",
        "province": "서울",
        "district": "강남구",
        "occupation": "소프트웨어 개발자",
        "education_level": "대학교(4년)",
        "persona": "활발한 직장인",
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
    def __init__(self, records: list[dict[str, Any]] | None = None, single_data: dict[str, Any] | None = None) -> None:
        self.records = records or FAKE_SEARCH_RESULTS
        self.single_data = single_data or {"total_count": 1}

    def run(self, query: str, **kwargs: object) -> FakeNeo4jResult:
        if "count(DISTINCT" in query:
            return FakeNeo4jResult(single_data=self.single_data)
        return FakeNeo4jResult(records=self.records)

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *args: object) -> None:
        pass


class FakeNeo4jSessionEmpty:
    def run(self, query: str, **kwargs: object) -> FakeNeo4jResult:
        if "count(DISTINCT" in query:
            return FakeNeo4jResult(single_data={"total_count": 0})
        return FakeNeo4jResult(records=[])

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *args: object) -> None:
        pass


class FakeNeo4jDriver:
    def __init__(self, session: object) -> None:
        self._session = session

    def session(self, **kwargs: object) -> object:
        return self._session

    def close(self) -> None:
        pass


def _make_driver() -> FakeNeo4jDriver:
    return FakeNeo4jDriver(FakeNeo4jSession())


def _make_empty_driver() -> FakeNeo4jDriver:
    return FakeNeo4jDriver(FakeNeo4jSessionEmpty())


class FakeNeo4jSessionCapture:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] = {}

    def run(self, query: str, **kwargs: object) -> FakeNeo4jResult:
        self.last_kwargs = kwargs
        if "count(DISTINCT" in query:
            return FakeNeo4jResult(single_data={"total_count": 1})
        return FakeNeo4jResult(records=FAKE_SEARCH_RESULTS)

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *args: object) -> None:
        pass


def _make_capture_driver(session: FakeNeo4jSessionCapture) -> FakeNeo4jDriver:
    return FakeNeo4jDriver(session)


class FakeQueryCounterSession:
    def __init__(self, first_count: int = 1, subsequent_count: int = 1) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self._count_calls = 0
        self._first_count = first_count
        self._subsequent_count = subsequent_count

    def run(self, query: str, **kwargs: object) -> FakeNeo4jResult:
        self.calls.append((query, kwargs))

        if "count(DISTINCT" in query:
            self._count_calls += 1
            if self._count_calls == 1:
                return FakeNeo4jResult(single_data={"total_count": self._first_count})
            return FakeNeo4jResult(single_data={"total_count": self._subsequent_count})

        return FakeNeo4jResult(records=FAKE_SEARCH_RESULTS)

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *args: object) -> None:
        pass


class FakeSemanticSearchSession(FakeQueryCounterSession):
    def __init__(self) -> None:
        super().__init__(first_count=0, subsequent_count=1)


def _make_query_counter_driver(session: FakeQueryCounterSession | FakeSemanticSearchSession) -> FakeNeo4jDriver:
    return FakeNeo4jDriver(session)


def test_search_no_filters(monkeypatch) -> None:
    monkeypatch.setattr(search, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/search")

    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 1
    assert body["page"] == 1
    assert body["page_size"] == 20
    assert body["total_pages"] == 1
    assert len(body["results"]) == 1
    assert body["results"][0]["uuid"] == "search-uuid-001"


def test_search_province_filter(monkeypatch) -> None:
    monkeypatch.setattr(search, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/search?province=서울")

    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 1
    assert len(body["results"]) == 1


def test_search_complex_filters(monkeypatch) -> None:
    monkeypatch.setattr(search, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/search?province=서울&age_group=30대&sex=여자")

    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 1
    assert len(body["results"]) == 1


def test_search_normalizes_input_filters(monkeypatch) -> None:
    session = FakeNeo4jSessionCapture()
    monkeypatch.setattr(search, "get_neo4j_driver", lambda: _make_capture_driver(session))
    client = TestClient(_create_test_app())

    response = client.get("/api/search?province=%EA%B2%BD%EA%B8%B0&age_group=%2020%EB%8C%80&sex=%EC%97%AC%EC%84%B1&occupation=%20%EC%B7%A8%EC%A4%80%EC%83%9D%20")

    assert response.status_code == 200
    assert session.last_kwargs["provinces"] == ["경기"]
    assert session.last_kwargs["age_groups"] == ["20대"]
    assert session.last_kwargs["sex"] == "여자"
    assert session.last_kwargs["occupation_0_raw"] == "취준생"
    assert session.last_kwargs["occupation_0_compact"] == "취준생"


def test_build_search_query_builds_or_clause_for_aliases() -> None:
    occupation_terms = search._normalize_occupation("취준생,무직")
    assert occupation_terms is not None

    _, count_query, params = search.build_search_query(occupation=occupation_terms)

    assert "occupation_0_raw" in params
    assert "occupation_1_raw" in params
    assert "occupation_2_raw" not in params
    assert "occupation_3_raw" not in params
    assert "(toLower(occ.name) CONTAINS $occupation_0_raw" in count_query
    assert "toLower(replace(replace(replace(replace(occ.name" in count_query


def test_build_search_query_supports_semantic_uuid_filter() -> None:
    _, count_query, params = search.build_search_query(keyword="직장인", semantic_persona_uuids=["uuid-a", "uuid-b"])

    assert "p.uuid IN $persona_uuids" in count_query
    assert params["persona_uuids"] == ["uuid-a", "uuid-b"]


def test_build_search_query_district_matches_name_or_key() -> None:
    _, count_query, params = search.build_search_query(district=["해운대구"])

    assert "(d.key IN $districts OR d.name IN $districts)" in count_query
    assert params["districts"] == ["해운대구"]


def test_search_runs_semantic_fallback_when_exact_filter_is_empty(monkeypatch) -> None:
    session = FakeSemanticSearchSession()
    monkeypatch.setattr(search, "get_neo4j_driver", lambda: _make_query_counter_driver(session))
    monkeypatch.setattr(search, "_collect_semantic_persona_uuids", lambda terms: ["uuid-semantic"])
    client = _create_test_app()
    app = TestClient(client)

    response = app.get("/api/search?keyword=%EC%9E%A5%EA%B9%80%EC%9E%A5")

    assert response.status_code == 200
    assert session.calls[1][0].count("p.uuid IN $persona_uuids") == 1


def test_search_keeps_exact_count_when_results_exist_without_fallback(monkeypatch) -> None:
    session = FakeQueryCounterSession()
    monkeypatch.setattr(search, "get_neo4j_driver", lambda: _make_query_counter_driver(session))
    monkeypatch.setattr(search, "_collect_semantic_persona_uuids", lambda terms: ["uuid-semantic"])
    client = _create_test_app()

    response = TestClient(client).get("/api/search?province=%EA%B2%BD%EA%B8%B0")

    assert response.status_code == 200
    assert len(session.calls) == 2
    assert "count(DISTINCT" in session.calls[0][0]
    assert "p.uuid IN $persona_uuids" not in session.calls[0][0]


def test_search_hobby_filter(monkeypatch) -> None:
    monkeypatch.setattr(search, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/search?hobby=등산")

    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 1
    assert len(body["results"]) == 1


def test_search_keyword(monkeypatch) -> None:
    monkeypatch.setattr(search, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/search?keyword=직장인")

    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 1
    assert len(body["results"]) == 1


def test_search_pagination(monkeypatch) -> None:
    monkeypatch.setattr(search, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/search?page=2&page_size=10")

    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 2
    assert body["page_size"] == 10


def test_search_empty_results(monkeypatch) -> None:
    monkeypatch.setattr(search, "get_neo4j_driver", _make_empty_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/search")

    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 0
    assert body["results"] == []
    assert body["total_pages"] == 0


def test_search_invalid_sort(monkeypatch) -> None:
    monkeypatch.setattr(search, "get_neo4j_driver", _make_driver)
    client = TestClient(_create_test_app())

    response = client.get("/api/search?sort_by=invalid")

    assert response.status_code == 400
    assert "유효하지 않은" in response.json()["error"]
