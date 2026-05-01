from typing import Any

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes import graph_quality


class FakeNeo4jResult:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def __iter__(self):
        return iter(self._records)


class FakeNeo4jSession:
    def run(self, query: str, **kwargs: object) -> FakeNeo4jResult:
        if "Country" in query:
            return FakeNeo4jResult([{"label": "대한민국", "count": 100}])
        if "MilitaryStatus" in query:
            return FakeNeo4jResult([{"label": "비현역", "count": 95}, {"label": "현역", "count": 5}])
        if "BachelorsField" in query:
            return FakeNeo4jResult([{"label": "해당없음", "count": 70}, {"label": "컴퓨터공학", "count": 30}])
        return FakeNeo4jResult([])

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


def test_graph_quality_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(graph_quality, "get_neo4j_driver", _make_driver)
    client = TestClient(create_app())

    response = client.get("/api/graph-quality")

    assert response.status_code == 200
    body = response.json()
    assert len(body["checks"]) == 3
    assert len(body["migration_plan"]) == 3
    country = next(check for check in body["checks"] if check["name"] == "country")
    assert country["cardinality"] == 1
    assert country["distribution"][0]["ratio"] == 1.0
    assert country["action"] == "remove_node"
    assert country["severity"] == "high"
    military = next(check for check in body["checks"] if check["name"] == "military_status")
    assert military["distribution"][0]["ratio"] == 0.95
    assert military["action"] == "hide_filter"
