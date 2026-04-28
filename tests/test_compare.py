import pytest
from src.api.schemas import (
    SegmentCompareRequest,
    SegmentDefinition,
    SegmentFilter,
)
from src.api.routes.compare import router
from fastapi.testclient import TestClient
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.api.exceptions import BadRequestException

app = FastAPI()

@app.exception_handler(BadRequestException)
async def bad_request_exception_handler(request: Request, exc: BadRequestException):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

app.include_router(router)
client = TestClient(app)

class FakeSession:
    def run(self, query: str, **kwargs):
        if "total" in query:
            class FakeTotalRecord:
                def single(self):
                    return {"total": 1000}
            return FakeTotalRecord()
        else:
            class FakeRecords:
                def __iter__(self):
                    if "filt_prov.name = $province" in query and kwargs.get("province") == "Seoul":
                        yield {"label": "Reading", "count": 100}
                        yield {"label": "Gaming", "count": 50}
                    else:
                        yield {"label": "Gaming", "count": 150}
                        yield {"label": "Cooking", "count": 30}
            return FakeRecords()

class FakeDriver:
    def session(self, database=None):
        class SessionContext:
            def __enter__(self):
                return FakeSession()
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return SessionContext()
    
    def close(self):
        pass

def fake_get_driver():
    return FakeDriver()

class FakeCompareChain:
    def analyze(self, *args, **kwargs):
        return "Mock AI Analysis"

@pytest.fixture
def override_deps(monkeypatch):
    monkeypatch.setattr("src.api.routes.compare.get_neo4j_driver", fake_get_driver)
    monkeypatch.setattr("src.api.routes.compare.CompareChain", FakeCompareChain)

def test_compare_valid(override_deps):
    req = SegmentCompareRequest(
        segment_a=SegmentDefinition(label="Group A", filters=SegmentFilter(province="Seoul")),
        segment_b=SegmentDefinition(label="Group B", filters=SegmentFilter(province="Busan")),
        dimensions=["hobby"],
        top_k=10
    )
    
    res = client.post("/api/compare/segments", json=req.model_dump())
    assert res.status_code == 200
    data = res.json()
    assert data["segment_a"]["count"] == 1000
    assert data["segment_b"]["count"] == 1000
    assert "hobby" in data["comparisons"]
    
    comp = data["comparisons"]["hobby"]
    assert "Gaming" in comp["common"]
    assert "Reading" in comp["only_a"]
    assert "Cooking" in comp["only_b"]
    
    assert data["ai_analysis"] == "Mock AI Analysis"

def test_compare_invalid_dimension(override_deps):
    req = SegmentCompareRequest(
        segment_a=SegmentDefinition(label="Group A"),
        segment_b=SegmentDefinition(label="Group B"),
        dimensions=["invalid_dim"]
    )
    res = client.post("/api/compare/segments", json=req.model_dump())
    assert res.status_code == 400

class FakeSessionZero:
    def run(self, query: str, **kwargs):
        if "total" in query:
            class FakeTotalRecord:
                def single(self):
                    return {"total": 0}
            return FakeTotalRecord()
        else:
            class FakeRecords:
                def __iter__(self):
                    return iter([])
            return FakeRecords()

class FakeDriverZero:
    def session(self, database=None):
        class SessionContext:
            def __enter__(self):
                return FakeSessionZero()
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return SessionContext()
    
    def close(self):
        pass

def test_compare_empty_segment(monkeypatch):
    monkeypatch.setattr("src.api.routes.compare.get_neo4j_driver", lambda: FakeDriverZero())
    monkeypatch.setattr("src.api.routes.compare.CompareChain", FakeCompareChain)
    
    req = SegmentCompareRequest(
        segment_a=SegmentDefinition(label="Empty A"),
        segment_b=SegmentDefinition(label="Empty B"),
        dimensions=["hobby"]
    )
    res = client.post("/api/compare/segments", json=req.model_dump())
    assert res.status_code == 200
    data = res.json()
    assert data["segment_a"]["count"] == 0
    assert data["segment_b"]["count"] == 0
    assert data["ai_analysis"] == ""
