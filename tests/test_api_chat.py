from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import exceptions as api_exceptions
from src.api.routes import chat
from src.rag.chat_graph import ChatGraph


class FakeChatGraph:
    def __init__(self) -> None:
        self.filters_by_session: dict[str, dict[str, str]] = {}
        self.turns_by_session: dict[str, int] = {}

    def invoke(self, session_id: str, message: str) -> dict[str, object]:
        filters = self.filters_by_session.setdefault(session_id, {})
        if "리셋" in message or "처음부터" in message:
            filters.clear()
        elif "서울" in message:
            filters["province"] = "서울"
        elif "부산" in message:
            filters["province"] = "부산"
        if "20대" in message:
            filters["age_group"] = "20대"
        self.turns_by_session[session_id] = self.turns_by_session.get(session_id, 0) + 1
        return {
            "response": "응답",
            "context_filters": dict(filters),
            "sources": [{"type": "fake"}],
            "turn_count": self.turns_by_session[session_id],
        }


def _create_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(chat.router)
    api_exceptions.add_exception_handlers(app)
    return app


def test_chat_endpoint_response_shape(monkeypatch) -> None:
    monkeypatch.setattr(chat, "get_chat_graph", lambda: FakeChatGraph())
    client = TestClient(_create_test_app())

    response = client.post("/api/chat", json={"session_id": "s1", "message": "서울 보여줘"})

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "응답"
    assert body["context_filters"]["province"] == "서울"
    assert body["sources"][0]["type"] == "fake"
    assert body["turn_count"] == 1


def test_chat_endpoint_returns_synthesized_chat_graph_response(monkeypatch) -> None:
    graph = ChatGraph()
    monkeypatch.setattr(
        graph,
        "_run_stats",
        lambda filters, requested_dimension=None: (
            "template stats",
            [{"type": "stats", "dimension": requested_dimension, "filters": filters}],
            [{"label": "독서", "count": 7}],
        ),
    )
    monkeypatch.setattr(graph, "_synthesize_response", lambda **kwargs: "API 합성 응답")
    monkeypatch.setattr(chat, "get_chat_graph", lambda: graph)
    client = TestClient(_create_test_app())

    response = client.post("/api/chat", json={"session_id": "s1", "message": "서울 20대 여성 취미는?"})

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "API 합성 응답"
    assert body["sources"][0]["type"] == "stats"
    assert body["context_filters"] == {"province": "서울", "age_group": "20대", "sex": "여자"}


def test_chat_endpoint_keeps_context_by_session(monkeypatch) -> None:
    fake = FakeChatGraph()
    monkeypatch.setattr(chat, "get_chat_graph", lambda: fake)
    client = TestClient(_create_test_app())

    client.post("/api/chat", json={"session_id": "s1", "message": "서울 보여줘"})
    response = client.post("/api/chat", json={"session_id": "s1", "message": "그중에서 20대만"})

    assert response.json()["context_filters"] == {"province": "서울", "age_group": "20대"}


def test_chat_endpoint_isolates_sessions(monkeypatch) -> None:
    fake = FakeChatGraph()
    monkeypatch.setattr(chat, "get_chat_graph", lambda: fake)
    client = TestClient(_create_test_app())

    response_a = client.post("/api/chat", json={"session_id": "a", "message": "서울 보여줘"})
    response_b = client.post("/api/chat", json={"session_id": "b", "message": "부산 보여줘"})

    assert response_a.json()["context_filters"] == {"province": "서울"}
    assert response_b.json()["context_filters"] == {"province": "부산"}


def test_chat_endpoint_reset(monkeypatch) -> None:
    fake = FakeChatGraph()
    monkeypatch.setattr(chat, "get_chat_graph", lambda: fake)
    client = TestClient(_create_test_app())

    client.post("/api/chat", json={"session_id": "s1", "message": "서울 보여줘"})
    response = client.post("/api/chat", json={"session_id": "s1", "message": "처음부터"})

    assert response.json()["context_filters"] == {}


def test_chat_endpoint_recommend_uses_selected_uuid_from_search(monkeypatch) -> None:
    graph = ChatGraph()

    def fake_run_search(filters):
        return (
            "search ok",
            [{"type": "search", "filters": filters, "total_count": 1}],
            [{"uuid": "person-9", "display_name": "김민수", "age": 33}],
        )

    def fake_run_recommend(message: str, selected_uuid: str | None = None):
        return (
            f"추천 대상: {selected_uuid}",
            [{"type": "recommend", "uuid": selected_uuid}],
            [{"item_name": "클라이밍", "reason": "reason", "reason_score": 0.2, "similar_users_count": 3}],
        )

    monkeypatch.setattr(graph, "_run_search", fake_run_search)
    monkeypatch.setattr(graph, "_run_recommend", fake_run_recommend)
    monkeypatch.setattr(chat, "get_chat_graph", lambda: graph)
    client = TestClient(_create_test_app())

    response_a = client.post("/api/chat", json={"session_id": "s1", "message": "서울 사람 보여줘"})
    response_b = client.post("/api/chat", json={"session_id": "s1", "message": "그 사람의 취미를 추천해줘"})

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert response_b.json()["response"] == "추천 대상: person-9"
    assert response_b.json()["sources"][0]["uuid"] == "person-9"


def test_chat_endpoint_rejects_empty_message(monkeypatch) -> None:
    monkeypatch.setattr(chat, "get_chat_graph", lambda: FakeChatGraph())
    client = TestClient(_create_test_app())

    response = client.post("/api/chat", json={"session_id": "s1", "message": ""})

    assert response.status_code == 422
