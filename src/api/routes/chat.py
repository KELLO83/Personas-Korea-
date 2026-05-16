from fastapi import APIRouter

from src.api.schemas import ChatRequest, ChatResponse
from src.rag.chat_graph import ChatGraph
from src.rag.tracing import trace_request, trace_span

router = APIRouter(prefix="/api", tags=["chat"])

_chat_graph = ChatGraph()


def get_chat_graph() -> ChatGraph:
    return _chat_graph


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    with trace_request("chat", session_id=request.session_id, question=request.message):
        with trace_span("chat_graph.invoke", {"session_id": request.session_id}):
            result = get_chat_graph().invoke(request.session_id, request.message)
    return ChatResponse(**result)
