from fastapi import APIRouter

from src.api.schemas import ChatRequest, ChatResponse
from src.rag.chat_graph import ChatGraph

router = APIRouter(prefix="/api", tags=["chat"])

_chat_graph = ChatGraph()


def get_chat_graph() -> ChatGraph:
    return _chat_graph


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = get_chat_graph().invoke(request.session_id, request.message)
    return ChatResponse(**result)
