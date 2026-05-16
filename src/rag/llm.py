from langchain_openai import ChatOpenAI

from src.config import settings
from src.rag.tracing import trace_span


class TracedChatOpenAI(ChatOpenAI):
    def invoke(self, input, config=None, **kwargs):  # type: ignore[no-untyped-def]
        metadata = {
            "model": getattr(self, "model_name", settings.LLM_MODEL),
            "input_length": len(str(input)) if input is not None else 0,
        }
        with trace_span("llm_called", metadata):
            return super().invoke(input, config=config, **kwargs)


def create_llm(temperature: float = 0.0) -> ChatOpenAI:
    return TracedChatOpenAI(
        api_key=settings.NVIDIA_API_KEY,
        base_url=settings.NVIDIA_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"thinking": False}},
    )
