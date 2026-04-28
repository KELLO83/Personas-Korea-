from langchain_openai import ChatOpenAI

from src.config import settings


def create_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.NVIDIA_API_KEY,
        base_url=settings.NVIDIA_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"thinking": False}},
    )
