import logging
from typing import Any

from src.embeddings.kure_model import KureEmbedder
from src.embeddings.vector_index import Neo4jVectorIndex
from src.rag.llm import create_llm

logger = logging.getLogger(__name__)


class VectorInsightChain:
    def __init__(self, embedder: KureEmbedder | None = None, vector_index: Neo4jVectorIndex | None = None) -> None:
        self.embedder = embedder or KureEmbedder()
        self.vector_index = vector_index or Neo4jVectorIndex()
        self.llm = create_llm(temperature=0.2)

    def search(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        embedding = self.embedder.encode_one(question)
        return self.vector_index.search(embedding, top_k=top_k)

    def ask(self, question: str, top_k: int = 5) -> dict[str, Any]:
        try:
            results = self.search(question, top_k=top_k)
        except Exception as exc:
            logger.warning("Vector search failed for question '%s': %s", question, exc)
            return {
                "answer": "",
                "sources": [],
                "query_type": "vector_error",
            }

        if not results:
            return {
                "answer": "관련된 페르소나를 찾을 수 없습니다. 질문을 다시 시도해 주세요.",
                "sources": [],
                "query_type": "vector_empty",
            }

        try:
            context = _format_context(results)
            prompt = _build_answer_prompt(question=question, context=context)
            response = self.llm.invoke(prompt)
            return {
                "answer": response.content,
                "sources": results,
                "query_type": "vector",
            }
        except Exception as exc:
            logger.warning("Vector LLM synthesis failed for question '%s': %s", question, exc)
            return {
                "answer": "",
                "sources": results,
                "query_type": "vector_error",
            }


def _format_context(results: list[dict[str, Any]]) -> str:
    lines = []
    for index, result in enumerate(results, start=1):
        lines.append(
            f"{index}. uuid={result.get('uuid')}, name={result.get('display_name')}, "
            f"score={result.get('score')}, persona={result.get('persona')}"
        )
    return "\n".join(lines)


def _build_answer_prompt(question: str, context: str) -> str:
    return (
        "다음은 한국어 페르소나 검색 결과입니다.\n"
        "검색 결과만 근거로 사용해 사용자의 질문에 한국어로 간결하게 답하세요.\n\n"
        f"질문:\n{question}\n\n"
        f"검색 결과:\n{context}"
    )
