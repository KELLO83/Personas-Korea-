import logging
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, StateGraph

from src.rag.cypher_chain import CypherInsightChain
from src.rag.vector_chain import VectorInsightChain

logger = logging.getLogger(__name__)

QueryRoute = Literal["cypher", "vector", "composite"]


class InsightState(TypedDict, total=False):
    question: str
    route: QueryRoute
    answer: str
    sources: list[Any]
    query_type: str


class InsightChain(Protocol):
    def ask(self, question: str) -> dict[str, Any]: ...


class InsightRouter:
    def __init__(
        self,
        cypher_chain: InsightChain | None = None,
        vector_chain: InsightChain | None = None,
    ) -> None:
        self._cypher_chain = cypher_chain
        self._vector_chain = vector_chain
        self.graph = self._build_graph()

    @property
    def cypher_chain(self) -> InsightChain:
        if self._cypher_chain is None:
            self._cypher_chain = CypherInsightChain()
        return self._cypher_chain

    @property
    def vector_chain(self) -> InsightChain:
        if self._vector_chain is None:
            self._vector_chain = VectorInsightChain()
        return self._vector_chain

    def ask(self, question: str) -> dict[str, Any]:
        result = self.graph.invoke({"question": question})
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "query_type": result.get("query_type", result.get("route", "unknown")),
        }

    def _build_graph(self):
        graph = StateGraph(InsightState)
        graph.add_node("classify", self._classify)
        graph.add_node("cypher", self._ask_cypher)
        graph.add_node("vector", self._ask_vector)
        graph.add_node("composite", self._ask_composite)
        graph.set_entry_point("classify")
        graph.add_conditional_edges(
            "classify",
            _route_from_state,
            {"cypher": "cypher", "vector": "vector", "composite": "composite"},
        )
        graph.add_edge("cypher", END)
        graph.add_edge("vector", END)
        graph.add_edge("composite", END)
        return graph.compile()

    def _classify(self, state: InsightState) -> InsightState:
        question = state.get("question", "")
        return {"route": classify_question(question)}

    def _ask_cypher(self, state: InsightState) -> InsightState:
        result = self.cypher_chain.ask(state.get("question", ""))
        if result.get("query_type") == "cypher_fallback":
            logger.info("Cypher failed, falling back to composite")
            return self._ask_composite(state)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "query_type": result["query_type"],
        }

    def _ask_vector(self, state: InsightState) -> InsightState:
        try:
            result = self.vector_chain.ask(state.get("question", ""))
        except Exception as exc:
            logger.warning("Vector chain failed before producing a result: %s", exc)
            return _vector_error_response()
        if result.get("query_type") == "vector_empty":
            logger.info("Vector returned empty, falling back to composite")
            return self._ask_composite(state)
        if result.get("query_type") == "vector_error" and not result.get("answer"):
            return _vector_error_response(result.get("sources", []))
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "query_type": result["query_type"],
        }

    def _ask_composite(self, state: InsightState) -> InsightState:
        question = state.get("question", "")
        cypher_result = self.cypher_chain.ask(question)
        try:
            vector_result = self.vector_chain.ask(question)
        except Exception as exc:
            logger.warning("Vector chain failed during composite route: %s", exc)
            vector_result = _vector_error_response()
        if vector_result.get("query_type") == "vector_error" and not vector_result.get("answer"):
            vector_result = _vector_error_response(vector_result.get("sources", []))

        sources = []
        if cypher_result.get("sources"):
            sources.append({"type": "cypher", "data": cypher_result["sources"]})
        if vector_result.get("sources"):
            sources.append({"type": "vector", "data": vector_result["sources"]})

        cypher_answer = cypher_result.get("answer", "") if cypher_result.get("query_type") != "cypher_fallback" else ""
        vector_answer = vector_result.get("answer", "") if vector_result.get("query_type") not in ("vector_empty", "vector_error") else ""

        if not cypher_answer and not vector_answer:
            answer = "질문에 답할 수 있는 정보를 찾지 못했습니다."
        elif not cypher_answer:
            answer = vector_answer
        elif not vector_answer:
            answer = cypher_answer
        else:
            answer = (
                f"[그래프 분석]\n{cypher_answer}\n\n"
                f"[유사 페르소나 기반]\n{vector_answer}"
            )

        return {
            "answer": answer,
            "sources": sources,
            "query_type": "composite",
        }


_insight_router: InsightRouter | None = None


def get_insight_router() -> InsightRouter:
    global _insight_router
    if _insight_router is None:
        _insight_router = InsightRouter()
    return _insight_router


def classify_question(question: str) -> QueryRoute:
    composite_keywords = [
        "비교",
        "차이",
        "왜",
        "어떻게",
        "설명",
        "분석",
    ]
    if any(keyword in question for keyword in composite_keywords):
        return "composite"

    cypher_keywords = [
        "몇 명",
        "몇명",
        "비율",
        "분포",
        "공통",
        "가장 많이",
        "상위",
        "평균",
        "지역",
        "직업",
        "취미는",
        "목표는",
    ]
    if any(keyword in question for keyword in cypher_keywords):
        return "cypher"
    return "vector"


def _route_from_state(state: InsightState) -> QueryRoute:
    return state.get("route", "vector")


def _vector_error_response(sources: list[Any] | None = None) -> InsightState:
    return {
        "answer": (
            "성격이나 성향처럼 페르소나 본문을 해석해야 하는 질문은 벡터 RAG 검색이 필요합니다. "
            "현재 벡터 검색 또는 LLM 합성 단계에서 오류가 발생해 근거를 만들지 못했습니다. "
            "임베딩 인덱스와 NVIDIA API 설정을 확인한 뒤 다시 시도해 주세요."
        ),
        "sources": sources or [],
        "query_type": "vector_error",
    }
