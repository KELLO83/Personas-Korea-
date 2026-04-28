from src.rag.router import InsightRouter, classify_question
from src.rag.vector_chain import _build_answer_prompt, _format_context


class FakeChain:
    def __init__(self, query_type: str) -> None:
        self.query_type = query_type

    def ask(self, question: str) -> dict[str, object]:
        return {"answer": f"{self.query_type}: {question}", "sources": [], "query_type": self.query_type}


class EmptyErrorChain:
    def ask(self, question: str) -> dict[str, object]:
        return {"answer": "", "sources": [], "query_type": "vector_error"}


class ExplodingChain:
    def ask(self, question: str) -> dict[str, object]:
        raise RuntimeError("chain unavailable")


def test_classify_question_routes_aggregate_questions_to_cypher() -> None:
    assert classify_question("광주 30대 남성들이 공통으로 즐기는 취미는?") == "cypher"


def test_classify_question_routes_semantic_questions_to_vector() -> None:
    assert classify_question("차분하지만 가족을 아끼는 사람을 찾아줘") == "vector"


def test_classify_question_routes_personality_questions_to_vector() -> None:
    assert classify_question("20대여성들의 성격은 어떤편?") == "vector"


def test_insight_router_uses_selected_chain() -> None:
    router = InsightRouter(cypher_chain=FakeChain("cypher"), vector_chain=FakeChain("vector"))

    cypher_result = router.ask("서울 여성들의 직업 분포는?")
    vector_result = router.ask("조용한 문화생활을 좋아하는 사람")

    assert cypher_result["query_type"] == "cypher"
    assert vector_result["query_type"] == "vector"


def test_insight_router_does_not_initialize_unused_cypher_chain_for_vector_route() -> None:
    router = InsightRouter(cypher_chain=ExplodingChain(), vector_chain=FakeChain("vector"))

    result = router.ask("20대여성들의 성격은 어떤편?")

    assert result["query_type"] == "vector"
    assert result["answer"] == "vector: 20대여성들의 성격은 어떤편?"


def test_insight_router_returns_non_empty_answer_for_vector_error() -> None:
    router = InsightRouter(cypher_chain=FakeChain("cypher"), vector_chain=EmptyErrorChain())

    result = router.ask("20대여성들의 성격은 어떤편?")

    assert result["query_type"] == "vector_error"
    assert "벡터 RAG 검색" in str(result["answer"])


def test_insight_router_composite_handles_vector_exception() -> None:
    router = InsightRouter(cypher_chain=FakeChain("cypher"), vector_chain=ExplodingChain())

    result = router.ask("20대 여성들의 성격을 왜 그렇게 볼 수 있는지 설명해줘")

    assert result["query_type"] == "composite"
    assert result["answer"] == "cypher: 20대 여성들의 성격을 왜 그렇게 볼 수 있는지 설명해줘"


def test_vector_prompt_uses_search_results_as_context() -> None:
    context = _format_context([{"uuid": "abc", "display_name": "최은지", "score": 0.9, "persona": "회계 사무원"}])
    prompt = _build_answer_prompt("질문", context)

    assert "abc" in prompt
    assert "최은지" in prompt
    assert "검색 결과만 근거" in prompt
