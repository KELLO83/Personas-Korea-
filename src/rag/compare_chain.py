from typing import Any
from langchain_core.prompts import PromptTemplate
from src.rag.llm import create_llm

COMPARE_PROMPT_TEMPLATE = """
다음은 두 인구 집단의 {dimension} 분포 비교 결과입니다.

[그룹 A: {label_a} ({count_a}명)]
{distribution_a}

[그룹 B: {label_b} ({count_b}명)]
{distribution_b}

두 그룹의 공통점과 차이점을 3~5문장으로 한국어로 분석해주세요.
결과에는 불필요한 인사말이나 부연 설명 없이 분석 내용만 포함해주세요.
"""

class CompareChain:
    def __init__(self) -> None:
        self.llm = create_llm(temperature=0.2)
        self.prompt = PromptTemplate(
            template=COMPARE_PROMPT_TEMPLATE,
            input_variables=["dimension", "label_a", "count_a", "distribution_a", "label_b", "count_b", "distribution_b"]
        )
        self.chain = self.prompt | self.llm

    def analyze(self, dimension: str, label_a: str, count_a: int, dist_a: list[dict[str, Any]], label_b: str, count_b: int, dist_b: list[dict[str, Any]]) -> str:
        dist_a_str = self._format_dist(dist_a)
        dist_b_str = self._format_dist(dist_b)
        
        try:
            response = self.chain.invoke({
                "dimension": dimension,
                "label_a": label_a,
                "count_a": count_a,
                "distribution_a": dist_a_str,
                "label_b": label_b,
                "count_b": count_b,
                "distribution_b": dist_b_str,
            })
            return str(response.content)
        except Exception:
            return ""

    def _format_dist(self, dist: list[dict[str, Any]]) -> str:
        if not dist:
            return "데이터 없음"
        return ", ".join(f"{item['name']}({float(item['ratio'])*100:.1f}%)" for item in dist)
