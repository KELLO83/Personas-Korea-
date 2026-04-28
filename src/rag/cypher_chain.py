import logging
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

from src.config import settings
from src.rag.llm import create_llm

logger = logging.getLogger(__name__)

CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

IMPORTANT DATA MATCHING RULES FOR KOREAN:
1. For sex/gender, use EXACTLY '남자' or '여자'. NEVER use '남성' or '여성'. Example: p.sex = '남자'
2. For age groups (e.g., '30대'), match against the `age_group` property: `p.age_group = '30대'`.
3. For regions (e.g., '광주'), match against the `Province` name. Note that province names are stored exactly like '광주', '서울', '부산'. Example: `prov.name = '광주'`.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

class CypherInsightChain:
    def __init__(self, max_retries: int = 1) -> None:
        graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE,
        )
        self.chain = GraphCypherQAChain.from_llm(
            llm=create_llm(temperature=0.0),
            graph=graph,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            validate_cypher=True,
            allow_dangerous_requests=True,
            return_intermediate_steps=True,
            verbose=False,
        )
        self.max_retries = max_retries

    def ask(self, question: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.chain.invoke({"query": question})
                return {
                    "answer": result.get("result", ""),
                    "sources": result.get("intermediate_steps", []),
                    "query_type": "cypher",
                }
            except Exception as exc:
                last_error = exc
                logger.warning("Cypher chain attempt %d failed: %s", attempt + 1, exc)
        return {
            "answer": f"죄송합니다, 질문을 처리하는 중 오류가 발생했습니다. ({last_error!s})",
            "sources": [],
            "query_type": "cypher_fallback",
        }
