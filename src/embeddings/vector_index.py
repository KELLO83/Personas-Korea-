from collections.abc import Iterable
from typing import Any

import pandas as pd
from neo4j import GraphDatabase

from src.config import settings

VECTOR_INDEX_NAME = "person_text_embedding_index"

CREATE_VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX person_text_embedding_index IF NOT EXISTS
FOR (p:Person) ON (p.text_embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: $dimensions,
    `vector.similarity_function`: 'cosine'
}}
"""

SET_PERSON_EMBEDDING_QUERY = """
UNWIND $rows AS row
MATCH (p:Person {uuid: row.uuid})
SET p.text_embedding = row.text_embedding
"""

VECTOR_SEARCH_QUERY = """
CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
YIELD node, score
RETURN node.uuid AS uuid,
       node.display_name AS display_name,
       node.persona AS persona,
       score
ORDER BY score DESC
"""

class Neo4jVectorIndex:
    def __init__(
        self,
        uri: str = settings.NEO4J_URI,
        user: str = settings.NEO4J_USER,
        password: str = settings.NEO4J_PASSWORD,
        database: str = settings.NEO4J_DATABASE,
        dimensions: int = settings.EMBEDDING_DIMENSION,
    ) -> None:
        self.database = database
        self.dimensions = dimensions
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def create_index(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(CREATE_VECTOR_INDEX_QUERY, dimensions=self.dimensions)

    def set_embeddings(self, rows: list[dict[str, Any]], batch_size: int = 500) -> int:
        total = 0
        with self.driver.session(database=self.database) as session:
            for batch in _batched(rows, batch_size):
                session.run(SET_PERSON_EMBEDDING_QUERY, rows=batch)
                total += len(batch)
        return total

    def search(self, embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                VECTOR_SEARCH_QUERY,
                index_name=VECTOR_INDEX_NAME,
                top_k=top_k,
                embedding=embedding,
            )
            return [dict(record) for record in result]

    def get_embedded_uuids(self, batch_size: int = 5000) -> set[str]:
        query = "MATCH (p:Person) WHERE p.text_embedding IS NOT NULL RETURN p.uuid AS uuid"
        uuids: set[str] = set()
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                uuid = record.get("uuid")
                if uuid:
                    uuids.add(str(uuid))
        return uuids


def build_embedding_rows(df: pd.DataFrame, embeddings: list[list[float]]) -> list[dict[str, Any]]:
    if len(df) != len(embeddings):
        raise ValueError("DataFrame row count and embedding count must match")

    rows = []
    for uuid, embedding in zip(df["uuid"].tolist(), embeddings, strict=True):
        rows.append({"uuid": str(uuid), "text_embedding": embedding})
    return rows


def _batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
