from typing import Any

from neo4j import GraphDatabase

from src.config import settings
from src.gds.fastrp import PERSONA_GRAPH_NAME
from src.embeddings.vector_index import VECTOR_INDEX_NAME

KNN_WRITE_QUERY = """
CALL gds.knn.write($graph_name, {
    nodeLabels: ['Person'],
    nodeProperties: ['fastrp_embedding'],
    topK: $top_k,
    sampleRate: 1.0,
    deltaThreshold: 0.001,
    writeRelationshipType: 'SIMILAR_TO',
    writeProperty: 'score'
})
YIELD nodesCompared, relationshipsWritten, similarityDistribution
RETURN nodesCompared, relationshipsWritten, similarityDistribution
"""

SIMILAR_PERSONAS_QUERY = """
MATCH (:Person {uuid: $uuid})-[relationship:SIMILAR_TO]->(target:Person)
RETURN target.uuid AS uuid,
       target.display_name AS display_name,
       target.persona AS persona,
       target.age AS age,
       target.sex AS sex,
       relationship.score AS similarity
ORDER BY similarity DESC
LIMIT $top_k
"""

TEXT_SIMILAR_PERSONAS_QUERY = """
MATCH (p:Person {uuid: $uuid})
CALL db.index.vector.queryNodes($index_name, $top_k, p.text_embedding)
YIELD node, score
WHERE node.uuid <> $uuid
RETURN node.uuid AS uuid,
       node.display_name AS display_name,
       node.persona AS persona,
       node.age AS age,
       node.sex AS sex,
       score AS similarity
ORDER BY score DESC
LIMIT $top_k
"""

SHARED_TRAITS_QUERY = """
MATCH (source:Person {uuid: $source_uuid})-->(shared)<--(target:Person {uuid: $target_uuid})
WHERE NOT shared:Person
RETURN labels(shared)[0] AS type, shared.name AS name
ORDER BY type, name
"""

SHARED_HOBBIES_QUERY = """
MATCH (source:Person {uuid: $source_uuid})-[:ENJOYS_HOBBY]->(hobby:Hobby)<-[:ENJOYS_HOBBY]-(target:Person {uuid: $target_uuid})
RETURN hobby.name AS name
ORDER BY hobby.name
"""

QUERY_PERSONA_QUERY = """
MATCH (p:Person {uuid: $uuid})
OPTIONAL MATCH (p)-[:LIVES_IN]->(district:District)
OPTIONAL MATCH (p)-[:WORKS_AS]->(occupation:Occupation)
RETURN p.uuid AS uuid,
       p.display_name AS display_name,
       p.age AS age,
       district.name AS district_name,
       occupation.name AS occupation_name
"""


class SimilarityService:
    def __init__(
        self,
        uri: str = settings.NEO4J_URI,
        user: str = settings.NEO4J_USER,
        password: str = settings.NEO4J_PASSWORD,
        database: str = settings.NEO4J_DATABASE,
        graph_name: str = PERSONA_GRAPH_NAME,
    ) -> None:
        self.database = database
        self.graph_name = graph_name
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def write_knn_relationships(self, top_k: int = settings.GDS_KNN_TOP_K) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            result = session.run(KNN_WRITE_QUERY, graph_name=self.graph_name, top_k=top_k)
            record = result.single()
            return dict(record) if record else {}

    def find_similar_personas(self, uuid: str, top_k: int = settings.GDS_KNN_TOP_K) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(SIMILAR_PERSONAS_QUERY, uuid=uuid, top_k=top_k)
            return [dict(record) for record in result]

    def find_text_similar_personas(self, uuid: str, top_k: int = settings.GDS_KNN_TOP_K) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                TEXT_SIMILAR_PERSONAS_QUERY,
                uuid=uuid,
                index_name=VECTOR_INDEX_NAME,
                top_k=top_k,
            )
            return [dict(record) for record in result]

    def find_shared_traits(self, source_uuid: str, target_uuid: str) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(SHARED_TRAITS_QUERY, source_uuid=source_uuid, target_uuid=target_uuid)
            return [dict(record) for record in result]

    def find_shared_hobbies(self, source_uuid: str, target_uuid: str) -> list[str]:
        with self.driver.session(database=self.database) as session:
            result = session.run(SHARED_HOBBIES_QUERY, source_uuid=source_uuid, target_uuid=target_uuid)
            return [record["name"] for record in result if record["name"]]

    def get_query_persona(self, uuid: str) -> dict[str, Any] | None:
        with self.driver.session(database=self.database) as session:
            record = session.run(QUERY_PERSONA_QUERY, uuid=uuid).single()
            return dict(record) if record else None
