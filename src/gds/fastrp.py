from typing import Any

from neo4j import GraphDatabase

from src.config import settings

PERSONA_GRAPH_NAME = "persona_graph"
PROJECT_RELATIONSHIP_TYPES = [
    "LIVES_IN",
    "IN_PROVINCE",
    "IN_COUNTRY",
    "WORKS_AS",
    "HAS_SKILL",
    "ENJOYS_HOBBY",
    "LIKES",
    "EDUCATED_AT",
    "MAJORED_IN",
    "MARITAL_STATUS",
    "MILITARY_STATUS",
    "LIVES_WITH",
    "LIVES_IN_HOUSING",
]

PROJECT_GRAPH_QUERY = """
CALL gds.graph.project(
    $graph_name,
    ['Person', 'District', 'Province', 'Country', 'Occupation', 'Skill', 'Hobby', 'EducationLevel', 'Field', 'MaritalStatus', 'MilitaryStatus', 'FamilyType', 'HousingType'],
    __RELATIONSHIP_PROJECTION__
)
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
"""

PROJECT_GRAPH_WITH_FASTRP_QUERY = """
CALL gds.graph.project(
    $graph_name,
    {
        Person: {properties: ['fastrp_embedding']},
        District: {},
        Province: {},
        Country: {},
        Occupation: {},
        Skill: {},
        Hobby: {},
        EducationLevel: {},
        Field: {},
        MaritalStatus: {},
        MilitaryStatus: {},
        FamilyType: {},
        HousingType: {}
    },
    __RELATIONSHIP_PROJECTION__
)
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
"""

DROP_GRAPH_QUERY = """
CALL gds.graph.drop($graph_name, false)
YIELD graphName
RETURN graphName
"""

FASTRP_WRITE_QUERY = """
CALL gds.fastRP.write($graph_name, {
    embeddingDimension: $dimension,
    iterationWeights: [0.0, 1.0, 1.0, 1.0],
    writeProperty: 'fastrp_embedding'
})
YIELD nodeCount, nodePropertiesWritten, preProcessingMillis, computeMillis, writeMillis
RETURN nodeCount, nodePropertiesWritten, preProcessingMillis, computeMillis, writeMillis
"""


RELATIONSHIP_TYPES_QUERY = """
MATCH ()-[r]->()
RETURN collect(DISTINCT type(r)) AS relationship_types
"""


def _build_relationship_projection(existing_relationship_types: set[str]) -> str:
    selected = [rel_type for rel_type in PROJECT_RELATIONSHIP_TYPES if rel_type in existing_relationship_types]
    if "ENJOYS_HOBBY" in selected and "LIKES" in selected:
        selected.remove("LIKES")
    if not selected:
        raise ValueError("No supported relationship types exist in Neo4j for GDS projection.")
    lines = [f"        {rel_type}: {{orientation: 'UNDIRECTED'}}" for rel_type in selected]
    return "{\n" + ",\n".join(lines) + "\n    }"


class FastRPService:
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

    def project_graph(self) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            query = self._with_available_relationship_projection(session, PROJECT_GRAPH_QUERY)
            result = session.run(query, graph_name=self.graph_name)
            record = result.single()
            return dict(record) if record else {}

    def project_graph_with_fastrp_embeddings(self) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            query = self._with_available_relationship_projection(session, PROJECT_GRAPH_WITH_FASTRP_QUERY)
            result = session.run(query, graph_name=self.graph_name)
            record = result.single()
            return dict(record) if record else {}

    def drop_graph(self) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            result = session.run(DROP_GRAPH_QUERY, graph_name=self.graph_name)
            record = result.single()
            return dict(record) if record else {}

    def write_embeddings(self, dimension: int = settings.GDS_FASTRP_DIMENSION) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            result = session.run(FASTRP_WRITE_QUERY, graph_name=self.graph_name, dimension=dimension)
            record = result.single()
            return dict(record) if record else {}

    def _with_available_relationship_projection(self, session: Any, query: str) -> str:
        record = session.run(RELATIONSHIP_TYPES_QUERY).single()
        existing = set(record["relationship_types"] if record else [])
        projection = _build_relationship_projection(existing)
        return query.replace("__RELATIONSHIP_PROJECTION__", projection)
