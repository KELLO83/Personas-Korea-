from typing import Any

from neo4j import GraphDatabase

from src.config import settings

PERSONA_GRAPH_NAME = "persona_graph"

PROJECT_GRAPH_QUERY = """
CALL gds.graph.project(
    $graph_name,
    ['Person', 'District', 'Province', 'Country', 'Occupation', 'Skill', 'Hobby', 'EducationLevel', 'Field', 'MaritalStatus', 'MilitaryStatus', 'FamilyType', 'HousingType'],
    {
        LIVES_IN: {orientation: 'UNDIRECTED'},
        IN_PROVINCE: {orientation: 'UNDIRECTED'},
        IN_COUNTRY: {orientation: 'UNDIRECTED'},
        WORKS_AS: {orientation: 'UNDIRECTED'},
        HAS_SKILL: {orientation: 'UNDIRECTED'},
        ENJOYS_HOBBY: {orientation: 'UNDIRECTED'},
        EDUCATED_AT: {orientation: 'UNDIRECTED'},
        MAJORED_IN: {orientation: 'UNDIRECTED'},
        MARITAL_STATUS: {orientation: 'UNDIRECTED'},
        MILITARY_STATUS: {orientation: 'UNDIRECTED'},
        LIVES_WITH: {orientation: 'UNDIRECTED'},
        LIVES_IN_HOUSING: {orientation: 'UNDIRECTED'}
    }
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
    {
        LIVES_IN: {orientation: 'UNDIRECTED'},
        IN_PROVINCE: {orientation: 'UNDIRECTED'},
        IN_COUNTRY: {orientation: 'UNDIRECTED'},
        WORKS_AS: {orientation: 'UNDIRECTED'},
        HAS_SKILL: {orientation: 'UNDIRECTED'},
        ENJOYS_HOBBY: {orientation: 'UNDIRECTED'},
        EDUCATED_AT: {orientation: 'UNDIRECTED'},
        MAJORED_IN: {orientation: 'UNDIRECTED'},
        MARITAL_STATUS: {orientation: 'UNDIRECTED'},
        MILITARY_STATUS: {orientation: 'UNDIRECTED'},
        LIVES_WITH: {orientation: 'UNDIRECTED'},
        LIVES_IN_HOUSING: {orientation: 'UNDIRECTED'}
    }
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
            result = session.run(PROJECT_GRAPH_QUERY, graph_name=self.graph_name)
            record = result.single()
            return dict(record) if record else {}

    def project_graph_with_fastrp_embeddings(self) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            result = session.run(PROJECT_GRAPH_WITH_FASTRP_QUERY, graph_name=self.graph_name)
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
