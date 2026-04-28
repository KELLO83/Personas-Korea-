from typing import Any

from neo4j import GraphDatabase

from src.config import settings
from src.gds.fastrp import PERSONA_GRAPH_NAME

LEIDEN_WRITE_QUERY = """
CALL gds.leiden.write($graph_name, {
    writeProperty: 'community_id',
    includeIntermediateCommunities: false
})
YIELD communityCount, modularity, ranLevels
RETURN communityCount, modularity, ranLevels
"""

COMMUNITY_SUMMARY_QUERY = """
MATCH (p:Person)
WHERE p.community_id IS NOT NULL
WITH p.community_id AS community_id, collect(p) AS people, count(p) AS size
WHERE size >= $min_size
WITH community_id, size, people, head(people).uuid AS representative_persona_uuid
CALL (people) {
    UNWIND people AS person
    OPTIONAL MATCH (person)-[:ENJOYS_HOBBY]->(hobby:Hobby)
    WITH hobby.name AS hobby_name, count(hobby.name) AS hobby_count
    WHERE hobby_name IS NOT NULL
    ORDER BY hobby_count DESC
    RETURN collect(hobby_name)[0..5] AS top_hobbies
}
CALL (people) {
    UNWIND people AS person
    OPTIONAL MATCH (person)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(province:Province)
    WITH province.name AS province_name, count(province.name) AS province_count
    WHERE province_name IS NOT NULL
    ORDER BY province_count DESC
    RETURN collect(province_name)[0..3] AS top_provinces
}
RETURN community_id AS id, size, top_hobbies, top_provinces, representative_persona_uuid
ORDER BY size DESC
"""


def _build_community_label(top_hobbies: list[str], top_provinces: list[str]) -> str:
    parts: list[str] = []
    if top_provinces:
        parts.append("/".join(top_provinces[:2]))
    if top_hobbies:
        parts.append(" + ".join(top_hobbies[:3]))
    return " ".join(parts) if parts else "미분류 커뮤니티"


def _build_top_traits(top_hobbies: list[str], top_provinces: list[str]) -> dict[str, Any]:
    return {
        "province": top_provinces[0] if top_provinces else None,
        "hobbies": top_hobbies[:5],
    }


class CommunityService:
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

    def write_communities(self) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            result = session.run(LEIDEN_WRITE_QUERY, graph_name=self.graph_name)
            record = result.single()
            return dict(record) if record else {}

    def summarize_communities(self, min_size: int = settings.GDS_LEIDEN_MIN_COMMUNITY_SIZE) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(COMMUNITY_SUMMARY_QUERY, min_size=min_size)
            rows = []
            for record in result:
                row = dict(record)
                row["label"] = _build_community_label(row.get("top_hobbies", []), row.get("top_provinces", []))
                row["top_traits"] = _build_top_traits(row.get("top_hobbies", []), row.get("top_provinces", []))
                rows.append(row)
            return rows
