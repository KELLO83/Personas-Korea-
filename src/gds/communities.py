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

COMMUNITY_PROFILE_QUERY = """
MATCH (p:Person)
WHERE p.community_id = $community_id
WITH collect(p) AS people, count(p) AS size
CALL (people, size) {
    UNWIND people AS person
    WITH person.age_group AS label, count(person) AS count, size
    WHERE label IS NOT NULL
    ORDER BY count DESC
    RETURN collect({label: label, count: count, ratio: toFloat(count) / size}) AS age_distribution
}
CALL (people, size) {
    UNWIND people AS person
    WITH person.sex AS label, count(person) AS count, size
    WHERE label IS NOT NULL
    ORDER BY count DESC
    RETURN collect({label: label, count: count, ratio: toFloat(count) / size}) AS sex_distribution
}
CALL (people) {
    UNWIND people AS person
    OPTIONAL MATCH (person)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(province:Province)
    WITH province.name AS label, count(province) AS count
    WHERE label IS NOT NULL
    ORDER BY count DESC
    RETURN collect({label: label, count: count})[0..8] AS top_provinces
}
CALL (people) {
    UNWIND people AS person
    OPTIONAL MATCH (person)-[:LIVES_IN]->(district:District)
    WITH coalesce(district.key, district.name) AS label, count(district) AS count
    WHERE label IS NOT NULL
    ORDER BY count DESC
    RETURN collect({label: label, count: count})[0..8] AS top_districts
}
CALL (people) {
    UNWIND people AS person
    OPTIONAL MATCH (person)-[:WORKS_AS]->(occupation:Occupation)
    WITH occupation.name AS label, count(occupation) AS count
    WHERE label IS NOT NULL
    ORDER BY count DESC
    RETURN collect({label: label, count: count})[0..8] AS top_occupations
}
CALL (people) {
    UNWIND people AS person
    OPTIONAL MATCH (person)-[:EDUCATED_AT]->(education:EducationLevel)
    WITH education.name AS label, count(education) AS count
    WHERE label IS NOT NULL
    ORDER BY count DESC
    RETURN collect({label: label, count: count})[0..8] AS top_education
}
CALL (people) {
    UNWIND people AS person
    OPTIONAL MATCH (person)-[:ENJOYS_HOBBY|LIKES]->(hobby:Hobby)
    WITH hobby.name AS label, count(hobby) AS count
    WHERE label IS NOT NULL
    ORDER BY count DESC
    RETURN collect({label: label, count: count})[0..10] AS top_hobbies
}
CALL (people) {
    UNWIND people AS person
    OPTIONAL MATCH (person)-->(skill:Skill)
    WITH skill.name AS label, count(skill) AS count
    WHERE label IS NOT NULL
    ORDER BY count DESC
    RETURN collect({label: label, count: count})[0..10] AS top_skills
}
CALL (people) {
    UNWIND people AS person
    OPTIONAL MATCH (person)-[:WORKS_AS]->(occupation:Occupation)
    OPTIONAL MATCH (person)-[:LIVES_IN]->(district:District)-[:IN_PROVINCE]->(province:Province)
    WITH person, occupation, province, district
    ORDER BY person.uuid
    RETURN collect({
        uuid: person.uuid,
        display_name: person.display_name,
        age: person.age,
        sex: person.sex,
        occupation: occupation.name,
        province: province.name,
        district: district.name,
        pagerank: null
    })[0..8] AS representative_personas
}
RETURN
    $community_id AS community_id,
    size,
    age_distribution,
    sex_distribution,
    top_provinces,
    top_districts,
    top_occupations,
    top_education,
    top_hobbies,
    top_skills,
    representative_personas
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

    def community_profile(self, community_id: int) -> dict[str, Any] | None:
        with self.driver.session(database=self.database) as session:
            record = session.run(COMMUNITY_PROFILE_QUERY, community_id=community_id).single()
            if not record:
                return None
            row = dict(record)
            row["label"] = _build_community_label(
                [item["label"] for item in row.get("top_hobbies", [])],
                [item["label"] for item in row.get("top_provinces", [])],
            )
            return row
