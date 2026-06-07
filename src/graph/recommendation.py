from __future__ import annotations

from typing import Any, LiteralString, cast

from neo4j import GraphDatabase, Query

from ..config import settings

VALID_RECOMMENDATION_CATEGORIES = {"hobby", "skill", "occupation", "district"}
VALID_CENTRALITY_METRICS = {"pagerank", "betweenness", "degree"}

PERSON_EXISTS_QUERY = """
MATCH (p:Person {uuid: $uuid})
RETURN p.uuid AS uuid
LIMIT 1
"""

SIMILAR_COUNT_QUERY = """
MATCH (:Person {uuid: $uuid})-[:SIMILAR_TO]->(sim:Person)
RETURN count(sim) AS count
"""

HOBBY_QUALITY_QUERY = """
MATCH (p:Person)
WHERE EXISTS { MATCH (p)-[:SIMILAR_TO]->(:Person) }
WITH p
ORDER BY p.uuid
LIMIT $sample_limit
MATCH (p)-[:SIMILAR_TO]->(sim:Person)-[:ENJOYS_HOBBY|LIKES]->(hobby:Hobby)
WITH p, hobby.name AS target, count(sim) AS supporters
ORDER BY p.uuid, supporters DESC
WITH p, collect({target: target, supporters: supporters})[0..10] AS recommendations
WITH collect(recommendations) AS recommendation_lists, count(p) AS sample_size
UNWIND recommendation_lists AS recommendations
UNWIND recommendations AS recommendation
WITH sample_size, recommendation.target AS target, recommendation.supporters AS supporters
WITH sample_size,
     target,
     count(*) AS frequency,
     avg(toFloat(supporters)) AS avg_supporters
WITH sample_size,
     collect({label: target, count: frequency}) AS targets,
     sum(frequency) AS recommendation_count,
     count(DISTINCT target) AS unique_target_count,
     max(frequency) AS max_frequency,
     sum(CASE WHEN avg_supporters <= 1.0 THEN frequency ELSE 0 END) AS weak_count
MATCH (catalog:Hobby)
RETURN sample_size,
       count(catalog) AS catalog_size,
       recommendation_count,
       unique_target_count,
       max_frequency,
       weak_count,
       targets
"""

PERSONA_SIMILARITY_QUALITY_QUERY = """
MATCH (p:Person)
WHERE EXISTS { MATCH (p)-[:SIMILAR_TO]->(:Person) }
WITH p
ORDER BY p.uuid
LIMIT $sample_limit
MATCH (p)-[similarity:SIMILAR_TO]->(target:Person)
OPTIONAL MATCH (target)-[:WORKS_AS]->(occupation:Occupation)
OPTIONAL MATCH (target)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(province:Province)
WITH p, target, similarity, occupation, province
ORDER BY p.uuid, coalesce(similarity.score, 0.0) DESC
WITH p,
     collect({
        target: coalesce(target.display_name, target.uuid),
        score: coalesce(similarity.score, 0.0),
        occupation: occupation.name,
        province: province.name
     })[0..10] AS recommendations
WITH collect(recommendations) AS recommendation_lists, count(p) AS sample_size
UNWIND recommendation_lists AS recommendations
UNWIND recommendations AS recommendation
WITH sample_size,
     recommendation.target AS target,
     recommendation.score AS score,
     recommendation.occupation AS occupation,
     recommendation.province AS province
WITH sample_size,
     target,
     occupation,
     province,
     count(*) AS frequency,
     avg(toFloat(score)) AS avg_score
WITH sample_size,
     collect({label: target, count: frequency}) AS targets,
     sum(frequency) AS recommendation_count,
     count(DISTINCT target) AS unique_target_count,
     count(DISTINCT occupation) AS occupation_diversity,
     count(DISTINCT province) AS province_diversity,
     max(frequency) AS max_frequency,
     sum(CASE WHEN avg_score < 0.1 THEN frequency ELSE 0 END) AS weak_count
MATCH (catalog:Person)
RETURN sample_size,
       count(catalog) AS catalog_size,
       recommendation_count,
       unique_target_count,
       occupation_diversity,
       province_diversity,
       max_frequency,
       weak_count,
       targets
"""

def _build_recommendation_query(category: str) -> str:
    relation = {
        "hobby": "ENJOYS_HOBBY",
        "skill": "HAS_SKILL",
        "occupation": "WORKS_AS",
        "district": "LIVES_IN",
    }[category]

    item_label = {
        "hobby": "Hobby",
        "skill": "Skill",
        "occupation": "Occupation",
        "district": "District",
    }[category]

    item_expr = {
        "hobby": "item.name",
        "skill": "item.name",
        "occupation": "item.name",
        "district": "coalesce(item.key, item.name)",
    }[category]

    return f"""
        MATCH (source:Person {{uuid: $uuid}})-[rel:SIMILAR_TO]->(sim:Person)
        WITH source, sim, coalesce(rel.score, 0.0) AS similarity
        MATCH (sim)-[:{relation}]->(item:{item_label})
        WHERE NOT (source)-[:{relation}]->(item)
        WITH {item_expr} AS item_name,
             sim,
             similarity
        WITH item_name,
             sim,
             similarity,
             CASE
                 WHEN $score_property IS NULL OR $score_property = '' THEN toFloat(similarity)
                 ELSE toFloat(similarity) * (1.0 + coalesce(sim[$score_property], 0.0))
             END AS weighted_similarity
        ORDER BY item_name, similarity DESC
        WITH item_name,
             count(DISTINCT sim) AS similar_users_count,
             sum(toFloat(weighted_similarity)) AS weighted_score,
             collect({{uuid: sim.uuid, display_name: sim.display_name, similarity: toFloat(similarity)}})[..5] AS supporting_personas
        MATCH (:Person {{uuid: $uuid}})-[:SIMILAR_TO]->(all_sim:Person)
        WITH item_name,
             similar_users_count,
             weighted_score,
             supporting_personas,
             count(DISTINCT all_sim) AS total_similar
        RETURN item_name,
               similar_users_count,
               CASE WHEN total_similar = 0 THEN 0.0 ELSE toFloat(similar_users_count) / total_similar END AS reason_score,
               weighted_score,
               supporting_personas
        ORDER BY reason_score DESC, weighted_score DESC, item_name
        LIMIT $top_n
    """


RECOMMENDATION_QUERIES: dict[str, str] = {
    "hobby": _build_recommendation_query("hobby"),
    "skill": _build_recommendation_query("skill"),
    "occupation": _build_recommendation_query("occupation"),
    "district": _build_recommendation_query("district"),
}

REASON_TEMPLATES = {
    "hobby": "당신과 유사한 {count}명 중 {ratio:.0%}가 '{item}'을(를) 취미로 가지고 있습니다.",
    "skill": "당신과 유사한 {count}명 중 {ratio:.0%}가 '{item}' 스킬을 보유하고 있습니다.",
    "occupation": "당신과 유사한 {count}명 중 {ratio:.0%}가 '{item}' 직업을 가지고 있습니다.",
    "district": "당신과 유사한 {count}명 중 {ratio:.0%}가 '{item}' 지역에 거주하고 있습니다.",
}


class RecommendationService:
    def __init__(
        self,
        uri: str = settings.NEO4J_URI,
        user: str = settings.NEO4J_USER,
        password: str = settings.NEO4J_PASSWORD,
        database: str = settings.NEO4J_DATABASE,
    ) -> None:
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def persona_exists(self, uuid: str) -> bool:
        with self.driver.session(database=self.database) as session:
            record = session.run(PERSON_EXISTS_QUERY, uuid=uuid).single()
            return bool(record)

    def has_similarity_data(self, uuid: str) -> bool:
        with self.driver.session(database=self.database) as session:
            record = session.run(SIMILAR_COUNT_QUERY, uuid=uuid).single()
            return bool(record and int(record["count"]) > 0)

    def recommend(
        self,
        uuid: str,
        category: str,
        top_n: int = 5,
        *,
        influence_metric: str | None = None,
    ) -> list[dict[str, Any]]:
        if category not in VALID_RECOMMENDATION_CATEGORIES:
            raise ValueError(f"Invalid recommendation category: {category}")
        if influence_metric is not None and influence_metric not in VALID_CENTRALITY_METRICS:
            raise ValueError(f"Invalid influence metric: {influence_metric}")
        query = RECOMMENDATION_QUERIES[category]
        with self.driver.session(database=self.database) as session:
            rows = [
                dict(record)
                for record in session.run(
                    Query(cast(LiteralString, query)),
                    uuid=uuid,
                    top_n=top_n,
                    score_property=influence_metric,
                )
            ]
        return [_format_recommendation(row, category, rank=index) for index, row in enumerate(rows, start=1)]

    def quality_snapshot(self, target: str) -> dict[str, Any]:
        query = {
            "hobby": HOBBY_QUALITY_QUERY,
            "persona_similarity": PERSONA_SIMILARITY_QUALITY_QUERY,
        }[target]
        with self.driver.session(database=self.database) as session:
            record = session.run(Query(cast(LiteralString, query)), sample_limit=500).single()
            return dict(record) if record else {}


def _format_recommendation(row: dict[str, Any], category: str, rank: int = 0) -> dict[str, Any]:
    item_name = str(row.get("item_name") or "")
    similar_users_count = int(row.get("similar_users_count") or 0)
    reason_score = round(float(row.get("reason_score") or 0.0), 4)
    weighted_score = round(float(row.get("weighted_score") or reason_score), 4)
    template = REASON_TEMPLATES[category]
    reason = template.format(count=similar_users_count, ratio=reason_score, item=item_name)
    return {
        "item_name": item_name,
        "reason": reason,
        "reason_score": reason_score,
        "similar_users_count": similar_users_count,
        "supporting_personas": _format_supporting_personas(row.get("supporting_personas")),
        "score": weighted_score,
        "rank": rank,
        "reason_cards": [
            {
                "type": "similar_person",
                "title": "유사 페르소나 기반",
                "detail": reason,
                "strength": reason_score,
            },
            {
                "type": "graph_frequency",
                "title": "그래프 빈도 기반",
                "detail": "Model is not promoted yet, so this item is ranked with Neo4j graph/rule fallback scoring.",
                "strength": weighted_score,
            },
        ],
        "already_known": False,
        "sources": ["similar_person", "graph_frequency"],
        "score_source": "fallback",
        "model_version": None,
        "graph_snapshot_id": None,
        "fallback_used": True,
        "fallback_reason": "recommendation_model_under_development",
    }


def _format_supporting_personas(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    personas: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        uuid = item.get("uuid")
        if not uuid:
            continue
        personas.append(
            {
                "uuid": str(uuid),
                "display_name": str(item.get("display_name") or ""),
                "similarity": round(float(item.get("similarity") or 0.0), 4),
            }
        )
    return personas
