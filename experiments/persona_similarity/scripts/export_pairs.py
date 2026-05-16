from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from neo4j import GraphDatabase

from experiments.persona_similarity.scripts.common import ensure_parent, load_config, mark_cache_hit, should_use_cache, stable_json_hash, write_json
from src.config import settings


EXPORT_QUERY = """
MATCH (source:Person)-[sim:SIMILAR_TO]->(target:Person)
WITH source, target, sim
ORDER BY source.uuid, sim.score DESC, target.uuid
WITH source, collect({target: target, score: sim.score})[..$candidate_top_n] AS candidates
UNWIND candidates AS candidate
WITH source, candidate.target AS target, candidate.score AS fastrp_score
OPTIONAL MATCH (source)-[:LIVES_IN]->(source_district:District)-[:IN_PROVINCE]->(source_province:Province)
OPTIONAL MATCH (target)-[:LIVES_IN]->(target_district:District)-[:IN_PROVINCE]->(target_province:Province)
OPTIONAL MATCH (source)-[:WORKS_AS]->(source_occupation:Occupation)
OPTIONAL MATCH (target)-[:WORKS_AS]->(target_occupation:Occupation)
OPTIONAL MATCH (source)-[:EDUCATED_AT]->(source_education:EducationLevel)
OPTIONAL MATCH (target)-[:EDUCATED_AT]->(target_education:EducationLevel)
OPTIONAL MATCH (source)-[:MAJORED_IN]->(source_field:Field)
OPTIONAL MATCH (target)-[:MAJORED_IN]->(target_field:Field)
OPTIONAL MATCH (source)-[:MARITAL_STATUS]->(source_marital:MaritalStatus)
OPTIONAL MATCH (target)-[:MARITAL_STATUS]->(target_marital:MaritalStatus)
OPTIONAL MATCH (source)-[:LIVES_WITH]->(source_family:FamilyType)
OPTIONAL MATCH (target)-[:LIVES_WITH]->(target_family:FamilyType)
OPTIONAL MATCH (source)-[:LIVES_IN_HOUSING]->(source_housing:HousingType)
OPTIONAL MATCH (target)-[:LIVES_IN_HOUSING]->(target_housing:HousingType)
CALL (source, target) {
    OPTIONAL MATCH (source)-[:ENJOYS_HOBBY|LIKES]->(shared_hobby:Hobby)<-[:ENJOYS_HOBBY|LIKES]-(target)
    RETURN collect(DISTINCT shared_hobby.name) AS shared_hobbies
}
CALL (source, target) {
    OPTIONAL MATCH (source)-->(shared_skill:Skill)<--(target)
    RETURN collect(DISTINCT shared_skill.name) AS shared_skills
}
RETURN
    source.uuid AS source_uuid,
    target.uuid AS target_uuid,
    fastrp_score,
    source.age AS source_age,
    target.age AS target_age,
    source.age_group AS source_age_group,
    target.age_group AS target_age_group,
    source.sex AS source_sex,
    target.sex AS target_sex,
    source_province.name AS source_province,
    target_province.name AS target_province,
    source_district.name AS source_district,
    target_district.name AS target_district,
    source_occupation.name AS source_occupation,
    target_occupation.name AS target_occupation,
    source_education.name AS source_education,
    target_education.name AS target_education,
    source_field.name AS source_field,
    target_field.name AS target_field,
    source_marital.name AS source_marital,
    target_marital.name AS target_marital,
    source_family.name AS source_family,
    target_family.name AS target_family,
    source_housing.name AS source_housing,
    target_housing.name AS target_housing,
    source.community_id AS source_community_id,
    target.community_id AS target_community_id,
    [hobby IN shared_hobbies WHERE hobby IS NOT NULL] AS shared_hobbies,
    [skill IN shared_skills WHERE skill IS NOT NULL] AS shared_skills
LIMIT coalesce($export_limit, 1000000000)
"""


def export_pairs(config: dict[str, Any]) -> pd.DataFrame:
    neo4j_config = config["neo4j"]
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    try:
        with driver.session(database=neo4j_config.get("database", settings.NEO4J_DATABASE)) as session:
            rows = [
                dict(record)
                for record in session.run(
                    EXPORT_QUERY,
                    candidate_top_n=int(neo4j_config["candidate_top_n"]),
                    export_limit=neo4j_config.get("export_limit"),
                )
            ]
    finally:
        driver.close()

    for row in rows:
        row["shared_hobbies"] = json.dumps(row.get("shared_hobbies") or [], ensure_ascii=False)
        row["shared_skills"] = json.dumps(row.get("shared_skills") or [], ensure_ascii=False)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/persona_similarity/configs/lightgbm_reranker.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    cache_metadata = {
        "stage": "export_pairs",
        "config_hash": stable_json_hash({"neo4j": config["neo4j"], "query": EXPORT_QUERY}),
        "candidate_top_n": config["neo4j"]["candidate_top_n"],
        "export_limit": config["neo4j"].get("export_limit"),
    }
    use_cache, cache_reason = should_use_cache(config["paths"]["candidate_pairs"], config["paths"]["export_status"], cache_metadata, args.force)
    if use_cache:
        mark_cache_hit(config["paths"]["export_status"], cache_metadata, config["paths"]["candidate_pairs"])
        return

    start_time = time.perf_counter()
    df = export_pairs(config)
    export_seconds = time.perf_counter() - start_time
    output_path = ensure_parent(config["paths"]["candidate_pairs"])
    df.to_parquet(output_path, index=False)
    write_json(
        config["paths"]["export_status"],
        {
            "rows": int(len(df)),
            **cache_metadata,
            "cache_hit": False,
            "cache_reason": cache_reason,
            "source_count": int(df["source_uuid"].nunique()) if not df.empty else 0,
            "target_count": int(df["target_uuid"].nunique()) if not df.empty else 0,
            "candidate_top_n": config["neo4j"]["candidate_top_n"],
            "export_seconds": export_seconds,
        },
    )


if __name__ == "__main__":
    main()
