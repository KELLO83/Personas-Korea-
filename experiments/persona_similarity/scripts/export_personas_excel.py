from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from neo4j import GraphDatabase

from experiments.persona_similarity.scripts.common import ensure_parent, write_json
from src.config import settings


PERSONA_QUERY = """
MATCH (p:Person)
OPTIONAL MATCH (p)-[:LIVES_IN]->(district:District)-[:IN_PROVINCE]->(province:Province)-[:IN_COUNTRY]->(country:Country)
OPTIONAL MATCH (p)-[:WORKS_AS]->(occupation:Occupation)
OPTIONAL MATCH (p)-[:EDUCATED_AT]->(education:EducationLevel)
OPTIONAL MATCH (p)-[:MAJORED_IN]->(field:Field)
OPTIONAL MATCH (p)-[:MARITAL_STATUS]->(marital:MaritalStatus)
OPTIONAL MATCH (p)-[:MILITARY_STATUS]->(military:MilitaryStatus)
OPTIONAL MATCH (p)-[:LIVES_WITH]->(family:FamilyType)
OPTIONAL MATCH (p)-[:LIVES_IN_HOUSING]->(housing:HousingType)
CALL (p) {
    OPTIONAL MATCH (p)-[:ENJOYS_HOBBY|LIKES]->(hobby:Hobby)
    RETURN collect(DISTINCT hobby.name) AS hobbies
}
CALL (p) {
    OPTIONAL MATCH (p)-->(skill:Skill)
    RETURN collect(DISTINCT skill.name) AS skills
}
CALL (p) {
    OPTIONAL MATCH (p)-[:SIMILAR_TO]->(sim:Person)
    RETURN count(sim) AS similar_to_out_count, max(sim.score) AS max_similar_to_score
}
CALL (p) {
    OPTIONAL MATCH (p)-[rel]-()
    RETURN count(rel) AS graph_degree
}
RETURN
    p.uuid AS uuid,
    p.display_name AS display_name,
    p.age AS age,
    p.age_group AS age_group,
    p.sex AS sex,
    country.name AS country,
    province.name AS province,
    district.name AS district,
    occupation.name AS occupation,
    education.name AS education,
    field.name AS field,
    marital.name AS marital_status,
    military.name AS military_status,
    family.name AS family_type,
    housing.name AS housing_type,
    p.community_id AS community_id,
    graph_degree,
    similar_to_out_count,
    max_similar_to_score,
    size([value IN hobbies WHERE value IS NOT NULL]) AS hobby_count,
    [value IN hobbies WHERE value IS NOT NULL] AS hobbies,
    size([value IN skills WHERE value IS NOT NULL]) AS skill_count,
    [value IN skills WHERE value IS NOT NULL] AS skills,
    p.persona AS persona,
    p.professional_persona AS professional_persona,
    p.sports_persona AS sports_persona,
    p.arts_persona AS arts_persona,
    p.travel_persona AS travel_persona,
    p.culinary_persona AS culinary_persona,
    p.family_persona AS family_persona,
    p.cultural_background AS cultural_background,
    p.career_goals_and_ambitions AS career_goals_and_ambitions,
    p.skills_and_expertise AS skills_and_expertise,
    p.hobbies_and_interests AS hobbies_and_interests
ORDER BY p.uuid
"""

COUNTS_QUERY = """
CALL () {
    MATCH (p:Person)
    RETURN 'Person' AS category, 'nodes' AS name, count(p) AS count
    UNION ALL
    MATCH (n:Hobby)
    RETURN 'Hobby' AS category, 'nodes' AS name, count(n) AS count
    UNION ALL
    MATCH (n:Skill)
    RETURN 'Skill' AS category, 'nodes' AS name, count(n) AS count
    UNION ALL
    MATCH (n:Occupation)
    RETURN 'Occupation' AS category, 'nodes' AS name, count(n) AS count
    UNION ALL
    MATCH (n:District)
    RETURN 'District' AS category, 'nodes' AS name, count(n) AS count
    UNION ALL
    MATCH ()-[r]->()
    RETURN 'Relationship' AS category, type(r) AS name, count(r) AS count
}
RETURN category, name, count
ORDER BY category, count DESC, name
"""


EXCEL_CELL_LIMIT = 32767


def clamp_excel_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return " | ".join(str(item) for item in value if item is not None)[:EXCEL_CELL_LIMIT]
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)[:EXCEL_CELL_LIMIT]
    if isinstance(value, str):
        return value[:EXCEL_CELL_LIMIT]
    return value


def fetch_frame(driver: Any, query: str, database: str) -> pd.DataFrame:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    with driver.session(database=database, fetch_size=1000) as session:
        result = session.run(query)
        records: list[dict[str, Any]] = []
        iterator = result
        if tqdm is not None:
            iterator = tqdm(result, desc="fetching Neo4j rows", unit="row")
        for record in iterator:
            records.append({key: clamp_excel_cell(value) for key, value in dict(record).items()})
    return pd.DataFrame(records)


def build_value_counts(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = [
        "age_group",
        "sex",
        "province",
        "district",
        "occupation",
        "education",
        "field",
        "marital_status",
        "military_status",
        "family_type",
        "housing_type",
        "community_id",
    ]
    for column in columns:
        if column not in frame.columns:
            continue
        counts = frame[column].fillna("<missing>").astype(str).value_counts(dropna=False)
        for value, count in counts.items():
            rows.append({"column": column, "value": value, "count": int(count)})
    return pd.DataFrame(rows)


def write_excel(personas: pd.DataFrame, counts: pd.DataFrame, value_counts: pd.DataFrame, metadata: dict[str, Any], output_path: Path) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        personas.to_excel(writer, sheet_name="personas", index=False)
        value_counts.to_excel(writer, sheet_name="value_counts", index=False)
        counts.to_excel(writer, sheet_name="graph_counts", index=False)
        pd.DataFrame([metadata]).to_excel(writer, sheet_name="metadata", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/persona_similarity/artifacts/datasets/current_neo4j_personas.xlsx")
    parser.add_argument("--metadata-output", default="experiments/persona_similarity/artifacts/metrics/current_neo4j_personas_export.json")
    parser.add_argument("--database", default=settings.NEO4J_DATABASE)
    args = parser.parse_args()

    start_time = time.perf_counter()
    output_path = ensure_parent(args.output)
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    try:
        personas = fetch_frame(driver, PERSONA_QUERY, args.database)
        counts = fetch_frame(driver, COUNTS_QUERY, args.database)
    finally:
        driver.close()

    value_counts = build_value_counts(personas)
    metadata = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "database": args.database,
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "persona_rows": int(len(personas)),
        "columns": list(personas.columns),
        "export_seconds": time.perf_counter() - start_time,
    }
    write_excel(personas, counts, value_counts, metadata, output_path)
    write_json(args.metadata_output, metadata)


if __name__ == "__main__":
    main()
