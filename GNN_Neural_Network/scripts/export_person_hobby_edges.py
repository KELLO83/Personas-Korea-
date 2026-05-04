from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from neo4j import GraphDatabase, Query

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402


EXPORT_QUERY = """
MATCH (p:Person)-[:LIKES]->(h)
WHERE p.uuid IS NOT NULL AND h.name IS NOT NULL
RETURN DISTINCT p.uuid AS person_uuid, h.name AS hobby_name
ORDER BY person_uuid, hobby_name
"""

CONTEXT_EXPORT_QUERY = """
MATCH (p:Person)
WHERE p.uuid IS NOT NULL
OPTIONAL MATCH (p)-[:WORKS_AS]->(occ:Occupation)
OPTIONAL MATCH (p)-[:LIVES_IN]->(d:District)
OPTIONAL MATCH (d)-[:IN_PROVINCE]->(prov:Province)
OPTIONAL MATCH (p)-[:LIVES_WITH]->(ft:FamilyType)
OPTIONAL MATCH (p)-[:LIVES_IN_HOUSING]->(ht:HousingType)
OPTIONAL MATCH (p)-[:EDUCATED_AT]->(edu:EducationLevel)
RETURN
  p.uuid AS person_uuid,
  p.age AS age,
  p.age_group AS age_group,
  p.sex AS sex,
  collect(DISTINCT occ.name)[0] AS occupation,
  collect(DISTINCT d.name)[0] AS district,
  collect(DISTINCT prov.name)[0] AS province,
  collect(DISTINCT ft.name)[0] AS family_type,
  collect(DISTINCT ht.name)[0] AS housing_type,
  collect(DISTINCT edu.name)[0] AS education_level,
  p.persona AS persona_text,
  p.professional_persona AS professional_text,
  p.sports_persona AS sports_text,
  p.arts_persona AS arts_text,
  p.travel_persona AS travel_text,
  p.culinary_persona AS culinary_text,
  p.family_persona AS family_text,
  p.hobbies_and_interests AS hobbies_text,
  p.skills_and_expertise AS skills_text,
  p.career_goals_and_ambitions AS career_goals,
  p.embedding_text AS embedding_text
ORDER BY person_uuid
"""

CONTEXT_FIELDS = [
    "person_uuid",
    "age",
    "age_group",
    "sex",
    "occupation",
    "district",
    "province",
    "family_type",
    "housing_type",
    "education_level",
    "persona_text",
    "professional_text",
    "sports_text",
    "arts_text",
    "travel_text",
    "culinary_text",
    "family_text",
    "hobbies_text",
    "skills_text",
    "career_goals",
    "embedding_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Person-Hobby edges for offline LightGCN training.")
    parser.add_argument("--output", type=Path, default=Path("GNN_Neural_Network/data/person_hobby_edges.csv"))
    parser.add_argument("--context-output", type=Path, default=Path("GNN_Neural_Network/data/person_context.csv"))
    parser.add_argument("--skip-context", action="store_true", help="Only export Person-Hobby edges.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke-test exports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.context_output.parent.mkdir(parents=True, exist_ok=True)
    query = EXPORT_QUERY
    context_query = CONTEXT_EXPORT_QUERY
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        query += "\nLIMIT $limit"
        context_query += "\nLIMIT $limit"

    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            # Edge 데이터 내보내기 (쿼리 객체 대신 직접 쿼리 문자열과 파라미터 사용)
            edge_result = session.run(query, limit=args.limit)
            edge_records = list(edge_result)
            with args.output.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["person_uuid", "hobby_name"])
                writer.writeheader()
                for record in edge_records:
                    writer.writerow({"person_uuid": record["person_uuid"], "hobby_name": record["hobby_name"]})
            edge_count = len(edge_records)

            context_count = 0
            if not args.skip_context:
                # Context 데이터 내보내기
                context_result = session.run(context_query, limit=args.limit)
                context_records = list(context_result)
                with args.context_output.open("w", encoding="utf-8", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=CONTEXT_FIELDS)
                    writer.writeheader()
                    for record in context_records:
                        writer.writerow({field: record[field] if record[field] is not None else "" for field in CONTEXT_FIELDS})
                context_count = len(context_records)
    finally:
        driver.close()
    print(f"Exported {edge_count:,} Person-Hobby edges to {args.output}")
    if not args.skip_context:
        print(f"Exported {context_count:,} person context rows to {args.context_output}")


if __name__ == "__main__":
    main()
