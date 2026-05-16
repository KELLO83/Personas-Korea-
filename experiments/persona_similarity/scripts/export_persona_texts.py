from __future__ import annotations

import argparse
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


TEXT_QUERY = """
MATCH (p:Person)
RETURN
    p.uuid AS uuid,
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
LIMIT coalesce($export_limit, 1000000000)
"""


def export_persona_texts(config: dict[str, Any]) -> pd.DataFrame:
    neo4j_config = config["neo4j"]
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    try:
        with driver.session(database=neo4j_config.get("database", settings.NEO4J_DATABASE), fetch_size=1000) as session:
            records = session.run(TEXT_QUERY, export_limit=neo4j_config.get("text_export_limit"))
            rows = [dict(record) for record in records]
    finally:
        driver.close()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/persona_similarity/configs/lightgbm_reranker.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    cache_metadata = {
        "stage": "export_persona_texts",
        "config_hash": stable_json_hash({"neo4j": config["neo4j"], "query": TEXT_QUERY}),
        "text_export_limit": config["neo4j"].get("text_export_limit"),
    }
    use_cache, cache_reason = should_use_cache(config["paths"]["persona_texts"], config["paths"]["persona_texts_status"], cache_metadata, args.force)
    if use_cache:
        mark_cache_hit(config["paths"]["persona_texts_status"], cache_metadata, config["paths"]["persona_texts"])
        return

    start_time = time.perf_counter()
    frame = export_persona_texts(config)
    output_path = ensure_parent(config["paths"]["persona_texts"])
    frame.to_parquet(output_path, index=False)
    write_json(
        config["paths"]["persona_texts_status"],
        {
            **cache_metadata,
            "cache_hit": False,
            "cache_reason": cache_reason,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "export_seconds": time.perf_counter() - start_time,
        },
    )


if __name__ == "__main__":
    main()
