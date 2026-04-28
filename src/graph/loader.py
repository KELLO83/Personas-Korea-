from collections.abc import Iterable
from typing import Any

import pandas as pd
from neo4j import GraphDatabase

from src.config import settings
from src.graph.queries import COUNT_GRAPH_QUERY, CREATE_PERSON_GRAPH_QUERY
from src.graph.schema import PERSON_PROPERTY_FIELDS, schema_queries


class GraphLoader:
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

    def create_schema(self) -> None:
        with self.driver.session(database=self.database) as session:
            for query in schema_queries():
                session.run(query)

    def load_personas(self, df: pd.DataFrame, batch_size: int = 1000) -> int:
        total = 0
        with self.driver.session(database=self.database) as session:
            for start in range(0, len(df), batch_size):
                rows = _to_graph_rows(df.iloc[start : start + batch_size])
                session.run(CREATE_PERSON_GRAPH_QUERY, rows=rows)
                total += len(rows)
        return total

    def count_nodes_by_label(self) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(COUNT_GRAPH_QUERY)
            return [dict(record) for record in result]


def _to_graph_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [_to_graph_row(row) for row in df.to_dict(orient="records")]


def _to_graph_row(row: dict[str, Any]) -> dict[str, Any]:
    province = _clean_value(row.get("province_cleaned")) or _clean_value(row.get("province"))
    district_name = _clean_value(row.get("district_cleaned")) or _clean_value(row.get("district"))
    country = _clean_value(row.get("country")) or "대한민국"
    district_key = f"{province}-{district_name}" if province and district_name else district_name

    return {
        "uuid": _clean_value(row.get("uuid")),
        "person_properties": _person_properties(row),
        "country": country,
        "province": province,
        "district_name": district_name,
        "district_key": district_key,
        "occupation": _clean_value(row.get("occupation")),
        "education_level": _clean_value(row.get("education_level")),
        "bachelors_field": _clean_value(row.get("bachelors_field")),
        "marital_status": _clean_value(row.get("marital_status")),
        "military_status": _clean_value(row.get("military_status")),
        "family_type": _clean_value(row.get("family_type")),
        "housing_type": _clean_value(row.get("housing_type")),
        "skills": _clean_list(row.get("skills_and_expertise_list")),
        "hobbies": _clean_list(row.get("hobbies_and_interests_list")),
    }


def _person_properties(row: dict[str, Any]) -> dict[str, Any]:
    properties = {}
    for field in PERSON_PROPERTY_FIELDS:
        value = _clean_value(row.get(field))
        if value is not None:
            properties[field] = value
    return properties


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return value


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
