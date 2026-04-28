from typing import Any

from fastapi import APIRouter, Query
from neo4j import GraphDatabase

from src.api.exceptions import BadRequestException
from src.api.schemas import PathResponse
from src.config import settings

router = APIRouter(prefix="/api", tags=["path"])

PATH_QUERY = """
MATCH (source:Person {uuid: $uuid1}), (target:Person {uuid: $uuid2})
MATCH path = shortestPath((source)-[*..4]-(target))
RETURN [node IN nodes(path) | {labels: labels(node), uuid: node.uuid, name: coalesce(node.name, node.display_name, node.uuid)}] AS nodes,
       [relationship IN relationships(path) | type(relationship)] AS relationships
"""

SHARED_NODES_QUERY = """
MATCH (source:Person {uuid: $uuid1})-->(shared)<--(target:Person {uuid: $uuid2})
WHERE NOT shared:Person
RETURN labels(shared)[0] AS type, shared.name AS name
ORDER BY type, name
"""


@router.get("/path/{uuid1}/{uuid2}", response_model=PathResponse)
def path(uuid1: str, uuid2: str, max_depth: int = Query(default=4, ge=1, le=6)) -> PathResponse:
    if uuid1 == uuid2:
        raise BadRequestException("동일한 UUID 간 경로 탐색은 지원하지 않습니다.")

    path_query = PATH_QUERY.replace("[*..4]", f"[*..{max_depth}]")
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            path_record = session.run(path_query, uuid1=uuid1, uuid2=uuid2).single()
            shared_nodes = [dict(record) for record in session.run(SHARED_NODES_QUERY, uuid1=uuid1, uuid2=uuid2)]
    finally:
        driver.close()

    if not path_record:
        return PathResponse(path_found=False, length=0, shared_nodes=shared_nodes)

    path_items = _format_path(path_record["nodes"], path_record["relationships"])
    return PathResponse(
        path_found=True,
        length=len(path_record["relationships"]),
        path=path_items,
        shared_nodes=shared_nodes,
        summary=_summarize_shared_nodes(shared_nodes),
    )


def _format_path(nodes: list[dict[str, Any]], relationships: list[str]) -> list[dict[str, Any]]:
    items = []
    for index, node in enumerate(nodes):
        label = node.get("labels", ["Node"])[0]
        name = node.get("name") or node.get("uuid", "Unknown")
        item: dict[str, Any] = {"node": f"{name} ({label})"}
        if index < len(relationships):
            item["edge"] = relationships[index]
        items.append(item)
    return items


def _summarize_shared_nodes(shared_nodes: list[dict[str, Any]]) -> str:
    if not shared_nodes:
        return "두 페르소나 사이에서 직접 공유되는 속성 노드는 발견되지 않았습니다."
    names = ", ".join(str(node.get("name")) for node in shared_nodes[:5])
    return f"두 페르소나는 {names} 항목을 공유합니다."
