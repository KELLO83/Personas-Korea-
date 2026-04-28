from fastapi import APIRouter, Query
from neo4j import GraphDatabase

from src.api.exceptions import BadRequestException, NotFoundException
from src.api.schemas import GraphEdge, GraphNode, SubgraphResponse
from src.config import settings
from src.graph.subgraph_queries import (
    PERSON_EXISTS_QUERY,
    SUBGRAPH_DEPTH1_QUERY,
    SUBGRAPH_DEPTH2_QUERY,
    SUBGRAPH_DEPTH3_QUERY,
)

router = APIRouter(prefix="/api", tags=["graph"])


def get_neo4j_driver():  # noqa: ANN201
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def _node_id(labels: list[str], record: dict) -> str:  # type: ignore[type-arg]
    label = labels[0] if labels else "Unknown"
    if label == "Person":
        return f"person_{record.get('n_uuid') or record.get('other_uuid', '')}"
    if label == "District":
        return f"district_{record.get('n_key', '')}"
    name = record.get("n_name") or record.get("entity_name", "")
    prefix = label.lower().replace(" ", "")
    return f"{prefix}_{name}"


@router.get("/graph/subgraph/{uuid}", response_model=SubgraphResponse)
def subgraph(
    uuid: str,
    depth: int = Query(default=1, ge=1, le=3),
    include_similar: bool = Query(default=False),
    max_nodes: int = Query(default=50, ge=1, le=200),
) -> SubgraphResponse:
    if depth > 3:
        raise BadRequestException("depth는 최대 3까지 지원합니다.")

    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            exists_record = session.run(PERSON_EXISTS_QUERY, uuid=uuid).single()
            if not exists_record:
                raise NotFoundException("해당 UUID의 페르소나를 찾을 수 없습니다.")

            depth1_records = [
                dict(r) for r in session.run(
                    SUBGRAPH_DEPTH1_QUERY,
                    uuid=uuid,
                    include_similar=include_similar,
                )
            ]

            depth2_records: list[dict[str, object]] = []
            depth3_records: list[dict[str, object]] = []
            if depth >= 2:
                remaining = max_nodes - 1 - len(depth1_records)
                max_secondary = max(remaining, 0)
                depth2_records = [
                    dict(r) for r in session.run(
                        SUBGRAPH_DEPTH2_QUERY,
                        uuid=uuid,
                        max_secondary=max_secondary,
                    )
                ]
            if depth >= 3:
                remaining = max_nodes - 1 - len(depth1_records) - len(depth2_records)
                max_tertiary = max(remaining, 0)
                depth3_records = [
                    dict(r) for r in session.run(
                        SUBGRAPH_DEPTH3_QUERY,
                        uuid=uuid,
                        max_tertiary=max_tertiary,
                    )
                ]
    finally:
        driver.close()

    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []

    center_label: str | None = None
    if depth1_records:
        center_label = str(depth1_records[0].get("center_label") or "")
    center_node_id = f"person_{uuid}"
    nodes[center_node_id] = GraphNode(
        id=center_node_id,
        label=center_label or uuid,
        type="Person",
        properties={},
    )

    for rec in depth1_records:
        node_labels = rec.get("node_labels", [])
        nid = _node_id(node_labels, rec)
        label_type = node_labels[0] if node_labels else "Unknown"

        if nid not in nodes:
            if label_type == "Person":
                props: dict[str, object] = {}
                if rec.get("n_age") is not None:
                    props["age"] = rec["n_age"]
                if rec.get("n_sex") is not None:
                    props["sex"] = rec["n_sex"]
                if rec.get("n_persona") is not None:
                    props["persona"] = rec["n_persona"]
                node_label = str(rec.get("n_display_name") or rec.get("n_name") or nid)
            else:
                props = {}
                node_label = str(rec.get("n_name") or nid)

            nodes[nid] = GraphNode(
                id=nid,
                label=node_label,
                type=label_type,
                properties=props,
            )

        rel_type = str(rec.get("rel_type", ""))
        edges.append(GraphEdge(source=center_node_id, target=nid, type=rel_type))

    if depth >= 2:
        for rec in depth2_records:
            entity_labels = rec.get("entity_labels", [])
            entity_name = str(rec.get("entity_name", ""))
            entity_label_type = entity_labels[0] if entity_labels else "Unknown"
            entity_id = _node_id(entity_labels, {"n_name": entity_name})

            if entity_id not in nodes:
                nodes[entity_id] = GraphNode(
                    id=entity_id,
                    label=entity_name,
                    type=entity_label_type,
                    properties={},
                )

            other_uuid = str(rec.get("other_uuid", ""))
            other_id = f"person_{other_uuid}"
            if other_id not in nodes:
                other_props: dict[str, object] = {}
                if rec.get("other_age") is not None:
                    other_props["age"] = rec["other_age"]
                if rec.get("other_sex") is not None:
                    other_props["sex"] = rec["other_sex"]
                nodes[other_id] = GraphNode(
                    id=other_id,
                    label=str(rec.get("other_display_name") or other_uuid),
                    type="Person",
                    properties=other_props,
                )

            rel2_type = str(rec.get("rel2_type", ""))
            edges.append(GraphEdge(source=entity_id, target=other_id, type=rel2_type))

    if depth >= 3:
        for rec in depth3_records:
            other_uuid = str(rec.get("other_uuid", ""))
            if not other_uuid:
                continue

            other_id = f"person_{other_uuid}"
            if other_id not in nodes:
                nodes[other_id] = GraphNode(
                    id=other_id,
                    label=str(rec.get("other_display_name") or other_uuid),
                    type="Person",
                    properties={},
                )

            next_labels = rec.get("next_entity_labels", [])
            next_name = str(rec.get("next_entity_name") or rec.get("next_entity_key") or "")
            if not next_name:
                continue
            next_type = next_labels[0] if next_labels else "Unknown"
            next_id = _node_id(next_labels, {"n_name": next_name, "n_key": rec.get("next_entity_key")})

            if next_id not in nodes:
                nodes[next_id] = GraphNode(
                    id=next_id,
                    label=next_name,
                    type=next_type,
                    properties={},
                )

            rel3_type = str(rec.get("rel3_type", ""))
            edges.append(GraphEdge(source=other_id, target=next_id, type=rel3_type))

    # Enforce max_nodes: keep center + first (max_nodes - 1) others
    node_list = list(nodes.values())
    if len(node_list) > max_nodes:
        kept_ids: set[str] = {center_node_id}
        for node in node_list:
            if node.id != center_node_id:
                kept_ids.add(node.id)
                if len(kept_ids) >= max_nodes:
                    break
        node_list = [n for n in node_list if n.id in kept_ids]
        edges = [e for e in edges if e.source in kept_ids and e.target in kept_ids]

    return SubgraphResponse(
        center_uuid=uuid,
        center_label=center_label,
        node_count=len(node_list),
        edge_count=len(edges),
        nodes=node_list,
        edges=edges,
    )
