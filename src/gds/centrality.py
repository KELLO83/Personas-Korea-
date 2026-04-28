from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, LiteralString, cast

from neo4j import GraphDatabase, Query
from neo4j.exceptions import Neo4jError

from ..config import settings
from .fastrp import DROP_GRAPH_QUERY, PERSONA_GRAPH_NAME, PROJECT_GRAPH_QUERY

VALID_CENTRALITY_METRICS = {"pagerank", "betweenness", "degree"}
CENTRALITY_STATUS_KEY = "centrality_batch"
SIMULATION_TIMEOUT_SECONDS = 10.0

GRAPH_EXISTS_QUERY = """
CALL gds.graph.exists($graph_name)
YIELD exists
RETURN exists
"""

PAGERANK_WRITE_QUERY = """
CALL gds.pageRank.write($graph_name, {
    writeProperty: 'pagerank_next',
    maxIterations: $max_iterations,
    dampingFactor: $damping_factor
})
YIELD nodePropertiesWritten, centralityDistribution, computeMillis, writeMillis
RETURN nodePropertiesWritten, centralityDistribution, computeMillis, writeMillis
"""

DEGREE_WRITE_QUERY = """
CALL gds.degree.write($graph_name, {
    writeProperty: 'degree_next'
})
YIELD nodePropertiesWritten, centralityDistribution, computeMillis, writeMillis
RETURN nodePropertiesWritten, centralityDistribution, computeMillis, writeMillis
"""

BETWEENNESS_WRITE_QUERY = """
CALL gds.betweenness.write($graph_name, {
    writeProperty: 'betweenness_next',
    samplingSize: $sampling_size,
    samplingSeed: $sampling_seed
})
YIELD nodePropertiesWritten, centralityDistribution, computeMillis, writeMillis
RETURN nodePropertiesWritten, centralityDistribution, computeMillis, writeMillis
"""

PROMOTE_PROPERTIES_QUERY = """
MATCH (p:Person)
WHERE p.pagerank_next IS NOT NULL OR p.degree_next IS NOT NULL OR p.betweenness_next IS NOT NULL
SET p.pagerank = coalesce(p.pagerank_next, p.pagerank),
    p.degree = coalesce(p.degree_next, p.degree),
    p.betweenness = coalesce(p.betweenness_next, p.betweenness),
    p.centrality_updated_at = datetime($updated_at)
REMOVE p.pagerank_next, p.degree_next, p.betweenness_next
RETURN count(p) AS promoted
"""

WRITE_STATUS_QUERY = """
MERGE (s:SystemStatus {key: $key})
SET s.status = $status,
    s.last_success_at = CASE WHEN $status = 'success' THEN datetime($timestamp) ELSE s.last_success_at END,
    s.last_failure_at = CASE WHEN $status = 'failed' THEN datetime($timestamp) ELSE s.last_failure_at END,
    s.last_error = $error,
    s.run_id = $run_id,
    s.metrics = $metrics
RETURN s.key AS key, s.status AS status, s.run_id AS run_id, s.last_success_at AS last_success_at
"""

READ_STATUS_QUERY = """
MATCH (s:SystemStatus {key: $key})
RETURN s.status AS status,
       s.run_id AS run_id,
       toString(s.last_success_at) AS last_success_at,
       s.metrics AS metrics
"""

TOP_INFLUENCE_QUERY = """
MATCH (p:Person)
WHERE p[$score_property] IS NOT NULL
  AND ($community_id IS NULL OR p.community_id = $community_id)
WITH p, p[$score_property] AS score
ORDER BY score DESC
LIMIT $limit
RETURN p.uuid AS uuid,
       p.display_name AS display_name,
       toFloat(score) AS score,
       p.community_id AS community_id
"""

SCORE_COUNT_QUERY = """
MATCH (p:Person)
WHERE p[$score_property] IS NOT NULL
RETURN count(p) AS count
"""

SUBGRAPH_SIMULATION_QUERY_TEMPLATE = """
MATCH (source:Person)
WHERE source.uuid IN $target_uuids
MATCH path = (source)-[*0..{max_depth}]-(neighbor)
WITH collect(DISTINCT neighbor) AS nodes
UNWIND nodes AS n
WITH nodes, collect(DISTINCT id(n)) AS node_ids
OPTIONAL MATCH (a)-[r]-(b)
WHERE id(a) IN node_ids AND id(b) IN node_ids
RETURN [node IN nodes | {{id: id(node), uuid: node.uuid, community_id: node.community_id}}] AS nodes,
       collect(DISTINCT {{source: id(a), target: id(b)}}) AS edges
"""


class SimulationTimeoutError(Exception):
    """Raised when removal simulation exceeds the configured query timeout."""


class CentralityService:
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

    def graph_exists(self) -> bool:
        with self.driver.session(database=self.database) as session:
            record = session.run(GRAPH_EXISTS_QUERY, graph_name=self.graph_name).single()
            return bool(record and record["exists"])

    def ensure_projection(self, recreate: bool = False) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            exists_record = session.run(GRAPH_EXISTS_QUERY, graph_name=self.graph_name).single()
            exists = bool(exists_record and exists_record["exists"])
            if exists and not recreate:
                return {"graphName": self.graph_name, "already_exists": True}
            if exists:
                session.run(DROP_GRAPH_QUERY, graph_name=self.graph_name).single()
            record = session.run(PROJECT_GRAPH_QUERY, graph_name=self.graph_name).single()
            return dict(record) if record else {}

    def write_pagerank(self, max_iterations: int = 20, damping_factor: float = 0.85) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            record = session.run(
                PAGERANK_WRITE_QUERY,
                graph_name=self.graph_name,
                max_iterations=max_iterations,
                damping_factor=damping_factor,
            ).single()
            return dict(record) if record else {}

    def write_degree(self) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            record = session.run(DEGREE_WRITE_QUERY, graph_name=self.graph_name).single()
            return dict(record) if record else {}

    def write_betweenness(self, sampling_size: int = 10000, sampling_seed: int = 42) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            record = session.run(
                BETWEENNESS_WRITE_QUERY,
                graph_name=self.graph_name,
                sampling_size=sampling_size,
                sampling_seed=sampling_seed,
            ).single()
            return dict(record) if record else {}

    def promote_next_properties(self, updated_at: str | None = None) -> int:
        timestamp = updated_at or _utc_now_iso()
        with self.driver.session(database=self.database) as session:
            record = session.run(PROMOTE_PROPERTIES_QUERY, updated_at=timestamp).single()
            return int(record["promoted"]) if record and record["promoted"] is not None else 0

    def write_status(self, status: str, run_id: str, metrics: list[str], error: str | None = None) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            record = session.run(
                WRITE_STATUS_QUERY,
                key=CENTRALITY_STATUS_KEY,
                status=status,
                timestamp=_utc_now_iso(),
                run_id=run_id,
                metrics=metrics,
                error=error,
            ).single()
            return dict(record) if record else {}

    def read_status(self) -> dict[str, Any] | None:
        with self.driver.session(database=self.database) as session:
            record = session.run(READ_STATUS_QUERY, key=CENTRALITY_STATUS_KEY).single()
            return dict(record) if record else None

    def find_top(self, metric: str, limit: int = 10, community_id: int | None = None) -> list[dict[str, Any]]:
        if metric not in VALID_CENTRALITY_METRICS:
            raise ValueError(f"Invalid centrality metric: {metric}")
        with self.driver.session(database=self.database) as session:
            result = session.run(
                TOP_INFLUENCE_QUERY,
                score_property=metric,
                limit=limit,
                community_id=community_id,
            )
            rows = []
            for index, record in enumerate(result, start=1):
                row = dict(record)
                row["rank"] = index
                rows.append(row)
            return rows

    def has_scores(self, metric: str) -> bool:
        if metric not in VALID_CENTRALITY_METRICS:
            raise ValueError(f"Invalid centrality metric: {metric}")
        with self.driver.session(database=self.database) as session:
            record = session.run(SCORE_COUNT_QUERY, score_property=metric).single()
            return bool(record and int(record["count"]) > 0)

    def simulate_removal(
        self,
        target_uuids: list[str],
        max_depth: int = 3,
        timeout_seconds: float = SIMULATION_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        query = SUBGRAPH_SIMULATION_QUERY_TEMPLATE.format(max_depth=max_depth)
        with self.driver.session(database=self.database) as session:
            try:
                record = session.run(
                    Query(cast(LiteralString, query), timeout=timeout_seconds),
                    target_uuids=target_uuids,
                ).single()
            except Neo4jError as exc:
                if _is_timeout_error(exc):
                    raise SimulationTimeoutError("Removal simulation exceeded the 10 second limit") from exc
                raise
        if not record:
            return _empty_simulation()

        nodes = [node for node in record["nodes"] if node.get("id") is not None]
        edges = [edge for edge in record["edges"] if edge.get("source") is not None and edge.get("target") is not None]
        target_ids = {node["id"] for node in nodes if node.get("uuid") in target_uuids}

        original_connectivity = _largest_component_ratio(nodes, edges, excluded_ids=set())
        current_connectivity = _largest_component_ratio(nodes, edges, excluded_ids=target_ids)
        affected_communities = sorted(
            {
                int(node["community_id"])
                for node in nodes
                if node.get("id") in target_ids and node.get("community_id") is not None
            }
        )
        return {
            "path_found": bool(nodes),
            "original_connectivity": original_connectivity,
            "current_connectivity": current_connectivity,
            "fragmentation_increase": round(max(original_connectivity - current_connectivity, 0.0), 4),
            "affected_communities": affected_communities,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_simulation() -> dict[str, Any]:
    return {
        "path_found": False,
        "original_connectivity": 0.0,
        "current_connectivity": 0.0,
        "fragmentation_increase": 0.0,
        "affected_communities": [],
    }


def _is_timeout_error(exc: Neo4jError) -> bool:
    code = getattr(exc, "code", "") or ""
    message = str(exc).lower()
    return "timeout" in code.lower() or "timed out" in message or "timeout" in message


def _largest_component_ratio(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], excluded_ids: set[int]) -> float:
    node_ids = {int(node["id"]) for node in nodes if int(node["id"]) not in excluded_ids}
    if not node_ids:
        return 0.0

    adjacency: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
    for edge in edges:
        source = int(edge["source"])
        target = int(edge["target"])
        if source in node_ids and target in node_ids:
            adjacency[source].add(target)
            adjacency[target].add(source)

    visited: set[int] = set()
    largest = 0
    for node_id in node_ids:
        if node_id in visited:
            continue
        size = 0
        queue: deque[int] = deque([node_id])
        visited.add(node_id)
        while queue:
            current = queue.popleft()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        largest = max(largest, size)
    return round(largest / len(node_ids), 4)
