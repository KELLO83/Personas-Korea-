import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, KeyboardEvent, MouseEvent, PointerEvent, WheelEvent } from "react";
import type { PersonaProfileResponse, SubgraphResponse } from "@/lib/api-types";
import type { Loadable } from "@/hooks/use-loadable";
import { NODE_COLORS, NODE_EMOJIS } from "@/lib/constants";
import { nodeTypeLabel, relationLabel } from "@/lib/formatters";

type GraphNode = SubgraphResponse["nodes"][number];
type GraphEdge = SubgraphResponse["edges"][number];

type PositionedNode = {
  node: GraphNode;
  left: number;
  top: number;
  depth: number;
  relationType: string | null;
  lane: LaneKey;
  degree: number;
};

type PositionedEdge = {
  edge: GraphEdge;
  source: PositionedNode;
  target: PositionedNode;
};

type LaneKey = "center" | "hobbies" | "work" | "place" | "life" | "similar" | "outer";

type LaneConfig = {
  label: string;
  angle: number;
  directRadius: number;
  outerRadius: number;
};

type NodeDragState = {
  pointerId: number;
  nodeId: string;
  startX: number;
  startY: number;
};

const GRAPH_NODE_LIMIT = 36;
const GRAPH_EDGE_LIMIT = 90;
const EDGE_LABEL_VIS_LIMIT = 18;
const DRAG_DISTANCE_THRESHOLD = 3;
const MIN_PADDING = 5;
const MAX_PADDING = 95;

const LANE_CONFIG: Record<LaneKey, LaneConfig> = {
  center: { label: "중심", angle: 0, directRadius: 0, outerRadius: 0 },
  hobbies: { label: "취미 / 관심", angle: -92, directRadius: 25, outerRadius: 39 },
  work: { label: "직업 / 기술", angle: -8, directRadius: 27, outerRadius: 42 },
  place: { label: "지역 / 주거", angle: 182, directRadius: 26, outerRadius: 40 },
  life: { label: "생활 / 배경", angle: 92, directRadius: 25, outerRadius: 39 },
  similar: { label: "유사 페르소나", angle: 42, directRadius: 34, outerRadius: 47 },
  outer: { label: "기타 관계", angle: 138, directRadius: 29, outerRadius: 43 },
};

const RELATION_LANE: Record<string, LaneKey> = {
  ENJOYS_HOBBY: "hobbies",
  HAS_SKILL: "work",
  WORKS_AS: "work",
  LIVES_IN: "place",
  LIVES_IN_HOUSING: "place",
  LIVES_WITH: "life",
  EDUCATED_AT: "life",
  MARITAL_STATUS: "life",
  MILITARY_STATUS: "life",
  SIMILAR_TO: "similar",
};

type EdgeStyle = {
  color: string;
  width: number;
  activeWidth: number;
  opacity: number;
};

const EDGE_STYLES: Record<string, EdgeStyle> = {
  ENJOYS_HOBBY: { color: "#8cff8f", width: 0.14, activeWidth: 0.34, opacity: 0.34 },
  HAS_SKILL: { color: "#ffd166", width: 0.15, activeWidth: 0.35, opacity: 0.38 },
  WORKS_AS: { color: "#cdb2ff", width: 0.16, activeWidth: 0.36, opacity: 0.4 },
  LIVES_IN: { color: "#74d6ff", width: 0.14, activeWidth: 0.32, opacity: 0.34 },
  LIVES_IN_HOUSING: { color: "#ffb8ff", width: 0.13, activeWidth: 0.3, opacity: 0.3 },
  LIVES_WITH: { color: "#ffb1d4", width: 0.13, activeWidth: 0.3, opacity: 0.3 },
  EDUCATED_AT: { color: "#6ec8f9", width: 0.14, activeWidth: 0.31, opacity: 0.34 },
  MARITAL_STATUS: { color: "#ff9f7b", width: 0.13, activeWidth: 0.3, opacity: 0.3 },
  MILITARY_STATUS: { color: "#babfc8", width: 0.12, activeWidth: 0.28, opacity: 0.28 },
  SIMILAR_TO: { color: "#ffffff", width: 0.13, activeWidth: 0.34, opacity: 0.42 },
};

const DEFAULT_EDGE_STYLE: EdgeStyle = { color: "#ffffff", width: 0.13, activeWidth: 0.3, opacity: 0.34 };

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function resolveCenterNodeId(graph: SubgraphResponse | null): string | null {
  if (!graph) return null;

  const plainId = graph.center_uuid;
  const prefixedId = `person_${plainId}`;
  const hasPlainId = graph.nodes.some((node) => node.id === plainId);
  const hasPrefixedId = graph.nodes.some((node) => node.id === prefixedId);

  if (hasPlainId) return plainId;
  if (hasPrefixedId) return prefixedId;
  return graph.nodes[0]?.id ?? null;
}

function getEdgeStyle(type: string) {
  return EDGE_STYLES[type] ?? DEFAULT_EDGE_STYLE;
}

function getNodeEmoji(type: string): string {
  return NODE_EMOJIS[type] ?? "•";
}

function getLaneForRelation(type: string | null): LaneKey {
  if (!type) return "outer";
  return RELATION_LANE[type] ?? "outer";
}

function degreesForEdges(edges: GraphEdge[]) {
  const degree = new Map<string, number>();
  for (const edge of edges) {
    degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1);
    degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1);
  }
  return degree;
}

function extractPersonaUuid(node: GraphNode): string | null {
  if (node.type !== "Person") return null;
  const uuid = typeof node.properties.uuid === "string" ? node.properties.uuid : null;
  if (uuid) return uuid;
  return node.id.startsWith("person_") ? node.id.slice("person_".length) : node.id;
}

function semanticLayout(graph: SubgraphResponse | null, manualPositions: Record<string, { left: number; top: number }>): PositionedNode[] {
  if (!graph) return [];

  const centerNodeId = resolveCenterNodeId(graph);
  const selectedNodes = graph.nodes.slice(0, GRAPH_NODE_LIMIT);
  const nodeIds = new Set(selectedNodes.map((node) => node.id));
  const visibleEdges = graph.edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target)).slice(0, GRAPH_EDGE_LIMIT);
  const degree = degreesForEdges(visibleEdges);
  const adjacency = new Map<string, Array<{ nodeId: string; relationType: string }>>();

  for (const node of selectedNodes) {
    adjacency.set(node.id, []);
  }
  for (const edge of visibleEdges) {
    adjacency.get(edge.source)?.push({ nodeId: edge.target, relationType: edge.type });
    adjacency.get(edge.target)?.push({ nodeId: edge.source, relationType: edge.type });
  }

  const depthById = new Map<string, number>();
  const relationById = new Map<string, string>();
  if (centerNodeId) {
    depthById.set(centerNodeId, 0);
    relationById.set(centerNodeId, "CENTER");
    const queue = [centerNodeId];
    for (let front = 0; front < queue.length; front += 1) {
      const current = queue[front];
      const currentDepth = depthById.get(current) ?? 0;
      if (currentDepth >= 2) continue;

      for (const neighbor of adjacency.get(current) ?? []) {
        if (depthById.has(neighbor.nodeId)) continue;
        depthById.set(neighbor.nodeId, currentDepth + 1);
        relationById.set(neighbor.nodeId, currentDepth === 0 ? neighbor.relationType : relationById.get(current) ?? neighbor.relationType);
        queue.push(neighbor.nodeId);
      }
    }
  }

  const laneGroups = new Map<LaneKey, GraphNode[]>();
  for (const node of selectedNodes) {
    if (node.id === centerNodeId) continue;
    const lane = getLaneForRelation(relationById.get(node.id) ?? null);
    const group = laneGroups.get(lane) ?? [];
    group.push(node);
    laneGroups.set(lane, group);
  }

  const positionById = new Map<string, PositionedNode>();
  for (const node of selectedNodes) {
    const isCenter = node.id === centerNodeId;
    const relationType = relationById.get(node.id) ?? null;
    const lane = isCenter ? "center" : getLaneForRelation(relationType);
    const depth = isCenter ? 0 : depthById.get(node.id) ?? 2;
    const config = LANE_CONFIG[lane];
    let left = 50;
    let top = 50;

    if (!isCenter) {
      const group = laneGroups.get(lane) ?? [node];
      const index = Math.max(0, group.findIndex((item) => item.id === node.id));
      const count = Math.max(1, group.length);
      const spread = lane === "similar" ? 72 : 54;
      const localAngle = config.angle + (index - (count - 1) / 2) * (spread / Math.max(count, 2));
      const radius = depth <= 1 ? config.directRadius : config.outerRadius;
      const wave = (index % 2 === 0 ? -1 : 1) * Math.min(3, Math.floor(index / 2) + 1);
      left = 50 + Math.cos((localAngle * Math.PI) / 180) * (radius + wave);
      top = 50 + Math.sin((localAngle * Math.PI) / 180) * (radius + wave * 0.7);
    }

    const manual = manualPositions[node.id];
    positionById.set(node.id, {
      node,
      left: manual ? manual.left : clamp(left, MIN_PADDING, MAX_PADDING),
      top: manual ? manual.top : clamp(top, MIN_PADDING, MAX_PADDING),
      depth,
      relationType: isCenter ? null : relationType,
      lane,
      degree: degree.get(node.id) ?? 0,
    });
  }

  return selectedNodes.map((node) => positionById.get(node.id)).filter((node): node is PositionedNode => Boolean(node));
}

function getWorldPercent(event: PointerEvent, pan: { x: number; y: number }, zoom: number, canvasRect: DOMRect) {
  const leftInPixels = (event.clientX - canvasRect.left - pan.x) / zoom;
  const topInPixels = (event.clientY - canvasRect.top - pan.y) / zoom;

  return {
    left: clamp((leftInPixels / canvasRect.width) * 100, 2, 98),
    top: clamp((topInPixels / canvasRect.height) * 100, 2, 98),
  };
}

interface GraphSectionProps {
  graph: Loadable<SubgraphResponse>;
  profile: PersonaProfileResponse | null;
  onSelectPersona?: (uuid: string, label: string | null) => void;
}

export function GraphSection({ graph, profile, onSelectPersona }: GraphSectionProps) {
  const [zoom, setZoom] = useState(1);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null);
  const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null);
  const [showAllEdgeLabels, setShowAllEdgeLabels] = useState(false);
  const [manualPositions, setManualPositions] = useState<Record<string, { left: number; top: number }>>({});
  const dragStartRef = useRef<{ pointerId: number; x: number; y: number; panX: number; panY: number } | null>(null);
  const nodeDragRef = useRef<NodeDragState | null>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const panRef = useRef({ x: 0, y: 0 });
  const zoomRef = useRef(1);
  const viewportFrameRef = useRef<number | null>(null);
  const zoomFrameRef = useRef<number | null>(null);
  const manualPositionFrameRef = useRef<number | null>(null);
  const pendingManualPositionRef = useRef<{ nodeId: string; position: { left: number; top: number } } | null>(null);
  const zoomLimits = { min: 0.6, max: 2.2 };
  const centerNodeId = useMemo(() => resolveCenterNodeId(graph.data), [graph.data]);

  function applyViewportTransform() {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const { x, y } = panRef.current;
    viewport.style.transform = `translate(${x}px, ${y}px) scale(${zoomRef.current})`;
  }

  function scheduleViewportTransform() {
    if (viewportFrameRef.current !== null) return;
    viewportFrameRef.current = window.requestAnimationFrame(() => {
      viewportFrameRef.current = null;
      applyViewportTransform();
    });
  }

  function scheduleZoomDisplay() {
    if (zoomFrameRef.current !== null) return;
    zoomFrameRef.current = window.requestAnimationFrame(() => {
      zoomFrameRef.current = null;
      setZoom(zoomRef.current);
    });
  }

  useEffect(() => {
    applyViewportTransform();
    return () => {
      if (viewportFrameRef.current !== null) {
        window.cancelAnimationFrame(viewportFrameRef.current);
      }
      if (zoomFrameRef.current !== null) {
        window.cancelAnimationFrame(zoomFrameRef.current);
      }
      if (manualPositionFrameRef.current !== null) {
        window.cancelAnimationFrame(manualPositionFrameRef.current);
      }
    };
  }, []);

  const positionedNodes = useMemo(() => semanticLayout(graph.data, manualPositions), [graph.data, manualPositions]);

  const positionedEdges = useMemo(() => {
    const nodeById = new Map(positionedNodes.map((item) => [item.node.id, item]));
    return (graph.data?.edges ?? [])
      .map((edge) => {
        const source = nodeById.get(edge.source);
        const target = nodeById.get(edge.target);
        return source && target ? { edge, source, target } : null;
      })
      .filter((item): item is PositionedEdge => item !== null)
      .slice(0, GRAPH_EDGE_LIMIT);
  }, [graph.data, positionedNodes]);

  const connectedByNode = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const { edge } of positionedEdges) {
      const sourceSet = map.get(edge.source) ?? new Set<string>();
      const targetSet = map.get(edge.target) ?? new Set<string>();
      sourceSet.add(edge.target);
      targetSet.add(edge.source);
      map.set(edge.source, sourceSet);
      map.set(edge.target, targetSet);
    }
    return map;
  }, [positionedEdges]);

  const activeNodeId = focusedNodeId ?? hoveredNodeId;
  const activeAdjacency = useMemo(() => {
    if (!activeNodeId) return new Set<string>();
    return new Set([activeNodeId, ...(connectedByNode.get(activeNodeId) ?? [])]);
  }, [activeNodeId, connectedByNode]);

  const activeNodeLabel = useMemo(() => {
    if (!activeNodeId) return null;
    return positionedNodes.find((item) => item.node.id === activeNodeId)?.node.label ?? null;
  }, [positionedNodes, activeNodeId]);

  const laneSummaries = useMemo(() => {
    const counts = new Map<LaneKey, number>();
    for (const item of positionedNodes) {
      if (item.lane === "center") continue;
      counts.set(item.lane, (counts.get(item.lane) ?? 0) + 1);
    }
    return (["hobbies", "work", "place", "life", "similar", "outer"] as LaneKey[]).filter((lane) => counts.has(lane)).map((lane) => ({ lane, count: counts.get(lane) ?? 0 }));
  }, [positionedNodes]);

  const shouldRenderEdgeLabels = Boolean(activeNodeId) || showAllEdgeLabels;
  const isNodeFocusMode = focusedNodeId !== null;

  function setZoomByLevel(nextZoom: number, anchor?: { x: number; y: number }) {
    const boundedZoom = clamp(nextZoom, zoomLimits.min, zoomLimits.max);
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect || !anchor) {
      zoomRef.current = boundedZoom;
      scheduleViewportTransform();
      scheduleZoomDisplay();
      return;
    }

    const currentPan = panRef.current;
    const anchorWorldX = (anchor.x - rect.left - currentPan.x) / zoomRef.current;
    const anchorWorldY = (anchor.y - rect.top - currentPan.y) / zoomRef.current;
    panRef.current = {
      x: anchor.x - rect.left - anchorWorldX * boundedZoom,
      y: anchor.y - rect.top - anchorWorldY * boundedZoom,
    };
    zoomRef.current = boundedZoom;
    scheduleViewportTransform();
    scheduleZoomDisplay();
  }

  function resetViewport() {
    zoomRef.current = 1;
    panRef.current = { x: 0, y: 0 };
    applyViewportTransform();
    setZoom(1);
    setHoveredNodeId(null);
    setFocusedNodeId(null);
    setManualPositions({});
  }

  function zoomStep(step: number) {
    const rect = canvasRef.current?.getBoundingClientRect();
    setZoomByLevel(zoomRef.current + step, rect ? { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 } : undefined);
  }

  function startPan(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragStartRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, panX: panRef.current.x, panY: panRef.current.y };
  }

  function movePan(event: PointerEvent<HTMLDivElement>) {
    const dragStart = dragStartRef.current;
    if (!dragStart || dragStart.pointerId !== event.pointerId) return;
    panRef.current = { x: dragStart.panX + event.clientX - dragStart.x, y: dragStart.panY + event.clientY - dragStart.y };
    scheduleViewportTransform();
  }

  function endPan(event: PointerEvent<HTMLDivElement>) {
    if (dragStartRef.current?.pointerId === event.pointerId) {
      dragStartRef.current = null;
    }
  }

  function wheelZoom(event: WheelEvent<HTMLDivElement>) {
    event.preventDefault();
    setZoomByLevel(zoomRef.current + (event.deltaY > 0 ? -0.12 : 0.12), { x: event.clientX, y: event.clientY });
  }

  function handleCanvasKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      zoomStep(0.12);
    } else if (event.key === "-") {
      event.preventDefault();
      zoomStep(-0.12);
    } else if (event.key === "Escape") {
      event.preventDefault();
      setFocusedNodeId(null);
      setHoveredNodeId(null);
    } else if (event.key === "0") {
      event.preventDefault();
      resetViewport();
    }
  }

  function startNodeDrag(event: PointerEvent<HTMLDivElement>, nodeId: string) {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    nodeDragRef.current = { pointerId: event.pointerId, nodeId, startX: event.clientX, startY: event.clientY };
    setDraggingNodeId(nodeId);
    setHoveredNodeId(nodeId);
  }

  function moveNodeDrag(event: PointerEvent<HTMLDivElement>) {
    const dragState = nodeDragRef.current;
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const world = getWorldPercent(event, panRef.current, zoomRef.current, rect);
    pendingManualPositionRef.current = { nodeId: dragState.nodeId, position: world };
    if (manualPositionFrameRef.current !== null) return;
    manualPositionFrameRef.current = window.requestAnimationFrame(() => {
      manualPositionFrameRef.current = null;
      const pending = pendingManualPositionRef.current;
      if (!pending) return;
      pendingManualPositionRef.current = null;
      setManualPositions((current) => ({ ...current, [pending.nodeId]: pending.position }));
    });
  }

  function endNodeDrag(event: PointerEvent<HTMLDivElement>) {
    const dragState = nodeDragRef.current;
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    const wasDragged = Math.abs(event.clientX - dragState.startX) > DRAG_DISTANCE_THRESHOLD || Math.abs(event.clientY - dragState.startY) > DRAG_DISTANCE_THRESHOLD;
    nodeDragRef.current = null;
    event.currentTarget.releasePointerCapture(event.pointerId);
    setDraggingNodeId(null);
    if (!wasDragged) {
      setFocusedNodeId((current) => (current === dragState.nodeId ? null : dragState.nodeId));
    }
  }

  function handleNodeKeyDown(event: KeyboardEvent<HTMLDivElement>, node: GraphNode) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      setFocusedNodeId((current) => (current === node.id ? null : node.id));
    }
  }

  function handleNodeDoubleClick(event: MouseEvent<HTMLDivElement>, node: GraphNode) {
    event.preventDefault();
    event.stopPropagation();
    const uuid = extractPersonaUuid(node);
    if (!uuid || node.id === centerNodeId) return;
    setManualPositions({});
    setFocusedNodeId(null);
    setHoveredNodeId(null);
    onSelectPersona?.(uuid, node.label);
  }

  return (
    <section className="grid graph-section">
      {graph.error && <div className="card error-box">{graph.error}</div>}
      <div className="grid two">
        <div className="card">
          <h2>{graph.data?.center_label ?? profile?.display_name ?? "관계 그래프"}</h2>
          <p className="muted">노드 {graph.data?.node_count ?? 0}개 · 관계 {graph.data?.edge_count ?? 0}개</p>
        </div>
        <div className="card">
          <h3>관계 관점</h3>
          <div className="pill-row">
            {laneSummaries.map(({ lane, count }) => <span className="pill" key={lane}>{LANE_CONFIG[lane].label} {count}</span>)}
          </div>
        </div>
      </div>
      <div className="graph-stage">
        <div className="graph-toolbar" aria-label="그래프 조작 도구">
          <button className="ghost-button" type="button" onClick={() => zoomStep(-0.12)}>축소</button>
          <span className="pill">{Math.round(zoom * 100)}%</span>
          <button className="ghost-button" type="button" onClick={() => zoomStep(0.12)}>확대</button>
          <button className="ghost-button" type="button" onClick={resetViewport}>리셋</button>
          <button className="ghost-button" type="button" onClick={() => setShowAllEdgeLabels((current) => !current)} aria-pressed={showAllEdgeLabels}>
            라벨: {showAllEdgeLabels ? "상시" : "선택시"}
          </button>
          <span className="pill" aria-live="polite">{activeNodeLabel ? `${activeNodeLabel.slice(0, 11)} 탐색 중` : "관계 지도"}</span>
        </div>
        <p className="graph-help small muted">
          {isNodeFocusMode ? "선택 관계 고정 · ESC 해제 · " : "호버: 관계 강조 · 클릭: 관계 고정 · "}
          페르소나 더블클릭: 새 중심 · 드래그: 보조 위치 조정 · 휠/버튼: 확대축소
        </p>
        <div
          ref={canvasRef}
          className="graph-canvas"
          role="img"
          aria-label="페르소나와 속성 간 관계 지도"
          tabIndex={0}
          onKeyDown={handleCanvasKeyDown}
          onPointerDown={startPan}
          onPointerMove={movePan}
          onPointerUp={endPan}
          onPointerCancel={endPan}
          onPointerLeave={endPan}
          onDoubleClick={resetViewport}
          onWheel={wheelZoom}
        >
          <div ref={viewportRef} className="graph-viewport">
            <div className="graph-lane-layer" aria-hidden="true">
              {(["hobbies", "work", "place", "life", "similar"] as LaneKey[]).map((lane) => {
                const config = LANE_CONFIG[lane];
                const left = 50 + Math.cos((config.angle * Math.PI) / 180) * (config.directRadius + 8);
                const top = 50 + Math.sin((config.angle * Math.PI) / 180) * (config.directRadius + 8);
                return <span className={`graph-lane-label lane-${lane}`} key={lane} style={{ left: `${left}%`, top: `${top}%` }}>{config.label}</span>;
              })}
            </div>
            <svg className="graph-edge-layer" aria-hidden="true" viewBox="0 0 100 100" preserveAspectRatio="none">
              {positionedEdges.map(({ edge, source, target }, index) => {
                const isActive = activeNodeId !== null && activeAdjacency.has(source.node.id) && activeAdjacency.has(target.node.id);
                const isDim = activeNodeId !== null && !isActive;
                const edgeStyle = getEdgeStyle(edge.type);
                return (
                  <line
                    className={`graph-edge ${isActive ? "is-active" : ""} ${isDim ? "is-dim" : ""}`}
                    key={`${edge.source}-${edge.target}-${edge.type}-${index}`}
                    style={{
                      stroke: edgeStyle.color,
                      strokeWidth: isActive ? edgeStyle.activeWidth : edgeStyle.width,
                      strokeOpacity: isDim ? 0.08 : isActive ? 0.98 : edgeStyle.opacity,
                      strokeDasharray: edge.type === "SIMILAR_TO" || source.depth > 1 || target.depth > 1 ? "1.8 1.6" : undefined,
                    }}
                    x1={source.left}
                    y1={source.top}
                    x2={target.left}
                    y2={target.top}
                  />
                );
              })}
            </svg>
            {positionedEdges.map(({ edge, source, target }, index) => {
              if (!shouldRenderEdgeLabels) return null;
              const isActive = activeNodeId !== null && activeAdjacency.has(source.node.id) && activeAdjacency.has(target.node.id);
              if (!showAllEdgeLabels && !isActive) return null;
              if (showAllEdgeLabels && index >= EDGE_LABEL_VIS_LIMIT) return null;

              return (
                <span
                  className={`graph-edge-label ${isActive ? "is-active" : ""}`}
                  key={`label-${edge.source}-${edge.target}-${edge.type}-${index}`}
                  style={{ left: `${(source.left + target.left) / 2}%`, top: `${(source.top + target.top) / 2}%` }}
                >
                  {relationLabel(edge.type)}
                </span>
              );
            })}
            {positionedNodes.map(({ node, left, top, depth, degree }) => {
              const isCenter = centerNodeId !== null && node.id === centerNodeId;
              const isActive = activeNodeId !== null && activeAdjacency.has(node.id);
              const isDimmed = activeNodeId !== null && !isActive;
              const isDragging = draggingNodeId === node.id;
              const isFocused = focusedNodeId === node.id;
              const canRecenter = Boolean(extractPersonaUuid(node)) && !isCenter;
              const nodeColor = NODE_COLORS[node.type] ?? "var(--line-strong)";
              const nodeClass = `graph-node depth-${Math.min(depth, 2)} ${isCenter ? "center" : ""} ${isActive ? "is-active" : ""} ${isDimmed ? "is-dim" : ""} ${isDragging ? "is-dragging" : ""} ${isFocused ? "is-focused" : ""} ${canRecenter ? "can-recenter" : ""}`;
              const scale = isCenter ? 1 : clamp(0.86 + degree * 0.035, 0.86, 1.12);

              return (
                <div
                  className={nodeClass}
                  key={node.id}
                  style={{
                    left: `${left}%`,
                    top: `${top}%`,
                    borderColor: nodeColor,
                    "--node-color": nodeColor,
                    "--node-scale": scale,
                  } as CSSProperties}
                  title={`${nodeTypeLabel(node.type)} · ${node.label}${canRecenter ? " · 더블클릭으로 중심 전환" : ""}`}
                  onPointerDown={(event) => startNodeDrag(event, node.id)}
                  onPointerMove={moveNodeDrag}
                  onPointerUp={endNodeDrag}
                  onPointerCancel={endNodeDrag}
                  onPointerEnter={() => setHoveredNodeId(node.id)}
                  onPointerLeave={() => setHoveredNodeId((current) => (current === node.id ? null : current))}
                  onFocus={() => setHoveredNodeId(node.id)}
                  onBlur={() => setHoveredNodeId((current) => (current === node.id ? null : current))}
                  onDoubleClick={(event) => handleNodeDoubleClick(event, node)}
                  onKeyDown={(event) => handleNodeKeyDown(event, node)}
                  tabIndex={0}
                >
                  <span className="node-orb" aria-hidden="true"><span>{getNodeEmoji(node.type)}</span></span>
                  <strong>{node.label.slice(0, isCenter ? 18 : 14)}</strong>
                  <span className="small muted">{nodeTypeLabel(node.type)}</span>
                </div>
              );
            })}
          </div>
        </div>
        {graph.loading && <div className="card" style={{ position: "absolute", left: 24, top: 24 }}>그래프를 불러오는 중…</div>}
      </div>
    </section>
  );
}
