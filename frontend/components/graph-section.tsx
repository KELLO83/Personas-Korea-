import { useMemo, useRef, useState } from "react";
import type { PointerEvent, WheelEvent } from "react";
import type { PersonaProfileResponse, SubgraphResponse } from "@/lib/api-types";
import type { Loadable } from "@/hooks/use-loadable";
import { NODE_COLORS } from "@/lib/constants";
import { nodeTypeLabel, relationLabel } from "@/lib/formatters";

interface GraphSectionProps {
  graph: Loadable<SubgraphResponse>;
  profile: PersonaProfileResponse | null;
}

export function GraphSection({ graph, profile }: GraphSectionProps) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const dragStartRef = useRef<{ pointerId: number; x: number; y: number; panX: number; panY: number } | null>(null);

  const positionedNodes = useMemo(() => {
    const nodes = graph.data?.nodes.slice(0, 28) ?? [];
    return nodes.map((node, index) => {
      const angle = (index / Math.max(nodes.length, 1)) * Math.PI * 2;
      const radius = index === 0 ? 0 : 18 + (index % 4) * 7;
      return {
        node,
        left: 50 + Math.cos(angle) * radius,
        top: 50 + Math.sin(angle) * radius,
      };
    });
  }, [graph.data]);

  const positionedEdges = useMemo(() => {
    const nodeById = new Map(positionedNodes.map((item) => [item.node.id, item]));
    return (graph.data?.edges ?? [])
      .map((edge) => {
        const source = nodeById.get(edge.source);
        const target = nodeById.get(edge.target);
        return source && target ? { edge, source, target } : null;
      })
      .filter((item): item is NonNullable<typeof item> => item !== null)
      .slice(0, 60);
  }, [graph.data, positionedNodes]);

  function changeZoom(nextZoom: number) {
    setZoom(Math.max(0.65, Math.min(nextZoom, 1.9)));
  }

  function resetViewport() {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }

  function startPan(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragStartRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y };
  }

  function movePan(event: PointerEvent<HTMLDivElement>) {
    const dragStart = dragStartRef.current;
    if (!dragStart || dragStart.pointerId !== event.pointerId) return;
    setPan({ x: dragStart.panX + event.clientX - dragStart.x, y: dragStart.panY + event.clientY - dragStart.y });
  }

  function endPan(event: PointerEvent<HTMLDivElement>) {
    if (dragStartRef.current?.pointerId === event.pointerId) {
      dragStartRef.current = null;
    }
  }

  function wheelZoom(event: WheelEvent<HTMLDivElement>) {
    event.preventDefault();
    changeZoom(zoom + (event.deltaY > 0 ? -0.08 : 0.08));
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
          <h3>관계 타입</h3>
          <div className="pill-row">
            {[...new Set((graph.data?.edges ?? []).map((edge) => edge.type))].slice(0, 10).map((type) => <span className="pill" key={type}>{relationLabel(type)}</span>)}
          </div>
        </div>
      </div>
      <div className="graph-stage">
        <div className="graph-toolbar" aria-label="그래프 조작 도구">
          <button className="ghost-button" type="button" onClick={() => changeZoom(zoom - 0.12)}>축소</button>
          <span className="pill">{Math.round(zoom * 100)}%</span>
          <button className="ghost-button" type="button" onClick={() => changeZoom(zoom + 0.12)}>확대</button>
          <button className="ghost-button" type="button" onClick={resetViewport}>리셋</button>
        </div>
        <p className="graph-help small muted">마우스 휠로 확대/축소 · 드래그로 이동</p>
        <div
          className="graph-canvas"
          role="img"
          aria-label="페르소나와 속성 간 관계 그래프"
          onPointerDown={startPan}
          onPointerMove={movePan}
          onPointerUp={endPan}
          onPointerCancel={endPan}
          onWheel={wheelZoom}
          style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}
        >
          <svg className="graph-edge-layer" aria-hidden="true" viewBox="0 0 100 100" preserveAspectRatio="none">
            <defs>
              <linearGradient id="graph-edge-gradient" x1="0" x2="1" y1="0" y2="0">
                <stop offset="0%" stopColor="#74d6ff" stopOpacity="0.72" />
                <stop offset="100%" stopColor="#8bffb0" stopOpacity="0.46" />
              </linearGradient>
            </defs>
            {positionedEdges.map(({ edge, source, target }, index) => (
              <line
                className="graph-edge"
                key={`${edge.source}-${edge.target}-${edge.type}-${index}`}
                x1={source.left}
                y1={source.top}
                x2={target.left}
                y2={target.top}
              />
            ))}
          </svg>
          {positionedEdges.slice(0, 18).map(({ edge, source, target }, index) => (
            <span
              className="graph-edge-label"
              key={`label-${edge.source}-${edge.target}-${edge.type}-${index}`}
              style={{ left: `${(source.left + target.left) / 2}%`, top: `${(source.top + target.top) / 2}%` }}
            >
              {relationLabel(edge.type)}
            </span>
          ))}
          {positionedNodes.map(({ node, left, top }, index) => (
            <div
              className={`graph-node ${index === 0 || node.id === graph.data?.center_uuid ? "center" : ""}`}
              key={node.id}
              style={{ left: `${left}%`, top: `${top}%`, borderColor: NODE_COLORS[node.type] ?? "var(--line-strong)" }}
              title={`${nodeTypeLabel(node.type)} · ${node.label}`}
            >
              <strong>{node.label.slice(0, 16)}</strong>
              <span className="small muted">{nodeTypeLabel(node.type)}</span>
            </div>
          ))}
        </div>
        {graph.loading && <div className="card" style={{ position: "absolute", left: 24, top: 24 }}>그래프를 불러오는 중…</div>}
      </div>
    </section>
  );
}
