import type { RagTraceListResponse, RecommendationStatusResponse } from "@/lib/api-types";
import type { Loadable } from "@/hooks/use-loadable";
import { MetricCard } from "./metric-card";

interface OperationsSectionProps {
  recommendationStatus: Loadable<RecommendationStatusResponse>;
  ragTraces: Loadable<RagTraceListResponse>;
  onRefresh: () => void;
}

export function OperationsSection({ recommendationStatus, ragTraces, onRefresh }: OperationsSectionProps) {
  const hobby = recommendationStatus.data?.hobby_recommender;
  const persona = recommendationStatus.data?.persona_similarity_recommender;
  const traceAdminDisabled = Boolean(ragTraces.error?.includes("disabled") || ragTraces.error?.includes("503"));
  const visibleTraces = ragTraces.error ? [] : (ragTraces.data?.traces ?? []);
  const traceCount = visibleTraces.length;
  const traceCaption = traceAdminDisabled
    ? "admin disabled"
    : ragTraces.data?.tracing_enabled
      ? "enabled"
      : "disabled";

  return (
    <section className="grid">
      <div className="section-toolbar">
        <div>
          <div className="eyebrow">Operations</div>
          <h2>추천 모델 상태와 RAG 관측성</h2>
        </div>
        <button className="ghost-button" onClick={onRefresh}>새로고침</button>
      </div>

      <div className="grid three">
        <MetricCard title="취미 추천" value={hobby?.status ?? "-"} caption={hobby?.score_source ?? "fallback"} loading={recommendationStatus.loading} />
        <MetricCard title="유사 페르소나" value={persona?.status ?? "-"} caption={persona?.score_source ?? "fallback"} loading={recommendationStatus.loading} />
        <MetricCard title="RAG traces" value={String(traceCount)} caption={traceCaption} loading={ragTraces.loading} />
      </div>

      {recommendationStatus.error && <div className="card error-box">{recommendationStatus.error}</div>}
      {ragTraces.error && (
        <div className={traceAdminDisabled ? "card" : "card error-box"}>
          {traceAdminDisabled
            ? "RAG trace 관리자 API는 기본 비활성화 상태입니다. 로컬/관리자 환경에서 확인하려면 RAG_TRACE_ADMIN_ENABLED=true로 켜세요."
            : ragTraces.error}
        </div>
      )}

      <div className="grid two">
        <ModelStatusCard title="취미 추천 모델" info={hobby} />
        <ModelStatusCard title="유사 페르소나 추천 모델" info={persona} />
      </div>

      <div className="card">
        <h3>RAG trace list</h3>
        <p className="muted small">기본값은 tracing disabled입니다. 환경변수로 켜면 chat/insight 요청의 route, span, latency를 확인합니다.</p>
        <div className="trace-list">
          {visibleTraces.map((trace) => (
            <div className="trace-row" key={trace.trace_id}>
              <div>
                <strong>{trace.route}</strong>
                <div className="small muted">{trace.trace_id}</div>
              </div>
              <span className={`badge ${trace.status === "error" ? "danger" : ""}`}>{trace.status}</span>
              <span className="small muted">{trace.latency_ms.toFixed(1)} ms</span>
              <span className="small muted">{trace.spans.length} spans</span>
            </div>
          ))}
          {traceCount === 0 && <p className="muted small">표시할 trace가 없습니다.</p>}
        </div>
      </div>
    </section>
  );
}

function ModelStatusCard({ title, info }: { title: string; info?: RecommendationStatusResponse["hobby_recommender"] }) {
  return (
    <div className="card">
      <h3>{title}</h3>
      <p>{info?.message ?? "상태 정보를 불러오지 못했습니다."}</p>
      <div className="meta-grid">
        <span>status</span><strong>{info?.status ?? "-"}</strong>
        <span>score_source</span><strong>{info?.score_source ?? "-"}</strong>
        <span>model_version</span><strong>{info?.model_version ?? "not promoted"}</strong>
        <span>fallback</span><strong>{info?.fallback_used ? "used" : "not used"}</strong>
      </div>
    </div>
  );
}
