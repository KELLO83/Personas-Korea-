import type {
  OperationsHealthResponse,
  OperationsReadinessResponse,
  OperationsWarningsResponse,
  RagTraceListResponse,
  RecommendationQualityResponse,
  RecommendationStatusResponse,
} from "@/lib/api-types";
import type { Loadable } from "@/hooks/use-loadable";
import { compactNumber, percent } from "@/lib/formatters";

interface OperationsSectionProps {
  recommendationStatus: Loadable<RecommendationStatusResponse>;
  recommendationQuality: Loadable<RecommendationQualityResponse>;
  operationsHealth: Loadable<OperationsHealthResponse>;
  operationsReadiness: Loadable<OperationsReadinessResponse>;
  operationsWarnings: Loadable<OperationsWarningsResponse>;
  ragTraces: Loadable<RagTraceListResponse>;
  onRefresh: () => void;
}

export function OperationsSection({
  recommendationStatus,
  recommendationQuality,
  operationsHealth,
  operationsReadiness,
  operationsWarnings,
  ragTraces,
  onRefresh,
}: OperationsSectionProps) {
  const hobby = recommendationStatus.data?.hobby_recommender;
  const persona = recommendationStatus.data?.persona_similarity_recommender;
  const hobbyQuality = recommendationQuality.data?.targets.find((target) => target.target === "hobby");
  const personaQuality = recommendationQuality.data?.targets.find((target) => target.target === "persona_similarity");
  const traceAdminDisabled = Boolean(ragTraces.error?.includes("disabled") || ragTraces.error?.includes("503"));
  const visibleTraces = ragTraces.error ? [] : (ragTraces.data?.traces ?? []);
  const traceCount = visibleTraces.length;
  const traceNotRequested = !ragTraces.data && !ragTraces.error && !ragTraces.loading;
  const traceCaption = traceAdminDisabled
    ? "admin disabled"
    : ragTraces.data?.tracing_enabled
      ? "enabled"
      : "disabled";
  const qualityLatencyMs = recommendationQuality.data?.targets
    .flatMap((target) => target.metrics)
    .filter((metric) => metric.name === "query_latency_ms")
    .reduce((total, metric) => total + metric.value, 0);
  const warningCount = operationsWarnings.data?.warnings.length ?? 0;
  const readinessReady = operationsReadiness.data?.metrics.filter((metric) => metric.ready).length ?? 0;
  const readinessTotal = operationsReadiness.data?.metrics.length ?? 0;
  const hasOperationsError = Boolean(
    recommendationQuality.error ||
      recommendationStatus.error ||
      operationsHealth.error ||
      operationsReadiness.error ||
      operationsWarnings.error ||
      operationsHealth.data?.status === "degraded" ||
      operationsWarnings.data?.status === "warning",
  );

  return (
    <section className="ops-console">
      <div className="ops-header">
        <div>
          <div className="eyebrow">Operations</div>
          <h2>추천 모델 상태와 RAG 관측성 콘솔</h2>
          <p className="muted small">관리자/개발자용 운영 모니터링 화면입니다. 추천 fallback, 품질 스냅샷, trace 상태를 한 곳에서 확인합니다.</p>
        </div>
        <div className="ops-actions">
          <span className={`ops-status-dot ${hasOperationsError ? "danger" : ""}`} />
          <span className="small muted">local monitor</span>
          <button className="ghost-button" onClick={onRefresh}>새로고침</button>
        </div>
      </div>

      <div className="ops-kpi-grid">
        <OpsKpi label="취미 추천" value={hobby?.status ?? "-"} detail={hobby?.score_source ?? "fallback"} loading={recommendationStatus.loading} />
        <OpsKpi label="유사 페르소나" value={persona?.status ?? "-"} detail={persona?.score_source ?? "fallback"} loading={recommendationStatus.loading} />
        <OpsKpi label="System Health" value={operationsHealth.data?.status ?? "-"} detail={operationsHealth.data?.neo4j.status ?? "neo4j"} loading={operationsHealth.loading} />
        <OpsKpi label="Readiness" value={readinessTotal ? `${readinessReady}/${readinessTotal}` : "-"} detail={operationsReadiness.data?.status ?? "coverage"} loading={operationsReadiness.loading} />
        <OpsKpi label="Warnings" value={String(warningCount)} detail={operationsWarnings.data?.status ?? "schema/rag"} loading={operationsWarnings.loading} />
        <OpsKpi label="품질 스냅샷" value={recommendationQuality.data ? "ready" : "-"} detail={recommendationQuality.error ?? "sample-based"} loading={recommendationQuality.loading} />
        <OpsKpi label="품질 쿼리 시간" value={qualityLatencyMs ? `${qualityLatencyMs.toFixed(0)}ms` : "-"} detail="hobby + persona" loading={recommendationQuality.loading} />
        <OpsKpi label="RAG trace" value={String(traceCount)} detail={traceCaption} loading={ragTraces.loading} />
      </div>

      {recommendationStatus.error && <div className="ops-alert danger">{recommendationStatus.error}</div>}
      {recommendationQuality.error && <div className="ops-alert danger">{recommendationQuality.error}</div>}
      {operationsHealth.error && <div className="ops-alert danger">{operationsHealth.error}</div>}
      {operationsReadiness.error && <div className="ops-alert danger">{operationsReadiness.error}</div>}
      {operationsWarnings.error && <div className="ops-alert danger">{operationsWarnings.error}</div>}
      {recommendationQuality.data && <div className="ops-alert">{recommendationQuality.data.dashboard_policy}</div>}

      <div className="ops-panel-grid">
        <SystemHealthPanel health={operationsHealth} />
        <DataReadinessPanel readiness={operationsReadiness} />
      </div>

      <WarningsPanel warnings={operationsWarnings} />

      <div className="ops-panel-grid">
        <QualityCard title="취미 추천 품질" target={hobbyQuality} loading={recommendationQuality.loading} />
        <QualityCard title="유사 페르소나 품질" target={personaQuality} loading={recommendationQuality.loading} />
      </div>

      {ragTraces.error && (
        <div className={traceAdminDisabled ? "ops-alert" : "ops-alert danger"}>
          {traceAdminDisabled
            ? "RAG trace 관리자 API는 기본 비활성화 상태입니다. 로컬/관리자 환경에서 확인하려면 RAG_TRACE_ADMIN_ENABLED=true로 켜세요."
            : ragTraces.error}
        </div>
      )}
      {traceNotRequested && (
        <div className="ops-alert">
          RAG trace 자동 조회는 기본 비활성화 상태입니다. 필요한 경우 로컬/관리자 환경에서 설정을 켠 뒤 새로고침하세요.
        </div>
      )}

      <div className="ops-panel-grid">
        <ModelStatusCard title="취미 추천 모델" info={hobby} />
        <ModelStatusCard title="유사 페르소나 추천 모델" info={persona} />
      </div>

      <div className="ops-panel">
        <div className="ops-panel-title">
          <div>
            <h3>RAG trace stream</h3>
            <p className="small muted">trace count {traceCount} · {traceCaption}</p>
          </div>
          <span className={`badge ${traceCaption === "enabled" ? "" : "danger"}`}>{traceCaption}</span>
        </div>
        <div className="ops-table">
          <div className="ops-table-row head">
            <span>route</span>
            <span>status</span>
            <span>latency</span>
            <span>spans</span>
          </div>
          {visibleTraces.map((trace) => (
            <div className="ops-table-row" key={trace.trace_id}>
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

function SystemHealthPanel({ health }: { health: Loadable<OperationsHealthResponse> }) {
  const data = health.data;
  return (
    <div className="ops-panel">
      <div className="ops-panel-title">
        <div>
          <h3>System Health</h3>
          <p className="small muted">FastAPI / Neo4j connectivity</p>
        </div>
        <span className={`badge ${data?.status === "ok" ? "" : "danger"}`}>{health.loading ? "loading" : (data?.status ?? "-")}</span>
      </div>
      {data && (
        <div className="ops-meta-grid">
          <span>api</span><strong>{data.api.status}</strong>
          <span>neo4j</span><strong>{data.neo4j.status} · {data.neo4j.latency_ms.toFixed(1)}ms</strong>
          <span>personas</span><strong>{compactNumber(data.total_personas)}</strong>
          <span>relationships</span><strong>{compactNumber(data.total_relationships)}</strong>
          <span>generated_at</span><strong>{new Date(data.generated_at).toLocaleTimeString("ko-KR")}</strong>
        </div>
      )}
      {!data && <p className="muted small">{health.loading ? "health check를 실행 중입니다." : "health 데이터가 없습니다."}</p>}
    </div>
  );
}

function DataReadinessPanel({ readiness }: { readiness: Loadable<OperationsReadinessResponse> }) {
  return (
    <div className="ops-panel">
      <div className="ops-panel-title">
        <div>
          <h3>Data Readiness</h3>
          <p className="small muted">추천/그래프 기능별 데이터 준비율</p>
        </div>
        <span className={`badge ${readiness.data?.status === "ready" ? "" : "danger"}`}>{readiness.loading ? "loading" : (readiness.data?.status ?? "-")}</span>
      </div>
      <div className="bar-list">
        {readiness.data?.metrics.map((metric) => (
          <div className="bar-row" key={metric.name}>
            <span>{metric.name}</span>
            <span className="bar-track"><span className="bar-fill" style={{ width: `${Math.min(metric.ratio, 1) * 100}%` }} /></span>
            <span className="small muted">{percent(metric.ratio)}</span>
          </div>
        ))}
      </div>
      {readiness.data?.metrics.map((metric) => (
        <p className="small muted" key={`${metric.name}-detail`}>{metric.name}: {metric.detail}</p>
      ))}
      {!readiness.data && <p className="muted small">{readiness.loading ? "readiness check를 실행 중입니다." : "readiness 데이터가 없습니다."}</p>}
    </div>
  );
}

function WarningsPanel({ warnings }: { warnings: Loadable<OperationsWarningsResponse> }) {
  const items = warnings.data?.warnings ?? [];
  return (
    <div className="ops-panel">
      <div className="ops-panel-title">
        <div>
          <h3>Warnings</h3>
          <p className="small muted">schema, RAG, model operation issue summary</p>
        </div>
        <span className={`badge ${items.some((item) => item.severity !== "info") ? "danger" : ""}`}>{warnings.loading ? "loading" : `${items.length} issues`}</span>
      </div>
      <div className="ops-table">
        <div className="ops-table-row head warnings">
          <span>issue</span>
          <span>severity</span>
          <span>action</span>
        </div>
        {items.map((item) => (
          <div className="ops-table-row warnings" key={item.code}>
            <div>
              <strong>{item.title}</strong>
              <div className="small muted">{item.code} · {item.detail}</div>
            </div>
            <span className={`badge ${item.severity === "info" ? "" : "danger"}`}>{item.severity}</span>
            <span className="small muted">{item.action}</span>
          </div>
        ))}
        {!warnings.loading && items.length === 0 && <p className="muted small">현재 표시할 warning이 없습니다.</p>}
      </div>
    </div>
  );
}

function OpsKpi({ label, value, detail, loading }: { label: string; value: string; detail: string; loading: boolean }) {
  return (
    <div className="ops-kpi">
      <span className="small muted">{label}</span>
      <strong>{loading ? "..." : value}</strong>
      <span className="small muted">{detail}</span>
    </div>
  );
}

function QualityCard({ title, target, loading }: { title: string; target?: RecommendationQualityResponse["targets"][number]; loading: boolean }) {
  const metrics = target?.metrics ?? [];
  const coverage = metrics.find((metric) => metric.name === "coverage");
  const diversity = metrics.find((metric) => metric.name === "diversity");
  const hub = metrics.find((metric) => metric.name === "hub_target_rate");
  const weak = metrics.find((metric) => metric.name === "weak_only_rate");
  return (
    <div className="ops-panel">
      <div className="ops-panel-title">
        <div>
          <h3>{title}</h3>
          <p className="small muted">{target ? `sample ${compactNumber(target.sample_size)} · catalog ${compactNumber(target.catalog_size)}` : "snapshot pending"}</p>
        </div>
        <span className="badge">{target?.score_source ?? "fallback"}</span>
      </div>
      {loading && <p className="muted small">품질 지표를 집계하는 중입니다.</p>}
      {!loading && !target && <p className="muted small">품질 스냅샷이 없습니다.</p>}
      {target && (
        <>
          <div className="ops-metric-grid">
            <QualityMetric title="Coverage" value={formatMetric(coverage)} caption={`catalog ${compactNumber(target.catalog_size)}`} />
            <QualityMetric title="Diversity" value={formatMetric(diversity)} caption={`sample ${compactNumber(target.sample_size)}`} />
            <QualityMetric title="Hub Target" value={formatMetric(hub)} caption="낮을수록 좋음" />
            <QualityMetric title="Weak-only" value={formatMetric(weak)} caption="낮을수록 좋음" />
          </div>
          {target.warnings.map((warning) => <div className="ops-alert danger" key={warning}>{warning}</div>)}
          <div className="bar-list" style={{ marginTop: 14 }}>
            {target.top_targets.slice(0, 6).map((item) => (
              <div className="bar-row" key={`${target.target}-${item.label}`}>
                <span>{item.label}</span>
                <span className="bar-track"><span className="bar-fill" style={{ width: `${Math.min(item.count, target.sample_size) / Math.max(target.sample_size, 1) * 100}%` }} /></span>
                <span className="small muted">{compactNumber(item.count)}</span>
              </div>
            ))}
          </div>
          <details className="small muted" style={{ marginTop: 12 }}>
            <summary>metric detail</summary>
            <pre>{JSON.stringify(target.metrics, null, 2)}</pre>
          </details>
        </>
      )}
    </div>
  );
}

function QualityMetric({ title, value, caption }: { title: string; value: string; caption: string }) {
  return (
    <div className="ops-metric-cell">
      <p className="small muted">{title}</p>
      <strong>{value}</strong>
      <p className="small muted">{caption}</p>
    </div>
  );
}

function formatMetric(metric?: RecommendationQualityResponse["targets"][number]["metrics"][number]) {
  if (!metric) return "-";
  if (metric.unit === "ms") return `${metric.value.toFixed(0)}ms`;
  return percent(metric.value);
}

function ModelStatusCard({ title, info }: { title: string; info?: RecommendationStatusResponse["hobby_recommender"] }) {
  return (
    <div className="ops-panel">
      <div className="ops-panel-title">
        <h3>{title}</h3>
        <span className={`badge ${info?.fallback_used ? "danger" : ""}`}>{info?.fallback_used ? "fallback" : "promoted"}</span>
      </div>
      <p>{info?.message ?? "상태 정보를 불러오지 못했습니다."}</p>
      <div className="ops-meta-grid">
        <span>status</span><strong>{info?.status ?? "-"}</strong>
        <span>score_source</span><strong>{info?.score_source ?? "-"}</strong>
        <span>model_version</span><strong>{info?.model_version ?? "not promoted"}</strong>
        <span>fallback_reason</span><strong>{info?.fallback_reason ?? "-"}</strong>
      </div>
    </div>
  );
}
