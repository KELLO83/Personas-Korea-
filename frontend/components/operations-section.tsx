import { useState } from "react";
import type {
  OperationsHealthResponse,
  OperationsReadinessResponse,
  OperationsWarningsResponse,
  RagTraceListResponse,
  RecommendationQualityResponse,
  RecommendationStatusResponse,
} from "@/lib/api-types";
import { personaApi } from "@/lib/api-client";
import type { Loadable } from "@/hooks/use-loadable";
import { compactNumber, percent, shortUuid } from "@/lib/formatters";

interface OperationsSectionProps {
  selectedUuid: string;
  recommendationStatus: Loadable<RecommendationStatusResponse>;
  recommendationQuality: Loadable<RecommendationQualityResponse>;
  operationsHealth: Loadable<OperationsHealthResponse>;
  operationsReadiness: Loadable<OperationsReadinessResponse>;
  operationsWarnings: Loadable<OperationsWarningsResponse>;
  ragTraces: Loadable<RagTraceListResponse>;
  onRefresh: () => void;
}

type ReplayStep = {
  name: string;
  status: "idle" | "ok" | "error";
  latencyMs: number | null;
  detail: string;
};

export function OperationsSection({
  selectedUuid,
  recommendationStatus,
  recommendationQuality,
  operationsHealth,
  operationsReadiness,
  operationsWarnings,
  ragTraces,
  onRefresh,
}: OperationsSectionProps) {
  const [replaySteps, setReplaySteps] = useState<ReplayStep[]>([]);
  const [replayRunning, setReplayRunning] = useState(false);
  const hobby = recommendationStatus.data?.hobby_recommender;
  const persona = recommendationStatus.data?.persona_similarity_recommender;
  const hobbyQuality = recommendationQuality.data?.targets.find((target) => target.target === "hobby");
  const personaQuality = recommendationQuality.data?.targets.find((target) => target.target === "persona_similarity");
  const traceAdminDisabled = Boolean(ragTraces.error?.includes("disabled") || ragTraces.error?.includes("503"));
  const visibleTraces = ragTraces.error ? [] : (ragTraces.data?.traces ?? []);
  const traceCount = visibleTraces.length;
  const traceNotRequested = !ragTraces.data && !ragTraces.error && !ragTraces.loading;
  const traceState = traceAdminDisabled ? "admin_disabled" : ragTraces.data?.tracing_enabled ? "enabled" : "disabled";
  const traceCaption = traceAdminDisabled
    ? "기록 비활성"
    : ragTraces.data?.tracing_enabled
      ? "기록 중"
      : "기록 꺼짐";
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

  async function runReplay() {
    setReplayRunning(true);
    setReplaySteps([]);
    const steps: Array<{ name: string; run: () => Promise<unknown> }> = [
      { name: "프로필 조회", run: () => personaApi.profile(selectedUuid) },
      { name: "관계 그래프", run: () => personaApi.graph(selectedUuid, { depth: 2, max_nodes: 24, include_similar: true, max_similar: 3 }) },
      { name: "추천 상태", run: () => personaApi.recommendationStatus() },
      { name: "품질 현황", run: () => personaApi.recommendationQuality() },
      { name: "서비스 상태", run: () => personaApi.operationsHealth() },
    ];

    for (const step of steps) {
      const startedAt = performance.now();
      try {
        await step.run();
        const latencyMs = performance.now() - startedAt;
        setReplaySteps((current) => [...current, { name: step.name, status: "ok", latencyMs, detail: "정상 응답" }]);
      } catch (error) {
        const latencyMs = performance.now() - startedAt;
        setReplaySteps((current) => [
          ...current,
          { name: step.name, status: "error", latencyMs, detail: error instanceof Error ? error.message : "failed" },
        ]);
      }
    }
    setReplayRunning(false);
  }

  return (
    <section className="ops-console">
      <div className="ops-header">
        <div>
          <div className="eyebrow">Service Operations</div>
          <h2>서비스 운영 상태와 추천 품질 관리</h2>
          <p className="muted small">운영자가 서비스 연결 상태, 데이터 준비도, 추천 품질, 대화 기록 상태를 한 화면에서 확인하는 관리 화면입니다.</p>
        </div>
        <div className="ops-actions">
          <span className={`ops-status-dot ${hasOperationsError ? "danger" : ""}`} />
          <span className="small muted">운영 모니터</span>
          <button className="ghost-button" onClick={onRefresh}>새로고침</button>
        </div>
      </div>

      <div className="ops-kpi-grid">
        <OpsKpi label="취미 추천" value={serviceStatusLabel(hobby?.status)} detail={scoreSourceLabel(hobby?.score_source)} loading={recommendationStatus.loading} />
        <OpsKpi label="유사 페르소나" value={serviceStatusLabel(persona?.status)} detail={scoreSourceLabel(persona?.score_source)} loading={recommendationStatus.loading} />
        <OpsKpi label="서비스 상태" value={serviceStatusLabel(operationsHealth.data?.status)} detail={operationsHealth.data?.neo4j.status === "ok" ? "그래프 DB 정상" : "그래프 DB 확인"} loading={operationsHealth.loading} />
        <OpsKpi label="데이터 준비도" value={readinessTotal ? `${readinessReady}/${readinessTotal}` : "-"} detail={readinessStatusLabel(operationsReadiness.data?.status)} loading={operationsReadiness.loading} />
        <OpsKpi label="상세 점검" value={String(warningCount)} detail={operationsWarnings.data?.status === "ok" ? "특이사항 없음" : "확인 필요"} loading={operationsWarnings.loading} />
        <OpsKpi label="품질 현황" value={recommendationQuality.data ? "확인 가능" : "-"} detail={recommendationQuality.error ?? "샘플 기준"} loading={recommendationQuality.loading} />
        <OpsKpi label="품질 응답 시간" value={qualityLatencyMs ? `${qualityLatencyMs.toFixed(0)}ms` : "-"} detail="추천 품질 조회" loading={recommendationQuality.loading} />
        <OpsKpi label="대화 기록" value={String(traceCount)} detail={traceCaption} loading={ragTraces.loading} />
      </div>

      {recommendationStatus.error && <div className="ops-alert danger">{recommendationStatus.error}</div>}
      {recommendationQuality.error && <div className="ops-alert danger">{recommendationQuality.error}</div>}
      {operationsHealth.error && <div className="ops-alert danger">{operationsHealth.error}</div>}
      {operationsReadiness.error && <div className="ops-alert danger">{operationsReadiness.error}</div>}
      {operationsWarnings.error && <div className="ops-alert danger">{operationsWarnings.error}</div>}
      {recommendationQuality.data && <div className="ops-alert">{recommendationQuality.data.dashboard_policy}</div>}

      <ReplayPanel steps={replaySteps} running={replayRunning} selectedUuid={selectedUuid} onRun={runReplay} />

      <div className="ops-panel-grid">
        <SystemHealthPanel health={operationsHealth} />
        <DataReadinessPanel readiness={operationsReadiness} />
      </div>

      <ModelSnapshotPanel status={recommendationStatus} quality={recommendationQuality} />

      <WarningsPanel warnings={operationsWarnings} />

      <div className="ops-panel-grid">
        <QualityCard title="취미 추천 품질" target={hobbyQuality} loading={recommendationQuality.loading} />
        <QualityCard title="유사 페르소나 품질" target={personaQuality} loading={recommendationQuality.loading} />
      </div>

      {ragTraces.error && (
        <div className={traceAdminDisabled ? "ops-alert" : "ops-alert danger"}>
          {traceAdminDisabled
            ? "대화 기록 상세 조회가 현재 꺼져 있습니다. 운영 검수가 필요할 때 관리자 설정에서 기록 조회를 켜고 새로고침하세요."
            : ragTraces.error}
        </div>
      )}
      {traceNotRequested && (
        <div className="ops-alert">
          대화 기록 상세 조회는 기본으로 자동 실행하지 않습니다. 운영 검수 시 관리자 설정을 켠 뒤 새로고침하세요.
        </div>
      )}

      <div className="ops-panel-grid">
        <ModelStatusCard title="취미 추천 운영 상태" info={hobby} />
        <ModelStatusCard title="유사 페르소나 운영 상태" info={persona} />
      </div>

      <div className="ops-panel">
        <div className="ops-panel-title">
          <div>
            <h3>대화 기록 흐름</h3>
            <p className="small muted">최근 대화 기록 {traceCount}건 · {traceCaption}</p>
          </div>
          <span className={`badge ${traceState === "enabled" ? "" : "danger"}`}>{traceCaption}</span>
        </div>
        <div className="ops-table">
          <div className="ops-table-row head">
            <span>대화 경로</span>
            <span>상태</span>
            <span>응답 시간</span>
            <span>단계</span>
          </div>
          {visibleTraces.map((trace) => (
            <div className="ops-table-row" key={trace.trace_id}>
              <div>
                <strong>{trace.route}</strong>
                <div className="small muted">{trace.trace_id}</div>
              </div>
              <span className={`badge ${trace.status === "error" ? "danger" : ""}`}>{trace.status === "error" ? "오류" : "정상"}</span>
              <span className="small muted">{trace.latency_ms.toFixed(1)} ms</span>
              <span className="small muted">{trace.spans.length}단계</span>
            </div>
          ))}
          {traceCount === 0 && <p className="muted small">표시할 대화 기록이 없습니다.</p>}
        </div>
      </div>
    </section>
  );
}

function ReplayPanel({
  steps,
  running,
  selectedUuid,
  onRun,
}: {
  steps: ReplayStep[];
  running: boolean;
  selectedUuid: string;
  onRun: () => void;
}) {
  return (
    <div className="ops-panel">
      <div className="ops-panel-title">
        <div>
          <h3>운영 점검 실행</h3>
          <p className="small muted">선택 대상 기준 주요 화면이 정상 응답하는지 빠르게 확인합니다.</p>
        </div>
        <button className="ghost-button" type="button" disabled={running} onClick={onRun}>{running ? "점검 중" : "점검 실행"}</button>
      </div>
      <p className="small muted">선택 대상 {shortUuid(selectedUuid)}</p>
      <div className="replay-list">
        {steps.map((step) => (
          <div className="replay-step" key={`${step.name}-${step.latencyMs}`}>
            <span className={`badge ${step.status === "error" ? "danger" : ""}`}>{step.status === "error" ? "오류" : "정상"}</span>
            <strong>{step.name}</strong>
            <span className="small muted">{step.latencyMs === null ? "-" : `${step.latencyMs.toFixed(1)}ms`}</span>
            <span className="small muted">{step.detail}</span>
          </div>
        ))}
        {steps.length === 0 && <p className="muted small">점검을 실행하면 주요 화면별 응답 시간과 실패 위치가 표시됩니다.</p>}
      </div>
    </div>
  );
}

function ModelSnapshotPanel({
  status,
  quality,
}: {
  status: Loadable<RecommendationStatusResponse>;
  quality: Loadable<RecommendationQualityResponse>;
}) {
  const hobby = status.data?.hobby_recommender;
  const persona = status.data?.persona_similarity_recommender;
  const hobbyQuality = quality.data?.targets.find((target) => target.target === "hobby");
  const personaQuality = quality.data?.targets.find((target) => target.target === "persona_similarity");

  return (
    <div className="ops-panel">
      <div className="ops-panel-title">
        <div>
          <h3>추천 서비스 스냅샷</h3>
          <p className="small muted">현재 운영 중인 추천 방식과 대체 상태를 비교합니다.</p>
        </div>
        <span className="badge">운영 비교</span>
      </div>
      <div className="model-snapshot-grid">
        <ModelSnapshotColumn title="취미 추천" status={serviceStatusLabel(hobby?.status)} scoreSource={scoreSourceLabel(hobby?.score_source)} metric={snapshotMetric(hobbyQuality)} />
        <ModelSnapshotColumn title="유사 페르소나" status={serviceStatusLabel(persona?.status)} scoreSource={scoreSourceLabel(persona?.score_source)} metric={snapshotMetric(personaQuality)} />
        <ModelSnapshotColumn title="대체 추천" status="사용 가능" scoreSource="그래프 기반" metric="무중단 대체" />
        <ModelSnapshotColumn title="후보 모델" status="준비 중" scoreSource="운영 미적용" metric="승격 근거 대기" />
      </div>
    </div>
  );
}

function ModelSnapshotColumn({ title, status, scoreSource, metric }: { title: string; status: string; scoreSource: string; metric: string }) {
  return (
    <div className="model-snapshot-cell">
      <span className="small muted">{title}</span>
      <strong>{status}</strong>
      <p className="small muted">{scoreSource}</p>
      <span className="pill">{metric}</span>
    </div>
  );
}

function SystemHealthPanel({ health }: { health: Loadable<OperationsHealthResponse> }) {
  const data = health.data;
  return (
    <div className="ops-panel">
      <div className="ops-panel-title">
        <div>
          <h3>서비스 연결 상태</h3>
          <p className="small muted">서비스 API와 그래프 데이터베이스 연결 상태</p>
        </div>
        <span className={`badge ${data?.status === "ok" ? "" : "danger"}`}>{health.loading ? "확인 중" : serviceStatusLabel(data?.status)}</span>
      </div>
      {data && (
        <div className="ops-meta-grid">
          <span>API</span><strong>{serviceStatusLabel(data.api.status)}</strong>
          <span>그래프 DB</span><strong>{serviceStatusLabel(data.neo4j.status)} · {data.neo4j.latency_ms.toFixed(1)}ms</strong>
          <span>페르소나</span><strong>{compactNumber(data.total_personas)}</strong>
          <span>관계</span><strong>{compactNumber(data.total_relationships)}</strong>
          <span>점검 시각</span><strong>{new Date(data.generated_at).toLocaleTimeString("ko-KR")}</strong>
        </div>
      )}
      {!data && <p className="muted small">{health.loading ? "서비스 상태를 확인 중입니다." : "서비스 상태 데이터가 없습니다."}</p>}
    </div>
  );
}

function snapshotMetric(target?: RecommendationQualityResponse["targets"][number]) {
  if (!target) return "snapshot 없음";
  const coverage = target.metrics.find((metric) => metric.name === "coverage");
  return coverage ? `커버리지 ${percent(coverage.value)}` : `샘플 ${compactNumber(target.sample_size)}`;
}

function DataReadinessPanel({ readiness }: { readiness: Loadable<OperationsReadinessResponse> }) {
  return (
    <div className="ops-panel">
      <div className="ops-panel-title">
        <div>
          <h3>데이터 준비도</h3>
          <p className="small muted">추천과 그래프 화면에 필요한 데이터가 충분한지 확인합니다.</p>
        </div>
        <span className={`badge ${readiness.data?.status === "ready" ? "" : "danger"}`}>{readiness.loading ? "확인 중" : readinessStatusLabel(readiness.data?.status)}</span>
      </div>
      <div className="bar-list">
        {readiness.data?.metrics.map((metric) => (
          <div className="bar-row" key={metric.name}>
            <span>{readinessMetricLabel(metric.name)}</span>
            <span className="bar-track"><span className="bar-fill" style={{ width: `${Math.min(metric.ratio, 1) * 100}%` }} /></span>
            <span className="small muted">{percent(metric.ratio)}</span>
          </div>
        ))}
      </div>
      {readiness.data?.metrics.map((metric) => (
        <p className="small muted" key={`${metric.name}-detail`}>{readinessMetricLabel(metric.name)}: {operatorReadinessDetail(metric.detail)}</p>
      ))}
      {!readiness.data && <p className="muted small">{readiness.loading ? "데이터 준비도를 확인 중입니다." : "데이터 준비도 정보가 없습니다."}</p>}
    </div>
  );
}

function WarningsPanel({ warnings }: { warnings: Loadable<OperationsWarningsResponse> }) {
  const items = warnings.data?.warnings ?? [];
  return (
    <div className="ops-panel">
      <div className="ops-panel-title">
        <div>
          <h3>운영 상세 점검</h3>
          <p className="small muted">서비스 운영 중 참고할 데이터/추천/대화 기록 점검 항목입니다.</p>
        </div>
        <span className={`badge ${items.some((item) => item.severity !== "info") ? "danger" : ""}`}>{warnings.loading ? "확인 중" : `${items.length}건`}</span>
      </div>
      <div className="ops-table">
        <div className="ops-table-row head warnings">
          <span>점검 항목</span>
          <span>중요도</span>
          <span>권장 확인</span>
        </div>
        {items.map((item) => {
          const display = operatorWarningCopy(item.code, item.title, item.detail, item.action);
          return (
            <div className="ops-table-row warnings" key={item.code}>
              <div>
                <strong>{display.title}</strong>
                <div className="small muted">{display.impact}</div>
              </div>
              <span className={`badge ${item.severity === "info" ? "" : "danger"}`}>{severityLabel(item.severity === "info" ? "info" : "warning")}</span>
              <span className="small muted">{display.action}</span>
            </div>
          );
        })}
        {!warnings.loading && items.length === 0 && <p className="muted small">현재 표시할 운영 점검 항목이 없습니다.</p>}
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
        <span className="badge">{scoreSourceLabel(target?.score_source)}</span>
      </div>
      {loading && <p className="muted small">품질 지표를 집계하는 중입니다.</p>}
      {!loading && !target && <p className="muted small">품질 스냅샷이 없습니다.</p>}
      {target && (
        <>
          <div className="ops-metric-grid">
            <QualityMetric title="커버리지" value={formatMetric(coverage)} caption={`대상 ${compactNumber(target.catalog_size)}`} />
            <QualityMetric title="다양성" value={formatMetric(diversity)} caption={`샘플 ${compactNumber(target.sample_size)}`} />
            <QualityMetric title="쏠림" value={formatMetric(hub)} caption="낮을수록 좋음" />
            <QualityMetric title="약한 근거" value={formatMetric(weak)} caption="낮을수록 좋음" />
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
            <summary>상세 지표</summary>
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
        <span className={`badge ${info?.fallback_used ? "danger" : ""}`}>{info?.fallback_used ? "대체 운영" : "운영 모델"}</span>
      </div>
      <p>{info?.message ?? "상태 정보를 불러오지 못했습니다."}</p>
      <div className="ops-meta-grid">
        <span>상태</span><strong>{serviceStatusLabel(info?.status)}</strong>
        <span>추천 방식</span><strong>{scoreSourceLabel(info?.score_source)}</strong>
        <span>모델 버전</span><strong>{info?.model_version ?? "운영 미적용"}</strong>
        <span>대체 사유</span><strong>{info?.fallback_reason ? "운영 모델 미확정" : "-"}</strong>
      </div>
    </div>
  );
}

function severityLabel(severity: "info" | "warning") {
  if (severity === "warning") return "확인";
  return "정보";
}

function serviceStatusLabel(status?: string | null) {
  if (!status) return "-";
  if (status === "ok" || status === "ready" || status === "available" || status === "promoted") return "정상";
  if (status === "degraded" || status === "warning") return "확인 필요";
  if (status === "fallback") return "대체 운영";
  if (status === "pending" || status === "under_development") return "준비 중";
  return status;
}

function readinessStatusLabel(status?: string | null) {
  if (!status) return "준비도";
  if (status === "ready" || status === "ok") return "준비 완료";
  if (status === "warning" || status === "degraded") return "확인 필요";
  return status;
}

function scoreSourceLabel(source?: string | null) {
  if (!source) return "-";
  if (source.includes("fallback")) return "안정 대체 방식";
  if (source.includes("model") || source.includes("promoted")) return "운영 모델";
  if (source.includes("graph") || source.includes("rule")) return "그래프 기반";
  if (source.includes("sample")) return "샘플 기준";
  return source;
}

function readinessMetricLabel(name: string) {
  const normalized = name.toLowerCase();
  if (normalized.includes("similar")) return "유사 관계";
  if (normalized.includes("community")) return "커뮤니티";
  if (normalized.includes("hobby")) return "취미 데이터";
  if (normalized.includes("skill")) return "스킬 데이터";
  if (normalized.includes("occupation")) return "직업 데이터";
  return name;
}

function operatorReadinessDetail(detail: string) {
  if (detail.includes("0/") || detail.includes("0 of")) return "현재 활용 가능한 데이터가 부족합니다.";
  return detail.replaceAll("relationship", "관계").replaceAll("coverage", "준비율");
}

function operatorWarningCopy(code: string, title: string, detail: string, action: string) {
  if (code.includes("skill") || title.includes("HAS_SKILL")) {
    return {
      title: "스킬 데이터 표시 방식 확인",
      impact: "스킬 관련 통계나 추천 근거가 일부 화면에서 비어 보일 수 있습니다.",
      action: "스킬 데이터가 최신 그래프 기준으로 연결되어 있는지 확인하세요.",
      source: "데이터 품질",
    };
  }
  if (code.includes("centrality") || detail.includes("pagerank") || detail.includes("degree")) {
    return {
      title: "중심성 지표 갱신 필요",
      impact: "대표 페르소나나 소모임 리더 선정 정확도가 낮아질 수 있습니다.",
      action: "그래프 분석 배치가 최근에 완료됐는지 확인하세요.",
      source: "추천 품질",
    };
  }
  if (code.includes("rag") || title.toLowerCase().includes("rag")) {
    return {
      title: "대화 기록 상세 조회 확인",
      impact: "대화형 탐색 장애 원인을 화면에서 바로 추적하기 어렵습니다.",
      action: "운영 검수 시 대화 기록 상세 조회 설정을 켜세요.",
      source: "대화 기록",
    };
  }
  return {
    title,
    impact: detail,
    action,
    source: "운영 점검",
  };
}
