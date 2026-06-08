"use client";

import { FormEvent, useEffect, useState } from "react";
import type {
  ChatMessage,
  GraphInsightsResponse,
  OperationsHealthResponse,
  OperationsReadinessResponse,
  OperationsWarningsResponse,
  PersonaProfileResponse,
  RagTraceListResponse,
  RecommendationQualityResponse,
  RecommendationStatusResponse,
  SearchResponse,
  StatsResponse,
  SubgraphResponse,
} from "@/lib/api-types";
import { personaApi } from "@/lib/api-client";
import { DEFAULT_PERSONA_UUID } from "@/lib/constants";
import { compactNumber, fullNumber, shortUuid, uuidWithName } from "@/lib/formatters";
import { chatSessionId, resetChatSessionId } from "@/lib/chat-session";
import { useLoadable } from "@/hooks/use-loadable";
import { DashboardSection } from "@/components/dashboard-section";
import { SearchFilters, SearchSection } from "@/components/search-section";
import { ProfileSection } from "@/components/profile-section";
import { GraphSection } from "@/components/graph-section";
import { ChatSection } from "@/components/chat-section";
import { InsightsSection } from "@/components/insights-section";
import { OperationsSection } from "@/components/operations-section";
import { RelationshipSection } from "@/components/relationship-section";
import { GraphInsightsSection } from "@/components/graph-insights-section";

type ViewKey = "dashboard" | "search" | "profile" | "graph" | "relationships" | "chat" | "insights" | "graphInsights" | "operations";
type ThemeMode = "dark" | "light";

const themeStorageKey = "persona-console-theme";
const defaultThemeMode: ThemeMode = "dark";

function getStoredTheme(): ThemeMode {
  if (typeof window === "undefined") return defaultThemeMode;
  return window.localStorage.getItem(themeStorageKey) === "light" ? "light" : defaultThemeMode;
}

function applyTheme(mode: ThemeMode) {
  document.documentElement.dataset.theme = mode;
  document.documentElement.style.colorScheme = mode;
  window.localStorage.setItem(themeStorageKey, mode);
}

const views: Array<{ key: ViewKey; label: string; caption: string }> = [
  { key: "dashboard", label: "대시보드", caption: "관계 관측소" },
  { key: "search", label: "검색/필터", caption: "세그먼트 탐색" },
  { key: "profile", label: "프로필", caption: "페르소나 맥락" },
  { key: "graph", label: "그래프", caption: "경로와 노드" },
  { key: "relationships", label: "관계형 추천", caption: "근거 기반 추천" },
  { key: "chat", label: "대화형 탐색", caption: "RAG 출처 확인" },
  { key: "insights", label: "확장 분석", caption: "F16-F18 검수" },
  { key: "graphInsights", label: "그래프 인사이트", caption: "정제/브릿지/경로" },
  { key: "operations", label: "운영 상태", caption: "품질과 준비도" },
];

const workflowSteps: Array<{ key: ViewKey; label: string; detail: string }> = [
  { key: "search", label: "01 검색", detail: "지역/취미/직업" },
  { key: "profile", label: "02 프로필", detail: "생활 맥락" },
  { key: "graph", label: "03 관계 그래프", detail: "경로 집중" },
  { key: "relationships", label: "04 추천 근거", detail: "유사/대조" },
  { key: "chat", label: "05 RAG 탐색", detail: "질문과 출처" },
];

const emptySearchFilters: SearchFilters = {
  province: "",
  age_group: "",
  sex: "",
  hobby: "",
  occupation: "",
  keyword: "",
};

const ragTraceAdminAutoLoad = process.env.NEXT_PUBLIC_RAG_TRACE_ADMIN_ENABLED === "true";

export default function Home() {
  const [activeView, setActiveView] = useState<ViewKey>("dashboard");
  const [themeMode, setThemeMode] = useState<ThemeMode>(defaultThemeMode);
  const [selectedUuid, setSelectedUuid] = useState(DEFAULT_PERSONA_UUID);
  const [selectedLabel, setSelectedLabel] = useState("기본 페르소나");
  const [stats, loadStats] = useLoadable<StatsResponse>();
  const [search, runSearch] = useLoadable<SearchResponse>();
  const [profile, loadProfile] = useLoadable<PersonaProfileResponse>();
  const [graph, loadGraph] = useLoadable<SubgraphResponse>();
  const [recommendationStatus, loadRecommendationStatus] = useLoadable<RecommendationStatusResponse>();
  const [recommendationQuality, loadRecommendationQuality] = useLoadable<RecommendationQualityResponse>();
  const [graphInsights, loadGraphInsights] = useLoadable<GraphInsightsResponse>();
  const [operationsHealth, loadOperationsHealth] = useLoadable<OperationsHealthResponse>();
  const [operationsReadiness, loadOperationsReadiness] = useLoadable<OperationsReadinessResponse>();
  const [operationsWarnings, loadOperationsWarnings] = useLoadable<OperationsWarningsResponse>();
  const [ragTraces, loadRagTraces] = useLoadable<RagTraceListResponse>();
  const [filters, setFilters] = useState<SearchFilters>(emptySearchFilters);
  const [page, setPage] = useState(1);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const selectedProfileName = profile.data?.uuid === selectedUuid ? profile.data.display_name : null;
  const selectedDisplayName = selectedProfileName ?? (selectedLabel === "기본 페르소나" ? "이름 미등록" : selectedLabel);
  const selectedProfile = profile.data?.uuid === selectedUuid ? profile.data : null;
  const selectedLocation = selectedProfile
    ? [selectedProfile.location.province, selectedProfile.location.district].filter(Boolean).join(" ") || "-"
    : "프로필 대기";
  const selectedOccupation = selectedProfile?.occupation ?? "-";
  const selectedConnectionCount = selectedProfile ? compactNumber(selectedProfile.graph_stats.total_connections) : "-";
  const graphNodeSummary = graph.data ? compactNumber(graph.data.node_count) : "-";
  const graphEdgeSummary = graph.data ? compactNumber(graph.data.edge_count) : "-";
  const totalPersonaSummary = stats.data ? fullNumber(stats.data.total_personas) : stats.loading ? "..." : "-";
  const topProvince = stats.data?.province_distribution[0]?.label ?? "-";

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const nextTheme = getStoredTheme();
      setThemeMode(nextTheme);
      applyTheme(nextTheme);
    }, 0);

    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    void loadStats(() => personaApi.stats());
    void loadRecommendationStatus(() => personaApi.recommendationStatus());
  }, [loadRecommendationStatus, loadStats]);

  useEffect(() => {
    if (activeView === "operations" && !recommendationQuality.data && !recommendationQuality.loading && !recommendationQuality.error) {
      void loadRecommendationQuality(() => personaApi.recommendationQuality());
    }
    if (activeView === "operations" && !operationsHealth.data && !operationsHealth.loading && !operationsHealth.error) {
      void loadOperationsHealth(() => personaApi.operationsHealth());
    }
    if (activeView === "operations" && !operationsReadiness.data && !operationsReadiness.loading && !operationsReadiness.error) {
      void loadOperationsReadiness(() => personaApi.operationsReadiness());
    }
    if (activeView === "operations" && !operationsWarnings.data && !operationsWarnings.loading && !operationsWarnings.error) {
      void loadOperationsWarnings(() => personaApi.operationsWarnings());
    }
  }, [
    activeView,
    loadOperationsHealth,
    loadOperationsReadiness,
    loadOperationsWarnings,
    loadRecommendationQuality,
    operationsHealth.data,
    operationsHealth.error,
    operationsHealth.loading,
    operationsReadiness.data,
    operationsReadiness.error,
    operationsReadiness.loading,
    operationsWarnings.data,
    operationsWarnings.error,
    operationsWarnings.loading,
    recommendationQuality.data,
    recommendationQuality.error,
    recommendationQuality.loading,
  ]);

  useEffect(() => {
    if (activeView === "graphInsights" && !graphInsights.data && !graphInsights.loading && !graphInsights.error) {
      void loadGraphInsights(() => personaApi.graphInsights({ limit: 12 }));
    }
  }, [activeView, graphInsights.data, graphInsights.error, graphInsights.loading, loadGraphInsights]);

  useEffect(() => {
    if (ragTraceAdminAutoLoad && activeView === "operations" && !ragTraces.data && !ragTraces.loading && !ragTraces.error) {
      void loadRagTraces(() => personaApi.ragTraces({ limit: 20 }));
    }
  }, [activeView, loadRagTraces, ragTraces.data, ragTraces.error, ragTraces.loading]);

  useEffect(() => {
    void loadProfile(() => personaApi.profile(selectedUuid));
    void loadGraph(() => personaApi.graph(selectedUuid, { depth: 2, max_nodes: 40, include_similar: true, max_similar: 5 }));
  }, [loadGraph, loadProfile, selectedUuid]);

  function updateFilter(key: keyof SearchFilters, value: string) {
    setFilters((current) => ({ ...current, [key]: value }));
  }

  async function submitSearch(targetPage = 1) {
    setPage(targetPage);
    await runSearch(() => personaApi.search({ ...filters, page: targetPage, page_size: 8, sort_by: "display_name", sort_order: "asc" }));
  }

  function selectPersona(uuid: string, label: string | null) {
    setSelectedUuid(uuid);
    setSelectedLabel(label || shortUuid(uuid));
    setActiveView("profile");
  }

  function selectGraphPersona(uuid: string, label: string | null) {
    setSelectedUuid(uuid);
    setSelectedLabel(label || shortUuid(uuid));
    setActiveView("graph");
  }

  function changeTheme(mode: ThemeMode) {
    setThemeMode(mode);
    applyTheme(mode);
  }

  function resetChat() {
    resetChatSessionId();
    setChatMessages([]);
    setChatInput("");
    setChatError(null);
    setChatLoading(false);
  }

  function refreshOperations() {
    void loadRecommendationStatus(() => personaApi.recommendationStatus());
    void loadRecommendationQuality(() => personaApi.recommendationQuality());
    void loadOperationsHealth(() => personaApi.operationsHealth());
    void loadOperationsReadiness(() => personaApi.operationsReadiness());
    void loadOperationsWarnings(() => personaApi.operationsWarnings());
    void loadRagTraces(() => personaApi.ragTraces({ limit: 20 }));
  }

  async function submitChat(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = chatInput.trim();
    if (!message) return;

    setChatInput("");
    setChatError(null);
    setChatLoading(true);
    setChatMessages((current) => [...current, { role: "user", content: message }]);

    try {
      const result = await personaApi.chat({ session_id: chatSessionId(), message, stream: false });
      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: result.response,
        filters: result.context_filters,
        sources: result.sources,
      };
      setChatMessages((current) => [...current, assistantMessage].slice(-12));
    } catch (error) {
      setChatError(error instanceof Error ? error.message : "채팅 응답을 가져오지 못했습니다.");
    } finally {
      setChatLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark">KG</div>
        <div className="eyebrow">Nemotron Personas</div>
        <h1 className="brand-title">Persona Knowledge Console</h1>
        <p className="brand-copy">한국 페르소나의 지역, 취미, 직업, 유사도 경로를 추적하는 지식그래프 워크스테이션</p>
        <div className="theme-switch" role="group" aria-label="테마 선택">
          <button className={themeMode === "dark" ? "active" : ""} type="button" aria-pressed={themeMode === "dark"} onClick={() => changeTheme("dark")}>
            다크
          </button>
          <button className={themeMode === "light" ? "active" : ""} type="button" aria-pressed={themeMode === "light"} onClick={() => changeTheme("light")}>
            라이트
          </button>
        </div>
        <nav className="nav-list" aria-label="주요 화면">
          {views.map((view) => (
            <button key={view.key} className={`nav-button ${activeView === view.key ? "active" : ""}`} onClick={() => setActiveView(view.key)}>
              <strong>{view.label}</strong>
              <br />
              <span className="small">{view.caption}</span>
            </button>
          ))}
        </nav>
      </aside>

      <main className="main-panel">
        {activeView === "dashboard" ? (
          <section className="hero observatory-hero">
            <div className="hero-card observatory-card">
              <div className="eyebrow">Korean Persona Knowledge Graph Observatory</div>
              <h1>사람, 취미, 직업, 지역의 관계 경로를 추적합니다</h1>
              <p>
                검색으로 세그먼트를 좁히고, 선택한 페르소나의 생활 맥락을 그래프 중심으로 전환한 뒤,
                추천 이유와 RAG 출처까지 같은 흐름에서 검증합니다.
              </p>
              <div className="route-rail" aria-label="관계 탐색 흐름">
                {workflowSteps.map((step) => (
                  <button
                    className={`route-step ${activeView === step.key ? "active" : ""}`}
                    key={step.key}
                    type="button"
                    onClick={() => setActiveView(step.key)}
                  >
                    <span>{step.label}</span>
                    <strong>{step.detail}</strong>
                  </button>
                ))}
              </div>
              <div className="signal-strip" aria-label="데이터 신호 요약">
                <div>
                  <span className="small muted">personas</span>
                  <strong>{totalPersonaSummary}</strong>
                </div>
                <div>
                  <span className="small muted">top region</span>
                  <strong>{topProvince}</strong>
                </div>
                <div>
                  <span className="small muted">current graph</span>
                  <strong>{graphNodeSummary} nodes</strong>
                </div>
                <div>
                  <span className="small muted">relations</span>
                  <strong>{graphEdgeSummary} edges</strong>
                </div>
              </div>
            </div>
            <div className="card status-card context-card">
              <div className="context-card-header">
                <span className="status-dot" />
                <span className="small muted">현재 선택</span>
              </div>
              <h2>{selectedDisplayName}</h2>
              <p className="muted small">UUID {uuidWithName(selectedUuid, selectedDisplayName)}</p>
              <div className="context-grid">
                <span>지역</span><strong>{selectedLocation}</strong>
                <span>직업</span><strong>{selectedOccupation}</strong>
                <span>연결</span><strong>{selectedConnectionCount}</strong>
              </div>
              <div className="context-actions">
                <button className="ghost-button" onClick={() => setActiveView("profile")}>프로필</button>
                <button className="primary-button" onClick={() => setActiveView("graph")}>관계 그래프</button>
              </div>
            </div>
          </section>
        ) : (
          <div className="hero-compact">
            <span className="status-dot" />
            <div>
              <div className="eyebrow">현재 선택</div>
              <h2>{selectedDisplayName} <span className="muted">({uuidWithName(selectedUuid, selectedDisplayName)})</span></h2>
            </div>
            <div className="compact-context">
              <span>{selectedLocation}</span>
              <span>{selectedOccupation}</span>
              <span>{selectedConnectionCount} links</span>
            </div>
            <button className="ghost-button compact-graph-button" onClick={() => setActiveView("graph")}>관계 그래프</button>
          </div>
        )}

        {activeView === "dashboard" && <DashboardSection stats={stats} />}
        {activeView === "search" && <SearchSection filters={filters} page={page} search={search} onFilterChange={updateFilter} onSearch={submitSearch} onSelect={selectPersona} />}
        {activeView === "profile" && <ProfileSection profile={profile} selectedUuid={selectedUuid} onUuidChange={(uuid) => selectPersona(uuid, null)} onSelectPersona={selectPersona} />}
        {activeView === "graph" && <GraphSection graph={graph} profile={profile.data} onSelectPersona={selectGraphPersona} />}
        {activeView === "relationships" && <RelationshipSection selectedUuid={selectedUuid} profile={profile} onSelectPersona={selectPersona} />}
        {activeView === "chat" && <ChatSection messages={chatMessages} input={chatInput} loading={chatLoading} error={chatError} onInputChange={setChatInput} onSubmit={submitChat} onReset={resetChat} />}
        {activeView === "insights" && <InsightsSection />}
        {activeView === "graphInsights" && <GraphInsightsSection insights={graphInsights} onSelectPersona={selectPersona} />}
        {activeView === "operations" && (
          <OperationsSection
            selectedUuid={selectedUuid}
            recommendationStatus={recommendationStatus}
            recommendationQuality={recommendationQuality}
            operationsHealth={operationsHealth}
            operationsReadiness={operationsReadiness}
            operationsWarnings={operationsWarnings}
            ragTraces={ragTraces}
            onRefresh={refreshOperations}
          />
        )}
      </main>
    </div>
  );
}
