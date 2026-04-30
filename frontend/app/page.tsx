"use client";

import { FormEvent, useEffect, useState } from "react";
import type { ChatMessage, PersonaProfileResponse, SearchResponse, StatsResponse, SubgraphResponse } from "@/lib/api-types";
import { personaApi } from "@/lib/api-client";
import { DEFAULT_PERSONA_UUID } from "@/lib/constants";
import { shortUuid } from "@/lib/formatters";
import { chatSessionId, resetChatSessionId } from "@/lib/chat-session";
import { useLoadable } from "@/hooks/use-loadable";
import { DashboardSection } from "@/components/dashboard-section";
import { SearchFilters, SearchSection } from "@/components/search-section";
import { ProfileSection } from "@/components/profile-section";
import { GraphSection } from "@/components/graph-section";
import { ChatSection } from "@/components/chat-section";

type ViewKey = "dashboard" | "search" | "profile" | "graph" | "chat";

const views: Array<{ key: ViewKey; label: string; caption: string }> = [
  { key: "dashboard", label: "대시보드", caption: "전체 분포" },
  { key: "search", label: "검색/필터", caption: "페르소나 탐색" },
  { key: "profile", label: "프로필", caption: "상세 정보" },
  { key: "graph", label: "그래프", caption: "관계 맵" },
  { key: "chat", label: "대화형 탐색", caption: "질문 기반 분석" },
];

const emptySearchFilters: SearchFilters = {
  province: "",
  age_group: "",
  sex: "",
  hobby: "",
  occupation: "",
  keyword: "",
};

export default function Home() {
  const [activeView, setActiveView] = useState<ViewKey>("dashboard");
  const [selectedUuid, setSelectedUuid] = useState(DEFAULT_PERSONA_UUID);
  const [selectedLabel, setSelectedLabel] = useState("기본 페르소나");
  const [stats, loadStats] = useLoadable<StatsResponse>();
  const [search, runSearch] = useLoadable<SearchResponse>();
  const [profile, loadProfile] = useLoadable<PersonaProfileResponse>();
  const [graph, loadGraph] = useLoadable<SubgraphResponse>();
  const [filters, setFilters] = useState<SearchFilters>(emptySearchFilters);
  const [page, setPage] = useState(1);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  useEffect(() => {
    void loadStats(() => personaApi.stats());
  }, [loadStats]);

  useEffect(() => {
    void loadProfile(() => personaApi.profile(selectedUuid));
    void loadGraph(() => personaApi.graph(selectedUuid, { depth: 2, max_nodes: 40, include_similar: true }));
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

  function resetChat() {
    resetChatSessionId();
    setChatMessages([]);
    setChatInput("");
    setChatError(null);
    setChatLoading(false);
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
          <section className="hero">
            <div className="hero-card">
              <div className="eyebrow">Graph Intelligence Dashboard</div>
              <h1>Nemotron-Personas-Korea 분석</h1>
            </div>
            <div className="card status-card">
              <span className="status-dot" /> <span className="small muted">현재 선택</span>
              <h2>{selectedLabel}</h2>
              <p className="muted small">UUID {shortUuid(selectedUuid)}</p>
              <button className="ghost-button" onClick={() => setActiveView("graph")}>관계 그래프로 보기</button>
            </div>
          </section>
        ) : (
          <div className="hero-compact">
            <span className="status-dot" />
            <div>
              <div className="eyebrow">현재 선택</div>
              <h2>{selectedLabel} <span className="muted">({shortUuid(selectedUuid)})</span></h2>
            </div>
            <button className="ghost-button" onClick={() => setActiveView("graph")} style={{ marginLeft: "auto" }}>관계 그래프</button>
          </div>
        )}

        {activeView === "dashboard" && <DashboardSection stats={stats} />}
        {activeView === "search" && <SearchSection filters={filters} page={page} search={search} onFilterChange={updateFilter} onSearch={submitSearch} onSelect={selectPersona} />}
        {activeView === "profile" && <ProfileSection profile={profile} selectedUuid={selectedUuid} onUuidChange={(uuid) => selectPersona(uuid, null)} onSelectPersona={selectPersona} />}
        {activeView === "graph" && <GraphSection graph={graph} profile={profile.data} onSelectPersona={selectGraphPersona} />}
        {activeView === "chat" && <ChatSection messages={chatMessages} input={chatInput} loading={chatLoading} error={chatError} onInputChange={setChatInput} onSubmit={submitChat} onReset={resetChat} />}
      </main>
    </div>
  );
}
