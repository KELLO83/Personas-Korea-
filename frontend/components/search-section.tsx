import { useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";
import type { PersonaProfileResponse, SearchResponse } from "@/lib/api-types";
import type { Loadable } from "@/hooks/use-loadable";
import { personaApi } from "@/lib/api-client";
import { fullNumber, joinDefined, shortUuid } from "@/lib/formatters";

export interface SearchFilters {
  province: string;
  age_group: string;
  sex: string;
  hobby: string;
  occupation: string;
  keyword: string;
}

interface SearchSectionProps {
  filters: SearchFilters;
  page: number;
  search: Loadable<SearchResponse>;
  onFilterChange: (key: keyof SearchFilters, value: string) => void;
  onSearch: (page: number) => Promise<void>;
  onSelect: (uuid: string, label: string | null) => void;
}

export function SearchSection({ filters, page, search, onFilterChange, onSearch, onSelect }: SearchSectionProps) {
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailProfile, setDetailProfile] = useState<PersonaProfileResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const modalRef = useRef<HTMLDivElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const detailTriggerRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (detailOpen) closeButtonRef.current?.focus();
  }, [detailOpen]);

  async function openDetail(uuid: string, trigger: HTMLButtonElement) {
    detailTriggerRef.current = trigger;
    setDetailOpen(true);
    setDetailProfile(null);
    setDetailError(null);
    setDetailLoading(true);
    try {
      setDetailProfile(await personaApi.profile(uuid));
    } catch (error) {
      setDetailError(error instanceof Error ? error.message : "상세 정보를 불러오지 못했습니다.");
    } finally {
      setDetailLoading(false);
    }
  }

  function closeDetail() {
    setDetailOpen(false);
    detailTriggerRef.current?.focus();
  }

  function handleModalKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") {
      closeDetail();
      return;
    }
    if (event.key !== "Tab" || !modalRef.current) return;

    const focusableElements = modalRef.current.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
    );
    if (focusableElements.length === 0) {
      event.preventDefault();
      return;
    }

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];
    if (event.shiftKey && document.activeElement === firstElement) {
      event.preventDefault();
      lastElement.focus();
    } else if (!event.shiftKey && document.activeElement === lastElement) {
      event.preventDefault();
      firstElement.focus();
    }
  }

  return (
    <section className="grid">
      <div className="card">
        <h2>검색/필터</h2>
        <div className="form-grid">
          <input className="input" aria-label="지역" placeholder="지역 예: 서울" value={filters.province} onChange={(event) => onFilterChange("province", event.target.value)} />
          <input className="input" aria-label="연령대" placeholder="연령대 예: 30대" value={filters.age_group} onChange={(event) => onFilterChange("age_group", event.target.value)} />
          <select className="select" aria-label="성별" value={filters.sex} onChange={(event) => onFilterChange("sex", event.target.value)}>
            <option value="">성별 전체</option>
            <option value="남자">남자</option>
            <option value="여자">여자</option>
          </select>
          <input className="input" aria-label="취미" placeholder="취미" value={filters.hobby} onChange={(event) => onFilterChange("hobby", event.target.value)} />
          <input className="input" aria-label="직업" placeholder="직업" value={filters.occupation} onChange={(event) => onFilterChange("occupation", event.target.value)} />
          <input className="input" aria-label="키워드" placeholder="키워드" value={filters.keyword} onChange={(event) => onFilterChange("keyword", event.target.value)} />
        </div>
        <div style={{ marginTop: 12 }}>
          <button className="primary-button" onClick={() => void onSearch(1)} disabled={search.loading}>{search.loading ? "검색 중…" : "검색하기"}</button>
        </div>
      </div>
      {search.error && <div className="card error-box">{search.error}</div>}
      <div className="card">
        <h3>검색 결과 {search.data ? fullNumber(search.data.total_count) : ""}</h3>
        <div className="results-list">
          {(search.data?.results ?? []).map((persona) => (
            <div className="result-card" key={persona.uuid}>
              <h3>{persona.display_name ?? shortUuid(persona.uuid)}</h3>
              <p className="muted small">{joinDefined([persona.age ? `${persona.age}세` : null, persona.sex, persona.province, persona.district, persona.occupation])}</p>
              <p>{persona.persona ? `${persona.persona.slice(0, 150)}${persona.persona.length > 150 ? "…" : ""}` : "페르소나 설명 없음"}</p>
              <div className="card-actions">
                <button className="ghost-button" onClick={() => onSelect(persona.uuid, persona.display_name)}>이 사람 선택</button>
                <button className="ghost-button" onClick={(event) => void openDetail(persona.uuid, event.currentTarget)}>상세보기</button>
              </div>
            </div>
          ))}
          {search.data?.results.length === 0 && <p className="muted">조건에 맞는 결과가 없습니다.</p>}
          {!search.data && <p className="muted">조건을 입력하고 검색을 실행하세요.</p>}
        </div>
        {search.data && search.data.total_pages > 1 && (
          <div className="pill-row" style={{ marginTop: 14 }}>
            <button className="ghost-button" disabled={page <= 1 || search.loading} onClick={() => void onSearch(page - 1)}>이전</button>
            <span className="pill">{search.data.page} / {search.data.total_pages}</span>
            <button className="ghost-button" disabled={page >= search.data.total_pages || search.loading} onClick={() => void onSearch(page + 1)}>다음</button>
          </div>
        )}
      </div>
      {detailOpen && (
        <div className="modal-backdrop" role="presentation" onClick={closeDetail}>
          <div ref={modalRef} className="modal-card" role="dialog" aria-modal="true" aria-labelledby="search-detail-title" onClick={(event) => event.stopPropagation()} onKeyDown={handleModalKeyDown}>
            <div className="modal-header">
              <div>
                <p className="eyebrow">Persona Detail</p>
                <h2 id="search-detail-title">{detailProfile?.display_name ?? "상세 정보"}</h2>
              </div>
              <button ref={closeButtonRef} className="ghost-button" onClick={closeDetail} aria-label="상세보기 닫기">닫기</button>
            </div>
            {detailLoading && <p className="muted">상세 정보를 불러오는 중입니다…</p>}
            {detailError && <div className="error-box modal-error">{detailError}</div>}
            {detailProfile && <SearchDetailProfile profile={detailProfile} onSelect={onSelect} />}
          </div>
        </div>
      )}
    </section>
  );
}

function SearchDetailProfile({ profile, onSelect }: { profile: PersonaProfileResponse; onSelect: (uuid: string, label: string | null) => void }) {
  const personaSections = [
    ["직업/전문성", profile.personas.professional],
    ["운동/건강", profile.personas.sports],
    ["문화/예술", profile.personas.arts],
    ["여행", profile.personas.travel],
    ["음식", profile.personas.culinary],
    ["가족", profile.personas.family],
  ].filter((entry): entry is [string, string] => Boolean(entry[1]));

  return (
    <div className="modal-body">
      <div className="grid two">
        <div className="detail-panel">
          <p className="small muted">UUID {shortUuid(profile.uuid)}</p>
          <h3>{profile.display_name ?? shortUuid(profile.uuid)}</h3>
          <p className="muted">
            {joinDefined([
              profile.demographics.age ? `${profile.demographics.age}세` : null,
              profile.demographics.sex,
              profile.location.province,
              profile.location.district,
              profile.occupation,
            ])}
          </p>
          <button className="primary-button" onClick={() => onSelect(profile.uuid, profile.display_name)}>이 사람 선택</button>
        </div>
        <div className="detail-panel">
          <h3>인구통계</h3>
          <div className="pill-row">
            <span className="pill">연령대 {profile.demographics.age_group ?? "-"}</span>
            <span className="pill">학력 {profile.demographics.education_level ?? "-"}</span>
            <span className="pill">전공 {profile.demographics.bachelors_field ?? "-"}</span>
            <span className="pill">혼인 {profile.demographics.marital_status ?? "-"}</span>
            <span className="pill">병역 {profile.demographics.military_status ?? "-"}</span>
            <span className="pill">가구 {profile.demographics.family_type ?? "-"}</span>
            <span className="pill">주거 {profile.demographics.housing_type ?? "-"}</span>
          </div>
        </div>
      </div>
      <div className="grid two">
        <div className="detail-panel">
          <h3>그래프 통계</h3>
          <div className="pill-row">
            <span className="pill">연결 {fullNumber(profile.graph_stats.total_connections)}</span>
            <span className="pill">취미 {fullNumber(profile.graph_stats.hobby_count)}</span>
            <span className="pill">스킬 {fullNumber(profile.graph_stats.skill_count)}</span>
            <span className="pill">커뮤니티 {profile.community.label ?? profile.community.community_id ?? "-"}</span>
          </div>
        </div>
        <div className="detail-panel">
          <h3>배경/목표</h3>
          <p><strong>문화 배경</strong>: {profile.cultural_background ?? "정보 없음"}</p>
          <p><strong>커리어 목표</strong>: {profile.career_goals ?? "정보 없음"}</p>
        </div>
      </div>
      <div className="detail-panel">
        <h3>성향 및 배경</h3>
        <p>{profile.personas.summary ?? "요약 정보가 없습니다."}</p>
      </div>
      {personaSections.length > 0 && (
        <div className="detail-panel">
          <h3>페르소나 상세</h3>
          <div className="results-list">
            {personaSections.map(([label, value]) => (
              <div className="result-card" key={label}>
                <strong>{label}</strong>
                <p>{value}</p>
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="grid two">
        <div className="detail-panel">
          <h3>취미</h3>
          <div className="pill-row">
            {profile.hobbies.slice(0, 12).map((hobby) => <span className="pill" key={hobby}>{hobby}</span>)}
            {profile.hobbies.length === 0 && <span className="muted small">표시할 취미가 없습니다.</span>}
          </div>
        </div>
        <div className="detail-panel">
          <h3>스킬</h3>
          <div className="pill-row">
            {profile.skills.slice(0, 12).map((skill) => <span className="pill" key={skill}>{skill}</span>)}
            {profile.skills.length === 0 && <span className="muted small">표시할 스킬이 없습니다.</span>}
          </div>
        </div>
      </div>
      <div className="detail-panel">
        <h3>유사 페르소나</h3>
        <div className="results-list">
          {profile.similar_preview.slice(0, 4).map((persona) => (
            <div className="result-card" key={persona.uuid}>
              <strong>{persona.display_name ?? shortUuid(persona.uuid)}</strong>
              <p className="muted small">{joinDefined([persona.age ? `${persona.age}세` : null, persona.similarity !== null ? `유사도 ${(persona.similarity * 100).toFixed(1)}%` : null])}</p>
            </div>
          ))}
          {profile.similar_preview.length === 0 && <p className="muted">표시할 유사 페르소나가 없습니다.</p>}
        </div>
      </div>
    </div>
  );
}
