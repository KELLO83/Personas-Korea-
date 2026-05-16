import { FormEvent, useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";
import type { PersonaProfileResponse, SimilarPreview, SimilarityExplanationResponse } from "@/lib/api-types";
import type { Loadable } from "@/hooks/use-loadable";
import { personaApi } from "@/lib/api-client";
import { lowPriorityLabel, profileExposurePolicy } from "@/lib/exposure-policy";
import { joinDefined, percent, shortUuid, uuidWithName } from "@/lib/formatters";
import { SearchDetailProfile, type SimilarityContext } from "@/components/search-section";

interface ProfileSectionProps {
  profile: Loadable<PersonaProfileResponse>;
  selectedUuid: string;
  onUuidChange: (uuid: string) => void;
  onSelectPersona: (uuid: string, label: string | null) => void;
}

export function ProfileSection({ profile, selectedUuid, onUuidChange, onSelectPersona }: ProfileSectionProps) {
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailProfile, setDetailProfile] = useState<PersonaProfileResponse | null>(null);
  const [detailSimilarityContext, setDetailSimilarityContext] = useState<SimilarityContext | null>(null);
  const [detailSimilarityExplanation, setDetailSimilarityExplanation] = useState<SimilarityExplanationResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const modalRef = useRef<HTMLDivElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const detailTriggerRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (detailOpen) closeButtonRef.current?.focus();
  }, [detailOpen]);

  function submitUuid(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const uuid = String(data.get("profileUuid") ?? "").trim();
    if (uuid) onUuidChange(uuid);
  }

  async function openDetail(uuid: string, trigger: HTMLButtonElement, similarityContext: SimilarityContext | null = null) {
    detailTriggerRef.current = trigger;
    setDetailOpen(true);
    setDetailProfile(null);
    setDetailSimilarityContext(similarityContext);
    setDetailSimilarityExplanation(null);
    setDetailError(null);
    setDetailLoading(true);
    try {
      const [profileResult, explanationResult] = await Promise.all([
        personaApi.profile(uuid),
        similarityContext ? personaApi.similarityExplanation(similarityContext.sourceProfile.uuid, uuid) : Promise.resolve(null),
      ]);
      setDetailProfile(profileResult);
      setDetailSimilarityExplanation(explanationResult);
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
        <h2>프로필 상세</h2>
        <form className="form-grid" style={{ gridTemplateColumns: "minmax(0, 1fr) 130px" }} onSubmit={submitUuid}>
          <input className="input" key={selectedUuid} name="profileUuid" defaultValue={selectedUuid} placeholder="페르소나 UUID" />
          <button className="primary-button">조회</button>
        </form>
      </div>
      {profile.error && <div className="card error-box">{profile.error}</div>}
      {profile.data ? <ProfileDetailBody profile={profile.data} onSelectPersona={onSelectPersona} onOpenDetail={openDetail} /> : <div className="card muted">{profile.loading ? "프로필을 불러오는 중" : "프로필을 불러오세요."}</div>}
      {detailOpen && (
        <div className="modal-backdrop" role="presentation" onClick={closeDetail}>
          <div ref={modalRef} className="modal-card" role="dialog" aria-modal="true" aria-labelledby="profile-detail-title" onClick={(event) => event.stopPropagation()} onKeyDown={handleModalKeyDown}>
            <div className="modal-header">
              <div>
                <p className="eyebrow">Persona Detail</p>
                <h2 id="profile-detail-title">{detailProfile?.display_name ?? "상세 정보"}</h2>
              </div>
              <button ref={closeButtonRef} className="ghost-button" onClick={closeDetail} aria-label="상세보기 닫기">닫기</button>
            </div>
            {detailLoading && <p className="muted">상세 정보를 불러오는 중입니다.</p>}
            {detailError && <div className="error-box modal-error">{detailError}</div>}
            {detailProfile && <SearchDetailProfile profile={detailProfile} onSelect={onSelectPersona} similarityContext={detailSimilarityContext} similarityExplanation={detailSimilarityExplanation} />}
          </div>
        </div>
      )}
    </section>
  );
}

function ProfileDetailBody({
  profile,
  onSelectPersona,
  onOpenDetail,
}: {
  profile: PersonaProfileResponse;
  onSelectPersona: (uuid: string, label: string | null) => void;
  onOpenDetail: (uuid: string, trigger: HTMLButtonElement, similarityContext: SimilarityContext | null) => Promise<void>;
}) {
  const personaSections = [
    ["직업/전문성", profile.personas.professional],
    ["운동/건강", profile.personas.sports],
    ["문화/예술", profile.personas.arts],
    ["여행", profile.personas.travel],
    ["음식", profile.personas.culinary],
    ["가족", profile.personas.family],
  ].filter((entry): entry is [string, string] => Boolean(entry[1]));

  return (
    <div className="grid">
      <div className="card">
        <div className="grid two">
          <div className="detail-panel">
            <p className="small muted">UUID {uuidWithName(profile.uuid, profile.display_name)}</p>
            <h3>{profile.display_name ?? shortUuid(profile.uuid)}</h3>
            <p className="muted">{joinDefined([profile.demographics.age ? `${profile.demographics.age}세` : null, profile.demographics.sex, profile.location.province, profile.location.district, profile.occupation])}</p>
            <p className="small muted">커뮤니티 {profile.community.label ?? profile.community.community_id ?? "미지정"}</p>
          </div>
          <div className="detail-panel">
            <h3>인구통계</h3>
            <div className="pill-row">
              <span className="pill">연령대 {profile.demographics.age_group ?? "-"}</span>
              <span className="pill">학력 {profile.demographics.education_level ?? "-"}</span>
              {profileExposurePolicy.lowerPriorityBachelorsField && <span className="pill muted">{lowPriorityLabel(profile.demographics.bachelors_field)}</span>}
              <span className="pill">혼인 {profile.demographics.marital_status ?? "-"}</span>
              {!profileExposurePolicy.hideMilitaryStatus && <span className="pill">병역 {profile.demographics.military_status ?? "-"}</span>}
              <span className="pill">가구 {profile.demographics.family_type ?? "-"}</span>
              <span className="pill">주거 {profile.demographics.housing_type ?? "-"}</span>
              {!profileExposurePolicy.hideCountry && <span className="pill">국가 {profile.location.country ?? "-"}</span>}
            </div>
            <p className="small muted">Country와 MilitaryStatus는 정보량 낮음 정책에 따라 기본 숨김 처리됩니다.</p>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>그래프 통계</h3>
        <div className="pill-row">
          <span className="pill">연결 {profile.graph_stats.total_connections}개</span>
          <span className="pill">취미 {profile.graph_stats.hobby_count}개</span>
          <span className="pill">스킬 {profile.graph_stats.skill_count}개</span>
        </div>
      </div>

      <div className="grid two">
        <div className="detail-panel">
          <h3>요약/배경</h3>
          <p>{profile.personas.summary ?? "요약 정보가 없습니다."}</p>
          <p><strong>문화 배경</strong>: {profile.cultural_background ?? "정보 없음"}</p>
          <p><strong>커리어 목표</strong>: {profile.career_goals ?? "정보 없음"}</p>
        </div>
        <div className="detail-panel">
          <h3>취미</h3>
          <div className="pill-row">
            {profile.hobbies.slice(0, 14).map((hobby) => <span className="pill" key={hobby}>{hobby}</span>)}
            {profile.hobbies.length === 0 && <span className="muted small">표시할 취미가 없습니다.</span>}
          </div>
        </div>
      </div>

      <div className="detail-panel">
        <h3>보유 스킬</h3>
        <div className="pill-row">
          {profile.skills.slice(0, 16).map((skill) => <span className="pill" key={skill}>{skill}</span>)}
          {profile.skills.length === 0 && <span className="muted small">표시할 스킬이 없습니다.</span>}
        </div>
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

      <div className="card">
        <h3>유사 페르소나 Preview</h3>
        <div className="results-list">
          {profile.similar_preview.slice(0, 6).map((persona) => (
            <div className="result-card" key={persona.uuid}>
              <strong>{persona.display_name ?? shortUuid(persona.uuid)}</strong>
              <p className="muted small">{joinDefined([persona.age ? `${persona.age}세` : null, persona.similarity !== null ? `유사도 ${percent(persona.similarity)}` : null])}</p>
              <div className="pill-row">
                {persona.shared_hobbies.slice(0, 7).map((hobby) => <span className="pill" key={hobby}>{hobby}</span>)}
              </div>
              <div className="card-actions">
                <button className="ghost-button" type="button" onClick={() => onSelectPersona(persona.uuid, persona.display_name)}>이 사람 선택</button>
                <button
                  className="ghost-button"
                  type="button"
                  onClick={(event) => void onOpenDetail(persona.uuid, event.currentTarget, buildSimilarityContext(profile, persona))}
                >
                  상세보기
                </button>
              </div>
            </div>
          ))}
          {profile.similar_preview.length === 0 && <p className="muted">표시할 유사 페르소나가 없습니다.</p>}
        </div>
      </div>
    </div>
  );
}

function buildSimilarityContext(profile: PersonaProfileResponse, persona: SimilarPreview): SimilarityContext {
  return {
    sourceLabel: profile.display_name ?? shortUuid(profile.uuid),
    similarity: persona.similarity,
    sharedHobbies: persona.shared_hobbies,
    sourceProfile: profile,
  };
}
