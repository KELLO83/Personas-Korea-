import { FormEvent } from "react";
import type { PersonaProfileResponse } from "@/lib/api-types";
import type { Loadable } from "@/hooks/use-loadable";
import { joinDefined, percent, shortUuid } from "@/lib/formatters";
import { MetricCard } from "./metric-card";

interface ProfileSectionProps {
  profile: Loadable<PersonaProfileResponse>;
  selectedUuid: string;
  onUuidChange: (uuid: string) => void;
  onSelectPersona: (uuid: string, label: string | null) => void;
}

export function ProfileSection({ profile, selectedUuid, onUuidChange, onSelectPersona }: ProfileSectionProps) {
  function submitUuid(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const uuid = String(data.get("profileUuid") ?? "").trim();
    if (uuid) onUuidChange(uuid);
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
      <div className="grid two">
        <div className="card">
          <p className="small muted">{profile.loading ? "불러오는 중" : "선택된 페르소나"}</p>
          <h2>{profile.data?.display_name ?? shortUuid(selectedUuid)}</h2>
          <p className="muted">{profile.data ? joinDefined([profile.data.demographics.age ? `${profile.data.demographics.age}세` : null, profile.data.demographics.sex, profile.data.location.province, profile.data.location.district, profile.data.occupation]) : "프로필을 불러오세요."}</p>
          <div className="pill-row">
            {(profile.data?.hobbies ?? []).slice(0, 8).map((hobby) => <span className="pill" key={hobby}>{hobby}</span>)}
          </div>
        </div>
        <div className="card">
          <h3>그래프 통계</h3>
          <div className="grid three">
            <MetricCard title="연결" value={String(profile.data?.graph_stats.total_connections ?? "-")} caption="total" loading={profile.loading} />
            <MetricCard title="취미" value={String(profile.data?.graph_stats.hobby_count ?? "-")} caption="hobby" loading={profile.loading} />
            <MetricCard title="스킬" value={String(profile.data?.graph_stats.skill_count ?? "-")} caption="skill" loading={profile.loading} />
          </div>
        </div>
      </div>
      <div className="card">
        <h3>페르소나 요약</h3>
        <p>{profile.data?.personas.summary ?? "요약 정보가 없습니다."}</p>
        <h3>보유 스킬</h3>
        <div className="pill-row">
          {(profile.data?.skills ?? []).slice(0, 12).map((skill) => <span className="pill" key={skill}>{skill}</span>)}
        </div>
      </div>
      <div className="card">
        <h3>유사 페르소나 Preview</h3>
        <div className="results-list">
          {(profile.data?.similar_preview ?? []).slice(0, 5).map((persona) => (
            <div className="result-card" key={persona.uuid}>
              <strong>{persona.display_name ?? shortUuid(persona.uuid)}</strong>
              <p className="muted small">{joinDefined([persona.age ? `${persona.age}세` : null, persona.similarity !== null ? `유사도 ${percent(persona.similarity)}` : null])}</p>
              <div className="pill-row">
                {persona.shared_hobbies.slice(0, 5).map((hobby) => <span className="pill" key={hobby}>{hobby}</span>)}
              </div>
              <div className="card-actions">
                <button className="ghost-button" type="button" onClick={() => onSelectPersona(persona.uuid, persona.display_name)}>이 사람 선택</button>
              </div>
            </div>
          ))}
          {profile.data && profile.data.similar_preview.length === 0 && <p className="muted">표시할 유사 페르소나가 없습니다.</p>}
        </div>
      </div>
    </section>
  );
}
