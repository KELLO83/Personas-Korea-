import { useEffect, useState } from "react";
import type {
  CommunityProfileResponse,
  LifeTrackResponse,
  PersonaGuildResponse,
  PersonaProfileResponse,
  SegmentCompareResponse,
  SimilarDiverseResponse,
} from "@/lib/api-types";
import { personaApi } from "@/lib/api-client";
import type { Loadable } from "@/hooks/use-loadable";
import { compactNumber, joinDefined, percent, shortUuid } from "@/lib/formatters";

type LoadState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
};

const initialState = { data: null, loading: false, error: null };

interface RelationshipSectionProps {
  selectedUuid: string;
  profile: Loadable<PersonaProfileResponse>;
  onSelectPersona: (uuid: string, label: string | null) => void;
}

export function RelationshipSection({ selectedUuid, profile, onSelectPersona }: RelationshipSectionProps) {
  const [guilds, setGuilds] = useState<LoadState<PersonaGuildResponse>>(initialState);
  const [diverse, setDiverse] = useState<LoadState<SimilarDiverseResponse>>(initialState);
  const [community, setCommunity] = useState<LoadState<CommunityProfileResponse>>(initialState);
  const [lifeTrack, setLifeTrack] = useState<LoadState<LifeTrackResponse>>(initialState);
  const [segmentInsight, setSegmentInsight] = useState<LoadState<SegmentCompareResponse>>(initialState);
  const [diversityAxis, setDiversityAxis] = useState("mixed");
  const [segments, setSegments] = useState({
    aLabel: "서울 20대",
    aProvince: "서울",
    aAgeGroup: "20대",
    bLabel: "부산 30대",
    bProvince: "부산",
    bAgeGroup: "30대",
  });
  const communityId = profile.data?.community.community_id ?? null;

  useEffect(() => {
    void run(setGuilds, () => personaApi.guilds(selectedUuid));
    void run(setDiverse, () => personaApi.similarDiverse(selectedUuid, { diversity_axis: diversityAxis, top_k: 8 }));
    void run(setLifeTrack, () => personaApi.lifeTrack(selectedUuid, { top_k: 8 }));
  }, [diversityAxis, selectedUuid]);

  useEffect(() => {
    if (communityId === null) {
      return;
    }
    void run(setCommunity, () => personaApi.communityProfile(communityId));
  }, [communityId]);

  async function run<T>(setter: (state: LoadState<T>) => void, loader: () => Promise<T>) {
    setter({ data: null, loading: true, error: null });
    try {
      setter({ data: await loader(), loading: false, error: null });
    } catch (error) {
      setter({ data: null, loading: false, error: error instanceof Error ? error.message : "요청에 실패했습니다." });
    }
  }

  function runSegmentInsight() {
    void run(setSegmentInsight, () =>
      personaApi.compareSegments({
        segment_a: { label: segments.aLabel, filters: { province: segments.aProvince, age_group: segments.aAgeGroup } },
        segment_b: { label: segments.bLabel, filters: { province: segments.bProvince, age_group: segments.bAgeGroup } },
        dimensions: ["hobby", "occupation", "education"],
        top_k: 8,
      }),
    );
  }

  return (
    <section className="grid">
      <div className="grid two">
        <div className="card">
          <div className="section-toolbar">
            <div>
              <div className="eyebrow">Virtual Guild</div>
              <h2>소모임 추천</h2>
            </div>
            <span className="pill">community {guilds.data?.source_community_id ?? "-"}</span>
          </div>
          <p className="muted small">{guilds.data?.scoring_policy ?? "SIMILAR_TO 기반 후보를 불러옵니다."}</p>
          <StatusMessage loading={guilds.loading} error={guilds.error} empty={guilds.data?.guilds.length === 0} />
          <div className="results-list">
            {guilds.data?.guilds.map((guild) => (
              <div className="result-card" key={guild.guild_id}>
                <h3>{guild.title}</h3>
                <p className="muted small">score {guild.score.toFixed(3)} · {guild.reason}</p>
                <Pills items={[...guild.shared_hobbies.slice(0, 5), ...guild.shared_skills.slice(0, 5)]} />
                <div className="results-list" style={{ marginTop: 12 }}>
                  {guild.members.map((member) => (
                    <button className="result-card" key={member.uuid} onClick={() => onSelectPersona(member.uuid, member.display_name)}>
                      <strong>{member.display_name ?? shortUuid(member.uuid)} {member.is_leader ? "· leader" : ""}</strong>
                      <p className="muted small">{joinDefined([member.age ? `${member.age}세` : null, member.province, member.district, member.occupation, `score ${member.score.toFixed(3)}`])}</p>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="section-toolbar">
            <div>
              <div className="eyebrow">Similar but Different</div>
              <h2>비슷하지만 다른 페르소나</h2>
            </div>
            <select className="select" style={{ width: 160 }} value={diversityAxis} onChange={(event) => setDiversityAxis(event.target.value)}>
              <option value="mixed">mixed</option>
              <option value="occupation">occupation</option>
              <option value="location">location</option>
              <option value="community">community</option>
              <option value="demographic">demographic</option>
            </select>
          </div>
          <p className="muted small">{diverse.data?.scoring_policy ?? "유사도에 다양성 보정을 더해 정렬합니다."}</p>
          <StatusMessage loading={diverse.loading} error={diverse.error} empty={diverse.data?.results.length === 0} />
          <div className="results-list">
            {diverse.data?.results.map((persona) => (
              <button className="result-card" key={persona.uuid} onClick={() => onSelectPersona(persona.uuid, persona.display_name)}>
                <strong>{persona.display_name ?? shortUuid(persona.uuid)}</strong>
                <p className="muted small">
                  {joinDefined([persona.age ? `${persona.age}세` : null, persona.sex, persona.province, persona.district, persona.occupation])}
                </p>
                <p className="small muted">similarity {percent(persona.similarity)} · diversity {percent(persona.diversity_score)} · final {persona.final_score.toFixed(3)}</p>
                <Pills items={persona.contrast_reasons.slice(0, 4)} />
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid two">
        <div className="card">
          <div className="eyebrow">Community Profile</div>
          <h2>{community.data?.label ?? "커뮤니티 프로필"}</h2>
          <StatusMessage loading={community.loading} error={community.error} empty={community.data === null && !community.loading} />
          {community.data && (
            <div className="grid">
              <p className="muted small">{community.data.summary}</p>
              <div className="grid two">
                <RankedList title="지역" items={community.data.top_provinces} />
                <RankedList title="직업" items={community.data.top_occupations} />
                <RankedList title="취미" items={community.data.top_hobbies} />
                <RankedList title="스킬" items={community.data.top_skills} />
              </div>
              <div className="results-list">
                {community.data.representative_personas.slice(0, 4).map((persona) => (
                  <button className="result-card" key={persona.uuid} onClick={() => onSelectPersona(persona.uuid, persona.display_name)}>
                    <strong>{persona.display_name ?? shortUuid(persona.uuid)}</strong>
                    <p className="muted small">{joinDefined([persona.age ? `${persona.age}세` : null, persona.province, persona.district, persona.occupation])}</p>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <div className="eyebrow">Life Track</div>
          <h2>롤모델 경로 탐색</h2>
          <p className="muted small">{lifeTrack.data?.interpretation_policy ?? "유사 older cohort를 찾습니다."}</p>
          <StatusMessage loading={lifeTrack.loading} error={lifeTrack.error} empty={lifeTrack.data?.role_models.length === 0} />
          {lifeTrack.data && (
            <div className="grid">
              <div className="grid two">
                <RankedList title="관찰 직업" items={lifeTrack.data.transitions.occupations ?? []} />
                <RankedList title="관찰 스킬" items={lifeTrack.data.transitions.skills ?? []} />
              </div>
              <div className="results-list">
                {lifeTrack.data.timeline.map((item) => (
                  <div className="result-card" key={item.age_band}>
                    <strong>{item.age_band} · evidence {compactNumber(item.evidence_count)}</strong>
                    <Pills items={[...item.representative_occupations, ...item.representative_skills.slice(0, 4), ...item.representative_hobbies.slice(0, 4)]} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="section-toolbar">
          <div>
            <div className="eyebrow">Segment Insight</div>
            <h2>세그먼트 자동 요약</h2>
          </div>
          <button className="primary-button" disabled={segmentInsight.loading} onClick={runSegmentInsight}>요약 생성</button>
        </div>
        <div className="form-grid">
          <input className="input" value={segments.aLabel} onChange={(event) => setSegments({ ...segments, aLabel: event.target.value })} />
          <input className="input" value={segments.aProvince} onChange={(event) => setSegments({ ...segments, aProvince: event.target.value })} />
          <input className="input" value={segments.aAgeGroup} onChange={(event) => setSegments({ ...segments, aAgeGroup: event.target.value })} />
          <input className="input" value={segments.bLabel} onChange={(event) => setSegments({ ...segments, bLabel: event.target.value })} />
          <input className="input" value={segments.bProvince} onChange={(event) => setSegments({ ...segments, bProvince: event.target.value })} />
          <input className="input" value={segments.bAgeGroup} onChange={(event) => setSegments({ ...segments, bAgeGroup: event.target.value })} />
        </div>
        <StatusMessage loading={segmentInsight.loading} error={segmentInsight.error} empty={false} />
        {segmentInsight.data && (
          <div className="result-card" style={{ marginTop: 12 }}>
            <p>{segmentInsight.data.deterministic_summary}</p>
            <p className="muted small">
              {segmentInsight.data.segment_a.label} {compactNumber(segmentInsight.data.segment_a.count)}명 · {segmentInsight.data.segment_b.label} {compactNumber(segmentInsight.data.segment_b.count)}명
            </p>
          </div>
        )}
      </div>
    </section>
  );
}

function StatusMessage({ loading, error, empty }: { loading: boolean; error: string | null; empty: boolean }) {
  if (loading) return <p className="muted small">불러오는 중입니다.</p>;
  if (error) return <div className="error-box modal-error">{error}</div>;
  if (empty) return <p className="muted small">표시할 결과가 없습니다.</p>;
  return null;
}

function Pills({ items }: { items: string[] }) {
  if (items.length === 0) return null;
  return <div className="pill-row">{items.map((item) => <span className="pill" key={item}>{item}</span>)}</div>;
}

function RankedList({ title, items }: { title: string; items: Array<{ label: string; count: number }> }) {
  return (
    <div className="result-card">
      <strong>{title}</strong>
      <div className="bar-list" style={{ marginTop: 8 }}>
        {items.slice(0, 6).map((item) => (
          <div className="bar-row" key={`${title}-${item.label}`}>
            <span>{item.label}</span>
            <span className="small muted">{compactNumber(item.count)}</span>
          </div>
        ))}
        {items.length === 0 && <p className="muted small">표시할 결과가 없습니다.</p>}
      </div>
    </div>
  );
}
