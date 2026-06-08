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

type ComparisonCandidate = {
  uuid: string;
  name: string;
  age: number | null;
  sex: string | null;
  occupation: string | null;
  province: string | null;
  district: string | null;
  similarity: number | null;
  diversity: number | null;
  finalScore: number | null;
  sharedHobbies: string[];
  sharedSkills: string[];
  contrastReasons: string[];
};

type SegmentPreset = {
  label: string;
  province: string;
  ageGroup: string;
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
  const [comparisonBoard, setComparisonBoard] = useState<ComparisonCandidate[]>([]);
  const [segmentPresets, setSegmentPresets] = useState<SegmentPreset[]>([]);
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

  function pinCandidate(candidate: ComparisonCandidate) {
    setComparisonBoard((current) => {
      const next = [candidate, ...current.filter((item) => item.uuid !== candidate.uuid)];
      return next.slice(0, 4);
    });
  }

  function removeCandidate(uuid: string) {
    setComparisonBoard((current) => current.filter((item) => item.uuid !== uuid));
  }

  function saveCurrentSegments() {
    const presets = [
      { label: segments.aLabel, province: segments.aProvince, ageGroup: segments.aAgeGroup },
      { label: segments.bLabel, province: segments.bProvince, ageGroup: segments.bAgeGroup },
    ].filter((preset) => preset.label.trim());
    setSegmentPresets((current) => [...presets, ...current].slice(0, 6));
  }

  function applyPreset(side: "a" | "b", preset: SegmentPreset) {
    setSegments((current) => side === "a"
      ? { ...current, aLabel: preset.label, aProvince: preset.province, aAgeGroup: preset.ageGroup }
      : { ...current, bLabel: preset.label, bProvince: preset.province, bAgeGroup: preset.ageGroup });
  }

  return (
    <section className="grid">
      <ComparisonBoard sourceName={profile.data?.display_name ?? shortUuid(selectedUuid)} candidates={comparisonBoard} onRemove={removeCandidate} onSelect={onSelectPersona} />

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
          <div className="results-list evidence-list">
            {guilds.data?.guilds.map((guild) => (
              <div className="result-card evidence-card" key={guild.guild_id}>
                <div className="evidence-head">
                  <h3>{guild.title}</h3>
                  <span className="evidence-score">{guild.score.toFixed(3)}</span>
                </div>
                <p className="muted small reason-line">{guild.reason}</p>
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
          <div className="results-list evidence-list">
            {diverse.data?.results.map((persona) => (
              <div className="result-card evidence-card" key={persona.uuid}>
                <div className="evidence-head">
                  <strong>{persona.display_name ?? shortUuid(persona.uuid)}</strong>
                  <span className="evidence-score">{persona.final_score.toFixed(3)}</span>
                </div>
                <p className="muted small">
                  {joinDefined([persona.age ? `${persona.age}세` : null, persona.sex, persona.province, persona.district, persona.occupation])}
                </p>
                <div className="reason-meter" aria-label={`similarity ${percent(persona.similarity)}, diversity ${percent(persona.diversity_score)}`}>
                  <span style={{ width: `${Math.min(persona.similarity, 1) * 100}%` }} />
                  <span style={{ width: `${Math.min(persona.diversity_score, 1) * 100}%` }} />
                </div>
                <p className="small muted">similarity {percent(persona.similarity)} · diversity {percent(persona.diversity_score)}</p>
                <Pills items={persona.contrast_reasons.slice(0, 4)} />
                <div className="card-actions">
                  <button className="ghost-button" type="button" onClick={() => pinCandidate(candidateFromDiverse(persona))}>비교 고정</button>
                  <button className="ghost-button" type="button" onClick={() => onSelectPersona(persona.uuid, persona.display_name)}>이 사람 선택</button>
                </div>
              </div>
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
        <div className="segment-builder-strip">
          <button className="ghost-button" type="button" onClick={saveCurrentSegments}>현재 세그먼트 저장</button>
          {segmentPresets.map((preset, index) => (
            <div className="segment-preset" key={`${preset.label}-${index}`}>
              <strong>{preset.label}</strong>
              <span>{joinDefined([preset.province, preset.ageGroup])}</span>
              <button type="button" onClick={() => applyPreset("a", preset)}>A</button>
              <button type="button" onClick={() => applyPreset("b", preset)}>B</button>
            </div>
          ))}
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

function ComparisonBoard({
  sourceName,
  candidates,
  onRemove,
  onSelect,
}: {
  sourceName: string;
  candidates: ComparisonCandidate[];
  onRemove: (uuid: string) => void;
  onSelect: (uuid: string, label: string | null) => void;
}) {
  return (
    <div className="card comparison-board">
      <div className="section-toolbar">
        <div>
          <div className="eyebrow">Recommendation Comparison</div>
          <h2>추천 이유 비교 보드</h2>
        </div>
        <span className="pill">{sourceName} 기준 · {candidates.length}/4</span>
      </div>
      {candidates.length === 0 ? (
        <p className="muted small">비슷하지만 다른 페르소나 카드에서 `비교 고정`을 눌러 후보를 모아보세요.</p>
      ) : (
        <div className="comparison-grid">
          {candidates.map((candidate) => (
            <div className="comparison-card" key={candidate.uuid}>
              <div className="evidence-head">
                <strong>{candidate.name}</strong>
                <span className="evidence-score">{candidate.finalScore?.toFixed(3) ?? "-"}</span>
              </div>
              <p className="small muted">{joinDefined([candidate.age ? `${candidate.age}세` : null, candidate.sex, candidate.province, candidate.district, candidate.occupation])}</p>
              <div className="comparison-matrix">
                <span>similarity</span><strong>{candidate.similarity === null ? "-" : percent(candidate.similarity)}</strong>
                <span>diversity</span><strong>{candidate.diversity === null ? "-" : percent(candidate.diversity)}</strong>
                <span>shared</span><strong>{candidate.sharedHobbies.length + candidate.sharedSkills.length}</strong>
                <span>contrast</span><strong>{candidate.contrastReasons.length}</strong>
              </div>
              <Pills items={[...candidate.sharedHobbies.slice(0, 3), ...candidate.sharedSkills.slice(0, 3)]} />
              <Pills items={candidate.contrastReasons.slice(0, 3)} />
              <div className="card-actions">
                <button className="ghost-button" type="button" onClick={() => onSelect(candidate.uuid, candidate.name)}>선택</button>
                <button className="ghost-button" type="button" onClick={() => onRemove(candidate.uuid)}>제거</button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function candidateFromDiverse(persona: SimilarDiverseResponse["results"][number]): ComparisonCandidate {
  return {
    uuid: persona.uuid,
    name: persona.display_name ?? shortUuid(persona.uuid),
    age: persona.age,
    sex: persona.sex,
    occupation: persona.occupation,
    province: persona.province,
    district: persona.district,
    similarity: persona.similarity,
    diversity: persona.diversity_score,
    finalScore: persona.final_score,
    sharedHobbies: persona.shared_hobbies,
    sharedSkills: persona.shared_skills,
    contrastReasons: persona.contrast_reasons,
  };
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
