import { useState } from "react";
import type { CareerTransitionResponse, GraphQualityResponse, LifestyleMapResponse, TargetPersonaResponse } from "@/lib/api-types";
import { personaApi } from "@/lib/api-client";
import { compactNumber, percent } from "@/lib/formatters";

type LoadState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
};

const initialState = { data: null, loading: false, error: null };

export function InsightsSection() {
  const [target, setTarget] = useState<LoadState<TargetPersonaResponse>>(initialState);
  const [lifestyle, setLifestyle] = useState<LoadState<LifestyleMapResponse>>(initialState);
  const [career, setCareer] = useState<LoadState<CareerTransitionResponse>>(initialState);
  const [quality, setQuality] = useState<LoadState<GraphQualityResponse>>(initialState);

  const [targetFilters, setTargetFilters] = useState({ age_group: "30대", province: "서울", occupation: "", hobby: "", skill: "", semantic_query: "" });
  const [lifestyleFilters, setLifestyleFilters] = useState({ source_field: "sports_persona", target_field: "culinary_persona", source_keyword: "러닝", candidate_keywords: "요리,맛집,베이킹" });
  const [careerFilters, setCareerFilters] = useState({ occupation: "개발자", compare_by: "age_group" });

  async function run<T>(setter: (state: LoadState<T>) => void, loader: () => Promise<T>) {
    setter({ data: null, loading: true, error: null });
    try {
      setter({ data: await loader(), loading: false, error: null });
    } catch (error) {
      setter({ data: null, loading: false, error: error instanceof Error ? error.message : "요청에 실패했습니다." });
    }
  }

  return (
    <section className="grid">
      <div className="hero-card">
        <div className="eyebrow">Phase 22 검수</div>
        <h1>확장 분석 콘솔</h1>
        <p className="muted">PRD F16-F18과 그래프 품질 정리 항목을 백엔드 API로 검수합니다.</p>
      </div>

      <div className="grid two">
        <div className="card">
          <h2>Target Persona Generator</h2>
          <div className="form-grid compact">
            <input className="input" placeholder="연령대" value={targetFilters.age_group} onChange={(event) => setTargetFilters({ ...targetFilters, age_group: event.target.value })} />
            <input className="input" placeholder="지역" value={targetFilters.province} onChange={(event) => setTargetFilters({ ...targetFilters, province: event.target.value })} />
            <input className="input" placeholder="직업" value={targetFilters.occupation} onChange={(event) => setTargetFilters({ ...targetFilters, occupation: event.target.value })} />
            <input className="input" placeholder="취미" value={targetFilters.hobby} onChange={(event) => setTargetFilters({ ...targetFilters, hobby: event.target.value })} />
            <input className="input" placeholder="스킬" value={targetFilters.skill} onChange={(event) => setTargetFilters({ ...targetFilters, skill: event.target.value })} />
            <input className="input" placeholder="semantic query" value={targetFilters.semantic_query} onChange={(event) => setTargetFilters({ ...targetFilters, semantic_query: event.target.value })} />
          </div>
          <button className="primary-button" style={{ marginTop: 12 }} disabled={target.loading} onClick={() => void run(setTarget, () => personaApi.targetPersona({ ...targetFilters, top_k: 5 }))}>대표 페르소나 생성</button>
          {target.error && <div className="error-box modal-error">{target.error}</div>}
          {target.data && (
            <div className="result-card" style={{ marginTop: 12 }}>
              <p className="small muted">{compactNumber(target.data.matched_count)}명 중 {target.data.sample_size}명 샘플 · {target.data.generation_method}</p>
              <p>{target.data.representative_persona}</p>
              <Pills title="취미" items={target.data.representative_hobbies} />
              <Pills title="스킬" items={target.data.representative_skills} />
              <p className="small muted">근거 UUID: {target.data.evidence_uuids.slice(0, 5).join(", ") || "-"}</p>
            </div>
          )}
        </div>

        <div className="card">
          <h2>Cross-domain Lifestyle Map</h2>
          <div className="form-grid compact">
            <input className="input" placeholder="source field" value={lifestyleFilters.source_field} onChange={(event) => setLifestyleFilters({ ...lifestyleFilters, source_field: event.target.value })} />
            <input className="input" placeholder="target field" value={lifestyleFilters.target_field} onChange={(event) => setLifestyleFilters({ ...lifestyleFilters, target_field: event.target.value })} />
            <input className="input" placeholder="source keyword" value={lifestyleFilters.source_keyword} onChange={(event) => setLifestyleFilters({ ...lifestyleFilters, source_keyword: event.target.value })} />
            <input className="input" placeholder="target keywords" value={lifestyleFilters.candidate_keywords} onChange={(event) => setLifestyleFilters({ ...lifestyleFilters, candidate_keywords: event.target.value })} />
          </div>
          <button className="primary-button" style={{ marginTop: 12 }} disabled={lifestyle.loading} onClick={() => void run(setLifestyle, () => personaApi.lifestyleMap(lifestyleFilters))}>라이프스타일 맵 계산</button>
          {lifestyle.error && <div className="error-box modal-error">{lifestyle.error}</div>}
          {lifestyle.data && <>
            <p className="small muted">{lifestyle.data.segment_policy}</p>
            <p className="small muted">{lifestyle.data.visualization_policy}</p>
            <RankedRatioList title={`${lifestyle.data.source_keyword} 조건부 연관`} items={lifestyle.data.edges.map((edge) => ({ label: edge.target_keyword, count: edge.overlap_count, ratio: edge.conditional_ratio }))} />
          </>}
        </div>
      </div>

      <div className="grid two">
        <div className="card">
          <h2>Career Transition Map</h2>
          <div className="form-grid compact">
            <input className="input" placeholder="직업" value={careerFilters.occupation} onChange={(event) => setCareerFilters({ ...careerFilters, occupation: event.target.value })} />
            <select className="select" value={careerFilters.compare_by} onChange={(event) => setCareerFilters({ ...careerFilters, compare_by: event.target.value })}>
              <option value="age_group">연령대</option>
              <option value="sex">성별</option>
              <option value="province">지역</option>
            </select>
          </div>
          <button className="primary-button" style={{ marginTop: 12 }} disabled={career.loading} onClick={() => void run(setCareer, () => personaApi.careerTransition({ ...careerFilters, top_k: 8 }))}>전환 패턴 분석</button>
          {career.error && <div className="error-box modal-error">{career.error}</div>}
          {career.data && (
            <div className="grid" style={{ marginTop: 12 }}>
              <p className="small muted">{career.data.analysis_scope}</p>
              <RankedRatioList title="목표" items={career.data.top_goals} />
              <RankedRatioList title="스킬" items={career.data.top_skills} />
              <RankedRatioList title="인접 직업" items={career.data.top_neighbor_occupations} />
              <RankedRatioList title="세그먼트" items={career.data.segment_distribution} />
            </div>
          )}
        </div>

        <div className="card">
          <h2>Graph Quality</h2>
          <p className="muted small">Country 제거, 병역/전공 필터 숨김 판단을 위한 진단입니다.</p>
          <button className="primary-button" disabled={quality.loading} onClick={() => void run(setQuality, () => personaApi.graphQuality())}>품질 진단 실행</button>
          {quality.error && <div className="error-box modal-error">{quality.error}</div>}
          <div className="results-list" style={{ marginTop: 12 }}>
            {quality.data?.checks.map((check) => (
              <div className="result-card" key={check.name}>
                <strong>{check.name}</strong>
                <p className="small muted">action={check.action} · severity={check.severity} · dominant={percent(check.dominant_ratio)}</p>
                <p>{check.recommendation}</p>
              </div>
            ))}
            {quality.data && quality.data.migration_plan.length > 0 && (
              <div className="result-card">
                <strong>Country 마이그레이션 검증 계획</strong>
                <div className="bar-list" style={{ marginTop: 8 }}>
                  {quality.data.migration_plan.map((step) => <p className="small muted" key={step.name}>{step.name}: {step.validation}</p>)}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function Pills({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <strong className="small muted">{title}</strong>
      <div className="pill-row">{items.map((item) => <span className="pill" key={item}>{item}</span>)}</div>
    </div>
  );
}

function RankedRatioList({ title, items }: { title: string; items: Array<{ name?: string; label?: string; count: number; ratio: number }> }) {
  return (
    <div className="result-card">
      <strong>{title}</strong>
      <div className="bar-list" style={{ marginTop: 8 }}>
        {items.slice(0, 6).map((item) => {
          const label = item.name ?? item.label ?? "-";
          return <div className="bar-row" key={label}><span>{label}</span><span className="small muted">{compactNumber(item.count)} · {percent(item.ratio)}</span></div>;
        })}
        {items.length === 0 && <p className="muted small">표시할 결과가 없습니다.</p>}
      </div>
    </div>
  );
}
