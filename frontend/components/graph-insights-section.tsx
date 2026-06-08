import type {
  BridgePersonaCandidate,
  CommunityLabelCandidate,
  GraphDataQualityIssue,
  GraphInsightsResponse,
  HobbyNormalizationCandidate,
  HobbyOccupationRegionPath,
  SkillExtractionCandidate,
} from "@/lib/api-types";
import type { Loadable } from "@/hooks/use-loadable";
import { compactNumber, fullNumber, joinDefined, percent, shortUuid } from "@/lib/formatters";

interface GraphInsightsSectionProps {
  insights: Loadable<GraphInsightsResponse>;
  onSelectPersona: (uuid: string, label: string | null) => void;
}

export function GraphInsightsSection({ insights, onSelectPersona }: GraphInsightsSectionProps) {
  const data = insights.data;

  return (
    <section className="graph-insights">
      <div className="insight-header">
        <div>
          <div className="eyebrow">Graph Data Intelligence</div>
          <h2>그래프 데이터 정제와 확장 후보</h2>
          <p className="muted small">현재 Neo4j 데이터에서 바로 관측되는 후보만 읽기 전용으로 보여줍니다.</p>
        </div>
        <span className="badge">5번 제외</span>
      </div>

      {insights.error && <div className="ops-alert danger">{insights.error}</div>}
      {insights.loading && <div className="ops-alert">그래프 인사이트를 불러오는 중입니다.</div>}
      {data && <div className="ops-alert">{data.dashboard_policy}</div>}

      <div className="insight-kpi-grid">
        <InsightKpi label="페르소나" value={summaryNumber(data, "personas")} detail="Person 노드" />
        <InsightKpi label="취미 노드" value={summaryNumber(data, "hobbies")} detail={`1인 취미 ${percent(data?.summary.singleton_hobby_ratio)}`} />
        <InsightKpi label="유사도 관계" value={summaryNumber(data, "similar_edges")} detail="SIMILAR_TO" />
        <InsightKpi label="커뮤니티" value={summaryNumber(data, "communities")} detail="community_id 기준" />
      </div>

      {data && (
        <>
          <DataQualityPanel issues={data.data_quality_issues} />

          <div className="insight-grid two">
            <HobbyNormalizationPanel candidates={data.hobby_normalization_candidates} />
            <SkillCandidatePanel candidates={data.skill_extraction_candidates} />
          </div>

          <CommunityLabelPanel communities={data.community_label_candidates} />

          <div className="insight-grid two">
            <BridgePersonaPanel personas={data.bridge_personas} onSelectPersona={onSelectPersona} />
            <PathPanel paths={data.hobby_occupation_region_paths} onSelectPersona={onSelectPersona} />
          </div>
        </>
      )}
    </section>
  );
}

function InsightKpi({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="insight-kpi">
      <span className="small muted">{label}</span>
      <strong>{value}</strong>
      <span className="small muted">{detail}</span>
    </div>
  );
}

function DataQualityPanel({ issues }: { issues: GraphDataQualityIssue[] }) {
  return (
    <div className="insight-panel">
      <PanelTitle title="데이터 품질/정제 대시보드" detail="운영자가 먼저 확인할 그래프 데이터 신호입니다." badge={`${issues.length} checks`} />
      <div className="quality-issue-grid">
        {issues.map((issue) => (
          <div className="quality-issue" key={issue.name}>
            <div className="quality-issue-head">
              <span className={`badge ${issue.severity === "warning" ? "danger" : ""}`}>{severityLabel(issue.severity)}</span>
              <strong>{issue.name}</strong>
            </div>
            <div className="quality-meter" aria-label={`${issue.name} 비율`}>
              <span style={{ width: `${Math.min(issue.ratio * 100, 100)}%` }} />
            </div>
            <div className="quality-numbers">
              <span>{fullNumber(Math.round(issue.value))}</span>
              <span>{percent(issue.ratio)}</span>
            </div>
            <p>{issue.impact}</p>
            <p className="muted small">{issue.recommendation}</p>
            {issue.examples.length > 0 && (
              <div className="chip-row">
                {issue.examples.slice(0, 4).map((example) => (
                  <span className="pill" key={example}>{example}</span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function HobbyNormalizationPanel({ candidates }: { candidates: HobbyNormalizationCandidate[] }) {
  return (
    <div className="insight-panel">
      <PanelTitle title="취미 정규화 후보" detail="문장형 취미를 canonical 후보로 묶어볼 수 있는 키워드입니다." badge={`${candidates.length} groups`} />
      <div className="candidate-list">
        {candidates.map((candidate) => (
          <div className="candidate-row" key={candidate.keyword}>
            <div>
              <strong>{candidate.canonical_label}</strong>
              <p className="small muted">{compactNumber(candidate.support_count)} mentions · {candidate.variant_count} variants</p>
            </div>
            <div className="variant-stack">
              {candidate.variants.slice(0, 3).map((variant) => (
                <span key={variant.name}>{variant.name} <em>{variant.count}</em></span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SkillCandidatePanel({ candidates }: { candidates: SkillExtractionCandidate[] }) {
  return (
    <div className="insight-panel">
      <PanelTitle title="스킬 그래프 생성 후보" detail="property 텍스트에서 Skill 노드 후보로 분리 가능한 표현입니다." badge={`${candidates.length} terms`} />
      <div className="skill-cloud">
        {candidates.map((candidate) => (
          <div className="skill-token" key={candidate.name}>
            <strong>{candidate.name}</strong>
            <span>{candidate.count}</span>
            {candidate.examples[0] && <p>{candidate.examples[0]}</p>}
          </div>
        ))}
      </div>
    </div>
  );
}

function CommunityLabelPanel({ communities }: { communities: CommunityLabelCandidate[] }) {
  return (
    <div className="insight-panel">
      <PanelTitle title="커뮤니티 자동 라벨링" detail="숫자 community_id 대신 운영자가 이해하기 쉬운 군집명을 제안합니다." badge={`${communities.length} communities`} />
      <div className="community-label-grid">
        {communities.map((community) => (
          <div className="community-label" key={community.community_id}>
            <span className="small muted">Community {community.community_id}</span>
            <strong>{community.label}</strong>
            <p>{community.summary}</p>
            <div className="chip-row">
              <span className="pill">{compactNumber(community.size)}명</span>
              <span className="pill">{joinDefined([community.top_province, community.top_occupation, community.top_hobby_keyword])}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function BridgePersonaPanel({
  personas,
  onSelectPersona,
}: {
  personas: BridgePersonaCandidate[];
  onSelectPersona: (uuid: string, label: string | null) => void;
}) {
  return (
    <div className="insight-panel">
      <PanelTitle title="브릿지 페르소나" detail="여러 community와 연결되는 유사도 허브 후보입니다." badge={`${personas.length} personas`} />
      <div className="insight-table">
        <div className="insight-table-row head">
          <span>이름</span>
          <span>커뮤니티</span>
          <span>연결</span>
          <span>평균</span>
        </div>
        {personas.map((persona) => (
          <button className="insight-table-row clickable" key={persona.uuid} type="button" onClick={() => onSelectPersona(persona.uuid, persona.display_name)}>
            <strong>{persona.display_name || shortUuid(persona.uuid)}</strong>
            <span>{persona.community_id ?? "-"}</span>
            <span>{persona.neighbor_community_count}개</span>
            <span>{percent(persona.average_similarity)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function PathPanel({
  paths,
  onSelectPersona,
}: {
  paths: HobbyOccupationRegionPath[];
  onSelectPersona: (uuid: string, label: string | null) => void;
}) {
  return (
    <div className="insight-panel">
      <PanelTitle title="취미-직업-지역 경로" detail="Person을 매개로 함께 나타나는 삼각 경로입니다." badge={`${paths.length} paths`} />
      <div className="path-stack">
        {paths.map((path) => (
          <div className="path-row" key={`${path.hobby_keyword}-${path.occupation}-${path.province}`}>
            <div>
              <strong>{path.hobby_keyword}</strong>
              <p className="small muted">{path.occupation} · {path.province}</p>
            </div>
            <span className="pill">{compactNumber(path.support_count)}명</span>
            {path.representative_persona_uuid && (
              <button className="ghost-button" type="button" onClick={() => onSelectPersona(path.representative_persona_uuid!, path.representative_persona_name)}>
                {path.representative_persona_name || shortUuid(path.representative_persona_uuid)}
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function PanelTitle({ title, detail, badge }: { title: string; detail: string; badge: string }) {
  return (
    <div className="ops-panel-title">
      <div>
        <h3>{title}</h3>
        <p className="small muted">{detail}</p>
      </div>
      <span className="badge">{badge}</span>
    </div>
  );
}

function summaryNumber(data: GraphInsightsResponse | null | undefined, key: string): string {
  const value = data?.summary[key];
  return typeof value === "number" ? compactNumber(value) : "-";
}

function severityLabel(value: string): string {
  if (value === "ok") return "정상";
  if (value === "warning") return "확인";
  return "정보";
}
