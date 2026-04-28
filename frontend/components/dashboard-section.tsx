import type { StatsResponse } from "@/lib/api-types";
import type { Loadable } from "@/hooks/use-loadable";
import { compactNumber, fullNumber, percent } from "@/lib/formatters";
import { MetricCard } from "./metric-card";

interface DashboardSectionProps {
  stats: Loadable<StatsResponse>;
}

export function DashboardSection({ stats }: DashboardSectionProps) {
  const topAge = stats.data?.age_distribution[0]?.label ?? "-";
  const apiStatus = stats.error ? "오류" : stats.data ? "연결됨" : "대기";
  return (
    <section className="grid">
      <div className="grid three">
        <MetricCard title="총 페르소나" value={stats.data ? fullNumber(stats.data.total_personas) : "-"} caption="Neo4j Person nodes" loading={stats.loading} />
        <MetricCard title="대표 연령대" value={topAge} caption="age_distribution 기준" loading={stats.loading} />
        <MetricCard title="API 상태" value={apiStatus} caption={stats.error ?? "FastAPI /api/stats 연결"} loading={stats.loading} />
      </div>
      {stats.error && <div className="card error-box">{stats.error}</div>}
      <div className="grid two">
        <DistributionCard title="연령대 분포" items={stats.data?.age_distribution ?? []} />
        <DistributionCard title="성별 분포" items={stats.data?.sex_distribution ?? []} />
        <DistributionCard title="지역 분포" items={stats.data?.province_distribution ?? []} />
        <RankedCard title="인기 취미" items={stats.data?.top_hobbies ?? []} />
        <RankedCard title="상위 직업" items={stats.data?.top_occupations ?? []} />
        <RankedCard title="상위 스킬" items={stats.data?.top_skills ?? []} />
      </div>
    </section>
  );
}

function DistributionCard({ title, items }: { title: string; items: Array<{ label: string; count: number; ratio: number }> }) {
  const max = Math.max(...items.map((item) => item.count), 1);
  return (
    <div className="card">
      <h3>{title}</h3>
      <div className="bar-list">
        {items.slice(0, 8).map((item) => {
          const level = item.count / max;
          const ratioLabel = percent(item.ratio);
          return (
          <div className="bar-row interactive" key={item.label} title={`${item.label}: ${ratioLabel}`} aria-label={`${item.label} ${ratioLabel}`}>
            <span>{item.label}</span>
            <span className="bar-track"><span className="bar-fill" style={{ width: `${level * 100}%`, background: barGradient(level) }} /></span>
            <span className="small muted">{ratioLabel}</span>
          </div>
          );
        })}
      </div>
    </div>
  );
}

function RankedCard({ title, items }: { title: string; items: Array<{ label: string; count: number }> }) {
  const max = Math.max(...items.map((item) => item.count), 1);
  return (
    <div className="card">
      <h3>{title}</h3>
      <div className="bar-list">
        {items.slice(0, 8).map((item) => {
          const level = item.count / max;
          return (
          <div className="bar-row interactive" key={item.label} title={`${item.label}: ${compactNumber(item.count)}`} aria-label={`${item.label} ${compactNumber(item.count)}`}>
            <span>{item.label}</span>
            <span className="bar-track"><span className="bar-fill" style={{ width: `${level * 100}%`, background: barGradient(level) }} /></span>
            <span className="small muted">{compactNumber(item.count)}</span>
          </div>
          );
        })}
      </div>
    </div>
  );
}

function barGradient(level: number) {
  const clamped = Math.max(0, Math.min(level, 1));
  if (clamped > 0.72) return "linear-gradient(90deg, #74d6ff 0%, #8bffb0 100%)";
  if (clamped > 0.42) return "linear-gradient(90deg, #74d6ff 0%, #ffd166 100%)";
  return "linear-gradient(90deg, #d7a8ff 0%, #74d6ff 100%)";
}
