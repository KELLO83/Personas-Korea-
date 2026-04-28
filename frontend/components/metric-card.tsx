interface MetricCardProps {
  title: string;
  value: string;
  caption: string;
  loading: boolean;
}

export function MetricCard({ title, value, caption, loading }: MetricCardProps) {
  return (
    <div className="card">
      <p className="small muted">{title}</p>
      <div className="metric">{loading ? "…" : value}</div>
      <p className="small muted">{caption}</p>
    </div>
  );
}
