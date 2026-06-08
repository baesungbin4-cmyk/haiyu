import { type SystemMetric } from "../services/mockData";

interface KpiCardProps {
  metric: SystemMetric;
}

export default function KpiCard({ metric }: KpiCardProps) {
  return (
    <article className={`kpi-card tone-${metric.tone}`}>
      <div className="kpi-card__meta">
        <span>{metric.label}</span>
        <em>{metric.tone === "red" ? "Review" : "Live"}</em>
      </div>
      <strong>{metric.value}</strong>
      <p>{metric.delta}</p>
    </article>
  );
}
