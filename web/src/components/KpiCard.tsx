import { ReactNode } from "react";

export default function KpiCard({ label, value, hint, icon, tone }: {
  label: string; value: ReactNode; hint?: string; icon: ReactNode; tone: "blue" | "green" | "orange" | "violet" | "slate";
}) {
  return (
    <div className="kpi">
      <div className={`kpi-icon tone-${tone}`}>{icon}</div>
      <div>
        <div className="kpi-label">{label}</div>
        <div className="kpi-value">{value}</div>
        {hint && <div className="kpi-hint">{hint}</div>}
      </div>
    </div>
  );
}
