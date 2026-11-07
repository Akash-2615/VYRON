import { ReactNode } from "react";

export default function KpiCard({ label, value, hint, icon, tone, badge, badgeTitle }: {
  label: string; value: ReactNode; hint?: string; icon: ReactNode;
  tone: "blue" | "green" | "orange" | "violet" | "slate" | "pink";
  badge?: { text: string; tone: "ok" | "warn" | "muted" };
  badgeTitle?: string;
}) {
  return (
    <div className="kpi">
      <div className={`kpi-icon tone-${tone}`}>{icon}</div>
      <div className="kpi-body">
        <div className="kpi-label">{label}</div>
        <div className="kpi-value">{value}</div>
        {hint && <div className="kpi-hint">{hint}</div>}
        {badge && (
          <span
            className={`kpi-verify kpi-verify-${badge.tone}`}
            title={badgeTitle || badge.text}
          >
            {badge.text}
          </span>
        )}
      </div>
    </div>
  );
}
