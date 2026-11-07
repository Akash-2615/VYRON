import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Timeline } from "../api";
import { useCssVar } from "../theme";
import { axisLabel, fmtDateTime } from "../utils";
import { Empty } from "./ui";

export default function ActivityChart({ data, height = 300, showLegend = true, fill = false }: {
  data: Timeline | null; height?: number; showLegend?: boolean; fill?: boolean;
}) {
  const grid = useCssVar("--chart-grid", "#2d2b45");
  const muted = useCssVar("--muted", "#9b97b0");
  const surface = useCssVar("--surface", "#1a182e");
  const border = useCssVar("--border", "#2d2b45");
  const text = useCssVar("--text", "#f4f2fb");
  const entry = useCssVar("--chart-entry", "#a855f7");
  const exit = useCssVar("--chart-exit", "#60a5fa");
  const occ = useCssVar("--chart-occ", "#34d399");

  if (!data || data.points.length === 0) {
    return <Empty title="No activity yet" hint="Events appear here once the tracker logs entries and exits." />;
  }
  const rows = data.points.map((p) => ({ ...p, label: axisLabel(p.t, data.bucket_seconds) }));
  const legendItems = [
    { key: "entries", label: "Entries", color: entry },
    { key: "exits", label: "Exits", color: exit },
    { key: "occupancy", label: "Inside now", color: occ },
  ];

  return (
    <div className={`activity-chart ${fill ? "is-fill" : ""}`}>
      {showLegend && (
        <ul className="chart-legend" aria-label="Chart legend">
          {legendItems.map((item) => (
            <li key={item.key}>
              <i style={{ background: item.color }} />
              <span>{item.label}</span>
            </li>
          ))}
        </ul>
      )}
      <div className="activity-chart-plot">
        <ResponsiveContainer width="100%" height={fill ? "100%" : height}>
          <ComposedChart data={rows} margin={{ top: 4, right: 8, left: -8, bottom: 2 }}>
            <defs>
              <linearGradient id="vyronEntryFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={entry} stopOpacity={0.45} />
                <stop offset="100%" stopColor={entry} stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="vyronExitFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={exit} stopOpacity={0.35} />
                <stop offset="100%" stopColor={exit} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={grid} vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize: 11, fill: muted }} tickLine={false} axisLine={{ stroke: grid }} minTickGap={24} />
            <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: muted }} tickLine={false} axisLine={false} width={28} />
            <Tooltip
              labelFormatter={(_, payload) => (payload?.[0] ? fmtDateTime((payload[0].payload as { t: string }).t) : "")}
              contentStyle={{ borderRadius: 10, border: `1px solid ${border}`, fontSize: 13, background: surface, color: text }}
            />
            <Area type="monotone" dataKey="entries" name="Entries" stroke={entry} fill="url(#vyronEntryFill)" strokeWidth={2} />
            <Area type="monotone" dataKey="exits" name="Exits" stroke={exit} fill="url(#vyronExitFill)" strokeWidth={2} />
            <Line dataKey="occupancy" name="Inside now" stroke={occ} strokeWidth={2.5} dot={false} type="monotone" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
