import { Bar, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Timeline } from "../api";
import { axisLabel, fmtDateTime } from "../utils";
import { Empty } from "./ui";

export default function ActivityChart({ data }: { data: Timeline | null }) {
  if (!data || data.points.length === 0) {
    return <Empty title="No activity yet" hint="Events appear here once the tracker logs entries and exits." />;
  }
  const rows = data.points.map((p) => ({ ...p, label: axisLabel(p.t, data.bucket_seconds) }));
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={rows} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e9f0" vertical={false} />
        <XAxis dataKey="label" tick={{ fontSize: 12, fill: "#64748b" }} tickLine={false} axisLine={{ stroke: "#e5e9f0" }} minTickGap={24} />
        <YAxis allowDecimals={false} tick={{ fontSize: 12, fill: "#64748b" }} tickLine={false} axisLine={false} />
        <Tooltip
          labelFormatter={(_, payload) => (payload?.[0] ? fmtDateTime((payload[0].payload as { t: string }).t) : "")}
          contentStyle={{ borderRadius: 8, border: "1px solid #e5e9f0", fontSize: 13 }}
        />
        <Legend iconType="circle" wrapperStyle={{ fontSize: 13 }} />
        <Bar dataKey="entries" name="Entries" fill="#22c55e" radius={[3, 3, 0, 0]} />
        <Bar dataKey="exits" name="Exits" fill="#f97316" radius={[3, 3, 0, 0]} />
        <Line dataKey="occupancy" name="Inside now" stroke="#2563eb" strokeWidth={2.5} dot={false} type="monotone" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
