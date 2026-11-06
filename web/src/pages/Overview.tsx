import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { Clock, LogIn, LogOut, Repeat, Shirt, UserCheck, Users, UserRound } from "lucide-react";
import { api, Summary } from "../api";
import ActivityChart from "../components/ActivityChart";
import KpiCard from "../components/KpiCard";
import LiveMonitor from "../components/LiveMonitor";
import { Badge, Card, Empty, Thumb } from "../components/ui";
import { usePolling } from "../usePolling";
import { fmtDuration, fmtTime } from "../utils";

export default function Overview({ summary, paused, onSelectVisitor }: {
  summary: Summary | null; paused: boolean; onSelectVisitor: (id: number) => void;
}) {
  const timeline = usePolling(api.timeline, [], 3000, paused);
  const recent = usePolling(() => api.events("", 8, 0), [], 3000, paused);
  const away = usePolling(api.awayVisitors, [], 4000, paused);
  const s = summary;
  const returning = s?.returning_visitors ?? 0;
  const mix = [
    { name: "New", value: Math.max(0, (s?.unique_visitors ?? 0) - returning), color: "#2563eb" },
    { name: "Returning", value: returning, color: "#8b5cf6" },
  ];

  return (
    <>
      <LiveMonitor paused={paused} />

      <div className="kpi-grid">
        <KpiCard label="Unique visitors" value={s?.unique_visitors ?? "—"} hint="Distinct faces registered" icon={<Users size={20} />} tone="blue" />
        <KpiCard label="Currently inside" value={s?.currently_inside ?? "—"} hint={`Peak ${s?.peak_occupancy ?? 0} at once`} icon={<UserCheck size={20} />} tone="green" />
        <KpiCard label="Total entries" value={s?.total_entries ?? "—"} hint="Includes re-entries" icon={<LogIn size={20} />} tone="orange" />
        <KpiCard label="Total exits" value={s?.total_exits ?? "—"} icon={<LogOut size={20} />} tone="slate" />
        <KpiCard label="Avg. time on camera" value={fmtDuration(s?.avg_dwell_seconds)} hint={`${returning} returning visitor${returning === 1 ? "" : "s"}`} icon={<Clock size={20} />} tone="violet" />
        <KpiCard
          label="Away walkers"
          value={s?.away_unique ?? "—"}
          hint={`${s?.away_currently ?? 0} in view · ${s?.away_with_clothes ?? 0} with clothes logged`}
          icon={<UserRound size={20} />}
          tone="orange"
        />
        <KpiCard
          label="Away clothing logged"
          value={s?.away_with_clothes ?? "—"}
          hint={`${s?.away_entries ?? 0} away entries / ${s?.away_exits ?? 0} exits`}
          icon={<Shirt size={20} />}
          tone="violet"
        />
      </div>

      <div className="grid-2">
        <Card title="Activity over time" className="span-2">
          <ActivityChart data={timeline.data} />
        </Card>
        <Card title="New vs returning">
          {(s?.unique_visitors ?? 0) === 0 ? (
            <Empty title="No visitors yet" />
          ) : (
            <>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={mix} dataKey="value" innerRadius={55} outerRadius={80} paddingAngle={3} stroke="none">
                    {mix.map((m) => <Cell key={m.name} fill={m.color} />)}
                  </Pie>
                  <Tooltip contentStyle={{ borderRadius: 8, fontSize: 13 }} />
                </PieChart>
              </ResponsiveContainer>
              <ul className="legend">
                {mix.map((m) => (
                  <li key={m.name}><i style={{ background: m.color }} />{m.name}<b>{m.value}</b></li>
                ))}
              </ul>
              <p className="muted small"><Repeat size={12} /> Returning = identified again after leaving the frame; they never add to the unique count.</p>
            </>
          )}
        </Card>
      </div>

      <Card title="Latest events">
        {recent.data && recent.data.items.length > 0 ? (
          <table className="table">
            <thead><tr><th>Face</th><th>Visitor</th><th>Event</th><th>Time</th></tr></thead>
            <tbody>
              {recent.data.items.map((e) => (
                <tr key={e.id} className="clickable" onClick={() => onSelectVisitor(e.face_id)}>
                  <td><Thumb src={e.image} size={36} /></td>
                  <td>Visitor #{e.face_id}</td>
                  <td><Badge kind={e.event_type}>{e.event_type === "entry" ? "Entry" : "Exit"}</Badge></td>
                  <td className="muted">{fmtTime(e.timestamp)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <Empty title="No events recorded" hint="Run the tracker (python main.py) to populate the dashboard." />
        )}
      </Card>

      <Card title="Away walkers (facing opposite / backs)">
        {away.data && away.data.length > 0 ? (
          <div className="visitor-grid">
            {away.data.slice(0, 12).map((v) => (
              <div key={v.away_id} className="visitor-card" style={{ cursor: "default" }}>
                <Thumb src={v.thumbnail} size={72} />
                <div className="visitor-info">
                  <div className="visitor-top">
                    <b>Away #{v.away_id}</b>
                    <Badge kind="left">Away</Badge>
                  </div>
                  {v.appearance_summary ? (
                    <div className="appearance-line" title={v.appearance_summary}>{v.appearance_summary}</div>
                  ) : (
                    <div className="muted small">Clothes not tagged yet</div>
                  )}
                  <div className="muted small">First {fmtTime(v.first_seen)}</div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <Empty
            title="No away walkers yet"
            hint="People walking with their back to the camera are detected with YOLO-person + clothing tags."
          />
        )}
      </Card>
    </>
  );
}
