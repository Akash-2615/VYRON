import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { Clock, Repeat, Shirt, UserRound, Users, Video } from "lucide-react";
import { api, Summary } from "../api";
import ActivityChart from "../components/ActivityChart";
import KpiCard from "../components/KpiCard";
import { Badge, Card, Empty, Thumb } from "../components/ui";
import { useCssVar } from "../theme";
import { usePolling } from "../usePolling";
import { fmtDuration, fmtTime } from "../utils";

function InFrameKpi({ paused }: { paused: boolean }) {
  const live = usePolling(api.liveStatus, [], 800, paused);
  const n = live.data?.persons_in_frame;
  const faces = live.data?.faces_in_frame ?? 0;
  const noFace = live.data?.no_face_in_frame ?? 0;
  const active = Boolean(live.data?.active);
  return (
    <KpiCard
      label="Persons in frame"
      value={active && n != null ? n : "—"}
      hint={!active ? "Waiting for live tracker" : `${faces} face · ${noFace} no-face`}
      icon={<Video size={20} />}
      tone="blue"
    />
  );
}

export default function Analytics({ summary, paused, onSelectVisitor }: {
  summary: Summary | null; paused: boolean; onSelectVisitor: (id: number) => void;
}) {
  const timeline = usePolling(api.timeline, [], 3000, paused);
  const away = usePolling(api.awayVisitors, [], 4000, paused);
  const surface = useCssVar("--surface", "#1a182e");
  const border = useCssVar("--border", "#2d2b45");
  const text = useCssVar("--text", "#f4f2fb");
  const entryColor = useCssVar("--chart-entry", "#a855f7");
  const returnColor = useCssVar("--blue", "#60a5fa");
  const s = summary;
  const returning = s?.returning_visitors ?? 0;
  const mix = [
    { name: "New", value: Math.max(0, (s?.unique_visitors ?? 0) - returning), color: entryColor },
    { name: "Returning", value: returning, color: returnColor },
  ];

  return (
    <>
      <div className="kpi-grid">
        <InFrameKpi paused={paused} />
        <KpiCard
          label="Avg time in frame"
          value={fmtDuration(s?.avg_time_in_frame_seconds ?? s?.avg_dwell_seconds)}
          hint={`${returning} returning visitors`}
          icon={<Clock size={20} />}
          tone="violet"
        />
        <KpiCard
          label="No-face people"
          value={s?.away_unique ?? "—"}
          hint={`${s?.away_currently ?? 0} in view · not in unique face count`}
          icon={<UserRound size={20} />}
          tone="orange"
        />
        <KpiCard
          label="No-face clothing logged"
          value={s?.away_with_clothes ?? "—"}
          hint={`${s?.away_entries ?? 0} entries / ${s?.away_exits ?? 0} exits`}
          icon={<Shirt size={20} />}
          tone="slate"
        />
        <KpiCard
          label="Returning visitors"
          value={returning}
          hint="Re-identified after leaving — never increases unique"
          icon={<Repeat size={20} />}
          tone="violet"
        />
        <KpiCard
          label="Peak occupancy"
          value={s?.peak_occupancy ?? "—"}
          hint={s?.hidden_now ? `${s.hidden_now} currently hidden` : "Max concurrent inside"}
          icon={<Users size={20} />}
          tone="blue"
        />
      </div>

      <div className="analytics-row">
        <Card title="Visitor flow">
          <ActivityChart data={timeline.data} />
        </Card>
        <Card title="New vs returning">
          {(s?.unique_visitors ?? 0) === 0 ? (
            <Empty title="No visitors yet" />
          ) : (
            <>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={mix} dataKey="value" innerRadius={55} outerRadius={80} paddingAngle={3} stroke="none">
                    {mix.map((m) => <Cell key={m.name} fill={m.color} />)}
                  </Pie>
                  <Tooltip contentStyle={{ borderRadius: 10, fontSize: 13, background: surface, border: `1px solid ${border}`, color: text }} />
                </PieChart>
              </ResponsiveContainer>
              <ul className="legend">
                {mix.map((m) => (
                  <li key={m.name}><i style={{ background: m.color }} />{m.name}<b>{m.value}</b></li>
                ))}
              </ul>
            </>
          )}
        </Card>
      </div>

      <Card title="No-face people (excluded from unique count)">
        {away.data && away.data.length > 0 ? (
          <div className="visitor-grid noface-grid">
            {away.data.slice(0, 16).map((v) => {
              const inside = v.status === "inside";
              return (
                <div key={v.away_id} className="visitor-card noface-card" style={{ cursor: "default" }}>
                  {(v.appearance_image || v.thumbnail) ? (
                    <img className="noface-thumb" src={v.appearance_image || v.thumbnail || ""} alt="" />
                  ) : (
                    <Thumb src={null} size={96} />
                  )}
                  <div className="visitor-info">
                    <div className="visitor-top">
                      <b>No-face #{v.away_id}</b>
                      <Badge kind={inside ? "inside" : "left"}>{inside ? "Inside" : "Left"}</Badge>
                    </div>
                    {v.appearance_summary ? (
                      <div className="appearance-line" title={v.appearance_summary}>{v.appearance_summary}</div>
                    ) : (
                      <div className="muted small">Clothes not tagged yet</div>
                    )}
                    {v.linked_face_id != null ? (
                      <div className="muted small">
                        Turned → face <a onClick={() => onSelectVisitor(Number(v.linked_face_id))}>#{v.linked_face_id}</a>
                      </div>
                    ) : (
                      <div className="muted small">
                        {inside ? "Still in frame" : "First"} {fmtTime(inside ? v.last_seen : v.first_seen)}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <Empty title="No no-face people yet" hint="Person boxes without a usable face land here." />
        )}
      </Card>
    </>
  );
}
