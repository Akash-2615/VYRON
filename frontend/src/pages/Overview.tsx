import { useState } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import {
  ArrowDownLeft, ArrowUpRight, BadgeCheck, Check, IdCard, LogIn, LogOut, Maximize2, UserCheck, Users,
} from "lucide-react";
import { api, Summary } from "../api";
import ActivityChart from "../components/ActivityChart";
import AiPipeline from "../components/AiPipeline";
import ChartModal from "../components/ChartModal";
import KpiCard from "../components/KpiCard";
import LiveMonitor from "../components/LiveMonitor";
import { Card, Empty, Thumb } from "../components/ui";
import { useCssVar } from "../theme";
import { usePolling } from "../api";
import { fmtTime } from "../utils";

type ExpandChart = "flow" | "status" | null;

export default function Overview({ summary, paused, onSelectVisitor }: {
  summary: Summary | null; paused: boolean; onSelectVisitor: (id: number) => void;
}) {
  const [expanded, setExpanded] = useState<ExpandChart>(null);
  const timeline = usePolling(api.timeline, [], 3000, paused);
  const recent = usePolling(() => api.events("", 12, 0), [], 3000, paused);
  const live = usePolling(api.liveStatus, [], 1000, paused);
  const tracker = usePolling(api.trackerStatus, [], 2000, paused);
  const countAudit = usePolling(() => api.count("latest"), [], 4000, paused);
  const health = usePolling(async () => {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error("health");
    return r.json() as Promise<{ ok: boolean; backend?: string; db_exists?: boolean }>;
  }, [], 5000, paused);

  const surface = useCssVar("--surface", "#1a182e");
  const border = useCssVar("--border", "#2d2b45");
  const text = useCssVar("--text", "#f4f2fb");
  const cInside = useCssVar("--green", "#34d399");
  const cEntry = useCssVar("--chart-entry", "#a855f7");
  const cExit = useCssVar("--pink", "#f472b6");

  const s = summary;
  const returning = s?.returning_visitors ?? 0;
  const statusMix = [
    { name: "Inside", value: s?.currently_inside ?? 0, color: cInside },
    { name: "Entered", value: Math.max(0, (s?.total_entries ?? 0) - (s?.currently_inside ?? 0)), color: cEntry },
    { name: "Exited", value: s?.total_exits ?? 0, color: cExit },
  ].filter((x) => x.value > 0);

  const fps = live.data?.fps;
  const latency = live.data?.latency_ms;
  const res = live.data?.width && live.data?.height
    ? `${live.data.width}×${live.data.height}`
    : "—";
  const sourceKind = tracker.data?.mode === "local" ? "Local video"
    : tracker.data?.mode === "webcam" ? `Webcam ${tracker.data.url ?? "0"}`
      : "RTSP";
  const camOk = Boolean(live.data?.active || tracker.data?.running);
  const dbOk = Boolean(health.data?.ok && health.data?.db_exists !== false);
  const pipeOk = Boolean(tracker.data?.running || live.data?.active);

  const tooltipStyle = { borderRadius: 10, fontSize: 12, background: surface, border: `1px solid ${border}`, color: text };

  const audit = countAudit.data;
  let uniqueBadge: { text: string; tone: "ok" | "warn" | "muted" } | undefined;
  let uniqueBadgeTitle: string | undefined;
  if (audit && audit.consistent === true && audit.derived_from_log) {
    uniqueBadge = { text: "Verified against log", tone: "ok" };
    uniqueBadgeTitle = `session ${audit.session_id ?? "—"} · as of ${audit.as_of}`;
  } else if (audit && audit.consistent === false) {
    // Prefer gallery signals (COUNT_SNAPSHOT / last_unique_total) over session-only
    // FACE_REGISTERED count — re-runs register 0 new faces but gallery stays correct.
    const log = audit.derived_from_log;
    const logGallery =
      log?.last_unique_total
      ?? log?.system_stop_unique
      ?? log?.last_count_snapshot?.unique_total
      ?? log?.unique_visitors;
    const dbU = audit.unique_visitors;
    const delta = (logGallery != null && dbU != null)
      ? Math.abs(dbU - logGallery)
      : (audit.diffs?.length ?? 1);
    uniqueBadge = { text: `Log differs by ${delta}`, tone: "warn" };
    uniqueBadgeTitle = (audit.diffs || []).join("\n") || "DB and log counts disagree";
  }

  return (
    <div className="overview-page">
      <div className="kpi-row-5">
        <KpiCard
          label="Unique Visitors"
          value={s?.unique_visitors ?? "—"}
          hint={returning ? `${returning} returning` : "Distinct faces"}
          icon={<Users size={18} />}
          tone="violet"
          badge={uniqueBadge}
          badgeTitle={uniqueBadgeTitle}
        />
        <KpiCard
          label="Total Entries"
          value={s?.total_entries ?? "—"}
          hint="One entry per visit"
          icon={<LogIn size={18} />}
          tone="green"
        />
        <KpiCard
          label="Total Exits"
          value={s?.total_exits ?? "—"}
          hint="One exit per visit"
          icon={<LogOut size={18} />}
          tone="pink"
        />
        <KpiCard
          label="Currently Inside"
          value={s?.currently_inside ?? "—"}
          hint={s?.hidden_now ? `${s.hidden_now} hidden` : `Peak ${s?.peak_occupancy ?? 0}`}
          icon={<UserCheck size={18} />}
          tone="blue"
        />
        <KpiCard
          label="Auto Registered"
          value={s?.unique_visitors ?? "—"}
          hint="faces = unique"
          icon={<IdCard size={18} />}
          tone="violet"
        />
      </div>

      <div className="dash-board">
        <div className="dash-main">
          <LiveMonitor paused={paused} compact />

          <div className="analytics-block" id="analytics-overview">
            <div className="section-label">
              <BadgeCheck size={14} /> Analytics
            </div>
            <div className="analytics-row">
              <Card
                className="flow-card"
                title="Visitor Flow (Today)"
                action={
                  <button
                    type="button"
                    className="icon-btn"
                    title="Maximize chart"
                    aria-label="Maximize Visitor Flow chart"
                    onClick={() => setExpanded("flow")}
                  >
                    <Maximize2 size={15} />
                  </button>
                }
              >
                <ActivityChart data={timeline.data} fill />
              </Card>
              <Card
                title="Visitor Status"
                className="status-card"
                action={
                  <button
                    type="button"
                    className="icon-btn"
                    title="Maximize chart"
                    aria-label="Maximize Visitor Status chart"
                    onClick={() => setExpanded("status")}
                  >
                    <Maximize2 size={15} />
                  </button>
                }
              >
                {statusMix.length === 0 ? (
                  <Empty title="No status yet" />
                ) : (
                  <div className="status-chart-fill">
                    <div className="status-chart-plot">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                          <Pie data={statusMix} dataKey="value" innerRadius="48%" outerRadius="88%" paddingAngle={2} stroke="none">
                            {statusMix.map((m) => <Cell key={m.name} fill={m.color} />)}
                          </Pie>
                          <Tooltip contentStyle={tooltipStyle} />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    <ul className="legend legend-compact">
                      {statusMix.map((m) => (
                        <li key={m.name}><i style={{ background: m.color }} />{m.name}<b>{m.value}</b></li>
                      ))}
                    </ul>
                  </div>
                )}
              </Card>
            </div>
          </div>
        </div>

        <div className="dash-side">
          <Card title="Live Events" className="events-card">
            {recent.data && recent.data.items.length > 0 ? (
              <ul className="event-feed compact">
                {recent.data.items.map((e) => {
                  const isEntry = e.event_type === "entry";
                  return (
                    <li key={e.id}>
                      <button type="button" className="event-row" onClick={() => onSelectVisitor(e.face_id)}>
                        <Thumb src={e.image} size={32} round />
                        <span className={`event-icon ${isEntry ? "entry" : "exit"}`}>
                          {isEntry ? <ArrowDownLeft size={14} /> : <ArrowUpRight size={14} />}
                        </span>
                        <span className="event-meta">
                          <b>ID: V{String(e.face_id).padStart(4, "0")}</b>
                          <span>{isEntry ? "Entry" : "Exit"}</span>
                        </span>
                        <span className="event-time">{fmtTime(e.timestamp)}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <Empty title="No events yet" hint="Start a stream to see entries and exits." />
            )}
          </Card>

          <Card title="Camera & Processing" className="cam-card">
            <ul className="spec-list">
              <li><span>Source</span><b>{sourceKind}</b></li>
              <li><span>Resolution</span><b>{res}</b></li>
              <li><span>FPS</span><b>{fps != null ? `${fps}` : "—"}</b></li>
              <li><span>Latency</span><b>{latency != null ? `${latency} ms` : "—"}</b></li>
              <li><span>Database</span><b>{health.data?.backend ?? "—"}</b></li>
              <li><span>In frame</span><b>{live.data?.persons_in_frame ?? "—"}</b></li>
            </ul>
            <div className="spec-checks">
              <span className={camOk ? "ok" : ""}><Check size={12} /> Camera</span>
              <span className={dbOk ? "ok" : ""}><Check size={12} /> Database</span>
              <span className={pipeOk ? "ok" : ""}><Check size={12} /> AI Pipeline</span>
            </div>
          </Card>
        </div>
      </div>

      <AiPipeline active={Boolean(live.data?.active || tracker.data?.running)} />

      <ChartModal title="Visitor Flow (Today)" open={expanded === "flow"} onClose={() => setExpanded(null)}>
        <ActivityChart data={timeline.data} height={420} />
      </ChartModal>

      <ChartModal title="Visitor Status" open={expanded === "status"} onClose={() => setExpanded(null)}>
        {statusMix.length === 0 ? (
          <Empty title="No status yet" />
        ) : (
          <div className="chart-modal-status">
            <ResponsiveContainer width="100%" height={360}>
              <PieChart>
                <Pie data={statusMix} dataKey="value" innerRadius={90} outerRadius={140} paddingAngle={3} stroke="none">
                  {statusMix.map((m) => <Cell key={m.name} fill={m.color} />)}
                </Pie>
                <Tooltip contentStyle={{ ...tooltipStyle, fontSize: 13 }} />
              </PieChart>
            </ResponsiveContainer>
            <ul className="legend legend-row">
              {statusMix.map((m) => (
                <li key={m.name}><i style={{ background: m.color }} />{m.name}<b>{m.value}</b></li>
              ))}
            </ul>
          </div>
        )}
      </ChartModal>
    </div>
  );
}
