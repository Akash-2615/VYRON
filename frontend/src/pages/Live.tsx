import { Check, LogIn, LogOut, UserCheck, Users } from "lucide-react";
import { api, Summary } from "../api";
import KpiCard from "../components/KpiCard";
import LiveEventsFeed from "../components/LiveEventsFeed";
import LiveMonitor from "../components/LiveMonitor";
import { usePolling } from "../api";

export default function Live({ summary, paused, onSelectVisitor }: {
  summary: Summary | null; paused: boolean;
  onSelectVisitor?: (id: number) => void;
}) {
  const live = usePolling(api.liveStatus, [], 800, paused);
  const tracker = usePolling(api.trackerStatus, [], 1500, paused);
  const health = usePolling(async () => {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error("health");
    return r.json() as Promise<{ ok: boolean; backend?: string }>;
  }, [], 5000, paused);

  const s = summary;
  const fps = live.data?.fps;
  const latency = live.data?.latency_ms;
  const res = live.data?.width && live.data?.height
    ? `${live.data.width}×${live.data.height}`
    : "—";
  const source = tracker.data?.mode === "local" ? "Local"
    : tracker.data?.mode === "webcam" ? "Webcam"
      : "RTSP";
  const camOk = Boolean(live.data?.active || tracker.data?.running);
  const dbOk = Boolean(health.data?.ok);
  const pipeOk = camOk;

  return (
    <div className="live-page">
      <div className="live-page-kpis">
        <KpiCard label="Unique" value={s?.unique_visitors ?? "—"} icon={<Users size={18} />} tone="violet" />
        <KpiCard label="Entries" value={s?.total_entries ?? "—"} icon={<LogIn size={18} />} tone="green" />
        <KpiCard label="Exits" value={s?.total_exits ?? "—"} icon={<LogOut size={18} />} tone="pink" />
        <KpiCard label="Inside" value={s?.currently_inside ?? "—"} icon={<UserCheck size={18} />} tone="blue" />
        <div className="live-tech-card">
          <div className="live-tech-row"><span>Source</span><b>{source}</b></div>
          <div className="live-tech-row"><span>Resolution</span><b>{res}</b></div>
          <div className="live-tech-row"><span>FPS</span><b>{fps != null ? fps : "—"}</b></div>
          <div className="live-tech-row"><span>Latency</span><b>{latency != null ? `${latency} ms` : "—"}</b></div>
          <div className="live-tech-row"><span>In frame</span><b>{live.data?.persons_in_frame ?? "—"}</b></div>
          <div className="spec-checks">
            <span className={camOk ? "ok" : ""}><Check size={12} /> Camera</span>
            <span className={dbOk ? "ok" : ""}><Check size={12} /> Database</span>
            <span className={pipeOk ? "ok" : ""}><Check size={12} /> Pipeline</span>
          </div>
        </div>
      </div>

      <div className="live-page-stage">
        <LiveMonitor paused={paused} fullscreen />
      </div>

      <div className="live-page-events">
        <LiveEventsFeed paused={paused} onSelectVisitor={onSelectVisitor} limit={16} />
      </div>
    </div>
  );
}
