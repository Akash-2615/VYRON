import { useEffect, useState } from "react";
import { Maximize2 } from "lucide-react";
import { api, LiveStatus, TrackerJobStatus } from "../api";
import StreamLauncher from "./StreamLauncher";
import { Card, Empty } from "./ui";

export default function LiveMonitor({ paused, compact = false, fullscreen = false }: {
  paused: boolean; compact?: boolean; fullscreen?: boolean;
}) {
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [job, setJob] = useState<TrackerJobStatus | null>(null);
  const [tick, setTick] = useState(0);
  const [freezeKey, setFreezeKey] = useState(0);

  useEffect(() => {
    if (paused) return;
    let cancelled = false;
    const pull = () => {
      api.liveStatus()
        .then((s) => { if (!cancelled) setStatus(s); })
        .catch(() => { if (!cancelled) setStatus(null); });
      api.trackerStatus()
        .then((s) => { if (!cancelled) setJob(s); })
        .catch(() => { /* ignore */ });
    };
    pull();
    const id = window.setInterval(pull, 800);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [paused]);

  useEffect(() => {
    if (!paused && status?.active) setTick((t) => t + 1);
  }, [paused, status?.active]);

  useEffect(() => {
    if (paused) setFreezeKey(Date.now());
  }, [paused]);

  const active = Boolean(status?.active);
  const running = Boolean(job?.running || job?.starting);
  const modeLabel = job?.mode === "local" ? "Local video"
    : job?.mode === "webcam" ? "Webcam" : "RTSP";
  const camLabel = job?.url
    ? (job.mode === "local"
      ? `File · ${job.url.split(/[/\\]/).pop()}`
      : job.mode === "webcam"
        ? `Webcam · device ${job.url}`
        : `Camera 01 (${modeLabel})`)
    : `Camera 01 (${modeLabel})`;

  async function onStop() {
    try { setJob(await api.trackerStop()); } catch { /* ignore */ }
  }

  return (
    <Card
      id="live-monitor"
      className={`${compact ? "live-compact" : ""} ${fullscreen ? "live-fullscreen" : ""}`}
      title={fullscreen ? "Full-screen Live Monitor" : "Live Monitor"}
      action={
        <div className="live-actions">
          <span className="live-cam-label">{camLabel}</span>
          {!fullscreen && (
            <button
              type="button"
              className="icon-btn"
              title="Focus live view"
              onClick={() => document.getElementById("live-monitor")?.scrollIntoView({ behavior: "smooth", block: "nearest" })}
            >
              <Maximize2 size={16} />
            </button>
          )}
          {running && (
            <button type="button" className="btn btn-danger btn-sm" onClick={() => void onStop()}>
              Stop
            </button>
          )}
        </div>
      }
    >
      <StreamLauncher paused={paused} />
      {active ? (
        <div className="live-frame">
          {paused ? (
            <img key={freezeKey} src={`/api/live.jpg?t=${freezeKey}`} alt="Frozen live frame" />
          ) : (
            <img key={tick} src={`/api/live?t=${tick}`} alt="Annotated live feed" />
          )}
          <div className="live-status-bar">
            <span className="live-status-ok">●</span>
            {status?.active ? (
              <>
                {status.faces_in_frame ?? 0} face{(status.faces_in_frame ?? 0) === 1 ? "" : "s"}
                {" · "}
                {status.no_face_in_frame ?? 0} no-face
                {" · "}
                {status.persons_in_frame ?? 0} person{(status.persons_in_frame ?? 0) === 1 ? "" : "s"}
              </>
            ) : (
              <>Waiting for frames…</>
            )}
          </div>
        </div>
      ) : (
        <Empty
          title="No live frame yet"
          hint="Choose RTSP, Local video, or Webcam above, then press Start."
        />
      )}
    </Card>
  );
}
