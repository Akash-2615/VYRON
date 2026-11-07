import { useEffect, useState } from "react";
import { Maximize2 } from "lucide-react";
import { api, LiveStatus, TrackerJobStatus } from "../api";
import StreamLauncher from "./StreamLauncher";
import { Card, Empty } from "./ui";

export default function LiveMonitor({ paused, compact = false, fullscreen = false, onMaximize }: {
  paused: boolean; compact?: boolean; fullscreen?: boolean; onMaximize?: () => void;
}) {
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [job, setJob] = useState<TrackerJobStatus | null>(null);
  const [tick, setTick] = useState(0);
  const [freezeKey, setFreezeKey] = useState(0);
  const [ppOn, setPpOn] = useState(false);
  const [ppSrc, setPpSrc] = useState<string | null>(null);

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
      api.preprocess()
        .then((p) => {
          if (cancelled) return;
          setPpOn(Boolean(p.enabled ?? p.settings?.enabled));
        })
        .catch(() => { if (!cancelled) setPpOn(false); });
    };
    pull();
    const id = window.setInterval(pull, 800);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [paused]);

  // Swap compare only after next JPEG loads — avoids remount flicker.
  useEffect(() => {
    if (paused || !ppOn || compact) {
      setPpSrc(null);
      return;
    }
    let cancelled = false;
    const pull = () => {
      const url = api.preprocessPreviewUrl();
      const img = new Image();
      img.onload = () => { if (!cancelled) setPpSrc(url); };
      img.src = url;
    };
    pull();
    const id = window.setInterval(pull, 1600);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [paused, ppOn, compact]);

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

  function handleMaximize() {
    if (onMaximize) {
      onMaximize();
      return;
    }
    // Fallback: browser fullscreen on this card.
    const el = document.getElementById("live-monitor");
    if (el && !document.fullscreenElement) {
      void el.requestFullscreen?.();
    } else if (document.fullscreenElement) {
      void document.exitFullscreen?.();
    }
  }

  return (
    <Card
      id="live-monitor"
      className={[
        compact ? "live-compact" : "",
        "live-fill",
        fullscreen ? "live-fullscreen" : "",
        active ? "live-active" : "",
      ].filter(Boolean).join(" ")}
      title={fullscreen ? "Full-screen Live Monitor" : "Live Monitor"}
      action={
        <div className="live-actions">
          {!running && <span className="live-cam-label">{camLabel}</span>}
          {!fullscreen && (
            <button
              type="button"
              className="icon-btn"
              title="Open full Live Monitor"
              onClick={handleMaximize}
            >
              <Maximize2 size={16} />
            </button>
          )}
          {/* Stop lives on the one-line stream banner while running */}
          {running && fullscreen && (
            <button type="button" className="btn btn-danger btn-sm" onClick={() => void onStop()}>
              Stop
            </button>
          )}
        </div>
      }
    >
      <StreamLauncher paused={paused} />
      {active ? (
        <div className="live-frame-wrap">
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
                  {ppOn ? " · preprocess on (see compare below)" : ""}
                </>
              ) : (
                <>Waiting for frames…</>
              )}
            </div>
          </div>
          {ppOn && !compact && (
            <div className="live-pp-compare" aria-label="Preprocess before and after">
              <div className="live-pp-compare-head">
                <strong>Original / Enhanced</strong>
                <span className="muted small">
                  Live feed above stays original. Right pane is Enhanced (detector input when ready).
                </span>
              </div>
              {ppSrc ? (
                <img src={ppSrc} alt="Preprocess before and after comparison" />
              ) : (
                <p className="muted small">Waiting for before/after preview…</p>
              )}
            </div>
          )}
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
