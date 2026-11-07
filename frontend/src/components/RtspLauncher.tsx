import { useEffect, useState } from "react";
import { Play, Square } from "lucide-react";
import { api, TrackerJobStatus } from "../api";

export default function RtspLauncher({ paused }: { paused: boolean }) {
  const [url, setUrl] = useState(() => localStorage.getItem("facetrack.rtsp") || "");
  const [job, setJob] = useState<TrackerJobStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (paused) return;
    let cancelled = false;
    const pull = () => {
      api.trackerStatus()
        .then((s) => { if (!cancelled) setJob(s); })
        .catch(() => { /* API may be old build without this route */ });
    };
    pull();
    const id = window.setInterval(pull, 1500);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [paused]);

  const running = Boolean(job?.running || job?.starting);

  async function onStart() {
    const trimmed = url.trim();
    setErr(null);
    if (!trimmed) {
      setErr("Paste an RTSP link first (rtsp://…)");
      return;
    }
    setBusy(true);
    try {
      localStorage.setItem("facetrack.rtsp", trimmed);
      const s = await api.trackerStart(trimmed, true);
      setJob(s);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onStop() {
    setErr(null);
    setBusy(true);
    try {
      const s = await api.trackerStop();
      setJob(s);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rtsp-launcher">
      <label className="rtsp-label" htmlFor="rtsp-url">RTSP camera link</label>
      <div className="rtsp-row">
        <input
          id="rtsp-url"
          className="rtsp-input"
          type="url"
          placeholder="rtsp://user:pass@camera-ip:554/stream"
          value={url}
          disabled={running || busy}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !running) void onStart(); }}
        />
        {!running ? (
          <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void onStart()}>
            <Play size={15} />
            Start
          </button>
        ) : (
          <button type="button" className="btn btn-danger" disabled={busy} onClick={() => void onStop()}>
            <Square size={15} />
            Stop
          </button>
        )}
      </div>
      <p className="muted small rtsp-hint">
        {job?.starting
          ? "Starting models (YOLO + InsightFace)… live view appears when the first frame arrives."
          : job?.running
            ? `Streaming ${job.url ?? ""} · detect → track → embed → log${job.backend ? ` · DB ${job.backend}` : ""}`
            : "Paste the interview RTSP link and press Start — detection, embeddings and logging run automatically."}
      </p>
      {job?.error && <p className="rtsp-error">{job.error}</p>}
      {err && <p className="rtsp-error">{err}</p>}
    </div>
  );
}
