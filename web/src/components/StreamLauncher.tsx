import { useEffect, useState } from "react";
import { Camera, Play, Square, Video, Wifi } from "lucide-react";
import { api, LocalVideoItem, TrackerJobStatus } from "../api";

export type StreamMode = "rtsp" | "local" | "webcam";
export type LocalPlaylist = "all" | "one";

const DEFAULT_LOCAL = "Video Datasets/video_sample1.mp4";
const WEBCAM_DEVICES = ["0", "1", "2", "3"];

export default function StreamLauncher({ paused }: { paused: boolean }) {
  const [mode, setMode] = useState<StreamMode>(() =>
    (localStorage.getItem("vyron.streamMode") as StreamMode) || "local",
  );
  const [url, setUrl] = useState(() => localStorage.getItem("facetrack.rtsp") || "");
  const [playlist, setPlaylist] = useState<LocalPlaylist>(() =>
    (localStorage.getItem("vyron.localPlaylist") as LocalPlaylist) || "one",
  );
  const [localPath, setLocalPath] = useState(
    () => localStorage.getItem("vyron.localVideo") || DEFAULT_LOCAL,
  );
  const [webcamId, setWebcamId] = useState(
    () => localStorage.getItem("vyron.webcam") || "0",
  );
  const [videos, setVideos] = useState<LocalVideoItem[]>([]);
  const [job, setJob] = useState<TrackerJobStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [clearDb, setClearDb] = useState(false);

  useEffect(() => {
    if (paused) return;
    let cancelled = false;
    const pull = () => {
      api.trackerStatus()
        .then((s) => { if (!cancelled) setJob(s); })
        .catch(() => { /* older API */ });
    };
    pull();
    const id = window.setInterval(pull, 1500);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [paused]);

  useEffect(() => {
    let cancelled = false;
    api.localVideos()
      .then((list) => {
        if (cancelled) return;
        const avail = list.available?.length
          ? list.available
          : list.items.filter((v) => v.exists);
        setVideos(avail);
        const saved = localStorage.getItem("vyron.localVideo");
        if (saved && avail.some((v) => v.path === saved)) {
          setLocalPath(saved);
        } else if (avail[0]?.path) {
          setLocalPath(avail[0].path);
        }
      })
      .catch(() => { /* endpoint may be missing on old server */ });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    try { localStorage.setItem("vyron.streamMode", mode); } catch { /* ignore */ }
  }, [mode]);

  useEffect(() => {
    try { localStorage.setItem("vyron.localPlaylist", playlist); } catch { /* ignore */ }
  }, [playlist]);

  useEffect(() => {
    try { localStorage.setItem("vyron.localVideo", localPath); } catch { /* ignore */ }
  }, [localPath]);

  const running = Boolean(job?.running || job?.starting);
  const available = videos.filter((v) => v.exists);

  async function onStart() {
    setErr(null);
    setBusy(true);
    try {
      if (mode === "rtsp") {
        const trimmed = url.trim();
        if (!trimmed) {
          setErr("Paste an RTSP link first (rtsp://…)");
          return;
        }
        localStorage.setItem("facetrack.rtsp", trimmed);
        setJob(await api.trackerStart(trimmed, false, "rtsp", { reset: clearDb }));
        return;
      }

      if (mode === "webcam") {
        const id = (webcamId || "0").trim();
        localStorage.setItem("vyron.webcam", id);
        setJob(await api.trackerStart(id, false, "webcam", { reset: clearDb }));
        return;
      }

      if (playlist === "all") {
        if (!available.length) {
          setErr("No local videos found — add files under Video Datasets/ or data/");
          return;
        }
        setJob(await api.trackerStart("", false, "local", { playlist: "all", reset: clearDb }));
        return;
      }

      const path = localPath.trim();
      if (!path) {
        setErr("Select a local video first");
        return;
      }
      localStorage.setItem("vyron.localVideo", path);
      setJob(await api.trackerStart(path, false, "local", { playlist: "one", reset: clearDb }));
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
      setJob(await api.trackerStop());
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const progress = job?.source_total && job.source_total > 1
    ? ` · video ${job.source_index ?? 1}/${job.source_total}`
    : "";

  // While streaming: one status line + Stop (hide the picker chrome).
  if (running) {
    const kind = job?.mode === "local" ? "local video"
      : job?.mode === "webcam" ? "webcam"
        : "RTSP";
    const label = job?.mode === "webcam"
      ? `device ${job?.url ?? ""}`
      : PathName(job?.url);
    const via = job?.starting
      ? "Starting models… live view appears on the first frame."
      : `Streaming via ${kind}${progress}${label ? ` · ${label}` : ""}${job?.backend ? ` · DB ${job.backend}` : ""}`;
    return (
      <div className="rtsp-launcher stream-launcher stream-launcher-live">
        <div className="rtsp-row stream-live-row">
          <p className="stream-via" title={via}>{via}</p>
          <button type="button" className="btn btn-danger" disabled={busy} onClick={() => void onStop()}>
            <Square size={15} /> Stop
          </button>
        </div>
        {job?.error && <p className="rtsp-error">{job.error}</p>}
        {err && <p className="rtsp-error">{err}</p>}
      </div>
    );
  }

  const modeHint = (() => {
    if (mode === "rtsp") return "RTSP mode: paste the camera link and press Start.";
    if (mode === "webcam") {
      return "Uses the Mac/PC webcam via OpenCV (device 0 = default). Allow camera access if prompted.";
    }
    if (playlist === "all") {
      return `Local playlist: ${available.length} video${available.length === 1 ? "" : "s"} will play in order.`;
    }
    return "Pick one file, then Start.";
  })();

  return (
    <div className="rtsp-launcher stream-launcher">
      <div className="stream-mode-toggle" role="tablist" aria-label="Stream source">
        <button
          type="button"
          role="tab"
          aria-selected={mode === "rtsp"}
          className={`stream-mode-btn ${mode === "rtsp" ? "active" : ""}`}
          disabled={busy}
          onClick={() => setMode("rtsp")}
        >
          <Wifi size={14} /> RTSP
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "local"}
          className={`stream-mode-btn ${mode === "local" ? "active" : ""}`}
          disabled={busy}
          onClick={() => setMode("local")}
        >
          <Video size={14} /> Local video
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "webcam"}
          className={`stream-mode-btn ${mode === "webcam" ? "active" : ""}`}
          disabled={busy}
          onClick={() => setMode("webcam")}
        >
          <Camera size={14} /> Webcam
        </button>
      </div>

      {mode === "rtsp" && (
        <>
          <label className="rtsp-label" htmlFor="rtsp-url">RTSP camera link</label>
          <div className="rtsp-row">
            <input
              id="rtsp-url"
              className="rtsp-input"
              type="url"
              placeholder="rtsp://user:pass@camera-ip:554/stream"
              value={url}
              disabled={busy}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") void onStart(); }}
            />
            <StartBtn busy={busy} onStart={onStart} />
          </div>
        </>
      )}

      {mode === "webcam" && (
        <>
          <label className="rtsp-label" htmlFor="webcam-device">Webcam device</label>
          <div className="rtsp-row">
            <select
              id="webcam-device"
              className="rtsp-input"
              value={webcamId}
              disabled={busy}
              onChange={(e) => setWebcamId(e.target.value)}
            >
              {WEBCAM_DEVICES.map((id) => (
                <option key={id} value={id}>
                  Camera {id}{id === "0" ? " (default)" : ""}
                </option>
              ))}
            </select>
            <StartBtn busy={busy} onStart={onStart} label="Start cam" />
          </div>
        </>
      )}

      {mode === "local" && (
        <>
          <div className="stream-mode-toggle local-playlist-toggle" role="tablist" aria-label="Local playlist">
            <button
              type="button"
              role="tab"
              aria-selected={playlist === "all"}
              className={`stream-mode-btn ${playlist === "all" ? "active" : ""}`}
              disabled={busy}
              onClick={() => setPlaylist("all")}
            >
              Play all ({available.length})
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={playlist === "one"}
              className={`stream-mode-btn ${playlist === "one" ? "active" : ""}`}
              disabled={busy}
              onClick={() => setPlaylist("one")}
            >
              Select one
            </button>
          </div>

          {playlist === "one" ? (
            <>
              <label className="rtsp-label" htmlFor="local-video-select">Choose video</label>
              <div className="rtsp-row">
                <select
                  id="local-video-select"
                  className="rtsp-input"
                  value={available.some((v) => v.path === localPath) ? localPath : (available[0]?.path || "")}
                  disabled={busy || available.length === 0}
                  onChange={(e) => setLocalPath(e.target.value)}
                >
                  {available.length === 0 && (
                    <option value="">No videos found</option>
                  )}
                  {available.map((v) => (
                    <option key={v.path} value={v.path}>{v.name}</option>
                  ))}
                </select>
                <StartBtn busy={busy} onStart={onStart} />
              </div>
            </>
          ) : (
            <div className="rtsp-row">
              <p className="muted small" style={{ margin: 0, flex: 1 }}>
                Plays every discovered local file once, same unique count across the session.
              </p>
              <button type="button" className="btn btn-primary" disabled={busy || !available.length} onClick={() => void onStart()}>
                <Play size={15} /> Start all
              </button>
            </div>
          )}
        </>
      )}

      <label className="clear-db-opt">
        <input
          type="checkbox"
          checked={clearDb}
          disabled={busy}
          onChange={(e) => setClearDb(e.target.checked)}
        />
        Clear registered faces on Start
      </label>

      <p className="muted small rtsp-hint">{modeHint}</p>
      {err && <p className="rtsp-error">{err}</p>}
    </div>
  );
}

function StartBtn({ busy, onStart, label = "Start" }: {
  busy: boolean; onStart: () => void; label?: string;
}) {
  return (
    <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void onStart()}>
      <Play size={15} /> {label}
    </button>
  );
}

function PathName(path: string | null | undefined): string {
  if (!path) return "";
  if (/^\d+$/.test(path)) return `cam ${path}`;
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1] || path;
}
