import { useState } from "react";
import { Moon, RefreshCw, Sun } from "lucide-react";
import { api } from "../api";
import { Card } from "../components/ui";
import { useTheme } from "../theme";
import { usePolling } from "../usePolling";

export default function Config({ paused = false }: { paused?: boolean }) {
  const { theme, setTheme } = useTheme();
  const [bust, setBust] = useState(0);
  const count = usePolling(() => api.count("latest"), [bust], 5000, paused);
  const audit = count.data;
  const log = audit?.derived_from_log;
  const dbUnique = audit?.source === "db" ? audit.unique_visitors : null;
  const logUnique = log?.unique_visitors ?? null;

  return (
    <div className="config-grid">
      <Card title="Appearance">
        <p className="muted small" style={{ marginTop: 0 }}>
          Purple brand tokens stay the same — only surfaces flip between dark and light.
        </p>
        <div className="stream-mode-toggle">
          <button type="button" className={`stream-mode-btn ${theme === "dark" ? "active" : ""}`} onClick={() => setTheme("dark")}>
            <Moon size={14} /> Dark
          </button>
          <button type="button" className={`stream-mode-btn ${theme === "light" ? "active" : ""}`} onClick={() => setTheme("light")}>
            <Sun size={14} /> Light
          </button>
        </div>
      </Card>
      <Card title="Stream sources">
        <p className="muted small" style={{ marginTop: 0 }}>
          Use the RTSP / Local video / Webcam toggle on Overview → Live Monitor. Local paths are resolved on the API host (repo root).
        </p>
        <ul className="spec-list">
          <li><span>RTSP</span><b>Live camera URL</b></li>
          <li><span>Local</span><b>MP4 / file path</b></li>
          <li><span>Webcam</span><b>Device index 0–3</b></li>
          <li><span>Stack</span><b>YOLOv8 · InsightFace · ByteTrack</b></li>
        </ul>
      </Card>
      <Card title="Invariants">
        <ul className="spec-list">
          <li><span>Unique count</span><b>rows in faces</b></li>
          <li><span>Visit rule</span><b>1 entry + 1 exit</b></li>
          <li><span>Returning</span><b>never new face_id</b></li>
        </ul>
      </Card>
      <Card
        title="Count audit"
        action={
          <button
            type="button"
            className="icon-btn"
            title="Re-check"
            aria-label="Re-check counts"
            onClick={() => setBust((n) => n + 1)}
          >
            <RefreshCw size={15} />
          </button>
        }
      >
        <p className="muted small" style={{ marginTop: 0 }}>
          Unique visitors must agree: <code>SELECT COUNT(*) FROM faces</code> vs distinct{" "}
          <code>FACE_REGISTERED</code> in the session log.
        </p>
        <ul className="spec-list">
          <li><span>Database</span><b>{dbUnique ?? audit?.unique_visitors ?? "—"}</b></li>
          <li><span>Log (derived)</span><b>{logUnique ?? "—"}</b></li>
          <li><span>Entries</span><b>db {audit?.entries ?? "—"} · log {log?.entries ?? "—"}</b></li>
          <li><span>Exits</span><b>db {audit?.exits ?? "—"} · log {log?.exits ?? "—"}</b></li>
          <li>
            <span>Verdict</span>
            <b className={
              audit?.consistent === true ? "count-ok"
                : audit?.consistent === false ? "count-warn" : ""
            }>
              {audit?.consistent === true ? "Consistent"
                : audit?.consistent === false ? "Mismatch"
                  : audit?.source === "none" ? "No data" : "Single source"}
            </b>
          </li>
          <li><span>Session</span><b className="mono-id">{audit?.session_id ?? "—"}</b></li>
        </ul>
        {audit?.diffs && audit.diffs.length > 0 && (
          <ul className="count-diffs">
            {audit.diffs.slice(0, 6).map((d) => <li key={d}>{d}</li>)}
          </ul>
        )}
      </Card>
    </div>
  );
}
