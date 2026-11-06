import { useState } from "react";
import { X } from "lucide-react";
import { api } from "../api";
import { usePolling } from "../usePolling";
import { fmtDateTime, fmtDuration } from "../utils";
import EmbeddingPanel from "./EmbeddingPanel";
import { Badge, Thumb } from "./ui";

export default function VisitorDrawer({ faceId, onClose }: { faceId: number; onClose: () => void }) {
  const { data, error } = usePolling(() => api.visitor(faceId), [faceId], 5000);
  const [tab, setTab] = useState<"face" | "activity">("face");
  return (
    <>
      <div className="overlay" onClick={onClose} />
      <aside className="drawer">
        <header>
          <h2>Visitor #{faceId}</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close"><X size={18} /></button>
        </header>
        {error && !data && <p className="error-text">{error}</p>}
        {data && (
          <div className="drawer-body">
            <div className="drawer-hero">
              <Thumb src={data.face_image ?? data.thumbnail} size={96} />
              <div className="stat-grid">
                <div><span>Status</span><Badge kind={data.status}>{data.status === "inside" ? "Inside" : "Left"}</Badge></div>
                <div><span>Visits</span><b>{data.visit_count}</b></div>
                <div><span>Total time</span><b>{fmtDuration(data.total_dwell_seconds)}</b></div>
                <div><span>First seen</span><b>{fmtDateTime(data.first_seen)}</b></div>
                <div><span>Last seen</span><b>{fmtDateTime(data.last_seen)}</b></div>
              </div>
            </div>
            {(data.appearance_summary || data.appearance) && (
              <div className="appearance-box">
                <h4>Appearance</h4>
                <p className="appearance-summary">{data.appearance_summary || data.appearance?.summary}</p>
                <div className="attr-chips">
                  {(data.appearance?.dominant_colors ?? []).map((c) => (
                    <span key={c} className="attr-chip">{c}</span>
                  ))}
                  {data.appearance?.upper_type && <span className="attr-chip">{data.appearance.upper_type}</span>}
                  {data.appearance?.lower_type && <span className="attr-chip">{data.appearance.lower_type}</span>}
                  {data.appearance?.accessories && <span className="attr-chip">{data.appearance.accessories}</span>}
                  {data.appearance?.hair && <span className="attr-chip">{data.appearance.hair}</span>}
                </div>
                {data.appearance?.model && (
                  <p className="muted small">Model: {data.appearance.model}</p>
                )}
              </div>
            )}
            <div className="tabs">
              <button className={tab === "face" ? "active" : ""} onClick={() => setTab("face")}>Face &amp; Embedding</button>
              <button className={tab === "activity" ? "active" : ""} onClick={() => setTab("activity")}>Activity timeline</button>
            </div>
            {tab === "face" && <EmbeddingPanel faceId={faceId} />}
            {tab === "activity" && <ol className="timeline">
              {data.events.map((e) => (
                <li key={e.id}>
                  <Thumb src={e.image} size={44} />
                  <div>
                    <Badge kind={e.event_type}>{e.event_type === "entry" ? "Entry" : "Exit"}</Badge>
                    <div className="muted">{fmtDateTime(e.timestamp)}</div>
                  </div>
                  {e.note && <span className="note">{e.note}</span>}
                </li>
              ))}
            </ol>}
          </div>
        )}
      </aside>
    </>
  );
}
