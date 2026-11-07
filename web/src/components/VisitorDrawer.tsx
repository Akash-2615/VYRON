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
                <div><span>Status</span>
                  <span className="visitor-badges">
                    <Badge kind={data.status}>{data.status === "inside" ? "Inside" : "Left"}</Badge>
                    {data.hidden_now && <Badge kind="left">Hidden</Badge>}
                  </span>
                </div>
                <div><span>Visits</span><b>{data.visit_count}</b></div>
                <div title="Entry→exit including brief occlusions"><span>Time in frame</span><b>{fmtDuration(data.time_in_frame_seconds ?? data.total_dwell_seconds)}</b></div>
                <div title="Seconds actually detected"><span>Visible time</span><b>{fmtDuration(data.visible_seconds ?? 0)}</b></div>
                <div><span>First seen</span><b>{fmtDateTime(data.first_seen)}</b></div>
                <div><span>Last seen</span><b>{fmtDateTime(data.last_seen)}</b></div>
              </div>
            </div>
            {data.visits && data.visits.length > 0 && (
              <div className="visit-gaps">
                <h4>Visits</h4>
                <ul className="visit-gap-list">
                  {data.visits.map((v) => (
                    <li key={v.visit_id}>
                      <span>#{v.visit_id}</span>
                      <span className="gap-bar" title={`visible ${v.visible_seconds}s`}>
                        <i style={{ width: `${Math.min(100, (v.visible_seconds / Math.max(1, v.visible_seconds + v.occluded_seconds)) * 100)}%` }} />
                        <em style={{ width: `${Math.min(100, (v.occluded_seconds / Math.max(1, v.visible_seconds + v.occluded_seconds)) * 100)}%` }} title="Occluded" />
                      </span>
                      <span className="muted small">{v.resume_count} resume{v.resume_count === 1 ? "" : "s"} · {v.exit_reason || "open"}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <div className="appearance-box">
              <h4>Appearance</h4>
              {(() => {
                const evidence = data.appearance_image || data.appearance?.evidence_image || null;
                const summary = data.appearance_summary || data.appearance?.summary || null;
                if (!evidence && !summary && !data.appearance) {
                  return (
                    <p className="muted small" style={{ margin: 0 }}>
                      No clothing tags or body crop yet — re-run the tracker with{" "}
                      <code>appearance.enabled</code> so each registration saves a person photo under{" "}
                      <code>logs/appearance/</code>.
                    </p>
                  );
                }
                return (
                  <>
                    {evidence && (
                      <div className="appearance-evidence">
                        <img src={evidence} alt="Person / clothing crop" />
                        <span className="muted small">Full-person crop used for clothing tags</span>
                      </div>
                    )}
                    {summary && <p className="appearance-summary">{summary}</p>}
                    <div className="attr-chips">
                      {(data.appearance?.dominant_colors ?? []).map((c) => (
                        <span key={c} className="attr-chip">{c}</span>
                      ))}
                      {data.appearance?.upper_color && <span className="attr-chip">{data.appearance.upper_color}</span>}
                      {data.appearance?.upper_type && <span className="attr-chip">{data.appearance.upper_type}</span>}
                      {data.appearance?.lower_type && <span className="attr-chip">{data.appearance.lower_type}</span>}
                      {data.appearance?.accessories && <span className="attr-chip">{data.appearance.accessories}</span>}
                      {data.appearance?.hair && <span className="attr-chip">{data.appearance.hair}</span>}
                    </div>
                    {data.appearance?.model && (
                      <p className="muted small">Model: {data.appearance.model}</p>
                    )}
                  </>
                );
              })()}
            </div>
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
