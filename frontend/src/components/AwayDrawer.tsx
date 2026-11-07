import { X } from "lucide-react";
import { api } from "../api";
import { usePolling } from "../api";
import { fmtDateTime } from "../utils";
import { Badge, Thumb } from "./ui";

export default function AwayDrawer({
  awayId,
  onClose,
  onOpenFace,
}: {
  awayId: number;
  onClose: () => void;
  onOpenFace?: (faceId: number) => void;
}) {
  const { data, error } = usePolling(() => api.awayVisitor(awayId), [awayId], 5000);
  return (
    <>
      <div className="overlay" onClick={onClose} />
      <aside className="drawer">
        <header>
          <h2>No-face #{awayId}</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close"><X size={18} /></button>
        </header>
        {error && !data && <p className="error-text">{error}</p>}
        {data && (
          <div className="drawer-body">
            <div className="drawer-hero">
              <Thumb src={data.appearance_image ?? data.thumbnail} size={96} />
              <div className="stat-grid">
                <div>
                  <span>Status</span>
                  <Badge kind={data.status === "inside" ? "inside" : "left"}>
                    {data.status === "inside" ? "Inside" : "Left"}
                  </Badge>
                </div>
                <div><span>Visits</span><b>{data.visit_count}</b></div>
                <div><span>First seen</span><b>{fmtDateTime(data.first_seen)}</b></div>
                <div><span>Last seen</span><b>{fmtDateTime(data.last_seen)}</b></div>
                {data.linked_face_id != null && (
                  <div>
                    <span>Turned to face</span>
                    <b>
                      {onOpenFace ? (
                        <a
                          href="#face"
                          onClick={(e) => {
                            e.preventDefault();
                            onOpenFace(Number(data.linked_face_id));
                          }}
                        >
                          Face #{data.linked_face_id}
                        </a>
                      ) : (
                        `#${data.linked_face_id}`
                      )}
                    </b>
                  </div>
                )}
              </div>
            </div>

            <p className="muted small" style={{ marginTop: 12 }}>
              Away walkers are people tracked by body only (no usable face). They are
              <strong> not</strong> counted in unique face visitors.
              {data.linked_face_id != null
                ? " “Turned → face” means this person later faced the camera and was linked to a registered face."
                : ""}
            </p>

            <div className="appearance-box">
              <h4>Appearance</h4>
              {(() => {
                const evidence = data.appearance_image || data.appearance?.evidence_image || data.thumbnail || null;
                const summary = data.appearance_summary || data.appearance?.summary || null;
                if (!evidence && !summary && !data.appearance) {
                  return (
                    <p className="muted small" style={{ margin: 0 }}>
                      No body crop or clothing tags yet. Enable{" "}
                      <code>appearance.enabled</code> and{" "}
                      <code>away_walkers.describe_away</code> in <code>config.json</code>, then
                      re-run so new away people get a clothing summary.
                    </p>
                  );
                }
                return (
                  <>
                    {evidence && (
                      <div className="appearance-evidence">
                        <img src={evidence} alt="Away person / clothing crop" />
                        <span className="muted small">Body crop (no-face / back-to-camera)</span>
                      </div>
                    )}
                    {summary ? (
                      <p className="appearance-summary">{summary}</p>
                    ) : (
                      <p className="muted small">Clothes not tagged yet (CLIP appearance off or not run).</p>
                    )}
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

            <h4>Activity timeline</h4>
            {data.events && data.events.length > 0 ? (
              <ol className="timeline">
                {data.events.map((e) => (
                  <li key={e.id}>
                    <Thumb src={e.image} size={44} />
                    <div>
                      <Badge kind={e.event_type === "exit" ? "exit" : "entry"}>
                        {e.event_type === "entry" ? "Entry" : "Exit"}
                      </Badge>
                      <div className="muted">{fmtDateTime(e.timestamp)}</div>
                    </div>
                    {e.note && <span className="note">{e.note}</span>}
                  </li>
                ))}
              </ol>
            ) : (
              <p className="muted small">No entry/exit events stored for this away person.</p>
            )}
          </div>
        )}
      </aside>
    </>
  );
}
