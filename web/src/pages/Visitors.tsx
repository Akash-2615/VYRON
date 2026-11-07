import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import { api } from "../api";
import { Badge, Card, Empty, Thumb } from "../components/ui";
import { usePolling } from "../usePolling";
import { fmtDateTime, fmtDuration } from "../utils";

type Filter = "all" | "inside" | "left" | "returning";

export default function Visitors({ paused, onSelectVisitor }: { paused: boolean; onSelectVisitor: (id: number) => void }) {
  const { data, error, loading } = usePolling(api.visitors, [], 5000, paused);
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<Filter>("all");

  const rows = useMemo(() => {
    return (data ?? []).filter((v) => {
      if (q && !String(v.face_id).includes(q.replace("#", ""))) return false;
      if (filter === "inside" || filter === "left") return v.status === filter;
      if (filter === "returning") return v.visit_count > 1;
      return true;
    });
  }, [data, q, filter]);

  return (
    <Card
      title={`Registered visitors${data ? ` (${data.length})` : ""}`}
      action={
        <div className="toolbar">
          <label className="search"><Search size={15} /><input placeholder="Search by ID" value={q} onChange={(e) => setQ(e.target.value)} /></label>
          <select value={filter} onChange={(e) => setFilter(e.target.value as Filter)}>
            <option value="all">All</option>
            <option value="inside">Inside now</option>
            <option value="left">Left</option>
            <option value="returning">Returning</option>
          </select>
        </div>
      }
    >
      {error && !data && <p className="error-text">{error}</p>}
      {!loading && rows.length === 0 ? (
        <Empty title="No visitors match" hint="Visitors are registered automatically when a new face is first detected." />
      ) : (
        <div className="visitor-grid">
          {rows.map((v) => (
            <button key={v.face_id} className="visitor-card" onClick={() => onSelectVisitor(v.face_id)}>
              <Thumb src={v.thumbnail} size={72} />
              <div className="visitor-info">
                <div className="visitor-top">
                  <b>Visitor #{v.face_id}</b>
                  <span className="visitor-badges">
                    <Badge kind={v.status}>{v.status === "inside" ? "Inside" : "Left"}</Badge>
                    {v.hidden_now && <Badge kind="left">Hidden</Badge>}
                  </span>
                </div>
                <div className="muted small" title="Time in frame includes brief occlusions; Visible is detected time only">
                  {v.visit_count} visit{v.visit_count === 1 ? "" : "s"} · {fmtDuration(v.time_in_frame_seconds ?? v.total_dwell_seconds)} in frame
                  {v.visible_seconds != null && <> · {fmtDuration(v.visible_seconds)} visible</>}
                </div>
                {v.appearance_summary && (
                  <div className="appearance-line" title={v.appearance_summary}>{v.appearance_summary}</div>
                )}
                <div className="muted small">First {fmtDateTime(v.first_seen)}</div>
                <div className="muted small">Last {fmtDateTime(v.last_seen)}</div>
              </div>
            </button>
          ))}
        </div>
      )}
    </Card>
  );
}
