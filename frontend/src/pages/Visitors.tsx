import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import { api, Visitor } from "../api";
import { Badge, Card, Empty, Thumb } from "../components/ui";
import { usePolling } from "../api";
import { fmtDateTime, fmtDuration } from "../utils";

type Filter = "all" | "inside" | "left" | "returning";
type Sort =
  | "last_seen_desc"
  | "last_seen_asc"
  | "first_seen_desc"
  | "first_seen_asc"
  | "id_desc"
  | "id_asc"
  | "visits_desc"
  | "dwell_desc";

function ts(value: string | null | undefined): number {
  if (!value) return 0;
  const n = Date.parse(value);
  return Number.isFinite(n) ? n : 0;
}

function dwell(v: Visitor): number {
  return v.time_in_frame_seconds ?? v.total_dwell_seconds ?? 0;
}

function compareVisitors(a: Visitor, b: Visitor, sort: Sort): number {
  switch (sort) {
    case "last_seen_asc":
      return ts(a.last_seen) - ts(b.last_seen) || a.face_id - b.face_id;
    case "first_seen_desc":
      return ts(b.first_seen) - ts(a.first_seen) || b.face_id - a.face_id;
    case "first_seen_asc":
      return ts(a.first_seen) - ts(b.first_seen) || a.face_id - b.face_id;
    case "id_desc":
      return b.face_id - a.face_id;
    case "id_asc":
      return a.face_id - b.face_id;
    case "visits_desc":
      return (b.visit_count - a.visit_count) || (ts(b.last_seen) - ts(a.last_seen));
    case "dwell_desc":
      return (dwell(b) - dwell(a)) || (ts(b.last_seen) - ts(a.last_seen));
    case "last_seen_desc":
    default:
      return ts(b.last_seen) - ts(a.last_seen) || b.face_id - a.face_id;
  }
}

export default function Visitors({ paused, onSelectVisitor }: { paused: boolean; onSelectVisitor: (id: number) => void }) {
  const { data, error, loading } = usePolling(api.visitors, [], 5000, paused);
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [sort, setSort] = useState<Sort>("last_seen_desc");

  const rows = useMemo(() => {
    const filtered = (data ?? []).filter((v) => {
      if (q && !String(v.face_id).includes(q.replace("#", ""))) return false;
      if (filter === "inside" || filter === "left") return v.status === filter;
      if (filter === "returning") return v.visit_count > 1;
      return true;
    });
    return [...filtered].sort((a, b) => compareVisitors(a, b, sort));
  }, [data, q, filter, sort]);

  return (
    <Card
      title={`Registered visitors${data ? ` (${data.length})` : ""}`}
      action={
        <div className="toolbar">
          <label className="search"><Search size={15} /><input placeholder="Search by ID" value={q} onChange={(e) => setQ(e.target.value)} /></label>
          <select value={filter} onChange={(e) => setFilter(e.target.value as Filter)} aria-label="Filter visitors">
            <option value="all">All</option>
            <option value="inside">Inside now</option>
            <option value="left">Left</option>
            <option value="returning">Returning</option>
          </select>
          <select value={sort} onChange={(e) => setSort(e.target.value as Sort)} aria-label="Sort visitors">
            <option value="last_seen_desc">Last seen · newest</option>
            <option value="last_seen_asc">Last seen · oldest</option>
            <option value="first_seen_desc">First seen · newest</option>
            <option value="first_seen_asc">First seen · oldest</option>
            <option value="id_desc">ID · high → low</option>
            <option value="id_asc">ID · low → high</option>
            <option value="visits_desc">Most visits</option>
            <option value="dwell_desc">Longest in frame</option>
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
