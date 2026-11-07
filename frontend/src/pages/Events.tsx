import { useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { api } from "../api";
import { Badge, Card, Empty, Thumb } from "../components/ui";
import { usePolling } from "../api";
import { fmtDateTime } from "../utils";

const PAGE_SIZE = 15;

export default function Events({ paused, onSelectVisitor }: { paused: boolean; onSelectVisitor: (id: number) => void }) {
  const [type, setType] = useState("");
  const [page, setPage] = useState(0);
  const { data, error } = usePolling(() => api.events(type, PAGE_SIZE, page * PAGE_SIZE), [type, page], 3000, paused);
  const pages = Math.max(1, Math.ceil((data?.total ?? 0) / PAGE_SIZE));

  return (
    <Card
      title={`Entry / exit log${data ? ` (${data.total})` : ""}`}
      action={
        <select value={type} onChange={(e) => { setType(e.target.value); setPage(0); }}>
          <option value="">All events</option>
          <option value="entry">Entries</option>
          <option value="exit">Exits</option>
        </select>
      }
    >
      {error && !data && <p className="error-text">{error}</p>}
      {data && data.items.length === 0 ? (
        <Empty title="No events" />
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr><th>Face</th><th>Visitor</th><th>Event</th><th>Timestamp</th><th>Track</th><th>Frame</th><th>Note</th></tr>
            </thead>
            <tbody>
              {data?.items.map((e) => (
                <tr key={e.id} className="clickable" onClick={() => onSelectVisitor(e.face_id)}>
                  <td><Thumb src={e.image} size={40} /></td>
                  <td>#{e.face_id}</td>
                  <td><Badge kind={e.event_type}>{e.event_type === "entry" ? "Entry" : "Exit"}</Badge></td>
                  <td>{fmtDateTime(e.timestamp)}</td>
                  <td className="muted">{e.track_id ?? "—"}</td>
                  <td className="muted">{e.frame_idx ?? "—"}</td>
                  <td className="muted">{e.note || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <footer className="pager">
        <span className="muted">Page {page + 1} of {pages}</span>
        <div>
          <button className="btn" disabled={page === 0} onClick={() => setPage(page - 1)}><ChevronLeft size={15} />Prev</button>
          <button className="btn" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Next<ChevronRight size={15} /></button>
        </div>
      </footer>
    </Card>
  );
}
