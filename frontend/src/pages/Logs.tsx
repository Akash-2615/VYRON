import { useState } from "react";
import { Search } from "lucide-react";
import { api2 } from "../api";
import { Card, Empty } from "../components/ui";
import { usePolling } from "../api";
import { eventTone } from "../utils";

export default function Logs({ paused, onSelectVisitor }: { paused: boolean; onSelectVisitor: (id: number) => void }) {
  const [event, setEvent] = useState("");
  const [level, setLevel] = useState("");
  const [q, setQ] = useState("");
  const [limit, setLimit] = useState(300);
  const [source, setSource] = useState<"events" | "away">("events");
  const { data, error } = usePolling(
    () => api2.logs(limit, event, level, q, source),
    [event, level, q, limit, source],
    3000,
    paused,
  );
  const counts = Object.entries(data?.counts ?? {}).sort((a, b) => b[1] - a[1]);

  return (
    <Card
      title={source === "away" ? "away_events.log" : "events.log"}
      action={
        <div className="toolbar">
          <select
            value={source}
            onChange={(e) => {
              setSource(e.target.value as "events" | "away");
              setEvent("");
            }}
            aria-label="Log source"
          >
            <option value="events">events.log</option>
            <option value="away">away_events.log</option>
          </select>
          <label className="search"><Search size={15} /><input placeholder="Search log" value={q} onChange={(e) => setQ(e.target.value)} /></label>
          <select value={level} onChange={(e) => setLevel(e.target.value)}>
            <option value="">All levels</option><option value="INFO">INFO</option><option value="WARNING">WARNING</option><option value="ERROR">ERROR</option>
          </select>
          <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
            <option value={100}>Last 100</option><option value={300}>Last 300</option><option value={1000}>Last 1000</option>
          </select>
        </div>
      }
    >
      {error && !data && <p className="error-text">{error}</p>}
      {data && !data.exists && <Empty title="No log file yet" hint={`Expected at ${data.path}. Run the tracker to create it.`} />}
      {data?.exists && (
        <>
          <div className="chips">
            <button className={`chip ${event === "" ? "active" : ""}`} onClick={() => setEvent("")}>All</button>
            {counts.map(([name, n]) => (
              <button key={name} className={`chip ${event === name ? "active" : ""}`} onClick={() => setEvent(event === name ? "" : name)}>
                {name}<b>{n}</b>
              </button>
            ))}
          </div>
          <div className="muted small log-meta">
            <span>{data.path}</span>
            <span>Showing {data.items.length} of {data.total} lines · newest first</span>
          </div>
          <div className="terminal">
            {data.items.length === 0 && <div className="term-empty">No log lines match.</div>}
            {data.items.map((l, i) => {
              const faceId = l.fields.face_id ? Number(l.fields.face_id) : null;
              return (
                <div key={`${l.ts}-${i}`} className="term-line">
                  <span className="t-ts">{l.ts.replace("T", " ")}</span>
                  <span className={`t-lvl lvl-${l.level}`}>{l.level}</span>
                  <span className={`t-ev ev-${eventTone(l.event, l.level)}`}>{l.event}</span>
                  <span className="t-msg">
                    {Object.entries(l.fields).map(([k, v]) => (
                      <span key={k} className="t-kv">
                        <i>{k}=</i>
                        {k === "face_id" && faceId !== null ? (
                          <a onClick={() => onSelectVisitor(faceId)}>{v}</a>
                        ) : (
                          <u>{v.length > 46 ? "…" + v.slice(-44) : v}</u>
                        )}
                      </span>
                    ))}
                  </span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </Card>
  );
}
