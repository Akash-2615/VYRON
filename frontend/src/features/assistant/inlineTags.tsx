import { ReactNode } from "react";
import { api } from "../../api";
import { usePolling } from "../../api";

const TAG_RE = /\[\[([a-z]+):([^\]]+)\]\]|\[\[([a-z]+)\]\]/g;

export type Segment =
  | { type: "text"; value: string }
  | { type: "visitor"; id: number }
  | { type: "kpis" }
  | { type: "trend" }
  | { type: "incident"; id: string }
  | { type: "unknown"; raw: string };

export function parseInlineTags(text: string): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  const re = new RegExp(TAG_RE.source, "g");
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push({ type: "text", value: text.slice(last, m.index) });
    if (m[3]) {
      const tag = m[3];
      if (tag === "kpis") out.push({ type: "kpis" });
      else if (tag === "trend") out.push({ type: "trend" });
      else out.push({ type: "unknown", raw: m[0] });
    } else {
      const tag = m[1];
      const arg = m[2];
      if (tag === "visitor" && /^\d+$/.test(arg)) out.push({ type: "visitor", id: Number(arg) });
      else if (tag === "incident") out.push({ type: "incident", id: arg });
      else out.push({ type: "unknown", raw: m[0] });
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ type: "text", value: text.slice(last) });
  return out;
}

function VisitorChip({ id, onOpen }: { id: number; onOpen: (id: number) => void }) {
  const { data } = usePolling(() => api.visitor(id), [id], 15_000);
  const img = data?.thumbnail || data?.face_image || null;
  return (
    <button type="button" className="aria-visitor-card" onClick={() => onOpen(id)} aria-label={`Open visitor ${id}`}>
      {img ? <img src={img} alt="" width={36} height={36} /> : <span className="aria-visitor-ph">#{id}</span>}
      <span>
        <strong>Visitor #{id}</strong>
        <small className="muted">
          {data ? `${data.status || "—"} · ${data.visit_count ?? "—"} visits` : "Loading…"}
        </small>
      </span>
    </button>
  );
}

function KpiStrip() {
  const { data } = usePolling(api.summary, [], 5000);
  if (!data) return <span className="muted small">KPIs loading…</span>;
  return (
    <ul className="aria-kpi-strip" aria-label="Live KPIs">
      <li><span>Unique</span><b>{data.unique_visitors ?? "—"}</b></li>
      <li><span>Inside</span><b>{data.currently_inside ?? "—"}</b></li>
      <li><span>Entries</span><b>{data.total_entries ?? "—"}</b></li>
      <li><span>Exits</span><b>{data.total_exits ?? "—"}</b></li>
    </ul>
  );
}

function TrendMini() {
  const { data } = usePolling(() => api.timeline(), [], 10_000);
  const pts = data?.points || [];
  const vals = pts.slice(-24).map((p) => Number(p.entries || 0) + Number(p.exits || 0));
  const max = Math.max(1, ...vals);
  return (
    <div className="aria-trend" aria-label="Activity sparkline">
      {vals.length === 0 ? (
        <span className="muted small">No activity yet</span>
      ) : (
        vals.map((v, i) => (
          <i key={i} style={{ height: `${Math.max(8, (v / max) * 100)}%` }} />
        ))
      )}
    </div>
  );
}

export function renderSegments(
  segments: Segment[],
  onOpenVisitor: (id: number) => void,
  renderText: (t: string) => ReactNode,
): ReactNode[] {
  return segments.map((s, i) => {
    if (s.type === "text") return <span key={i}>{renderText(s.value)}</span>;
    if (s.type === "visitor") return <VisitorChip key={i} id={s.id} onOpen={onOpenVisitor} />;
    if (s.type === "kpis") return <KpiStrip key={i} />;
    if (s.type === "trend") return <TrendMini key={i} />;
    if (s.type === "incident") {
      return <span key={i} className="aria-incident-chip">Incident {s.id}</span>;
    }
    return <span key={i}>{s.raw}</span>;
  });
}
