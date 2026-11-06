import { useMemo, useState } from "react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis } from "recharts";
import { Copy } from "lucide-react";
import { api2 } from "../api";
import { usePolling } from "../usePolling";
import { divergingColor, fmtDateTime, histogram } from "../utils";
import { Empty, Thumb } from "./ui";

const COLS = 32;

function Heatmap({ vector }: { vector: number[] }) {
  const max = useMemo(() => Math.max(...vector.map(Math.abs), 1e-9), [vector]);
  const rows = Math.ceil(vector.length / COLS);
  const cell = 9;
  return (
    <svg className="heatmap" viewBox={`0 0 ${COLS * cell} ${rows * cell}`} role="img" aria-label="Embedding heatmap">
      {vector.map((v, i) => (
        <rect key={i} x={(i % COLS) * cell} y={Math.floor(i / COLS) * cell} width={cell - 1} height={cell - 1} rx={1.5}
          fill={divergingColor(v, max)}>
          <title>{`dim ${i}: ${v.toFixed(4)}`}</title>
        </rect>
      ))}
    </svg>
  );
}

export default function EmbeddingPanel({ faceId }: { faceId: number }) {
  // paused=true -> fetch once when the tab opens (vectors are immutable)
  const { data, error } = usePolling(() => api2.embeddings(faceId), [faceId], 5000, true);
  const [sel, setSel] = useState<number | "mean">("mean");
  const [copied, setCopied] = useState(false);

  if (error && !data) return <p className="error-text">{error}</p>;
  if (!data) return <p className="muted">Loading…</p>;
  if (data.count === 0) return <Empty title="No embeddings stored" hint="Embeddings are created when the face is registered." />;

  const sample = typeof sel === "number" ? data.samples.find((s) => s.id === sel) : undefined;
  const vector = sample ? sample.vector : data.mean;
  const norm = Math.sqrt(vector.reduce((a, v) => a + v * v, 0));
  const hist = histogram(vector);
  const copy = async () => {
    await navigator.clipboard.writeText(JSON.stringify(vector));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="emb">
      <div className="emb-hero">
        <div>
          <Thumb src={data.face_image ?? data.samples[0]?.image ?? null} size={150} />
          <div className="muted small center">Registered close-up</div>
        </div>
        <div className="stat-grid">
          <div><span>Model</span><b>ArcFace · InsightFace</b></div>
          <div><span>Dimensions</span><b>{data.dim}</b></div>
          <div><span>Stored samples</span><b>{data.count}</b></div>
          <div><span>Vector L2 norm</span><b>{norm.toFixed(3)}</b></div>
          <div><span>Intra-identity sim.</span><b>{data.intra_similarity == null ? "—" : data.intra_similarity.toFixed(3)}</b></div>
          <div><span>Match threshold</span><b>{data.match_threshold}</b></div>
        </div>
      </div>

      <h4>Embedding samples <span className="muted small">— each stored with the close-up it came from</span></h4>
      <div className="samples">
        <button className={`sample ${sel === "mean" ? "active" : ""}`} onClick={() => setSel("mean")}>
          <div className="sample-mean">μ</div><span>Mean</span>
        </button>
        {data.samples.map((s) => (
          <button key={s.id} className={`sample ${sel === s.id ? "active" : ""}`} onClick={() => setSel(s.id)}>
            <Thumb src={s.image} size={56} />
            <span>{fmtDateTime(s.created_at).split(", ").pop()}</span>
          </button>
        ))}
      </div>

      <h4>Vector heatmap <span className="muted small">— {vector.length} values, blue negative · red positive</span></h4>
      <Heatmap vector={vector} />

      <div className="emb-row">
        <div className="emb-hist">
          <h4>Value distribution</h4>
          <ResponsiveContainer width="100%" height={110}>
            <BarChart data={hist} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
              <XAxis dataKey="x" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} interval={5} />
              <Tooltip contentStyle={{ borderRadius: 8, fontSize: 12 }} />
              <Bar dataKey="count" fill="#6366f1" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="emb-raw">
          <h4>Raw values</h4>
          <code className="mono">[{vector.slice(0, 8).map((v) => v.toFixed(4)).join(", ")}, … +{vector.length - 8}]</code>
          <button className="btn" onClick={copy}><Copy size={14} />{copied ? "Copied" : "Copy vector"}</button>
        </div>
      </div>

      <h4>Closest other identities <span className="muted small">— cosine similarity to this face</span></h4>
      {data.similar.length === 0 ? (
        <p className="muted small">No other visitors registered yet.</p>
      ) : (
        <ul className="sim-list">
          {data.similar.map((s) => (
            <li key={s.face_id}>
              <Thumb src={s.thumbnail} size={34} round />
              <span className="sim-id">#{s.face_id}</span>
              <div className="sim-bar">
                <div className={`sim-fill ${s.similarity >= data.match_threshold ? "danger" : ""}`}
                  style={{ width: `${Math.max(0, Math.min(1, s.similarity)) * 100}%` }} />
                <i className="sim-thr" style={{ left: `${data.match_threshold * 100}%` }} title="match threshold" />
              </div>
              <b>{s.similarity.toFixed(3)}</b>
            </li>
          ))}
        </ul>
      )}
      <p className="muted small">Scores left of the dashed line are treated as different people. A high score here would mean two identities risk being merged.</p>
    </div>
  );
}
