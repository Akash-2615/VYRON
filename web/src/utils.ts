export function fmtDuration(sec: number | null | undefined): string {
  if (sec == null) return "—";
  const s = Math.round(sec);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

export const fmtTime = (iso: string | null) =>
  iso ? new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "—";

export const fmtDateTime = (iso: string | null) =>
  iso ? new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "—";

export function axisLabel(iso: string, bucketSeconds: number): string {
  const d = new Date(iso);
  if (bucketSeconds < 60) return d.toLocaleTimeString([], { minute: "2-digit", second: "2-digit" });
  if (bucketSeconds < 86400) return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

/** Diverging blue-white-red colour for an embedding component in [-max, +max]. */
export function divergingColor(v: number, max: number): string {
  const t = Math.max(-1, Math.min(1, v / (max || 1)));
  const k = Math.abs(t);
  const [r, g, b] = t >= 0 ? [220, 38, 38] : [37, 99, 235];
  const mix = (c: number) => Math.round(255 + (c - 255) * k);
  return `rgb(${mix(r)},${mix(g)},${mix(b)})`;
}

export function histogram(values: number[], bins = 24): { x: number; count: number }[] {
  if (values.length === 0) return [];
  const lo = Math.min(...values), hi = Math.max(...values);
  const w = (hi - lo) / bins || 1;
  const out = Array.from({ length: bins }, (_, i) => ({ x: +(lo + (i + 0.5) * w).toFixed(3), count: 0 }));
  for (const v of values) out[Math.min(bins - 1, Math.floor((v - lo) / w))].count++;
  return out;
}

export function eventTone(event: string, level: string): string {
  if (level === "ERROR") return "red";
  if (level === "WARNING") return "amber";
  if (event === "ENTRY") return "green";
  if (event === "EXIT") return "orange";
  if (event === "FACE_REGISTERED") return "violet";
  if (event === "FACE_RECOGNIZED") return "blue";
  if (event.startsWith("EMBED")) return "cyan";
  return "slate";
}
