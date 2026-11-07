/** Tiny Node smoke test (no vitest): node src/features/assistant/parseTags.spec.mjs */
import assert from "node:assert/strict";

const TAG_RE = /\[\[([a-z]+):([^\]]+)\]\]|\[\[([a-z]+)\]\]/g;
const ALLOWED = new Set(["visitor", "kpis", "trend", "incident"]);

function parse(text) {
  const out = [];
  let last = 0;
  const re = new RegExp(TAG_RE.source, "g");
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push({ type: "text", value: text.slice(last, m.index) });
    if (m[3]) {
      out.push(ALLOWED.has(m[3]) ? { type: m[3] } : { type: "unknown", raw: m[0] });
    } else if (m[1] === "visitor" && /^\d+$/.test(m[2])) {
      out.push({ type: "visitor", id: Number(m[2]) });
    } else if (ALLOWED.has(m[1])) {
      out.push({ type: m[1], id: m[2] });
    } else {
      out.push({ type: "unknown", raw: m[0] });
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ type: "text", value: text.slice(last) });
  return out;
}

const segs = parse("Hi [[visitor:12]] [[kpis]] [[evil:1]]");
assert.equal(segs.find((s) => s.type === "visitor").id, 12);
assert.ok(segs.some((s) => s.type === "kpis"));
assert.ok(segs.some((s) => s.type === "unknown"));
console.log("parseTags.spec.mjs OK");
