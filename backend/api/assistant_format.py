"""Human-readable formatting for ARIA tool results (server-side math)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional


def fmt_duration(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return None
    if s < 0:
        s = 0.0
    if s < 60:
        return f"{int(round(s))} s"
    m = int(s // 60)
    rem = int(round(s - m * 60))
    if m < 60:
        return f"{m} min {rem} s" if rem else f"{m} min"
    h = m // 60
    m = m % 60
    return f"{h} h {m} min" if m else f"{h} h"


def fmt_pct(part: Optional[float], whole: Optional[float]) -> Optional[str]:
    if part is None or whole is None or float(whole) <= 0:
        return None
    p = 100.0 * float(part) / float(whole)
    if abs(p - round(p)) < 0.05:
        return f"{int(round(p))}%"
    return f"{p:.0f}%"


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    raw = str(ts).strip().replace(" ", "T", 1)
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw[:26], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw[:19])
    except ValueError:
        return None


def fmt_relative(ts: Optional[str], *, now: Optional[str] = None) -> Optional[str]:
    a = _parse_ts(ts)
    b = _parse_ts(now) or datetime.utcnow()
    if a is None:
        return None
    delta = (b - a).total_seconds()
    if delta < 0:
        delta = -delta
        suffix = "from now"
    else:
        suffix = "ago"
    if delta < 60:
        return f"just now" if suffix == "ago" else f"in {int(delta)} s"
    if delta < 3600:
        m = int(delta // 60)
        return f"about {m} minute{'s' if m != 1 else ''} {suffix}"
    if delta < 86400:
        h = int(delta // 3600)
        return f"about {h} hour{'s' if h != 1 else ''} {suffix}"
    d = int(delta // 86400)
    return f"about {d} day{'s' if d != 1 else ''} {suffix}"


def fmt_window(start: Optional[str], end: Optional[str]) -> str:
    if not start and not end:
        return "no footage window yet"
    if start and end and start[:10] == end[:10]:
        return f"footage from {start[11:16] if len(start) >= 16 else start} to {end[11:16] if len(end) >= 16 else end}"
    bits = []
    if start:
        bits.append(f"from {start[:16].replace('T', ' ')}")
    if end:
        bits.append(f"to {end[:16].replace('T', ' ')}")
    return " ".join(bits) or "current footage"


def stamp_result(payload: dict, *, data_as_of: Optional[str], window: Optional[str],
                 plain_summary: Optional[str] = None) -> dict:
    out = dict(payload)
    out["data_as_of"] = data_as_of
    out["window"] = window or "current footage"
    if plain_summary:
        out["plain_summary"] = plain_summary
    return out


def enrich_visitor_row(row: dict, *, data_end: Optional[str]) -> dict:
    r = dict(row)
    tif = r.get("time_in_frame_seconds")
    r["time_in_frame_human"] = fmt_duration(tif)
    if r.get("last_seen"):
        r["last_seen_relative"] = fmt_relative(r.get("last_seen"), now=data_end)
    return r


METRIC_GLOSSARY = {
    "unique_visitors": {
        "name": "unique visitors",
        "plain": "How many distinct people the system has registered. Each person gets one ID; coming back later does not add another unique.",
    },
    "entries": {
        "name": "entries",
        "plain": "How many times someone was recorded walking in (one entry per visit).",
    },
    "exits": {
        "name": "exits",
        "plain": "How many times someone was recorded leaving (one exit per visit).",
    },
    "time_in_frame": {
        "name": "time in frame",
        "plain": "Total time a person was present in a visit: visible face time plus time they were hidden or occluded.",
    },
    "visible_time": {
        "name": "visible time",
        "plain": "Time the face was actually seen — excludes hidden/occluded gaps.",
    },
    "returning": {
        "name": "returning visitors",
        "plain": "People whose visit_count is greater than one — same ID, another visit.",
    },
    "hidden": {
        "name": "hidden / occluded",
        "plain": "A visit that is paused because the face disappeared briefly; it can resume without counting a new entry.",
    },
    "away_walkers": {
        "name": "away walkers",
        "plain": "People seen mostly from behind or without a usable face. Approximate KPIs, never part of the unique visitor count.",
    },
    "count_audit": {
        "name": "count audit",
        "plain": "Compares the database gallery size with the latest session log. A DB that kept older faces can disagree with a fresh log until you Fresh-start or finish a clean run. Mid-session without SYSTEM_STOP is normal.",
    },
    "match_threshold": {
        "name": "match threshold",
        "plain": "How similar a new face embedding must be to reuse an existing ID. Lower merges more faces; higher splits more easily.",
    },
    "detection_skip": {
        "name": "detection skip frames",
        "plain": "How many frames to skip between detector runs — higher is faster, a bit less responsive.",
    },
    "preprocessing": {
        "name": "video preprocessing",
        "plain": "Optional dull/dark/noisy assist for the face detector only. Crops, preview, and disturbance stats always use the original frame.",
    },
}


ALLOWED_INLINE_TAGS = frozenset({"visitor", "kpis", "trend", "incident"})

ALLOWED_ACTIONS = frozenset({"open_visitor", "goto", "open_log"})

ALLOWED_PAGES = frozenset({
    "overview", "live", "visitors", "events", "analytics", "config", "logs",
})
