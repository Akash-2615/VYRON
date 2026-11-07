"""Read-only Groq tool schemas + execution against AnalyticsService."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator

from api.assistant_format import (
    METRIC_GLOSSARY,
    enrich_visitor_row,
    fmt_duration,
    fmt_pct,
    fmt_relative,
    fmt_window,
    stamp_result,
)
from api.assistant_prompt import TOOL_STATUS_LABELS
from api.services import MAX_TOOL_ROWS, AnalyticsService


class ListVisitorsArgs(BaseModel):
    status: Optional[str] = Field(None, description="inside | left")
    min_visits: int = Field(0, ge=0, le=1000)
    sort_by: str = Field("last_seen", description="last_seen | dwell | visits | face_id")
    limit: int = Field(20, ge=1, le=MAX_TOOL_ROWS)

    @field_validator("status")
    @classmethod
    def _status(cls, v):
        if v is None or v in ("inside", "left"):
            return v
        raise ValueError("status must be inside or left")

    @field_validator("sort_by")
    @classmethod
    def _sort(cls, v):
        if v not in ("last_seen", "dwell", "visits", "face_id"):
            raise ValueError("invalid sort_by")
        return v


class GetVisitorArgs(BaseModel):
    face_id: int = Field(..., ge=1, le=10_000_000)


class QueryEventsArgs(BaseModel):
    type: Optional[str] = Field(None, description="entry | exit")
    face_id: Optional[int] = Field(None, ge=1, le=10_000_000)
    from_ts: Optional[str] = Field(None, alias="from", max_length=40)
    to_ts: Optional[str] = Field(None, alias="to", max_length=40)
    limit: int = Field(25, ge=1, le=MAX_TOOL_ROWS)
    offset: int = Field(0, ge=0, le=100_000)

    model_config = {"populate_by_name": True}

    @field_validator("type")
    @classmethod
    def _type(cls, v):
        if v is None or v in ("entry", "exit"):
            return v
        raise ValueError("type must be entry or exit")


class SearchLogsArgs(BaseModel):
    event: Optional[str] = Field(None, max_length=64)
    level: Optional[str] = Field(None, max_length=16)
    query: Optional[str] = Field(None, max_length=200)
    limit: int = Field(30, ge=1, le=MAX_TOOL_ROWS)


class SimilarArgs(BaseModel):
    face_id: int = Field(..., ge=1, le=10_000_000)


class TimelineArgs(BaseModel):
    bucket_seconds: Optional[int] = Field(None, ge=1, le=86400)


class GetCountArgs(BaseModel):
    session: str = Field("latest", description="latest | all | session_id")


class ExplainMetricArgs(BaseModel):
    name: str = Field(..., min_length=2, max_length=64)


class TopDwellArgs(BaseModel):
    limit: int = Field(5, ge=1, le=20)
    window_minutes: Optional[int] = Field(None, ge=1, le=24 * 60)


class BusiestArgs(BaseModel):
    window_minutes: int = Field(60, ge=5, le=24 * 60)


class CompareWindowsArgs(BaseModel):
    a_minutes: int = Field(30, ge=1, le=24 * 60, description="Recent window length in minutes")
    b_minutes: int = Field(30, ge=1, le=24 * 60, description="Previous window length in minutes")


def _schema(model: type[BaseModel]) -> dict:
    return model.model_json_schema()


TOOLS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": "Unique visitors, entries, exits, inside now, returning, avg dwell — with human-readable fields.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_count",
            "description": "Reconciled unique count: database vs session log.",
            "parameters": _schema(GetCountArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_count_audit",
            "description": "Plain-language count audit (DB gallery vs latest session log).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_timeline",
            "description": "Bucketed entry/exit/occupancy over the footage window.",
            "parameters": _schema(TimelineArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_visitors",
            "description": "List visitors with optional status/min_visits filters.",
            "parameters": _schema(ListVisitorsArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_visitor",
            "description": "One visitor: visits, time in frame, clothing tags, recent events.",
            "parameters": _schema(GetVisitorArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_events",
            "description": "Query entry/exit events with optional filters.",
            "parameters": _schema(QueryEventsArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": "Search the session log (warnings, tracks, disturbances).",
            "parameters": _schema(SearchLogsArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_identities",
            "description": "Top similarity scores vs other visitors. Never returns raw vectors.",
            "parameters": _schema(SimilarArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_away_stats",
            "description": "Away/no-face walker KPIs (approximate; not in unique count).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_config_summary",
            "description": "Non-secret tuning values (thresholds, skip frames, grace).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_metric",
            "description": "Plain glossary entry for a metric name (unique visitors, count audit, preprocessing, …).",
            "parameters": _schema(ExplainMetricArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_dwell",
            "description": "Visitors ranked by time in frame (human durations).",
            "parameters": _schema(TopDwellArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_busiest_periods",
            "description": "Busiest buckets of activity in the footage window.",
            "parameters": _schema(BusiestArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_windows",
            "description": "Compare entries/exits in the latest N minutes vs the N minutes before that.",
            "parameters": _schema(CompareWindowsArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_health",
            "description": "Source status, backend, recent log pulse, preprocess on/off.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_preprocess_status",
            "description": "Read-only video preprocessing status (enabled, metrics).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_incidents",
            "description": "Warnings, disturbances, identity mismatches — plain language.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


def status_label(name: str) -> str:
    return TOOL_STATUS_LABELS.get(name, "Checking the analytics…")


def _meta(svc: AnalyticsService) -> Tuple[Optional[str], Optional[str], str]:
    start, end = svc.data_window()
    return start, end, fmt_window(start, end)


def _wrap(svc: AnalyticsService, payload: dict, plain: str) -> dict:
    start, end, window = _meta(svc)
    return stamp_result(payload, data_as_of=end, window=window, plain_summary=plain)


def live_snapshot(svc: AnalyticsService) -> dict:
    """Compact snapshot injected into the system prompt every turn."""
    s = svc.get_summary()
    start, end = s.get("data_start"), s.get("data_end")
    unique = s.get("unique_visitors") or 0
    inside = s.get("currently_inside") or 0
    entries = s.get("total_entries") or 0
    exits = s.get("total_exits") or 0
    returning = s.get("returning_visitors") or 0
    avg = s.get("avg_time_in_frame_seconds")
    warnings = 0
    try:
        logs = svc.search_logs(level="WARNING", limit=20)
        warnings = int(logs.get("total") or 0)
    except Exception:  # noqa: BLE001
        pass
    source = "unknown"
    try:
        from src.core.paths import runtime_path
        p = runtime_path("live_stats.json")
        if p.is_file():
            raw = json.loads(p.read_text(encoding="utf-8"))
            source = str(raw.get("source") or raw.get("mode") or "live")
    except Exception:  # noqa: BLE001
        source = "file/live"
    return {
        "visitors": unique,
        "inside_now": inside,
        "entries": entries,
        "exits": exits,
        "returning": returning,
        "avg_time_in_frame": fmt_duration(avg),
        "avg_time_in_frame_seconds": avg,
        "warnings_count": warnings,
        "source_status": source,
        "data_as_of": end,
        "window": fmt_window(start, end),
        "plain_summary": (
            f"Right now: {unique} unique visitor{'s' if unique != 1 else ''}, "
            f"{inside} inside, {entries} entries / {exits} exits"
            + (f", avg time in frame {fmt_duration(avg)}" if avg else "")
            + "."
        ),
    }


def _tool_get_summary(svc: AnalyticsService) -> dict:
    s = svc.get_summary()
    unique = s.get("unique_visitors") or 0
    inside = s.get("currently_inside") or 0
    plain = (
        f"{unique} unique visitor{'s' if unique != 1 else ''}, "
        f"{inside} inside now, "
        f"{s.get('total_entries') or 0} entries / {s.get('total_exits') or 0} exits"
    )
    if s.get("avg_time_in_frame_seconds") is not None:
        plain += f", average time in frame {fmt_duration(s.get('avg_time_in_frame_seconds'))}"
    out = {
        **s,
        "avg_time_in_frame_human": fmt_duration(s.get("avg_time_in_frame_seconds")),
        "avg_dwell_human": fmt_duration(s.get("avg_dwell_seconds")),
    }
    return _wrap(svc, out, plain + ".")


def _tool_get_count(svc: AnalyticsService, session: str) -> dict:
    c = svc.get_count(session=session)
    ok = c.get("consistent")
    plain = (
        "Database and session log agree on the unique count."
        if ok is True else
        "Database and session log disagree — often because the DB kept faces from earlier runs while the log was reset, or the session is still live."
        if ok is False else
        "Only one count source is available right now."
    )
    return _wrap(svc, c, plain)


def _tool_get_count_audit(svc: AnalyticsService) -> dict:
    c = svc.get_count(session="latest")
    db_u = c.get("unique_visitors")
    log = c.get("derived_from_log") or {}
    log_u = log.get("last_unique_total") or log.get("unique_visitors")
    notes = c.get("notes") or []
    plain = (
        f"Count audit: database gallery {db_u if db_u is not None else '—'}, "
        f"latest-session log {log_u if log_u is not None else '—'}. "
    )
    if c.get("consistent") is True:
        plain += "They match."
    elif c.get("consistent") is False:
        plain += "They don't match — usual causes: older faces still in the DB after a log reset, or a live session without SYSTEM_STOP yet. Fresh start aligns them."
    else:
        plain += "Not enough data on both sides to compare."
    return _wrap(svc, {
        "consistent": c.get("consistent"),
        "db_unique": db_u,
        "log_unique": log_u,
        "notes": notes[:6],
        "diffs": (c.get("diffs") or [])[:6],
        "session_id": c.get("session_id"),
    }, plain)


def _tool_explain_metric(name: str) -> dict:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "unique": "unique_visitors",
        "uniques": "unique_visitors",
        "dwell": "time_in_frame",
        "tif": "time_in_frame",
        "audit": "count_audit",
        "preprocess": "preprocessing",
        "pre_processing": "preprocessing",
        "away": "away_walkers",
        "skip": "detection_skip",
        "detection_skip_frames": "detection_skip",
        "match": "match_threshold",
        "occluded": "hidden",
    }
    key = aliases.get(key, key)
    entry = METRIC_GLOSSARY.get(key)
    if not entry:
        return {
            "error": "unknown_metric",
            "known": sorted(METRIC_GLOSSARY.keys()),
            "plain_summary": "I don't have a glossary entry for that name — try unique_visitors, count_audit, or preprocessing.",
            "data_as_of": None,
            "window": "n/a",
        }
    return {
        "metric": key,
        "label": entry["name"],
        "explanation": entry["plain"],
        "plain_summary": entry["plain"],
        "data_as_of": None,
        "window": "n/a",
    }


def _tool_top_dwell(svc: AnalyticsService, limit: int) -> dict:
    rows = svc.list_visitors(sort_by="dwell", limit=limit)
    _, end, window = _meta(svc)
    enriched = [enrich_visitor_row(r, data_end=end) for r in rows]
    for i, r in enumerate(enriched, 1):
        r["rank"] = i
    if not enriched:
        plain = "No visitors to rank yet."
    else:
        top = enriched[0]
        plain = (
            f"Longest stay: visitor #{top['face_id']} at "
            f"{top.get('time_in_frame_human') or 'unknown'}."
        )
    return _wrap(svc, {"visitors": enriched, "limit": limit}, plain)


def _tool_busiest(svc: AnalyticsService, window_minutes: int) -> dict:
    tl = svc.get_timeline()
    points = tl.get("points") or []
    ranked = sorted(points, key=lambda p: int(p.get("entries") or 0) + int(p.get("exits") or 0), reverse=True)
    top = ranked[:5]
    for i, p in enumerate(top, 1):
        p["rank"] = i
        p["activity"] = int(p.get("entries") or 0) + int(p.get("exits") or 0)
        p["when_relative"] = fmt_relative(p.get("t"), now=tl.get("data_end"))
    plain = (
        f"Busiest stretch had {top[0]['activity']} entry/exit events ({top[0].get('when_relative')})."
        if top else "No activity buckets yet."
    )
    return _wrap(svc, {"periods": top, "window_minutes": window_minutes}, plain)


def _tool_compare(svc: AnalyticsService, a_min: int, b_min: int) -> dict:
    from datetime import datetime, timedelta

    summary = svc.get_summary()
    end = summary.get("data_end")
    if not end:
        return _wrap(svc, {"error": "no_data"}, "No footage window to compare yet.")
    try:
        end_dt = datetime.fromisoformat(str(end).replace(" ", "T", 1)[:26])
    except ValueError:
        return _wrap(svc, {"error": "bad_data_end"}, "Couldn't read the footage end time.")
    a_start = (end_dt - timedelta(minutes=a_min)).isoformat()
    b_end = a_start
    b_start = (end_dt - timedelta(minutes=a_min + b_min)).isoformat()
    a = svc.query_events(from_ts=a_start, to_ts=end, limit=50)
    b = svc.query_events(from_ts=b_start, to_ts=b_end, limit=50)
    a_entries = sum(1 for e in a.get("items") or [] if e.get("event_type") == "entry")
    b_entries = sum(1 for e in b.get("items") or [] if e.get("event_type") == "entry")
    if b_entries > 0:
        change = (a_entries - b_entries) / b_entries
        cmp = f"up {change:.0%}" if change >= 0 else f"down {abs(change):.0%}"
        cmp += " vs previous window"
    else:
        cmp = "no earlier activity to compare"
        change = None
    plain = (
        f"Last {a_min} min: {a_entries} entries. Previous {b_min} min: {b_entries} entries "
        f"({cmp})."
    )
    return _wrap(svc, {
        "recent": {"minutes": a_min, "entries": a_entries, "events_sampled": a.get("total")},
        "previous": {"minutes": b_min, "entries": b_entries, "events_sampled": b.get("total")},
        "comparison": cmp,
        "change_ratio": change,
        "pct_change_human": fmt_pct(a_entries - b_entries, b_entries) if b_entries else None,
    }, plain)


def _tool_health(svc: AnalyticsService) -> dict:
    snap = live_snapshot(svc)
    pp = _tool_preprocess(svc)
    logs = svc.search_logs(limit=5)
    last_events = [
        {"event": i.get("event"), "level": i.get("level"), "relative": fmt_relative(i.get("ts"), now=snap.get("data_as_of"))}
        for i in (logs.get("items") or [])[:5]
    ]
    plain = (
        f"System looks {'quiet' if (snap.get('warnings_count') or 0) == 0 else 'a bit noisy'} — "
        f"backend from config, source {snap.get('source_status')}, "
        f"preprocess {'on' if pp.get('enabled') else 'off'}."
    )
    return _wrap(svc, {
        "snapshot": snap,
        "preprocess_enabled": pp.get("enabled"),
        "recent_log_events": last_events,
        "warnings_count": snap.get("warnings_count"),
    }, plain)


def _tool_preprocess(svc: AnalyticsService) -> dict:
    enabled = False
    metrics: dict = {}
    try:
        from src.core.paths import runtime_path
        from src.detection.preprocess import PreprocessSettings, default_preprocess_dict
        status_p = runtime_path("preprocess_status.json")
        override = runtime_path("preprocess.json")
        if status_p.is_file():
            metrics = json.loads(status_p.read_text(encoding="utf-8"))
            enabled = bool(metrics.get("enabled") or metrics.get("settings_enabled"))
        elif override.is_file():
            raw = json.loads(override.read_text(encoding="utf-8"))
            s = PreprocessSettings.from_dict(raw)
            enabled = bool(s.enabled)
        else:
            enabled = bool(default_preprocess_dict().get("enabled"))
    except Exception as exc:  # noqa: BLE001
        return _wrap(svc, {"available": False, "detail": str(exc)[:120]},
                     "Preprocess status isn't available right now.")
    plain = (
        f"Video preprocessing is {'on' if enabled else 'off'}"
        + (f"; worker cost about {metrics.get('total_avg_ms', 0):.1f} ms avg" if metrics.get("total_avg_ms") else "")
        + "."
    )
    return _wrap(svc, {
        "available": True,
        "enabled": enabled,
        "metrics": {
            "total_avg_ms": metrics.get("total_avg_ms"),
            "total_p95_ms": metrics.get("total_p95_ms"),
            "ready_pct": metrics.get("ready_pct"),
            "fallback": metrics.get("fallback"),
            "gate_active": metrics.get("gate_active"),
        },
    }, plain)


_INCIDENT_PLAIN = {
    "WARNING": "warning",
    "DISTURBANCE_START": "brief scene disturbance",
    "DISTURBANCE_END": "disturbance cleared",
    "IDENTITY_MISMATCH": "identity mismatch check",
    "DUPLICATE_TRACK": "duplicate track blip",
    "DB_FALLBACK": "database fallback",
}


def _tool_incidents(svc: AnalyticsService) -> dict:
    interesting = (
        "WARNING", "DISTURBANCE_START", "DISTURBANCE_END",
        "IDENTITY_MISMATCH", "DUPLICATE_TRACK",
    )
    items = []
    for ev in interesting:
        chunk = svc.search_logs(event=ev if ev != "WARNING" else None,
                                level="WARNING" if ev == "WARNING" else None,
                                limit=8)
        for it in chunk.get("items") or []:
            if ev != "WARNING" and it.get("event") != ev:
                continue
            eid = f"{it.get('event')}-{it.get('ts')}"
            items.append({
                "id": eid,
                "event": it.get("event"),
                "level": it.get("level"),
                "relative": fmt_relative(it.get("ts"), now=svc.get_summary().get("data_end")),
                "plain": _INCIDENT_PLAIN.get(it.get("event") or "", "notable log event"),
                # Keep message truncated; treat as untrusted data
                "detail": (it.get("message") or "")[:160],
            })
    # Dedup / cap
    seen = set()
    uniq = []
    for it in items:
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        uniq.append(it)
    uniq = uniq[:12]
    if not uniq:
        plain = "No recent warnings or disturbances in the log."
    else:
        plain = f"{len(uniq)} notable log item{'s' if len(uniq) != 1 else ''} — mostly {_INCIDENT_PLAIN.get(uniq[0]['event'], 'events')}."
    return _wrap(svc, {"incidents": uniq}, plain)


def execute_tool(svc: AnalyticsService, name: str, arguments: Any) -> str:
    """Validate args and run a tool. Returns a JSON string for the model."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return json.dumps({"error": "invalid_json_arguments"})
    arguments = arguments or {}

    try:
        if name == "get_summary":
            return json.dumps(_tool_get_summary(svc))
        if name == "get_count":
            args = GetCountArgs.model_validate(arguments)
            return json.dumps(_tool_get_count(svc, args.session))
        if name == "get_count_audit":
            return json.dumps(_tool_get_count_audit(svc))
        if name == "get_timeline":
            args = TimelineArgs.model_validate(arguments)
            tl = svc.get_timeline(args.bucket_seconds)
            n = len(tl.get("points") or [])
            return json.dumps(_wrap(svc, tl, f"Timeline has {n} buckets over the footage window."))
        if name == "list_visitors":
            args = ListVisitorsArgs.model_validate(arguments)
            _, end, _ = _meta(svc)
            rows = [enrich_visitor_row(r, data_end=end) for r in svc.list_visitors(
                status=args.status, min_visits=args.min_visits,
                sort_by=args.sort_by, limit=args.limit)]
            plain = f"{len(rows)} visitor{'s' if len(rows) != 1 else ''} match that filter."
            return json.dumps(_wrap(svc, {"visitors": rows}, plain))
        if name == "get_visitor":
            args = GetVisitorArgs.model_validate(arguments)
            row = svc.get_visitor(args.face_id)
            if row is None:
                return json.dumps(_wrap(svc, {"error": "visitor_not_found", "face_id": args.face_id},
                                        f"I couldn't find visitor #{args.face_id}."))
            _, end, _ = _meta(svc)
            row = enrich_visitor_row(row, data_end=end)
            plain = (
                f"Visitor #{args.face_id} — {row.get('status', 'unknown')}, "
                f"{row.get('visit_count', 0)} visit(s), "
                f"about {row.get('time_in_frame_human') or 'unknown'} in frame."
            )
            return json.dumps(_wrap(svc, row, plain))
        if name == "query_events":
            args = QueryEventsArgs.model_validate(arguments)
            q = svc.query_events(
                type=args.type, face_id=args.face_id,
                from_ts=args.from_ts, to_ts=args.to_ts,
                limit=args.limit, offset=args.offset)
            return json.dumps(_wrap(svc, q, f"{q.get('total', 0)} matching entry/exit events."))
        if name == "search_logs":
            args = SearchLogsArgs.model_validate(arguments)
            logs = svc.search_logs(
                event=args.event, level=args.level, query=args.query, limit=args.limit)
            return json.dumps(_wrap(svc, logs, f"{logs.get('total', 0)} matching log lines."))
        if name == "get_similar_identities":
            args = SimilarArgs.model_validate(arguments)
            sim = svc.get_similar_identities(args.face_id)
            return json.dumps(_wrap(svc, sim, f"Top similarity matches for visitor #{args.face_id}."))
        if name == "get_away_stats":
            away = svc.away_stats()
            plain = (
                f"About {away.get('away_unique', 0)} away/no-face walkers "
                f"(not part of the unique count)."
            )
            return json.dumps(_wrap(svc, away, plain))
        if name == "get_config_summary":
            cfg = svc.get_config_summary()
            return json.dumps(_wrap(svc, cfg, "Non-secret tuning values only."))
        if name == "explain_metric":
            args = ExplainMetricArgs.model_validate(arguments)
            return json.dumps(_tool_explain_metric(args.name))
        if name == "get_top_dwell":
            args = TopDwellArgs.model_validate(arguments)
            return json.dumps(_tool_top_dwell(svc, args.limit))
        if name == "get_busiest_periods":
            args = BusiestArgs.model_validate(arguments)
            return json.dumps(_tool_busiest(svc, args.window_minutes))
        if name == "compare_windows":
            args = CompareWindowsArgs.model_validate(arguments)
            return json.dumps(_tool_compare(svc, args.a_minutes, args.b_minutes))
        if name == "get_system_health":
            return json.dumps(_tool_health(svc))
        if name == "get_preprocess_status":
            return json.dumps(_tool_preprocess(svc))
        if name == "get_recent_incidents":
            return json.dumps(_tool_incidents(svc))
        return json.dumps({"error": "unknown_tool", "name": name})
    except ValidationError as e:
        return json.dumps({"error": "invalid_arguments", "detail": e.errors()[:5]})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": "tool_unavailable", "detail": str(e)[:200]})


def execute_tools_parallel(
    svc: AnalyticsService,
    calls: List[Tuple[str, str, Any]],
    *,
    cache: Optional[Dict[str, str]] = None,
    timeout_sec: float = 8.0,
) -> List[Tuple[str, str, str]]:
    """Run independent tool calls in parallel. calls = [(tool_call_id, name, arguments)]."""
    cache = cache if cache is not None else {}
    results: Dict[str, str] = {}
    pending: List[Tuple[str, str, Any]] = []
    for cid, name, args in calls:
        key = f"{name}:{args if isinstance(args, str) else json.dumps(args, sort_keys=True)}"
        if key in cache:
            results[cid] = cache[key]
        else:
            pending.append((cid, name, args, key))

    if pending:
        with ThreadPoolExecutor(max_workers=min(4, len(pending))) as pool:
            futs = {
                pool.submit(execute_tool, svc, name, args): (cid, key)
                for cid, name, args, key in pending
            }
            deadline = time.time() + timeout_sec
            for fut in as_completed(futs, timeout=max(0.1, deadline - time.time())):
                cid, key = futs[fut]
                try:
                    out = fut.result(timeout=max(0.1, deadline - time.time()))
                except Exception as exc:  # noqa: BLE001
                    out = json.dumps({"error": "tool_timeout", "detail": str(exc)[:120]})
                cache[key] = out
                results[cid] = out

    ordered = []
    for cid, name, args in calls:
        ordered.append((cid, name, results.get(cid) or json.dumps({"error": "tool_missing"})))
    return ordered
