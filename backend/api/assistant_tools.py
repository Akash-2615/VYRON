"""Read-only Groq tool schemas + execution against AnalyticsService."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

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


def _schema(model: type[BaseModel]) -> dict:
    return model.model_json_schema()


TOOLS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": "Unique visitors, entries, exits, currently inside, returning, avg dwell, peak, first/last event.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_count",
            "description": "Reconciled unique visitor count: DB vs events.log (FACE_REGISTERED). Returns consistent flag and diffs.",
            "parameters": _schema(GetCountArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_timeline",
            "description": "Bucketed entry/exit/occupancy timeline over the data window.",
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
            "description": "Stats, visit history, clothing tags, and recent entry/exit for one face_id.",
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
            "description": "Search the parsed events.log tail (TRACK_*, EMBEDDING_*, WARNING, etc.).",
            "parameters": _schema(SearchLogsArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_identities",
            "description": "Top cosine similarity scores vs other visitors. Never returns raw vectors.",
            "parameters": _schema(SimilarArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_away_stats",
            "description": "Away/no-face walker KPIs (approximate; not part of unique count).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_config_summary",
            "description": "Non-secret tuning values (thresholds, skip frames, grace seconds).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


def status_label(name: str) -> str:
    return TOOL_STATUS_LABELS.get(name, f"Running {name}…")


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
            return json.dumps(svc.get_summary())
        if name == "get_count":
            args = GetCountArgs.model_validate(arguments)
            return json.dumps(svc.get_count(session=args.session))
        if name == "get_timeline":
            args = TimelineArgs.model_validate(arguments)
            return json.dumps(svc.get_timeline(args.bucket_seconds))
        if name == "list_visitors":
            args = ListVisitorsArgs.model_validate(arguments)
            return json.dumps({"visitors": svc.list_visitors(
                status=args.status, min_visits=args.min_visits,
                sort_by=args.sort_by, limit=args.limit)})
        if name == "get_visitor":
            args = GetVisitorArgs.model_validate(arguments)
            row = svc.get_visitor(args.face_id)
            if row is None:
                return json.dumps({"error": "visitor_not_found", "face_id": args.face_id})
            return json.dumps(row)
        if name == "query_events":
            args = QueryEventsArgs.model_validate(arguments)
            return json.dumps(svc.query_events(
                type=args.type, face_id=args.face_id,
                from_ts=args.from_ts, to_ts=args.to_ts,
                limit=args.limit, offset=args.offset))
        if name == "search_logs":
            args = SearchLogsArgs.model_validate(arguments)
            return json.dumps(svc.search_logs(
                event=args.event, level=args.level, query=args.query, limit=args.limit))
        if name == "get_similar_identities":
            args = SimilarArgs.model_validate(arguments)
            return json.dumps(svc.get_similar_identities(args.face_id))
        if name == "get_away_stats":
            return json.dumps(svc.away_stats())
        if name == "get_config_summary":
            return json.dumps(svc.get_config_summary())
        return json.dumps({"error": "unknown_tool", "name": name})
    except ValidationError as e:
        return json.dumps({"error": "invalid_arguments", "detail": e.errors()[:5]})
    except Exception as e:  # noqa: BLE001
        # Missing tables / optional features → soft skip
        return json.dumps({"error": "tool_unavailable", "detail": str(e)[:200]})
