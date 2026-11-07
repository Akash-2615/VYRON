"""Optional Groq analytics assistant (SSE chat over POST) — ARIA."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Generator, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.assistant_format import ALLOWED_ACTIONS, ALLOWED_INLINE_TAGS, ALLOWED_PAGES
from api.assistant_prompt import ASSISTANT_SYSTEM_PROMPT, PROMPT_VERSION
from api.assistant_tools import TOOLS, execute_tools_parallel, live_snapshot, status_label
from api.services import AnalyticsService

log = logging.getLogger("assistant")

MAX_MESSAGES = 12
MAX_CONTENT_CHARS = 4000
DEFAULT_MODEL = "openai/gpt-oss-120b"
OVERALL_TIMEOUT_SEC = 45.0
FALLBACK_VOICE = (
    "I couldn't reach the analytics just now — give me a moment and try again."
)

_TAG_RE = re.compile(r"\[\[([a-z]+):([^\]]+)\]\]|\[\[([a-z]+)\]\]")


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatContext(BaseModel):
    page: Optional[str] = None
    selected_face_id: Optional[int] = None
    filters: Optional[Dict[str, Any]] = None
    context_dismissed: Optional[bool] = False


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    context: Optional[ChatContext] = None


class RateLimiter:
    def __init__(self, max_per_minute: int = 20):
        self.max = max_per_minute
        self._hits: Dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        q = self._hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.max:
            return False
        q.append(now)
        return True


_rate = RateLimiter(20)


def _env_bool(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def assistant_settings() -> dict:
    enabled = _env_bool("ASSISTANT_ENABLED", True)
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    model = (os.environ.get("GROQ_MODEL") or DEFAULT_MODEL).strip()
    rounds = int(os.environ.get("ASSISTANT_MAX_TOOL_ROUNDS") or "5")
    configured = bool(key)
    groq_importable = False
    try:
        import groq  # noqa: F401
        groq_importable = True
    except Exception:  # noqa: BLE001
        groq_importable = False
    return {
        "enabled": enabled,
        "configured": configured and groq_importable,
        "model": model,
        "max_tool_rounds": max(1, min(rounds, 8)),
        "groq_installed": groq_importable,
        "has_api_key": configured,
        "prompt_version": PROMPT_VERSION,
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _trim_messages(messages: List[ChatMessage]) -> List[dict]:
    out = []
    for m in messages[-MAX_MESSAGES:]:
        role = m.role if m.role in ("user", "assistant", "system") else "user"
        content = (m.content or "")[:MAX_CONTENT_CHARS]
        out.append({"role": role, "content": content})
    return out


def _build_system(svc: AnalyticsService, context: Optional[ChatContext]) -> str:
    snap = live_snapshot(svc)
    summary = svc.get_summary()
    prompt = ASSISTANT_SYSTEM_PROMPT.format(
        prompt_version=PROMPT_VERSION,
        data_start=summary.get("data_start") or "unknown",
        data_end=summary.get("data_end") or "unknown",
        data_as_of=snap.get("data_as_of") or summary.get("data_end") or "unknown",
        kpi_snapshot=json.dumps(snap, indent=2),
    )
    ctx_bits = []
    if context and not context.context_dismissed:
        if context.page:
            ctx_bits.append(f"Looking at page: {context.page}")
        if context.selected_face_id:
            ctx_bits.append(f"Looking at visitor #{context.selected_face_id}")
        if context.filters:
            ctx_bits.append(f"Active filters: {json.dumps(context.filters)[:300]}")
    if ctx_bits:
        prompt += "\n## UI context\n- " + "\n- ".join(ctx_bits)
    return prompt


def extract_actions_from_answer(text: str, context: Optional[ChatContext] = None) -> List[dict]:
    """Build server-validated navigation actions from whitelist inline tags."""
    actions: List[dict] = []
    seen = set()
    for m in _TAG_RE.finditer(text or ""):
        if m.group(3):
            tag, arg = m.group(3), None
        else:
            tag, arg = m.group(1), m.group(2)
        if tag not in ALLOWED_INLINE_TAGS:
            continue
        if tag == "visitor" and arg and arg.isdigit():
            key = ("open_visitor", arg)
            if key not in seen and "open_visitor" in ALLOWED_ACTIONS:
                seen.add(key)
                actions.append({
                    "type": "open_visitor",
                    "face_id": int(arg),
                    "label": f"Open visitor #{arg}",
                })
        elif tag == "kpis":
            key = ("goto", "overview")
            if key not in seen:
                seen.add(key)
                actions.append({"type": "goto", "page": "overview", "label": "Open Overview"})
        elif tag == "trend":
            key = ("goto", "analytics")
            if key not in seen:
                seen.add(key)
                actions.append({"type": "goto", "page": "analytics", "label": "Open Analytics"})
        elif tag == "incident":
            key = ("open_log", arg or "WARNING")
            if key not in seen and "open_log" in ALLOWED_ACTIONS:
                seen.add(key)
                actions.append({
                    "type": "open_log",
                    "level": "WARNING",
                    "event": arg,
                    "label": "Open System Log",
                })
    # Context-aware soft suggestion (still whitelist)
    if context and context.selected_face_id and not any(
            a.get("type") == "open_visitor" for a in actions):
        fid = int(context.selected_face_id)
        actions.append({
            "type": "open_visitor",
            "face_id": fid,
            "label": f"Open visitor #{fid}",
        })
    # Validate pages
    out = []
    for a in actions[:6]:
        if a["type"] == "goto" and a.get("page") not in ALLOWED_PAGES:
            continue
        out.append(a)
    return out


def whitelist_tags_only(text: str) -> List[str]:
    """Return unknown tag names found (for tests)."""
    unknown = []
    for m in _TAG_RE.finditer(text or ""):
        tag = m.group(3) or m.group(1)
        if tag and tag not in ALLOWED_INLINE_TAGS:
            unknown.append(tag)
    return unknown


def _call_with_retry(fn: Callable, *, retries: int = 1):
    last = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            name = type(e).__name__.lower()
            msg = str(e).lower()
            retryable = ("429" in msg or "rate" in msg or "timeout" in msg
                         or "5" in name or "server" in msg or "unavailable" in msg)
            if attempt >= retries or not retryable:
                raise
            time.sleep(0.6 * (attempt + 1))
    raise last  # type: ignore[misc]


def run_agent_sse(
    svc: AnalyticsService,
    messages: List[dict],
    context: Optional[ChatContext],
    *,
    client: Any = None,
    model: Optional[str] = None,
    max_rounds: Optional[int] = None,
) -> Generator[str, None, None]:
    settings = assistant_settings()
    if not settings["enabled"]:
        yield _sse("error", {"message": "ARIA is turned off right now (ASSISTANT_ENABLED=false)."})
        return
    # Real Groq needs a key; injected test clients skip that gate.
    if client is None and not settings["configured"]:
        yield _sse("error", {
            "message": "ARIA isn't set up yet — add GROQ_API_KEY to .env and install groq, then restart the API.",
        })
        return

    model = model or settings["model"]
    max_rounds = max_rounds or settings["max_tool_rounds"]
    if client is None:
        from groq import Groq
        client = Groq(api_key=(os.environ.get("GROQ_API_KEY") or "").strip())

    try:
        system = _build_system(svc, context)
    except Exception:  # noqa: BLE001
        log.exception("aria_snapshot_failed")
        yield _sse("token", {"text": FALLBACK_VOICE})
        yield _sse("done", {"actions": []})
        return

    msgs: List[dict] = [{"role": "system", "content": system}] + messages
    started = time.time()
    tool_cache: Dict[str, str] = {}
    answer_parts: List[str] = []

    try:
        for _round_i in range(max_rounds):
            if time.time() - started > OVERALL_TIMEOUT_SEC:
                yield _sse("token", {"text": FALLBACK_VOICE})
                yield _sse("done", {"actions": []})
                return

            def _create():
                return client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                    stream=False,
                )

            resp = _call_with_retry(_create, retries=1)
            choice = resp.choices[0].message
            tool_calls = getattr(choice, "tool_calls", None) or []

            if tool_calls:
                msgs.append({
                    "role": "assistant",
                    "content": choice.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in tool_calls
                    ],
                })
                # Friendly status (one line) then parallel execution
                names = [tc.function.name for tc in tool_calls]
                yield _sse("status", {"message": status_label(names[0] if len(names) == 1 else "get_summary")})
                if len(names) > 1:
                    yield _sse("status", {"message": "Pulling a few numbers together…"})

                call_specs = [
                    (tc.id, tc.function.name, tc.function.arguments or "{}")
                    for tc in tool_calls
                ]
                remaining = max(1.0, OVERALL_TIMEOUT_SEC - (time.time() - started))
                results = execute_tools_parallel(
                    svc, call_specs, cache=tool_cache, timeout_sec=min(8.0, remaining))
                for cid, _name, result in results:
                    if len(result) > 12_000:
                        result = result[:12_000] + '…"}'
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": cid,
                        "content": result,
                    })
                continue

            yield _sse("status", {"message": "Putting that into plain words…"})

            def _stream():
                return client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    temperature=0.2,
                    stream=True,
                )

            stream = _call_with_retry(_stream, retries=1)
            for chunk in stream:
                if time.time() - started > OVERALL_TIMEOUT_SEC:
                    yield _sse("token", {"text": " " + FALLBACK_VOICE})
                    yield _sse("done", {"actions": []})
                    return
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                if text:
                    answer_parts.append(text)
                    yield _sse("token", {"text": text})
            full = "".join(answer_parts)
            actions = extract_actions_from_answer(full, context)
            yield _sse("done", {"actions": actions, "prompt_version": PROMPT_VERSION})
            return

        yield _sse("token", {"text": FALLBACK_VOICE})
        yield _sse("done", {"actions": []})
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        msg = str(e)
        log.warning("assistant_error type=%s detail=%s", name, msg[:240])
        low = msg.lower()
        if "model" in low and ("not found" in low or "does not exist" in low or "decommissioned" in low):
            yield _sse("error", {
                "message": (
                    f"Groq model `{model}` isn't available on this key — "
                    "set GROQ_MODEL in .env to a model you can access, then restart the API."
                ),
            })
        elif "429" in low or "rate" in low:
            yield _sse("error", {
                "message": "ARIA is getting a lot of questions — wait a minute and try again.",
            })
        elif "connection" in low or "timeout" in low:
            yield _sse("error", {
                "message": "Couldn't reach Groq just now — check the network and retry.",
            })
        else:
            yield _sse("token", {"text": FALLBACK_VOICE})
        yield _sse("done", {"actions": []})


def create_assistant_router(
    svc: AnalyticsService,
    *,
    client_factory: Optional[Callable[[], Any]] = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/assistant", tags=["assistant"])

    @router.get("/status")
    def status():
        s = assistant_settings()
        snap = {}
        try:
            snap = live_snapshot(svc)
        except Exception:  # noqa: BLE001
            pass
        return {
            "enabled": s["enabled"],
            "configured": s["configured"],
            "model": s["model"],
            "groq_installed": s["groq_installed"],
            "has_api_key": s["has_api_key"],
            "prompt_version": s["prompt_version"],
            "live_summary": snap.get("plain_summary"),
            "live_snapshot": {
                "visitors": snap.get("visitors"),
                "inside_now": snap.get("inside_now"),
                "warnings_count": snap.get("warnings_count"),
            } if snap else None,
        }

    @router.post("/chat")
    def chat(body: ChatRequest, request: Request):
        settings = assistant_settings()
        if not settings["enabled"]:
            raise HTTPException(503, "ARIA is turned off (ASSISTANT_ENABLED=false).")
        if client_factory is None and not settings["configured"]:
            raise HTTPException(
                503,
                "ARIA isn't configured. Add GROQ_API_KEY to .env and install groq "
                "(pip install -r requirements-api.txt). Model names change — set GROQ_MODEL if needed.",
            )
        ip = request.client.host if request.client else "unknown"
        if not _rate.allow(ip):
            raise HTTPException(429, "ARIA is getting a lot of questions — wait a minute and try again.")

        messages = _trim_messages(body.messages or [])
        if not messages:
            raise HTTPException(400, "messages required")
        messages = [m for m in messages if m["role"] in ("user", "assistant")]
        if not messages or messages[-1]["role"] != "user":
            raise HTTPException(400, "Last message must be from the user.")

        client = client_factory() if client_factory else None

        def gen():
            yield from run_agent_sse(
                svc, messages, body.context,
                client=client,
            )

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store",
                "X-Accel-Buffering": "no",
            },
        )

    return router
