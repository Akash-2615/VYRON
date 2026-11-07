"""Optional Groq analytics assistant (SSE chat over POST)."""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Generator, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.assistant_prompt import ASSISTANT_SYSTEM_PROMPT
from api.assistant_tools import TOOLS, execute_tool, status_label
from api.services import AnalyticsService

log = logging.getLogger("assistant")

MAX_MESSAGES = 12
MAX_CONTENT_CHARS = 4000
DEFAULT_MODEL = "llama-3.3-70b-versatile"
OVERALL_TIMEOUT_SEC = 45.0


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatContext(BaseModel):
    page: Optional[str] = None
    selected_face_id: Optional[int] = None
    filters: Optional[Dict[str, Any]] = None


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
    summary = svc.get_summary()
    data_start = summary.get("data_start")
    data_end = summary.get("data_end")
    kpi = {
        "unique_visitors": summary.get("unique_visitors"),
        "total_entries": summary.get("total_entries"),
        "total_exits": summary.get("total_exits"),
        "currently_inside": summary.get("currently_inside"),
        "returning_visitors": summary.get("returning_visitors"),
        "avg_time_in_frame_seconds": summary.get("avg_time_in_frame_seconds"),
        "peak_occupancy": summary.get("peak_occupancy"),
        "away_unique": summary.get("away_unique"),
    }
    ctx_bits = []
    if context:
        if context.page:
            ctx_bits.append(f"UI page: {context.page}")
        if context.selected_face_id:
            ctx_bits.append(f"Selected visitor: #{context.selected_face_id}")
        if context.filters:
            ctx_bits.append(f"Filters: {json.dumps(context.filters)[:300]}")
    prompt = ASSISTANT_SYSTEM_PROMPT.format(
        data_start=data_start or "unknown",
        data_end=data_end or "unknown",
        kpi_snapshot=json.dumps(kpi, indent=2),
    )
    if ctx_bits:
        prompt += "\nUI context\n- " + "\n- ".join(ctx_bits)
    return prompt


class GroqClientProtocol:
    """Minimal surface used by the agent loop (real SDK or test fake)."""

    def chat_completions_create(self, **kwargs) -> Any:  # pragma: no cover - protocol
        raise NotImplementedError


def _real_groq_client(api_key: str):
    from groq import Groq
    return Groq(api_key=api_key)


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
        yield _sse("error", {"message": "Assistant is disabled (ASSISTANT_ENABLED=false)."})
        return
    if not settings["configured"]:
        yield _sse("error", {
            "message": "Assistant not configured. Set GROQ_API_KEY in .env and pip install groq.",
        })
        return

    model = model or settings["model"]
    max_rounds = max_rounds or settings["max_tool_rounds"]
    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if client is None:
        client = _real_groq_client(api_key)

    system = _build_system(svc, context)
    msgs: List[dict] = [{"role": "system", "content": system}] + messages
    started = time.time()

    try:
        for round_i in range(max_rounds):
            if time.time() - started > OVERALL_TIMEOUT_SEC:
                yield _sse("error", {"message": "Assistant timed out. Try a narrower question."})
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
                for tc in tool_calls:
                    name = tc.function.name
                    yield _sse("status", {"message": status_label(name)})
                    result = execute_tool(svc, name, tc.function.arguments)
                    # Cap tool payload size sent back to the model
                    if len(result) > 12_000:
                        result = result[:12_000] + '…"}'
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue

            # Final streamed answer
            yield _sse("status", {"message": "Writing answer…"})

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
                    yield _sse("error", {"message": "Assistant timed out while streaming."})
                    return
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                if text:
                    yield _sse("token", {"text": text})
            yield _sse("done", {})
            return

        yield _sse("error", {"message": "Too many tool rounds. Please rephrase your question."})
    except Exception as e:  # noqa: BLE001
        log.warning("assistant_error type=%s", type(e).__name__)
        yield _sse("error", {
            "message": "The assistant hit a temporary error. Please try again in a moment.",
        })


def create_assistant_router(
    svc: AnalyticsService,
    *,
    client_factory: Optional[Callable[[], Any]] = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/assistant", tags=["assistant"])

    @router.get("/status")
    def status():
        s = assistant_settings()
        return {
            "enabled": s["enabled"],
            "configured": s["configured"],
            "model": s["model"],
            "groq_installed": s["groq_installed"],
            "has_api_key": s["has_api_key"],
        }

    @router.post("/chat")
    def chat(body: ChatRequest, request: Request):
        settings = assistant_settings()
        if not settings["enabled"]:
            raise HTTPException(503, "Assistant is disabled (ASSISTANT_ENABLED=false).")
        if not settings["configured"]:
            raise HTTPException(
                503,
                "Assistant not configured. Add GROQ_API_KEY to .env and install groq "
                "(pip install -r requirements-api.txt). Model names change — check the Groq console.",
            )
        ip = request.client.host if request.client else "unknown"
        if not _rate.allow(ip):
            raise HTTPException(429, "Too many assistant requests. Please wait a minute.")

        messages = _trim_messages(body.messages or [])
        if not messages:
            raise HTTPException(400, "messages required")
        # Drop accidental system roles from client
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
