import { useEffect, useMemo, useRef, useState } from "react";
import { Plus, X } from "lucide-react";
import { AssistantContext } from "../../api";
import Composer from "./Composer";
import MessageList from "./MessageList";
import { ChatAction, useAssistantChat } from "./useAssistantChat";

const WIDTH_KEY = "vyron.aria.width";
const CTX_DISMISS_KEY = "vyron.aria.ctxDismiss";

function suggestionChips(ctx: AssistantContext, warnings: number | undefined, empty: boolean): string[] {
  const chips: string[] = [];
  if (empty) {
    chips.push("How do I start counting visitors?");
  }
  if (ctx.page === "config") {
    chips.push("Explain the count audit", "What does preprocessing do?");
  } else if (ctx.page === "logs") {
    chips.push("Any warnings?");
  } else if (ctx.page === "visitors" || ctx.page === "analytics") {
    chips.push("Who stayed longest?");
  } else {
    chips.push("How many unique visitors so far?");
  }
  if (ctx.selected_face_id) {
    chips.unshift(`Explain visitor #${ctx.selected_face_id}`);
  }
  if ((warnings || 0) > 0) {
    chips.push("Any warnings in the log?");
  }
  if (!chips.includes("Summarize the last 10 minutes")) {
    chips.push("Summarize the last 10 minutes");
  }
  return [...new Set(chips)].slice(0, 5);
}

export default function AssistantPanel({
  open, onClose, context, onOpenVisitor, onNavigate,
}: {
  open: boolean;
  onClose: () => void;
  context: AssistantContext;
  onOpenVisitor: (id: number) => void;
  onNavigate?: (page: string) => void;
}) {
  const chat = useAssistantChat(context);
  const {
    messages, statusLine, streaming, error, cfg, feedback,
    send, stop, reset, setError, rate, regenerate,
  } = chat;
  const [width, setWidth] = useState(() => {
    try {
      const n = Number(localStorage.getItem(WIDTH_KEY) || 420);
      return Math.min(640, Math.max(360, n || 420));
    } catch {
      return 420;
    }
  });
  const [ctxDismissed, setCtxDismissed] = useState(() => {
    try { return localStorage.getItem(CTX_DISMISS_KEY) === "1"; } catch { return false; }
  });
  const dragRef = useRef<{ startX: number; startW: number } | null>(null);

  useEffect(() => {
    try { localStorage.setItem(WIDTH_KEY, String(width)); } catch { /* ignore */ }
  }, [width]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      const dx = dragRef.current.startX - e.clientX;
      setWidth(Math.min(640, Math.max(360, dragRef.current.startW + dx)));
    };
    const onUp = () => { dragRef.current = null; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  const configured = Boolean(cfg?.configured);
  const emptyData = (cfg?.live_snapshot?.visitors ?? 0) === 0;
  const chips = useMemo(
    () => suggestionChips(context, cfg?.live_snapshot?.warnings_count, emptyData),
    [context, cfg, emptyData],
  );

  const lookingAt = !ctxDismissed && (context.selected_face_id
    ? `Visitor #${context.selected_face_id}`
    : context.page
      ? context.page.charAt(0).toUpperCase() + context.page.slice(1)
      : null);

  const onAction = (a: ChatAction) => {
    if (a.type === "open_visitor" && a.face_id) onOpenVisitor(a.face_id);
    else if (a.type === "goto" && a.page && onNavigate) onNavigate(a.page);
    else if (a.type === "open_log" && onNavigate) onNavigate("logs");
  };

  const dismissCtx = () => {
    setCtxDismissed(true);
    try { localStorage.setItem(CTX_DISMISS_KEY, "1"); } catch { /* ignore */ }
  };

  // Keep mounted for persistence across pages; hide when closed.
  return (
    <aside
      className={`assistant-panel ${open ? "is-open" : "is-closed"}`}
      aria-label="ARIA assistant"
      aria-hidden={!open}
      style={open ? { width } : undefined}
    >
      {open && (
        <div
          className="assistant-resize"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize ARIA panel"
          onMouseDown={(e) => {
            dragRef.current = { startX: e.clientX, startW: width };
          }}
        />
      )}
      <header className="assistant-header">
        <div className="assistant-title">
          <img className="assistant-logo" src="/favicon.png" alt="" width={28} height={28} />
          <div>
            <strong>ARIA</strong>
            <small className="muted">{cfg?.model ? `Powered by Groq · ${cfg.model}` : "VYRON analytics companion"}</small>
          </div>
        </div>
        <div className="assistant-header-actions">
          <button type="button" className="icon-btn" title="New chat" onClick={reset} aria-label="New chat">
            <Plus size={16} />
          </button>
          <button type="button" className="icon-btn" title="Close" onClick={onClose} aria-label="Close assistant">
            <X size={16} />
          </button>
        </div>
      </header>

      {lookingAt && (
        <div className="assistant-context-chip">
          <span>Looking at: <b>{lookingAt}</b></span>
          <button type="button" className="icon-btn" aria-label="Dismiss context" onClick={dismissCtx}>
            <X size={12} />
          </button>
        </div>
      )}

      {!configured ? (
        <div className="assistant-setup">
          <h3>Almost there</h3>
          <p className="muted">
            Add <code>GROQ_API_KEY</code> to <code>.env</code>, install API deps, and restart uvicorn.
            Model names change — set <code>GROQ_MODEL</code> if needed.
          </p>
          <pre className="assistant-code">{`GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-120b
ASSISTANT_ENABLED=true`}</pre>
        </div>
      ) : (
        <div className="assistant-body">
          {messages.length === 0 && (
            <div className="assistant-welcome">
              <p>
                Hi — I’m ARIA. {cfg?.live_summary
                  || "Ask me about unique visitors, dwell time, logs, or the count audit."}
              </p>
            </div>
          )}
          <div className="assistant-chips">
            {chips.map((q) => (
              <button key={q} type="button" className="assistant-chip" disabled={streaming}
                onClick={() => void send(q)}>
                {q}
              </button>
            ))}
          </div>
          <MessageList
            messages={messages}
            statusLine={statusLine}
            feedback={feedback}
            onOpenVisitor={onOpenVisitor}
            onCopy={(t) => { void navigator.clipboard.writeText(t).catch(() => undefined); }}
            onRate={rate}
            onRegenerate={regenerate}
            onAction={onAction}
          />
          <div className="assistant-footer">
            {error && (
              <div className="assistant-error" role="alert">
                <p className="assistant-error-text">{error}</p>
                <div className="assistant-error-actions">
                  <button type="button" className="btn" onClick={() => setError(null)}>Dismiss</button>
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={() => {
                      const lastUser = [...messages].reverse().find((m) => m.role === "user");
                      setError(null);
                      if (lastUser) void send(lastUser.content);
                    }}
                  >
                    Retry
                  </button>
                </div>
              </div>
            )}
            <Composer disabled={!configured} streaming={streaming} onSend={(t) => void send(t)} onStop={stop} />
          </div>
        </div>
      )}
    </aside>
  );
}
