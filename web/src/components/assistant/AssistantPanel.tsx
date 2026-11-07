import { useEffect } from "react";
import { Plus, X } from "lucide-react";
import { AssistantContext } from "../../api";
import Composer from "./Composer";
import MessageList from "./MessageList";
import { useAssistantChat } from "./useAssistantChat";

const SUGGESTIONS = [
  "How many unique visitors so far?",
  "Who stayed longest?",
  "Any warnings or errors in the log?",
  "Summarize the last 10 minutes",
];

export default function AssistantPanel({
  open, onClose, context, onOpenVisitor,
}: {
  open: boolean;
  onClose: () => void;
  context: AssistantContext;
  onOpenVisitor: (id: number) => void;
}) {
  const { messages, statusLine, streaming, error, cfg, send, stop, reset, setError } =
    useAssistantChat(context);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const configured = Boolean(cfg?.configured);
  const chips = [...SUGGESTIONS];
  if (context.selected_face_id) {
    chips.unshift(`Explain visitor #${context.selected_face_id}`);
  }

  return (
    <aside className="assistant-panel" aria-label="ARIA assistant">
      <header className="assistant-header">
        <div className="assistant-title">
          <img className="assistant-logo" src="/favicon.png" alt="" width={28} height={28} />
          <div>
            <strong>ARIA</strong>
            <small className="muted">{cfg?.model ? cfg.model : "Powered by ARIA · VYRON"}</small>
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

      {!configured ? (
        <div className="assistant-setup">
          <h3>Set up Groq</h3>
          <p className="muted">
            Add <code>GROQ_API_KEY</code> to a <code>.env</code> file in the repo root
            (see <code>.env.example</code>), install API deps, and restart uvicorn.
            Model names change — check the Groq console.
          </p>
          <pre className="assistant-code">{`GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
ASSISTANT_ENABLED=true`}</pre>
        </div>
      ) : (
        <>
          <div className="assistant-chips">
            {chips.map((q) => (
              <button key={q} type="button" className="assistant-chip" disabled={streaming}
                onClick={() => void send(q)}>
                {q}
              </button>
            ))}
          </div>
          <MessageList messages={messages} statusLine={statusLine} onOpenVisitor={onOpenVisitor} />
          {error && (
            <div className="assistant-error">
              <span>{error}</span>
              <button type="button" className="btn" onClick={() => setError(null)}>Dismiss</button>
              <button type="button" className="btn btn-primary" onClick={() => {
                const lastUser = [...messages].reverse().find((m) => m.role === "user");
                setError(null);
                if (lastUser) void send(lastUser.content);
              }}>Retry</button>
            </div>
          )}
          <Composer disabled={!configured} streaming={streaming} onSend={(t) => void send(t)} onStop={stop} />
        </>
      )}
    </aside>
  );
}
