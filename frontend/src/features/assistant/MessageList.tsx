import { useEffect, useRef, useState } from "react";
import { Copy, RefreshCw, ThumbsDown, ThumbsUp } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChatAction, ChatMessage } from "./useAssistantChat";
import { parseInlineTags, renderSegments } from "./inlineTags";

function visitorMarkdown(text: string): string {
  return text.replace(/#(\d+)\b/g, "[#$1](#visitor-$1)");
}

function stripTagsForMd(text: string): string {
  return text.replace(/\[\[[^\]]+\]\]/g, "");
}

export default function MessageList({
  messages, statusLine, feedback, onOpenVisitor, onCopy, onRate, onRegenerate, onAction,
}: {
  messages: ChatMessage[];
  statusLine: string | null;
  feedback: Record<string, "up" | "down">;
  onOpenVisitor: (id: number) => void;
  onCopy: (text: string) => void;
  onRate: (id: string, v: "up" | "down") => void;
  onRegenerate: () => void;
  onAction: (a: ChatAction) => void;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);
  const [showJump, setShowJump] = useState(false);

  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const onScroll = () => {
      const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
      stickRef.current = nearBottom;
      setShowJump(!nearBottom);
    };
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (stickRef.current) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, statusLine]);

  return (
    <div className="assistant-messages" ref={boxRef} aria-live="polite">
      {messages.map((m, idx) => {
        // Hide blank assistant placeholders (failed stream left an empty bubble).
        if (m.role === "assistant" && !(m.content || "").trim() && !(statusLine && idx === messages.length - 1)) {
          return null;
        }
        const isLastAssistant = m.role === "assistant" && idx === messages.length - 1;
        const segments = m.role === "assistant" ? parseInlineTags(m.content || "") : [];
        const hasTags = segments.some((s) => s.type !== "text");
        return (
          <div key={m.id} className={`assistant-bubble ${m.role}`}>
            <div className="assistant-bubble-head">
              <span title={m.createdAt ? new Date(m.createdAt).toLocaleString() : undefined}>
                {m.role === "user" ? "You" : "ARIA"}
              </span>
              <div className="assistant-bubble-actions">
                {m.content && (
                  <button type="button" className="icon-btn" title="Copy" aria-label="Copy"
                    onClick={() => onCopy(m.content)}>
                    <Copy size={13} />
                  </button>
                )}
                {m.role === "assistant" && m.content && (
                  <>
                    <button type="button" className={`icon-btn ${feedback[m.id] === "up" ? "active" : ""}`}
                      title="Helpful" aria-label="Thumbs up" onClick={() => onRate(m.id, "up")}>
                      <ThumbsUp size={13} />
                    </button>
                    <button type="button" className={`icon-btn ${feedback[m.id] === "down" ? "active" : ""}`}
                      title="Not helpful" aria-label="Thumbs down" onClick={() => onRate(m.id, "down")}>
                      <ThumbsDown size={13} />
                    </button>
                    {isLastAssistant && (
                      <button type="button" className="icon-btn" title="Regenerate" aria-label="Regenerate"
                        onClick={onRegenerate}>
                        <RefreshCw size={13} />
                      </button>
                    )}
                  </>
                )}
              </div>
            </div>
            {m.role === "assistant" ? (
              <div className="assistant-md">
                {hasTags && (
                  <div className="aria-inline-blocks">
                    {renderSegments(segments.filter((s) => s.type !== "text"), onOpenVisitor, (t) => t)}
                  </div>
                )}
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    a: ({ href, children }) => {
                      const mVisitor = href && /^#visitor-(\d+)$/.exec(href);
                      if (mVisitor) {
                        const id = Number(mVisitor[1]);
                        return (
                          <button type="button" className="assistant-face-link"
                            onClick={() => onOpenVisitor(id)}>
                            {children}
                          </button>
                        );
                      }
                      return <span>{children}</span>;
                    },
                  }}
                >
                  {visitorMarkdown(stripTagsForMd(m.content || (statusLine && isLastAssistant ? "…" : "")))}
                </ReactMarkdown>
                {m.actions && m.actions.length > 0 && (
                  <div className="assistant-action-chips">
                    {m.actions.map((a, i) => (
                      <button key={i} type="button" className="assistant-chip"
                        onClick={() => onAction(a)}>
                        {a.label || a.type}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <p>{m.content}</p>
            )}
          </div>
        );
      })}
      {statusLine && <p className="assistant-status" role="status">{statusLine}</p>}
      <div ref={bottomRef} />
      {showJump && (
        <button type="button" className="assistant-jump" onClick={() => {
          bottomRef.current?.scrollIntoView({ behavior: "smooth" });
          setShowJump(false);
        }}>
          Jump to latest
        </button>
      )}
    </div>
  );
}
