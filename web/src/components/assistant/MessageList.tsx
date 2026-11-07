import { useEffect, useRef } from "react";
import { Copy } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChatMessage } from "./useAssistantChat";

/** Turn `#12` into markdown links the renderer can intercept. */
function visitorMarkdown(text: string): string {
  return text.replace(/#(\d+)\b/g, "[#$1](#visitor-$1)");
}

export default function MessageList({
  messages, statusLine, onOpenVisitor,
}: {
  messages: ChatMessage[];
  statusLine: string | null;
  onOpenVisitor: (id: number) => void;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);

  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const onScroll = () => {
      stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
    };
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (stickRef.current) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, statusLine]);

  async function copyText(text: string) {
    try { await navigator.clipboard.writeText(text); } catch { /* ignore */ }
  }

  return (
    <div className="assistant-messages" ref={boxRef} aria-live="polite">
      {messages.length === 0 && (
        <p className="muted small assistant-empty">Ask about unique visitors, dwell time, or log warnings.</p>
      )}
      {messages.map((m) => (
        <div key={m.id} className={`assistant-bubble ${m.role}`}>
          <div className="assistant-bubble-head">
            <span>{m.role === "user" ? "You" : "ARIA"}</span>
            {m.content && (
              <button type="button" className="icon-btn" title="Copy" onClick={() => void copyText(m.content)}>
                <Copy size={13} />
              </button>
            )}
          </div>
          {m.role === "assistant" ? (
            <div className="assistant-md">
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
                    return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
                  },
                }}
              >
                {visitorMarkdown(m.content || (statusLine ? "…" : ""))}
              </ReactMarkdown>
            </div>
          ) : (
            <p>{m.content}</p>
          )}
        </div>
      ))}
      {statusLine && <p className="assistant-status" role="status">{statusLine}</p>}
      <div ref={bottomRef} />
    </div>
  );
}
