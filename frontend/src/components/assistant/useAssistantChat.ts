import { useCallback, useEffect, useRef, useState } from "react";
import { api, AssistantContext, AssistantStatus, streamAssistantChat } from "../../api";

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
}

const MAX_MESSAGES = 12;

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function useAssistantChat(context: AssistantContext) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [statusLine, setStatusLine] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cfg, setCfg] = useState<AssistantStatus | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    api.assistantStatus()
      .then(setCfg)
      .catch(() => setCfg({ enabled: false, configured: false, model: "" }));
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStreaming(false);
    setStatusLine(null);
  }, []);

  const reset = useCallback(() => {
    stop();
    setMessages([]);
    setError(null);
    setStatusLine(null);
  }, [stop]);

  const send = useCallback(async (text: string) => {
    const content = text.trim();
    if (!content || streaming) return;
    setError(null);
    const userMsg: ChatMessage = { id: uid(), role: "user", content };
    const assistantId = uid();
    setMessages((prev) => {
      const next: ChatMessage[] = [
        ...prev,
        userMsg,
        { id: assistantId, role: "assistant", content: "" },
      ];
      return next.slice(-MAX_MESSAGES);
    });
    setStreaming(true);
    setStatusLine("Thinking…");
    const ac = new AbortController();
    abortRef.current = ac;
    const history = [...messages, userMsg]
      .filter((m) => m.content)
      .slice(-MAX_MESSAGES)
      .map((m) => ({ role: m.role, content: m.content }));
    try {
      await streamAssistantChat(
        history,
        context,
        ({ event, data }) => {
          if (event === "status") {
            setStatusLine(String(data.message || "Working…"));
          } else if (event === "token") {
            setStatusLine(null);
            const t = String(data.text || "");
            setMessages((prev) => prev.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + t } : m));
          } else if (event === "error") {
            setError(String(data.message || "Assistant error"));
            setStatusLine(null);
          } else if (event === "done") {
            setStatusLine(null);
          }
        },
        ac.signal,
      );
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStreaming(false);
      setStatusLine(null);
      abortRef.current = null;
    }
  }, [context, messages, streaming]);

  return { messages, statusLine, streaming, error, cfg, send, stop, reset, setError };
}
