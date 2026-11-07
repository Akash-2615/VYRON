import { useCallback, useEffect, useRef, useState } from "react";
import { api, AssistantContext, AssistantStatus, streamAssistantChat } from "../../api";

export type ChatRole = "user" | "assistant";

export interface ChatAction {
  type: string;
  label?: string;
  face_id?: number;
  page?: string;
  level?: string;
  event?: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  actions?: ChatAction[];
  createdAt?: number;
}

const MAX_MESSAGES = 12;
const STORAGE_KEY = "vyron.aria.messages.v1";
const FEEDBACK_KEY = "vyron.aria.feedback.v1";

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function loadMessages(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatMessage[];
    return Array.isArray(parsed) ? parsed.slice(-MAX_MESSAGES) : [];
  } catch {
    return [];
  }
}

function saveMessages(msgs: ChatMessage[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(msgs.slice(-MAX_MESSAGES)));
  } catch { /* ignore */ }
}

export function useAssistantChat(context: AssistantContext) {
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadMessages());
  const [statusLine, setStatusLine] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cfg, setCfg] = useState<AssistantStatus | null>(null);
  const [feedback, setFeedback] = useState<Record<string, "up" | "down">>(() => {
    try {
      return JSON.parse(localStorage.getItem(FEEDBACK_KEY) || "{}");
    } catch {
      return {};
    }
  });
  const abortRef = useRef<AbortController | null>(null);
  const lastUserRef = useRef<string>("");

  useEffect(() => {
    api.assistantStatus()
      .then(setCfg)
      .catch(() => setCfg({ enabled: false, configured: false, model: "" }));
  }, []);

  useEffect(() => {
    saveMessages(messages);
  }, [messages]);

  useEffect(() => {
    try {
      localStorage.setItem(FEEDBACK_KEY, JSON.stringify(feedback));
    } catch { /* ignore */ }
  }, [feedback]);

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
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
  }, [stop]);

  const rate = useCallback((id: string, value: "up" | "down") => {
    setFeedback((prev) => ({ ...prev, [id]: value }));
  }, []);

  const send = useCallback(async (text: string, opts?: { regenerate?: boolean }) => {
    const content = text.trim();
    if (!content || streaming) return;
    setError(null);
    lastUserRef.current = content;
    const userMsg: ChatMessage = { id: uid(), role: "user", content, createdAt: Date.now() };
    const assistantId = uid();
    setMessages((prev) => {
      const base = opts?.regenerate
        ? prev.filter((m, i, arr) => !(m.role === "assistant" && i === arr.length - 1))
        : prev;
      const withUser = opts?.regenerate ? base : [...base, userMsg];
      const next: ChatMessage[] = [
        ...withUser,
        { id: assistantId, role: "assistant", content: "", createdAt: Date.now() },
      ];
      return next.slice(-MAX_MESSAGES);
    });
    setStreaming(true);
    setStatusLine("Thinking…");
    const ac = new AbortController();
    abortRef.current = ac;
    const historyBase = opts?.regenerate
      ? messages.filter((m, i, arr) => !(m.role === "assistant" && i === arr.length - 1))
      : [...messages, userMsg];
    const history = historyBase
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
            setError(String(data.message || "ARIA hit a snag — try again in a moment."));
            setStatusLine(null);
            // Drop the empty assistant placeholder so the thread doesn't fill with blank bubbles.
            setMessages((prev) => prev.filter((m) =>
              !(m.id === assistantId && !m.content.trim())));
          } else if (event === "done") {
            setStatusLine(null);
            const actions = Array.isArray(data.actions) ? data.actions as ChatAction[] : [];
            if (actions.length) {
              setMessages((prev) => prev.map((m) =>
                m.id === assistantId ? { ...m, actions } : m));
            }
          }
        },
        ac.signal,
      );
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      const msg = e instanceof Error ? e.message : String(e);
      if (/429/.test(msg)) {
        setError("ARIA is getting a lot of questions — wait a minute and try again.");
      } else if (/503|not configured|GROQ/i.test(msg)) {
        setError(msg);
      } else {
        setError("I couldn't reach the analytics just now — check the connection and retry.");
      }
      setMessages((prev) => prev.filter((m) =>
        !(m.id === assistantId && !m.content.trim())));
    } finally {
      setStreaming(false);
      setStatusLine(null);
      abortRef.current = null;
    }
  }, [context, messages, streaming]);

  const regenerate = useCallback(() => {
    const lastUser = lastUserRef.current
      || [...messages].reverse().find((m) => m.role === "user")?.content;
    if (lastUser) void send(lastUser, { regenerate: true });
  }, [messages, send]);

  return {
    messages, statusLine, streaming, error, cfg, feedback,
    send, stop, reset, setError, rate, regenerate, lastUser: lastUserRef.current,
  };
}
