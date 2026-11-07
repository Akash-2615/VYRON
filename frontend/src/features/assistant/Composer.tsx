import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { Send, Square } from "lucide-react";

export default function Composer({
  disabled, streaming, onSend, onStop,
}: {
  disabled?: boolean;
  streaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(88, Math.max(36, el.scrollHeight))}px`;
  }, [text]);

  function submit(e?: FormEvent) {
    e?.preventDefault();
    const t = text.trim();
    if (!t || disabled || streaming) return;
    onSend(t);
    setText("");
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <form className="assistant-composer" onSubmit={submit}>
      <textarea
        ref={ref}
        rows={1}
        value={text}
        disabled={disabled}
        placeholder={disabled ? "ARIA not configured" : "Ask ARIA about visitors, dwell time, logs…"}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        aria-label="ARIA message"
      />
      {streaming ? (
        <button type="button" className="btn btn-danger" onClick={onStop} aria-label="Stop">
          <Square size={14} /> Stop
        </button>
      ) : (
        <button type="submit" className="btn btn-primary" disabled={disabled || !text.trim()} aria-label="Send">
          <Send size={14} /> Send
        </button>
      )}
    </form>
  );
}
