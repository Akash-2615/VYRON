import { ReactNode, useState } from "react";
import { User } from "lucide-react";

export function Card({ title, action, children, className = "", id, style }: {
  title?: string; action?: ReactNode; children: ReactNode; className?: string; id?: string;
  style?: React.CSSProperties;
}) {
  return (
    <section className={`card ${className}`} id={id} style={style}>
      {title && (
        <header className="card-head">
          <h3>{title}</h3>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function Badge({ kind, children }: {
  kind: "entry" | "exit" | "inside" | "left" | "returning" | "new" | "reentry";
  children: ReactNode;
}) {
  return <span className={`badge badge-${kind}`}>{children}</span>;
}

export function Thumb({ src, size = 40, round = false }: { src: string | null; size?: number; round?: boolean }) {
  const [failed, setFailed] = useState(false);
  const style = { width: size, height: size, borderRadius: round ? "50%" : 8 };
  if (!src || failed) {
    return <div className="thumb thumb-empty" style={style}><User size={size * 0.5} /></div>;
  }
  return <img className="thumb" style={style} src={src} alt="face" loading="lazy" onError={() => setFailed(true)} />;
}

export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="empty">
      <strong>{title}</strong>
      {hint && <span>{hint}</span>}
    </div>
  );
}
