import { ReactNode, useEffect } from "react";
import { X } from "lucide-react";

export default function ChartModal({ title, open, onClose, children }: {
  title: string; open: boolean; onClose: () => void; children: ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="chart-modal-backdrop" role="dialog" aria-modal="true" aria-label={title} onClick={onClose}>
      <div className="chart-modal" onClick={(e) => e.stopPropagation()}>
        <header className="chart-modal-head">
          <h3>{title}</h3>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </header>
        <div className="chart-modal-body">{children}</div>
      </div>
    </div>
  );
}
