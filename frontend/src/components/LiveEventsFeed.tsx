import { ArrowDownLeft, ArrowUpRight } from "lucide-react";
import { api } from "../api";
import { usePolling } from "../api";
import { useStreamSync } from "../hooks/useStreamSync";
import { entryKindLabel, fmtTime } from "../utils";
import { Card, Empty, Thumb } from "./ui";

/** Recent entry/exit feed. Polls in lockstep with Events / Logs while streaming. */
export default function LiveEventsFeed({
  paused,
  onSelectVisitor,
  limit = 12,
  className = "",
}: {
  paused: boolean;
  onSelectVisitor?: (id: number) => void;
  limit?: number;
  className?: string;
}) {
  const { streaming, bust } = useStreamSync(paused);
  const recent = usePolling(
    () => api.events("", limit, 0),
    [bust, limit],
    streaming ? 800 : 2500,
    paused,
  );

  return (
    <Card title="Live Events" className={`events-card ${className}`.trim()}>
      {recent.data && recent.data.items.length > 0 ? (
        <ul className="event-feed compact">
          {recent.data.items.map((e) => {
            const kind = entryKindLabel(e.event_type, e.note);
            const isEntry = e.event_type === "entry";
            return (
              <li key={`${e.id}-${e.timestamp}`}>
                <button
                  type="button"
                  className="event-row"
                  onClick={() => onSelectVisitor?.(e.face_id)}
                >
                  <Thumb src={e.image} size={32} round />
                  <span className={`event-icon ${kind.kind}`}>
                    {isEntry ? <ArrowDownLeft size={14} /> : <ArrowUpRight size={14} />}
                  </span>
                  <span className="event-meta">
                    <b>ID: V{String(e.face_id).padStart(4, "0")}</b>
                    <span className={`event-kind event-kind-${kind.kind}`}>{kind.label}</span>
                  </span>
                  <span className="event-time">{fmtTime(e.timestamp)}</span>
                </button>
              </li>
            );
          })}
        </ul>
      ) : (
        <Empty title="No events yet" hint="Start a stream to see entries and exits." />
      )}
    </Card>
  );
}
