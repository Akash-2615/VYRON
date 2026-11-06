import { useEffect, useState } from "react";
import { api, LiveStatus } from "../api";
import { Card, Empty } from "./ui";

export default function LiveMonitor({ paused }: { paused: boolean }) {
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [tick, setTick] = useState(0);
  const [freezeKey, setFreezeKey] = useState(0);

  useEffect(() => {
    if (paused) return;
    let cancelled = false;
    const pull = () => {
      api.liveStatus()
        .then((s) => { if (!cancelled) setStatus(s); })
        .catch(() => { if (!cancelled) setStatus(null); });
    };
    pull();
    const id = window.setInterval(pull, 1500);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [paused]);

  // Remount the MJPEG <img> when the tracker comes back online.
  useEffect(() => {
    if (!paused && status?.active) setTick((t) => t + 1);
  }, [paused, status?.active]);

  // On pause: drop the MJPEG connection and freeze on a single JPEG snapshot.
  useEffect(() => {
    if (paused) setFreezeKey(Date.now());
  }, [paused]);

  const active = Boolean(status?.active);
  const pill = paused ? "Paused" : active ? "Streaming" : "Waiting for tracker";
  const pillClass = paused ? "off" : active ? "on" : "off";

  return (
    <Card
      title="Live monitor"
      action={
        <span className={`live-pill ${pillClass}`}>
          <i />
          {pill}
        </span>
      }
    >
      {active ? (
        <div className="live-frame">
          {paused ? (
            <img
              key={freezeKey}
              src={`/api/live.jpg?t=${freezeKey}`}
              alt="Frozen live frame"
            />
          ) : (
            <img
              key={tick}
              src={`/api/live?t=${tick}`}
              alt="Annotated live feed with face boxes and confidence"
            />
          )}
          <p className="muted small live-legend">
            Magenta = walking away (back) · Outer box = person/clothes · Inner = face · Green = owner
          </p>
        </div>
      ) : (
        <Empty
          title="No live frame yet"
          hint="Run python main.py — annotated frames (boxes + confidence) stream here automatically."
        />
      )}
    </Card>
  );
}
