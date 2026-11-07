import {
  Camera, Check, Crosshair, Database, Layers, Scan, UserPlus,
} from "lucide-react";

const STEPS = [
  { title: "Video", sub: "RTSP / File / Cam", Icon: Camera },
  { title: "Face Detection", sub: "YOLOv8 + Retina", Icon: Crosshair },
  { title: "Embedding", sub: "InsightFace", Icon: Layers },
  { title: "Tracking", sub: "ByteTrack", Icon: Scan },
  { title: "Auto Registration", sub: "New Face", Icon: UserPlus },
  { title: "Logging & Counting", sub: "DB + Filesystem", Icon: Database },
];

export default function AiPipeline({ active = false }: { active?: boolean }) {
  return (
    <section className={`ai-pipeline ${active ? "is-active" : ""}`} aria-label="AI Pipeline">
      <header className="ai-pipeline-head">
        <h3>AI Pipeline <span>(Real-time)</span></h3>
        <small className={active ? "on" : ""}>{active ? "Live" : "Idle"}</small>
      </header>
      <div className="ai-pipeline-track">
        {STEPS.map((step, i) => (
          <div key={step.title} className="ai-pipeline-node-wrap">
            {i > 0 && (
              <span className="ai-pipeline-arrow" aria-hidden>
                <i />
              </span>
            )}
            <article className="ai-pipeline-node">
              <span className="ai-pipeline-ok" aria-hidden>
                <Check size={10} strokeWidth={3} />
              </span>
              <span className="ai-pipeline-icon">
                <step.Icon size={20} strokeWidth={1.8} />
              </span>
              <b>{step.title}</b>
              <small>{step.sub}</small>
            </article>
          </div>
        ))}
      </div>
    </section>
  );
}
