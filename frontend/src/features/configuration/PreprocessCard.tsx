import { useEffect, useMemo, useRef, useState } from "react";
import { Info } from "lucide-react";
import { Card } from "../../components/ui";
import { api } from "../../api";
import { usePreprocess } from "./usePreprocess";

const PRESET_LABELS: Record<string, string> = {
  off: "Off",
  dull_low_contrast: "Dull / low contrast",
  low_light: "Low light",
  noisy_cctv: "Noisy CCTV",
};

const DETECTOR_RES_HELP =
  "Downscale to the detector input size first, then enhance that smaller image. Live Monitor stays on the original. Preview panes: Original | Enhanced.";

function stepEnabled(settings: Record<string, unknown> | undefined, name: string): boolean {
  if (!settings) return false;
  if (name === "quality_assist") {
    return Boolean((settings.quality_assist as { enabled?: boolean })?.enabled);
  }
  const block = settings[name] as { enabled?: boolean } | undefined;
  return Boolean(block?.enabled);
}

function isFrameCatalogStep(step: { name: string; kind?: string }): boolean {
  if (step.kind) return step.kind === "gate" || step.kind === "frame";
  return step.name !== "quality_assist";
}

export function PreprocessCard({ paused = false }: { paused?: boolean }) {
  const { data, error, saving, savedAt, save, saveDebounced, reset } = usePreprocess(paused);
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const helpRef = useRef<HTMLDivElement | null>(null);
  const settings = data?.settings;
  const metrics = data?.metrics || {};
  const writable = data?.writable !== false;
  const masterOn = Boolean(settings?.enabled ?? data?.enabled);

  const statusChip = useMemo(() => {
    if (!masterOn) return { label: "Off", cls: "" };
    if (metrics.gate_active) return { label: "Active", cls: "count-ok" };
    return { label: "On · gate idle (video looks fine)", cls: "count-ok" };
  }, [masterOn, metrics.gate_active]);

  const frameSteps = useMemo(
    () => (data?.step_catalog || []).filter(isFrameCatalogStep),
    [data?.step_catalog],
  );
  const blurAssist = useMemo(
    () => (data?.step_catalog || []).find((s) => s.name === "quality_assist" || s.kind === "blur_gate"),
    [data?.step_catalog],
  );

  const anyStepOn = useMemo(() => {
    if (!settings) return false;
    return frameSteps.some((s) => stepEnabled(settings as unknown as Record<string, unknown>, s.name));
  }, [settings, frameSteps]);

  const setMaster = (on: boolean) => {
    void save({ enabled: on });
  };

  const setStepEnabled = (name: string, on: boolean) => {
    if (on) setExpanded((prev) => ({ ...prev, [name]: true }));
    if (name === "quality_assist") {
      void save({ quality_assist: { enabled: on } });
      return;
    }
    void save({ [name]: { enabled: on } });
  };

  const setParam = (step: string, key: string, value: unknown) => {
    if (!writable) return;
    saveDebounced({ [step]: { [key]: value } });
  };

  const applyPreset = (name: string) => {
    void save({ preset: name });
  };

  const toggleExpand = (name: string) => {
    setExpanded((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  // Swap preview only after the next JPEG loads — avoids remount flicker.
  useEffect(() => {
    if (!masterOn || paused) {
      setPreviewSrc(null);
      return;
    }
    let cancelled = false;
    const pull = () => {
      const url = api.preprocessPreviewUrl();
      const img = new Image();
      img.onload = () => { if (!cancelled) setPreviewSrc(url); };
      img.src = url;
    };
    pull();
    const id = window.setInterval(pull, 1600);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [masterOn, paused]);

  useEffect(() => {
    if (!helpOpen) return;
    const onDoc = (e: MouseEvent) => {
      if (helpRef.current && !helpRef.current.contains(e.target as Node)) setHelpOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setHelpOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [helpOpen]);

  return (
    <Card title="Video preprocessing">
      <p className="muted small" style={{ marginTop: 0 }}>
        Changes apply within ~1 second — no restart. Original video is never modified;
        enhancement feeds the face detector only (optional blur-gate assist).
      </p>

      {error && <p className="count-warn small" role="alert">{error}</p>}
      {!data && !error && <p className="muted small">Loading…</p>}

      {data && (
        <div className="preprocess-body">
          {/* Master row */}
          <div className="preprocess-master">
            <label className={`preprocess-switch ${masterOn ? "on" : ""}`}>
              <input
                type="checkbox"
                checked={masterOn}
                disabled={!writable}
                aria-label="Enable video preprocessing"
                onChange={(e) => setMaster(e.target.checked)}
              />
              <span className="preprocess-switch-track" aria-hidden="true" />
              <span className="preprocess-switch-label">
                {masterOn ? "Preprocessing on" : "Preprocessing off"}
              </span>
            </label>

            <span className={`preprocess-status-chip small ${statusChip.cls}`}>{statusChip.label}</span>
            {saving && <span className="muted small">Saving…</span>}
            {savedAt && !saving && (
              <span className="muted small">Saved · v{data.version}</span>
            )}
            {!writable && (
              <span className="count-warn small">Read-only (loopback or X-Admin-Token required)</span>
            )}
          </div>

          {masterOn && !anyStepOn && (
            <p className="muted small preprocess-hint">
              Master is on, but every step below is off — pick a preset or enable CLAHE / gamma to affect detection.
            </p>
          )}

          {/* Metrics + presets */}
          <div className="preprocess-control-row">
            <ul className="preprocess-metrics" aria-label="Preprocess metrics">
              <li><span>Cost</span><b>{Number(metrics.total_avg_ms ?? 0).toFixed(1)}/{Number(metrics.total_p95_ms ?? 0).toFixed(1)} ms</b></li>
              <li><span>Ready</span><b>{Number(metrics.ready_pct ?? 100).toFixed(0)}%</b></li>
              <li><span>Fallback</span><b>{String(metrics.fallback ?? 0)}</b></li>
              <li><span>Dropped</span><b>{String(metrics.dropped ?? 0)}</b></li>
              <li><span>Source</span><b>{data.source}</b></li>
            </ul>

            <div className="stream-mode-toggle preprocess-presets" role="group" aria-label="Presets">
              {Object.keys(PRESET_LABELS).map((key) => (
                <button
                  key={key}
                  type="button"
                  className="stream-mode-btn"
                  disabled={!writable}
                  aria-label={`Preset ${PRESET_LABELS[key]}`}
                  onClick={() => applyPreset(key)}
                >
                  {PRESET_LABELS[key]}
                </button>
              ))}
              <button type="button" className="stream-mode-btn" disabled={!writable} onClick={() => void reset()}>
                Reset
              </button>
            </div>
          </div>

          {/* Frame filters + blur-gate assist (separate) */}
          <div className={`preprocess-main ${masterOn ? "" : "is-off"}`}>
            <ol className="preprocess-steps">
              {frameSteps.map((step, idx) => {
                const on = stepEnabled(settings as unknown as Record<string, unknown>, step.name);
                const block = (settings as unknown as Record<string, Record<string, unknown>>)?.[step.name] || {};
                const stepMs = (metrics.step_ms as Record<string, number> | undefined)?.[step.name];
                const open = expanded[step.name] === true;
                return (
                  <li
                    key={step.name}
                    className={`preprocess-step ${on ? "is-on" : "is-off"}`}
                  >
                    <div className="preprocess-step-head">
                      <span className="preprocess-order" aria-hidden="true">{idx + 1}</span>
                      <input
                        type="checkbox"
                        checked={on}
                        disabled={!writable || !masterOn}
                        aria-label={`Toggle ${step.label}`}
                        onChange={(e) => setStepEnabled(step.name, e.target.checked)}
                      />
                      <button
                        type="button"
                        className="preprocess-step-title"
                        onClick={() => toggleExpand(step.name)}
                        aria-expanded={open}
                      >
                        {step.label}
                      </button>
                      {stepMs != null && <span className="muted small">{stepMs.toFixed(1)} ms</span>}
                    </div>
                    {open && (
                      <div className="preprocess-step-body">
                        <p className="muted small preprocess-blurb">{step.what}</p>
                        <p className="muted small preprocess-blurb"><em>{step.why}</em></p>
                        {step.params.length > 0 && (
                          <div className="preprocess-params">
                            {step.params.map((p) => {
                              const val = block[p.key] ?? p.default;
                              if (p.type === "select") {
                                return (
                                  <label key={p.key} className="small">
                                    {p.label}
                                    <select
                                      value={String(val)}
                                      disabled={!writable || !masterOn || !on}
                                      aria-label={p.label}
                                      onChange={(e) => setParam(step.name, p.key, e.target.value)}
                                    >
                                      {(p.options || []).map((o) => <option key={o} value={o}>{o}</option>)}
                                    </select>
                                  </label>
                                );
                              }
                              return (
                                <label key={p.key} className="small">
                                  {p.label}{p.unit ? ` (${p.unit})` : ""}: {String(val)}
                                  <input
                                    type="range"
                                    min={p.min}
                                    max={p.max}
                                    step={p.type === "slider" ? 0.05 : 1}
                                    value={Number(val)}
                                    disabled={!writable || !masterOn || !on}
                                    aria-label={p.label}
                                    onChange={(e) => setParam(step.name, p.key, Number(e.target.value))}
                                  />
                                </label>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    )}
                  </li>
                );
              })}
            </ol>

            <aside className="preprocess-aside">
              <div className="preprocess-advanced-row" ref={helpRef}>
                <label className="small preprocess-advanced">
                  <input
                    type="checkbox"
                    checked={Boolean(settings?.process_at_detector_resolution)}
                    disabled={!writable || !masterOn}
                    aria-label="Process at detector resolution"
                    onChange={(e) => void save({ process_at_detector_resolution: e.target.checked })}
                  />
                  Process at detector resolution
                </label>
                <button
                  type="button"
                  className="preprocess-info-btn"
                  aria-label="About process at detector resolution"
                  aria-expanded={helpOpen}
                  title="About this option"
                  onClick={() => setHelpOpen((v) => !v)}
                >
                  <Info size={14} />
                </button>
                {helpOpen && (
                  <div className="preprocess-info-pop" role="tooltip">
                    {DETECTOR_RES_HELP}
                  </div>
                )}
              </div>
              {blurAssist && (
                <div className="preprocess-blur-assist">
                  <label className="small preprocess-advanced">
                    <input
                      type="checkbox"
                      checked={stepEnabled(settings as unknown as Record<string, unknown>, "quality_assist")}
                      disabled={!writable || !masterOn}
                      aria-label="Quality-gate assist"
                      onChange={(e) => setStepEnabled("quality_assist", e.target.checked)}
                    />
                    {blurAssist.label}
                  </label>
                  <p className="muted small" style={{ margin: 0 }}>
                    Blur-gate branch only — contrast on the 112×112 crop from the <b>original</b> frame. Not a detector frame filter.
                  </p>
                </div>
              )}
            </aside>
          </div>

          <div className="preprocess-preview-hero">
            <div className="preprocess-preview-hero-head">
              <h4>Original / Enhanced</h4>
              <span className="muted small">
                Left = Original (identity · live · crops) · Right = Enhanced (detector when ready)
              </span>
            </div>
            {masterOn ? (
              previewSrc ? (
                <img
                  className="preprocess-preview-hero-img"
                  src={previewSrc}
                  alt="Preprocess before and after"
                />
              ) : (
                <p className="muted small">Waiting for first before/after frame…</p>
              )
            ) : (
              <p className="muted small">Turn preprocessing on to see a live before/after comparison.</p>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}
