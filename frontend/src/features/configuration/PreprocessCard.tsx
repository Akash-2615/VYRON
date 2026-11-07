import { useMemo, useState } from "react";
import { Card } from "../../components/ui";
import { api } from "../../api";
import { usePreprocess } from "./usePreprocess";

const PRESET_LABELS: Record<string, string> = {
  off: "Off",
  dull_low_contrast: "Dull / low contrast",
  low_light: "Low light",
  noisy_cctv: "Noisy CCTV",
};

function stepEnabled(settings: Record<string, unknown> | undefined, name: string): boolean {
  if (!settings) return false;
  if (name === "quality_assist") {
    return Boolean((settings.quality_assist as { enabled?: boolean })?.enabled);
  }
  const block = settings[name] as { enabled?: boolean } | undefined;
  return Boolean(block?.enabled);
}

export function PreprocessCard({ paused = false }: { paused?: boolean }) {
  const { data, error, saving, savedAt, save, saveDebounced, reset } = usePreprocess(paused);
  const [previewBust, setPreviewBust] = useState(0);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const settings = data?.settings;
  const metrics = data?.metrics || {};
  const writable = data?.writable !== false;
  const masterOn = Boolean(settings?.enabled ?? data?.enabled);

  const statusChip = useMemo(() => {
    if (!masterOn) return { label: "Off", cls: "" };
    if (metrics.gate_active) return { label: "Active", cls: "count-ok" };
    return { label: "On · gate idle (video looks fine)", cls: "count-ok" };
  }, [masterOn, metrics.gate_active]);

  const anyStepOn = useMemo(() => {
    if (!settings) return false;
    return (data?.step_catalog || []).some((s) => stepEnabled(settings as unknown as Record<string, unknown>, s.name));
  }, [settings, data?.step_catalog]);

  const setMaster = (on: boolean) => {
    void save({ enabled: on });
    setPreviewBust((n) => n + 1);
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
    setPreviewBust((n) => n + 1);
  };

  const toggleExpand = (name: string) => {
    setExpanded((prev) => ({ ...prev, [name]: !prev[name] }));
  };

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

          {/* Steps + preview */}
          <div className={`preprocess-main ${masterOn ? "" : "is-off"}`}>
            <ol className="preprocess-steps">
              {(data.step_catalog || []).map((step, idx) => {
                const on = stepEnabled(settings as unknown as Record<string, unknown>, step.name);
                const block = (settings as unknown as Record<string, Record<string, unknown>>)?.[step.name] || {};
                const stepMs = (metrics.step_ms as Record<string, number> | undefined)?.[step.name];
                // Keep collapsed by default so all 7 steps stay on-screen; expand on click / enable.
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

              <div className="preprocess-preview">
                <p className="muted small" style={{ marginBottom: 6 }}>Before / after</p>
                {masterOn ? (
                  <img
                    key={previewBust}
                    src={`${api.preprocessPreviewUrl()}&b=${previewBust}`}
                    alt="Preprocess before and after"
                    onError={(e) => { (e.target as HTMLImageElement).style.opacity = "0.35"; }}
                    onLoad={(e) => { (e.target as HTMLImageElement).style.opacity = "1"; }}
                  />
                ) : (
                  <p className="muted small">Turn preprocessing on to see a live preview.</p>
                )}
              </div>
            </aside>
          </div>
        </div>
      )}
    </Card>
  );
}
