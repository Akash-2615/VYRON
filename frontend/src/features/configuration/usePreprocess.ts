import { useCallback, useEffect, useRef, useState } from "react";
import { api, type PreprocessPayload } from "../../api";

export function usePreprocess(paused = false) {
  const [data, setData] = useState<PreprocessPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mounted = useRef(true);
  const saveGen = useRef(0);
  const savingRef = useRef(false);

  const refresh = useCallback(async () => {
    if (paused || (typeof document !== "undefined" && document.hidden)) return;
    try {
      const p = await api.preprocess();
      if (!mounted.current) return;
      // Don't clobber an in-flight optimistic save with a stale poll.
      if (savingRef.current) return;
      setData(p);
      setError(null);
    } catch (e) {
      if (!mounted.current) return;
      setError(e instanceof Error ? e.message : "API unreachable");
    }
  }, [paused]);

  useEffect(() => {
    mounted.current = true;
    void refresh();
    const id = setInterval(() => void refresh(), 3000);
    return () => {
      mounted.current = false;
      clearInterval(id);
      if (timer.current) clearTimeout(timer.current);
    };
  }, [refresh]);

  const cancelDebounce = useCallback(() => {
    if (timer.current) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  const save = useCallback(async (partial: Record<string, unknown>, opts?: { optimistic?: boolean }) => {
    cancelDebounce();
    const gen = ++saveGen.current;
    savingRef.current = true;
    setSaving(true);
    setError(null);

    // Optimistic master / nested toggles so Enable/Disable feels instant.
    if (opts?.optimistic !== false) {
      setData((prev) => {
        if (!prev) return prev;
        const nextSettings = { ...prev.settings } as PreprocessPayload["settings"] & Record<string, unknown>;
        for (const [k, v] of Object.entries(partial)) {
          if (k === "preset") continue;
          if (v && typeof v === "object" && !Array.isArray(v) && typeof nextSettings[k] === "object") {
            nextSettings[k] = { ...(nextSettings[k] as object), ...(v as object) };
          } else {
            (nextSettings as Record<string, unknown>)[k] = v;
          }
        }
        const enabled = typeof nextSettings.enabled === "boolean" ? nextSettings.enabled : prev.enabled;
        return { ...prev, enabled, settings: nextSettings };
      });
    }

    try {
      const p = await api.preprocessPut(partial);
      if (!mounted.current || gen !== saveGen.current) return p;
      setData(p);
      setSavedAt(Date.now());
      return p;
    } catch (e) {
      if (!mounted.current || gen !== saveGen.current) return;
      setError(e instanceof Error ? e.message : "Save failed");
      try {
        const p = await api.preprocess();
        if (mounted.current && gen === saveGen.current) setData(p);
      } catch { /* keep optimistic until next poll */ }
      return undefined;
    } finally {
      if (gen === saveGen.current) {
        savingRef.current = false;
        setSaving(false);
      }
    }
  }, [cancelDebounce]);

  const saveDebounced = useCallback((partial: Record<string, unknown>, ms = 400) => {
    if (timer.current) clearTimeout(timer.current);
    // Optimistic local param update while debouncing network write.
    setData((prev) => {
      if (!prev) return prev;
      const nextSettings = { ...prev.settings } as PreprocessPayload["settings"] & Record<string, unknown>;
      for (const [k, v] of Object.entries(partial)) {
        if (v && typeof v === "object" && !Array.isArray(v) && typeof nextSettings[k] === "object") {
          nextSettings[k] = { ...(nextSettings[k] as object), ...(v as object) };
        } else {
          (nextSettings as Record<string, unknown>)[k] = v;
        }
      }
      return { ...prev, settings: nextSettings };
    });
    timer.current = setTimeout(() => {
      void save(partial, { optimistic: false });
    }, ms);
  }, [save]);

  const reset = useCallback(async () => {
    cancelDebounce();
    const gen = ++saveGen.current;
    savingRef.current = true;
    setSaving(true);
    setError(null);
    try {
      const p = await api.preprocessReset();
      if (!mounted.current || gen !== saveGen.current) return;
      setData(p);
      setSavedAt(Date.now());
    } catch (e) {
      if (!mounted.current) return;
      setError(e instanceof Error ? e.message : "Reset failed");
    } finally {
      if (gen === saveGen.current) {
        savingRef.current = false;
        setSaving(false);
      }
    }
  }, [cancelDebounce]);

  return { data, error, saving, savedAt, refresh, save, saveDebounced, reset, cancelDebounce };
}
