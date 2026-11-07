import { useEffect, useState } from "react";

/** Fetches immediately, then every `intervalMs` unless paused. Keeps last good data on errors. */
export function usePolling<T>(fn: () => Promise<T>, deps: unknown[], intervalMs = 3000, paused = false) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  useEffect(() => {
    let alive = true;
    const run = async () => {
      try {
        const d = await fn();
        if (!alive) return;
        setData(d);
        setError(null);
        setUpdatedAt(new Date());
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (alive) setLoading(false);
      }
    };
    run();
    if (paused) return () => { alive = false; };
    const id = setInterval(run, intervalMs);
    return () => { alive = false; clearInterval(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, paused, intervalMs]);

  return { data, error, loading, updatedAt };
}
