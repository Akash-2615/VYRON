import { api } from "../api";
import { usePolling } from "../api";

/** Shared stream activity key so Events / Logs / Live Events / KPIs refresh together. */
export function useStreamSync(paused: boolean) {
  const tracker = usePolling(api.trackerStatus, [], 1000, paused);
  const live = usePolling(api.liveStatus, [], 1000, paused);
  const streaming = Boolean(
    tracker.data?.running
    || tracker.data?.starting
    || tracker.data?.stopping
    || tracker.data?.thread_alive,
  );
  const bust = [
    tracker.data?.frames ?? 0,
    tracker.data?.unique_visitors ?? "",
    live.data?.faces_in_frame ?? 0,
    live.data?.persons_in_frame ?? 0,
    Math.floor((live.data?.age_sec ?? 99) * 2),
    streaming ? 1 : 0,
  ].join(":");
  return { tracker, live, streaming, bust };
}
