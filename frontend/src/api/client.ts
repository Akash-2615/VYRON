export interface Summary {
  unique_visitors: number;
  total_entries: number;
  total_exits: number;
  currently_inside: number;
  hidden_now?: number;
  returning_visitors: number;
  avg_dwell_seconds: number | null;
  avg_time_in_frame_seconds?: number | null;
  peak_occupancy: number;
  first_event: string | null;
  last_event: string | null;
  away_unique: number;
  away_entries: number;
  away_exits: number;
  away_currently: number;
  away_with_clothes: number;
}

export interface VisitRow {
  visit_id: number;
  entry_ts: string;
  exit_ts: string | null;
  visible_seconds: number;
  occluded_seconds: number;
  resume_count: number;
  exit_reason: string | null;
}

export interface AwayVisitor {
  away_id: number;
  first_seen: string;
  last_seen: string;
  visit_count: number;
  status?: "inside" | "left";
  appearance_summary: string | null;
  appearance: AppearanceInfo | null;
  appearance_image: string | null;
  thumbnail: string | null;
  linked_face_id?: number | null;
}

export interface TimelinePoint {
  t: string;
  entries: number;
  exits: number;
  occupancy: number;
}
export interface Timeline {
  bucket_seconds: number;
  points: TimelinePoint[];
}

export interface AppearanceInfo {
  summary?: string;
  dominant_colors?: string[];
  upper_color?: string;
  upper_type?: string;
  lower_type?: string;
  accessories?: string;
  hair?: string;
  model?: string;
  scores?: Record<string, number>;
  evidence_image?: string;
}

export interface Visitor {
  face_id: number;
  first_seen: string;
  last_seen: string;
  visit_count: number;
  total_dwell_seconds: number;
  time_in_frame_seconds?: number;
  visible_seconds?: number;
  occluded_seconds?: number;
  resume_count?: number;
  hidden_now?: boolean;
  status: "inside" | "left";
  face_image: string | null;
  thumbnail: string | null;
  latest_image: string | null;
  appearance_summary: string | null;
  appearance_image: string | null;
  appearance: AppearanceInfo | null;
}

export interface VisitorEvent {
  id: number;
  event_type: "entry" | "exit";
  timestamp: string;
  image: string | null;
  note: string | null;
}
export interface VisitorDetail extends Visitor {
  events: VisitorEvent[];
  visits?: VisitRow[];
}

export interface EventItem {
  id: number;
  face_id: number;
  event_type: "entry" | "exit";
  timestamp: string;
  image: string | null;
  track_id: number | null;
  frame_idx: number | null;
  note: string | null;
}
export interface EventPage {
  total: number;
  items: EventItem[];
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} (${path})`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown = {}): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

async function put<T>(path: string, body: unknown = {}): Promise<T> {
  const res = await fetch(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export interface PreprocessParamSpec {
  key: string;
  label: string;
  type: "slider" | "number" | "select";
  min?: number;
  max?: number;
  default?: number | string;
  unit?: string;
  options?: string[];
}

export interface PreprocessStepCatalog {
  name: string;
  label: string;
  what: string;
  why: string;
  params: PreprocessParamSpec[];
}

export interface PreprocessSettings {
  enabled: boolean;
  process_at_detector_resolution: boolean;
  mode: string;
  max_wait_ms: number;
  auto_gate: Record<string, unknown>;
  denoise: Record<string, unknown>;
  white_balance: Record<string, unknown>;
  gamma: Record<string, unknown>;
  clahe: Record<string, unknown>;
  sharpen: Record<string, unknown>;
  quality_assist: Record<string, unknown>;
  version?: number;
}

export interface PreprocessPayload {
  enabled: boolean;
  settings: PreprocessSettings;
  defaults: PreprocessSettings;
  step_catalog: PreprocessStepCatalog[];
  presets?: Record<string, unknown>;
  metrics: Record<string, unknown>;
  writable: boolean;
  version: number;
  source: "override" | "config" | string;
}

export interface LiveStatus {
  active: boolean;
  path: string | null;
  age_sec: number | null;
  bytes?: number;
  persons_in_frame?: number | null;
  faces_in_frame?: number | null;
  no_face_in_frame?: number | null;
  faces_turned_away?: number | null;
  stats_age_sec?: number | null;
  fps?: number | null;
  latency_ms?: number | null;
  width?: number | null;
  height?: number | null;
}

export interface TrackerJobStatus {
  ok?: boolean;
  running: boolean;
  starting: boolean;
  url: string | null;
  mode?: string | null;
  playlist?: string | null;
  sources?: string[];
  source_index?: number;
  source_total?: number;
  error: string | null;
  started_at: number | null;
  frames: number;
  unique_visitors: number | null;
  backend: string | null;
  thread_alive?: boolean;
}

export interface LocalVideoItem {
  path: string;
  name: string;
  exists: boolean;
  size_bytes: number | null;
}

export interface LocalVideoList {
  items: LocalVideoItem[];
  available: LocalVideoItem[];
  count: number;
  available_count: number;
}

export interface CountAudit {
  unique_visitors: number | null;
  entries: number | null;
  exits: number | null;
  currently_inside: number | null;
  returning_visitors: number | null;
  open_visits: number | null;
  source: "db" | "log" | "none" | string;
  derived_from_log: {
    unique_visitors: number;
    entries: number;
    exits: number;
    open_visits: number;
    session_id?: string | null;
    available?: boolean;
    last_unique_total?: number | null;
    system_stop_unique?: number | null;
    last_count_snapshot?: {
      unique_total?: number;
      entries?: number;
      exits?: number;
    } | null;
  } | null;
  consistent: boolean | null;
  diffs: string[];
  notes?: string[];
  session_id: string | null;
  as_of: string;
}

export const api = {
  summary: () => get<Summary>("/api/summary"),
  count: (session = "latest") => get<CountAudit>(`/api/count?session=${encodeURIComponent(session)}`),
  timeline: () => get<Timeline>("/api/timeline"),
  visitors: () => get<Visitor[]>("/api/visitors"),
  visitor: (id: number) => get<VisitorDetail>(`/api/visitors/${id}`),
  events: (type: string, limit: number, offset: number) =>
    get<EventPage>(`/api/events?limit=${limit}&offset=${offset}${type ? `&type=${type}` : ""}`),
  liveStatus: () => get<LiveStatus>("/api/live/status"),
  awayVisitors: () => get<AwayVisitor[]>("/api/away-visitors"),
  awayVisitor: (id: number) =>
    get<AwayVisitor & {
      events: { id: number; event_type: string; timestamp: string; image: string | null; note: string | null; track_id?: number | null; frame_idx?: number | null }[];
    }>(`/api/away-visitors/${id}`),
  trackerStatus: () => get<TrackerJobStatus>("/api/tracker/status"),
  localVideos: () => get<LocalVideoList>("/api/videos"),
  trackerStart: (
    url: string,
    reset = false,
    mode: "rtsp" | "local" | "webcam" = "rtsp",
    opts?: { playlist?: "one" | "all"; sources?: string[]; reset?: boolean },
  ) =>
    post<TrackerJobStatus>("/api/tracker/start", {
      url,
      reset: opts?.reset ?? reset,
      mode,
      playlist: opts?.playlist ?? "one",
      sources: opts?.sources,
    }),
  trackerStop: () => post<TrackerJobStatus>("/api/tracker/stop"),
  assistantStatus: () => get<AssistantStatus>("/api/assistant/status"),
  preprocess: () => get<PreprocessPayload>("/api/preprocess"),
  preprocessPut: (body: Record<string, unknown>) => put<PreprocessPayload>("/api/preprocess", body),
  preprocessReset: () => post<PreprocessPayload>("/api/preprocess/reset"),
  preprocessPreviewUrl: () => `/api/preprocess/preview?t=${Date.now()}`,
};

export interface AssistantStatus {
  enabled: boolean;
  configured: boolean;
  model: string;
  groq_installed?: boolean;
  has_api_key?: boolean;
  prompt_version?: string;
  live_summary?: string;
  live_snapshot?: {
    visitors?: number;
    inside_now?: number;
    warnings_count?: number;
  } | null;
}

export interface AssistantContext {
  page?: string;
  selected_face_id?: number | null;
  filters?: Record<string, unknown>;
}

export type AssistantSseHandler = (ev: {
  event: string;
  data: Record<string, unknown>;
}) => void;

/** POST /api/assistant/chat as SSE (fetch + ReadableStream). */
export async function streamAssistantChat(
  messages: { role: string; content: string }[],
  context: AssistantContext,
  onEvent: AssistantSseHandler,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/assistant/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ messages, context }),
    signal,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  if (!res.body) throw new Error("No response body");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() || "";
    for (const block of parts) {
      const lines = block.split("\n");
      let event = "message";
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      try {
        onEvent({ event, data: JSON.parse(data) as Record<string, unknown> });
      } catch {
        onEvent({ event, data: { raw: data } });
      }
    }
  }
}

export interface EmbeddingSample {
  id: number;
  created_at: string;
  image: string | null;
  norm: number;
  vector: number[];
}
export interface SimilarFace {
  face_id: number;
  similarity: number;
  thumbnail: string | null;
}
export interface EmbeddingInfo {
  face_id: number;
  dim: number;
  count: number;
  match_threshold: number;
  intra_similarity: number | null;
  face_image: string | null;
  mean: number[];
  samples: EmbeddingSample[];
  similar: SimilarFace[];
}

export interface LogEntry {
  ts: string;
  level: string;
  event: string;
  message: string;
  fields: Record<string, string>;
}
export interface LogPage {
  path: string;
  exists: boolean;
  source?: string;
  total: number;
  counts: Record<string, number>;
  items: LogEntry[];
}

export const api2 = {
  embeddings: (id: number) => get<EmbeddingInfo>(`/api/visitors/${id}/embeddings`),
  logs: (limit: number, event: string, level: string, q: string, source: "events" | "away" = "events") => {
    const p = new URLSearchParams({ limit: String(limit), source });
    if (event) p.set("event", event);
    if (level) p.set("level", level);
    if (q) p.set("q", q);
    return get<LogPage>(`/api/logs?${p.toString()}`);
  },
};
