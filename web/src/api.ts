export interface Summary {
  unique_visitors: number;
  total_entries: number;
  total_exits: number;
  currently_inside: number;
  returning_visitors: number;
  avg_dwell_seconds: number | null;
  peak_occupancy: number;
  first_event: string | null;
  last_event: string | null;
  away_unique: number;
  away_entries: number;
  away_exits: number;
  away_currently: number;
  away_with_clothes: number;
}

export interface AwayVisitor {
  away_id: number;
  first_seen: string;
  last_seen: string;
  visit_count: number;
  appearance_summary: string | null;
  appearance: AppearanceInfo | null;
  thumbnail: string | null;
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
}

export interface Visitor {
  face_id: number;
  first_seen: string;
  last_seen: string;
  visit_count: number;
  total_dwell_seconds: number;
  status: "inside" | "left";
  face_image: string | null;
  thumbnail: string | null;
  latest_image: string | null;
  appearance_summary: string | null;
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

export interface LiveStatus {
  active: boolean;
  path: string | null;
  age_sec: number | null;
  bytes?: number;
}

export const api = {
  summary: () => get<Summary>("/api/summary"),
  timeline: () => get<Timeline>("/api/timeline"),
  visitors: () => get<Visitor[]>("/api/visitors"),
  visitor: (id: number) => get<VisitorDetail>(`/api/visitors/${id}`),
  events: (type: string, limit: number, offset: number) =>
    get<EventPage>(`/api/events?limit=${limit}&offset=${offset}${type ? `&type=${type}` : ""}`),
  liveStatus: () => get<LiveStatus>("/api/live/status"),
  awayVisitors: () => get<AwayVisitor[]>("/api/away-visitors"),
};

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
  total: number;
  counts: Record<string, number>;
  items: LogEntry[];
}

export const api2 = {
  embeddings: (id: number) => get<EmbeddingInfo>(`/api/visitors/${id}/embeddings`),
  logs: (limit: number, event: string, level: string, q: string) => {
    const p = new URLSearchParams({ limit: String(limit) });
    if (event) p.set("event", event);
    if (level) p.set("level", level);
    if (q) p.set("q", q);
    return get<LogPage>(`/api/logs?${p.toString()}`);
  },
};
