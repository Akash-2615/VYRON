import { ReactNode, useEffect, useState } from "react";
import {
  BarChart3, Check, IdCard, LayoutDashboard, List, Moon, PanelLeftClose, PanelLeftOpen,
  Pause, Play, Settings, Sun, Terminal, Users, Video,
} from "lucide-react";
import { api } from "../api";
import { useTheme } from "../theme";
import { usePolling } from "../usePolling";
import BrandMark from "./BrandMark";

export type Page = "overview" | "live" | "visitors" | "events" | "analytics" | "logs" | "config";

const SIDEBAR_KEY = "vyron.sidebarOpen";

const NAV: { id: Page; label: string; icon: ReactNode }[] = [
  { id: "overview", label: "Overview", icon: <LayoutDashboard size={18} /> },
  { id: "live", label: "Live Monitor", icon: <Video size={18} /> },
  { id: "visitors", label: "Visitors", icon: <Users size={18} /> },
  { id: "events", label: "Event Logs", icon: <List size={18} /> },
  { id: "visitors", label: "Registered Faces", icon: <IdCard size={18} /> },
  { id: "analytics", label: "Analytics", icon: <BarChart3 size={18} /> },
  { id: "config", label: "Configuration", icon: <Settings size={18} /> },
  { id: "logs", label: "System Log", icon: <Terminal size={18} /> },
];

const TITLES: Record<Page, { title: string; sub: string }> = {
  overview: {
    title: "Overview",
    sub: "Auto Registration · Face Recognition · Real-time Tracking · Visitor Counting",
  },
  live: {
    title: "Live Monitor",
    sub: "Full-screen feed · stream controls · live KPIs",
  },
  visitors: {
    title: "Registered Faces",
    sub: "Unique identities in the gallery",
  },
  events: {
    title: "Event Logs",
    sub: "Entry and exit events with crop evidence",
  },
  analytics: {
    title: "Analytics",
    sub: "Secondary KPIs, no-face walkers, and flow detail",
  },
  logs: {
    title: "System Log",
    sub: "Pipeline diagnostics and tracker events",
  },
  config: {
    title: "Configuration",
    sub: "Theme, stream modes, and system invariants",
  },
};

export default function Layout({ page, onNavigate, paused, onTogglePause, connected, updatedAt, children }: {
  page: Page; onNavigate: (p: Page) => void; paused: boolean; onTogglePause: () => void;
  connected: boolean; updatedAt: Date | null; children: ReactNode;
}) {
  const { theme, toggle } = useTheme();
  const meta = TITLES[page];
  const [navFocus, setNavFocus] = useState<string>("Overview");
  const [clock, setClock] = useState(() => new Date());
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    try { return localStorage.getItem(SIDEBAR_KEY) !== "0"; } catch { return true; }
  });

  useEffect(() => {
    const id = window.setInterval(() => setClock(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    try { localStorage.setItem(SIDEBAR_KEY, sidebarOpen ? "1" : "0"); } catch { /* ignore */ }
  }, [sidebarOpen]);

  const live = usePolling(api.liveStatus, [], 1500, paused);
  const tracker = usePolling(api.trackerStatus, [], 2000, paused);
  const health = usePolling(async () => {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error("health");
    return r.json() as Promise<{ ok: boolean; backend?: string; db_exists?: boolean }>;
  }, [], 5000, paused);

  useEffect(() => {
    const match = NAV.find((n) => n.id === page);
    if (match) setNavFocus(match.label);
  }, [page]);

  const camOk = Boolean(live.data?.active || tracker.data?.running);
  const dbOk = Boolean(health.data?.ok);
  const pipeOk = Boolean(tracker.data?.running || live.data?.active);
  const fps = live.data?.fps;
  const latency = live.data?.latency_ms;

  function go(n: (typeof NAV)[number]) {
    setNavFocus(n.label);
    onNavigate(n.id);
  }

  return (
    <div className={`shell ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <aside className={`sidebar ${sidebarOpen ? "" : "is-collapsed"}`} aria-hidden={!sidebarOpen}>
        <div className="brand sidebar-brand">
          <BrandMark />
        </div>

        <nav>
          {NAV.map((n) => (
            <button
              key={n.label}
              type="button"
              className={`nav-item ${navFocus === n.label ? "active" : ""}`}
              onClick={() => go(n)}
              tabIndex={sidebarOpen ? 0 : -1}
            >
              {n.icon}
              <span>{n.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-foot">
          <div className="sys-card">
            <div className={`sys-pill ${connected ? "online" : "offline"}`}>
              <i />
              {connected ? "System Online" : "API Offline"}
            </div>
            <ul className="sys-checks">
              <li className={camOk ? "ok" : ""}><Check size={13} /> Camera {tracker.data?.mode === "local" ? "(File)" : tracker.data?.mode === "webcam" ? "(Webcam)" : "(RTSP)"}</li>
              <li className={dbOk ? "ok" : ""}><Check size={13} /> Database</li>
              <li className={pipeOk ? "ok" : ""}><Check size={13} /> AI Pipeline</li>
            </ul>
            <div className="sys-metrics">
              <span><b>{fps != null ? fps : "—"}</b> FPS</span>
              <span><b>{latency != null ? latency : "—"}</b> ms</span>
            </div>
          </div>
          <div className="sidebar-wave holo-wave" aria-hidden>
            <svg viewBox="0 0 280 90" preserveAspectRatio="none">
              <defs>
                <linearGradient id="holoFill" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#7c3aed" stopOpacity="0.15" />
                  <stop offset="45%" stopColor="#a855f7" stopOpacity="0.45" />
                  <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.2" />
                </linearGradient>
                <linearGradient id="holoStroke" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#c084fc" stopOpacity="0.95" />
                  <stop offset="50%" stopColor="#e879f9" stopOpacity="0.9" />
                  <stop offset="100%" stopColor="#67e8f9" stopOpacity="0.85" />
                </linearGradient>
                <filter id="holoGlow" x="-20%" y="-40%" width="140%" height="180%">
                  <feGaussianBlur stdDeviation="2.2" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              <path className="wave-path wave-fill" fill="url(#holoFill)" d="M0,58 C35,40 70,74 110,52 C150,30 190,70 240,48 C260,40 270,46 280,42 L280,90 L0,90 Z" />
              <path className="wave-path wave-1" fill="none" stroke="url(#holoStroke)" strokeWidth="1.6" filter="url(#holoGlow)" d="M-20,46 C30,22 70,68 120,40 C170,12 210,62 280,34" />
              <path className="wave-path wave-2" fill="none" stroke="url(#holoStroke)" strokeWidth="1.3" filter="url(#holoGlow)" d="M-20,54 C40,32 85,72 140,48 C195,24 230,66 300,44" />
              <path className="wave-path wave-3" fill="none" stroke="url(#holoStroke)" strokeWidth="1.1" filter="url(#holoGlow)" d="M-20,62 C45,44 95,78 155,56 C210,36 245,72 300,52" />
              <path className="wave-path wave-4" fill="none" stroke="#a855f7" strokeOpacity="0.45" strokeWidth="0.9" d="M-20,38 C50,18 100,58 160,36 C220,14 250,50 300,28" />
            </svg>
          </div>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="topbar-left">
            <div className={`topbar-brand ${sidebarOpen ? "" : "is-visible"}`} aria-hidden={sidebarOpen}>
              <BrandMark compact />
            </div>
            <button
              type="button"
              className="btn btn-ghost sidebar-toggle"
              onClick={() => setSidebarOpen((v) => !v)}
              title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
              aria-label={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
              aria-expanded={sidebarOpen}
            >
              {sidebarOpen ? <PanelLeftClose size={17} /> : <PanelLeftOpen size={17} />}
            </button>
            <div className="topbar-title">
              <h1 className="page-title">{meta.title}</h1>
              <p className="topbar-tagline">{meta.sub}</p>
            </div>
          </div>
          <div className="topbar-right">
            <span className={`status ${connected ? (paused ? "paused" : "live") : "offline"}`}>
              <i />
              {!connected ? "API offline" : paused ? "Paused" : "Live"}
            </span>
            <time className="topbar-clock" dateTime={clock.toISOString()} title={updatedAt ? `Data updated ${updatedAt.toLocaleTimeString()}` : undefined}>
              {clock.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}
              {" · "}
              {clock.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
            </time>
            <button
              type="button"
              className="btn btn-ghost theme-toggle"
              onClick={toggle}
              title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            >
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
              {theme === "dark" ? "Light" : "Dark"}
            </button>
            <button type="button" className="btn" onClick={onTogglePause}>
              {paused ? <Play size={15} /> : <Pause size={15} />}
              {paused ? "Resume" : "Pause"}
            </button>
          </div>
        </header>
        <main className={`content ${page === "overview" ? "overview-fit" : ""} ${page === "live" ? "live-fit" : ""}`}>
          {children}
        </main>
      </div>
    </div>
  );
}
