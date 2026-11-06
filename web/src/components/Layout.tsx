import { ReactNode } from "react";
import { Activity, LayoutDashboard, List, Pause, Play, Terminal, Users } from "lucide-react";

export type Page = "overview" | "visitors" | "events" | "logs";

const NAV: { id: Page; label: string; icon: ReactNode }[] = [
  { id: "overview", label: "Overview", icon: <LayoutDashboard size={18} /> },
  { id: "visitors", label: "Visitors", icon: <Users size={18} /> },
  { id: "events", label: "Event Log", icon: <List size={18} /> },
  { id: "logs", label: "System Log", icon: <Terminal size={18} /> },
];

const TITLES: Record<Page, string> = {
  overview: "Overview",
  visitors: "Visitors",
  events: "Event Log",
  logs: "System Log",
};

export default function Layout({ page, onNavigate, paused, onTogglePause, connected, updatedAt, children }: {
  page: Page; onNavigate: (p: Page) => void; paused: boolean; onTogglePause: () => void;
  connected: boolean; updatedAt: Date | null; children: ReactNode;
}) {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <Activity size={22} />
          <span>FaceTrack<small>Analytics</small></span>
        </div>
        <nav>
          {NAV.map((n) => (
            <button key={n.id} className={`nav-item ${page === n.id ? "active" : ""}`} onClick={() => onNavigate(n.id)}>
              {n.icon}
              {n.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">Real-time face detection, recognition &amp; visitor counting</div>
      </aside>

      <div className="main">
        <header className="topbar">
          <h1>{TITLES[page]}</h1>
          <div className="topbar-right">
            <span className={`status ${connected ? (paused ? "paused" : "live") : "offline"}`}>
              <i />
              {!connected ? "API offline" : paused ? "Paused" : "Live"}
              {updatedAt && connected && <small>updated {updatedAt.toLocaleTimeString()}</small>}
            </span>
            <button className="btn" onClick={onTogglePause}>
              {paused ? <Play size={15} /> : <Pause size={15} />}
              {paused ? "Resume" : "Pause"}
            </button>
          </div>
        </header>
        <main className="content">{children}</main>
      </div>
    </div>
  );
}
