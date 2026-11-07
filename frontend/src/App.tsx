import { useEffect, useState } from "react";
import { api } from "./api";
import AssistantPanel from "./components/assistant/AssistantPanel";
import Layout, { Page } from "./components/Layout";
import AwayDrawer from "./components/AwayDrawer";
import VisitorDrawer from "./components/VisitorDrawer";
import Analytics from "./pages/Analytics";
import Config from "./pages/Config";
import Events from "./pages/Events";
import Live from "./pages/Live";
import Logs from "./pages/Logs";
import Overview from "./pages/Overview";
import Visitors from "./pages/Visitors";
import { usePolling } from "./api";

const ASSISTANT_KEY = "facetrack.assistantOpen";

export default function App() {
  const [page, setPage] = useState<Page>("overview");
  const [paused, setPaused] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [selectedAway, setSelectedAway] = useState<number | null>(null);
  const [assistantOpen, setAssistantOpen] = useState(() => {
    try { return localStorage.getItem(ASSISTANT_KEY) === "1"; } catch { return false; }
  });
  const summary = usePolling(api.summary, [], 3000, paused);

  useEffect(() => {
    try { localStorage.setItem(ASSISTANT_KEY, assistantOpen ? "1" : "0"); } catch { /* ignore */ }
  }, [assistantOpen]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setAssistantOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const wrapClass = [
    "app-frame",
    assistantOpen ? "assistant-open" : "",
    (selected !== null || selectedAway !== null) && assistantOpen ? "drawer-with-assistant" : "",
  ].filter(Boolean).join(" ");

  return (
    <div className={wrapClass}>
      <Layout
        page={page}
        onNavigate={setPage}
        paused={paused}
        onTogglePause={() => setPaused((p) => !p)}
        connected={!summary.error}
        updatedAt={summary.updatedAt}
        assistantOpen={assistantOpen}
        onToggleAssistant={() => setAssistantOpen((v) => !v)}
      >
        {summary.error && !summary.data && (
          <div className="banner">
            Cannot reach the API ({summary.error}). Start it with <code>uvicorn api.server:app --port 8000</code>.
          </div>
        )}
        {page === "overview" && (
          <Overview
            summary={summary.data}
            paused={paused}
            onSelectVisitor={setSelected}
            onOpenLive={() => setPage("live")}
          />
        )}
        {page === "live" && <Live summary={summary.data} paused={paused} />}
        {page === "visitors" && <Visitors paused={paused} onSelectVisitor={setSelected} />}
        {page === "events" && <Events paused={paused} onSelectVisitor={setSelected} />}
        {page === "analytics" && (
          <Analytics
            summary={summary.data}
            paused={paused}
            onSelectVisitor={(id) => { setSelectedAway(null); setSelected(id); }}
            onSelectAway={(id) => { setSelected(null); setSelectedAway(id); }}
          />
        )}
        {page === "config" && <Config paused={paused} />}
        {page === "logs" && <Logs paused={paused} onSelectVisitor={setSelected} />}
      </Layout>

      <AssistantPanel
        open={assistantOpen}
        onClose={() => setAssistantOpen(false)}
        context={{ page, selected_face_id: selected }}
        onOpenVisitor={(id) => { setSelectedAway(null); setSelected(id); }}
        onNavigate={(p) => setPage(p as Page)}
      />

      {selected !== null && (
        <VisitorDrawer faceId={selected} onClose={() => setSelected(null)} />
      )}
      {selectedAway !== null && (
        <AwayDrawer
          awayId={selectedAway}
          onClose={() => setSelectedAway(null)}
          onOpenFace={(id) => { setSelectedAway(null); setSelected(id); }}
        />
      )}

    </div>
  );
}
