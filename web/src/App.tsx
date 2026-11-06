import { useState } from "react";
import { api } from "./api";
import Layout, { Page } from "./components/Layout";
import VisitorDrawer from "./components/VisitorDrawer";
import Events from "./pages/Events";
import Logs from "./pages/Logs";
import Overview from "./pages/Overview";
import Visitors from "./pages/Visitors";
import { usePolling } from "./usePolling";

export default function App() {
  const [page, setPage] = useState<Page>("overview");
  const [paused, setPaused] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const summary = usePolling(api.summary, [], 3000, paused);

  return (
    <Layout
      page={page}
      onNavigate={setPage}
      paused={paused}
      onTogglePause={() => setPaused((p) => !p)}
      connected={!summary.error}
      updatedAt={summary.updatedAt}
    >
      {summary.error && !summary.data && (
        <div className="banner">
          Cannot reach the API ({summary.error}). Start it with <code>uvicorn api.server:app --port 8000</code>.
        </div>
      )}
      {page === "overview" && <Overview summary={summary.data} paused={paused} onSelectVisitor={setSelected} />}
      {page === "visitors" && <Visitors paused={paused} onSelectVisitor={setSelected} />}
      {page === "events" && <Events paused={paused} onSelectVisitor={setSelected} />}
      {page === "logs" && <Logs paused={paused} onSelectVisitor={setSelected} />}
      {selected !== null && <VisitorDrawer faceId={selected} onClose={() => setSelected(null)} />}
    </Layout>
  );
}
