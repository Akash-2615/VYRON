# Frontend

Vite + React 18 + TypeScript dashboard (VYRON).

```
frontend/
  public/
    brand/           logos + marks (canonical brand assets)
    favicon.png      served at /favicon.png
  src/
    api/             HTTP client + polling hook
      client.ts
      usePolling.ts
      index.ts
    pages/           route-level screens
    components/      shared UI (layout, live monitor, drawers, assistant/)
    App.tsx          shell + routing
    main.tsx         entry
    theme.tsx        light/dark
    styles.css       design tokens + layout
    utils.ts
  index.html
  package.json
  vite.config.ts     dev proxy → :8000 /api and /media
```

Dev: `npm run dev` (from this folder). Build: `npm run build` → `dist/` (API can serve it).
