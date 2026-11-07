# Architecture — modular layout

Behaviour-preserving package layout introduced on branch `refactor/modular`.
Old import paths remain via module-alias shims (see `src/*.py` files marked deprecated).

## Layer diagram

```mermaid
flowchart BT
  subgraph entry["Entrypoints"]
    main["backend/main.py / backend/scripts/run_rtsp.py"]
    scripts["backend/scripts/*"]
    api_entry["api.server:app (--app-dir backend)"]
  end
  subgraph api_layer["api/"]
    routers["routers / assistant"]
    services["services"]
  end
  subgraph high["src high"]
    pipeline["pipeline/"]
  end
  subgraph mid["src mid"]
    presence["presence/"]
    reporting["reporting/"]
  end
  subgraph leaf["src leaf"]
    detection["detection/"]
    recognition["recognition/"]
    tracking["tracking/"]
    video_io["video_io/"]
    viz["viz/"]
  end
  subgraph base["src base"]
    core["core/"]
    storage["storage/"]
    event_logging["event_logging/"]
    session_svc["session_svc/"]
  end
  main --> pipeline
  scripts --> reporting
  scripts --> storage
  api_entry --> routers
  routers --> services
  services --> storage
  services --> reporting
  pipeline --> presence
  pipeline --> detection
  pipeline --> recognition
  pipeline --> tracking
  pipeline --> video_io
  presence --> tracking
  detection --> core
  recognition --> core
  tracking --> core
  video_io --> core
  storage --> core
  event_logging --> core
  session_svc --> storage
  session_svc --> event_logging
```

## Request / data flow

```mermaid
sequenceDiagram
  participant UI as Dashboard
  participant API as api.server
  participant Job as tracker_job
  participant Pipe as FaceTrackingPipeline
  participant DB as storage.Database
  participant Log as event_logging
  UI->>API: POST /api/tracker/start
  API->>Job: start(url)
  Job->>Pipe: process_frame loop
  Pipe->>DB: faces/events/visits
  Pipe->>Log: events.log + crops
  UI->>API: GET /api/summary
  API->>DB: readonly queries
```

## Rules

- Lower layers never import higher ones (`backend/scripts/check_layers.py`).
- Heavy ML imports stay lazy inside detector / recognizer / appearance / retina constructors.
- Pipeline stays injectable (detector/embedder passed in) for tests without models.
- Do not change log event names, DB schema, API paths, or CLI flags while moving code.
