# VYRON demo video script (Loom / YouTube)

**Target length:** ~8–10 minutes (cut to 6 if needed).  
**Before record:** `make db` · `make api` · `make web` · gallery already has faces (or run 30s of sample video first) · optional `GROQ_API_KEY` for ARIA.

**Screen layout tip:** Loom with camera PIP + browser full screen. Cursor highlight on. Speak slowly; pause 1s after each “show”.

---

## 0. Cold open (0:00–0:30)

> Hi — I’m Akash, and this is **VYRON**, powered by **ARIA**.  
> It’s an AI visitor counter for the Katomaran hackathon.  
> We detect faces with **YOLOv8**, recognize with **InsightFace ArcFace**, track with a **ByteTrack-style** tracker, and enforce **exactly one entry and one exit per visit**.  
> Unique visitors equals the number of rows in the **faces** table — so re-identification never invents a new person.  
> I’ll walk through architecture, config, live streaming, gallery, events, counting, preprocess, and ARIA.

*(Show README architecture diagram for 3 seconds, then switch to dashboard Overview.)*

---

## 1. Problem + invariants (0:30–1:15)

> The hard requirements were:  
> **one ENTRY and one EXIT per visit**,  
> **unique count that does not grow on return**,  
> support for **local video, webcam, and RTSP**,  
> and triple output — **Postgres**, **events.log**, and **crop images**.  
> Away walkers without a usable face are tracked separately — they do **not** inflate unique faces.

*(Point at Overview KPIs: Unique / Entries / Exits / Inside / Auto Registered.)*

---

## 2. Architecture in one breath (1:15–2:00)

*(Show `docs/assets/vyron-architecture.png` or README Architecture section.)*

> Pipeline: **Video / webcam / RTSP** → optional preprocess for the **detector only** → **YOLOv8-face** (+ Retina assist if enabled) → boxes mapped back to the **original** frame → **ByteTrack** → quality gate → crop from **original** → **InsightFace** embed → gallery match or auto-register → presence state machine → ENTRY/EXIT → Postgres + logs + crops.  
> Critical invariant: enhanced frames never feed embeddings or Live Monitor — identity evidence stays honest.

---

## 3. Config — frame-skip & tuning (2:00–2:40)

*(Open `config.json` briefly, highlight `detection_skip_frames`, `match_threshold`, `stream.process_max_width`.)*

> Config lives in `config.json`.  
> `detection_skip_frames` controls how often YOLO runs — tracks coast between detections.  
> `match_threshold` is around **0.52** for buffalo_l — raise it to reduce merges, lower it to reduce twin IDs.  
> Postgres DSN is **only** in `.env` as `DATABASE_URL` — never in config.

---

## 4. Live Monitor — start a stream (2:40–4:00)

*(Go to Live Monitor. Show RTSP / Local / Webcam tabs.)*

> Dashboard Start supports three sources: **RTSP**, **local MP4**, and **webcam**.  
> Gallery is **shared** across sources unless I tick **Clear registered faces**.  
> I’ll start a **local video**.

*(Click Start. Wait for “Streaming” + annotated feed.)*

> Live feed shows face IDs and confidence.  
> People facing away can appear as **NO-FACE** — counted as persons, **not** as unique faces.  
> Side KPIs update: Unique, Entries, Exits, Inside.  
> **Live Events** on the right show New entry, Re-entry, and Exit in sync with the stream.  
> Stop is safe — we fixed cross-thread webcam release so the API doesn’t crash on Stop.

*(Optional: Stop → switch to Webcam → Start 5s → Stop. Or skip if time-tight.)*

---

## 5. Overview — KPIs + charts + pipeline (4:00–4:40)

*(Overview page.)*

> Overview is the ops view: unique visitors, entries, exits, currently inside, auto-registered.  
> Visitor Flow and Visitor Status charts.  
> Bottom strip is the AI pipeline: Video → Face Detection → Embedding → Tracking → Auto Registration → Logging.  
> If the count badge says **Verified against log**, DB and `events.log` agree.

---

## 6. Registered Faces — gallery + appearance + embeddings (4:40–5:50)

*(Registered Faces. Search / filter / sort briefly.)*

> Each card is one unique `face_id`. Visits, time in frame, visible time, first/last seen, and **OpenCLIP** clothing tags.

*(Open one visitor drawer.)*

> Visit history: multiple visits, same ID — returning never creates a new face.  
> Appearance crop and tags from **open_clip ViT-B-32**.  
> Face & Embedding tab: ArcFace **512-d**, stored samples, mean vector heatmap, match threshold, and **closest other identities** — scores stay below threshold so we don’t merge strangers.

---

## 7. Event Logs + System Log (5:50–6:30)

*(Event Logs.)*

> Entry/exit table with face crops, timestamps, track id, frame index, and notes like `stream_end` or re-entry.

*(System Log.)*

> This is the live tail of `events.log`: `TRACK_STARTED`, `FACE_RECOGNIZED`, `ENTRY`, `EXIT`, `COUNT_SNAPSHOT`, with `sid=` on every line. Crops are written under `logs/entries` and `logs/exits`.

---

## 8. Analytics — away / no-face (6:30–7:00)

*(Analytics.)*

> Analytics shows dwell, returning visitors, peak occupancy, and the **no-face / away** grid.  
> These people are excluded from the unique face count. Some cards show **Turned → face #N** when we later link a face.

---

## 9. Configuration — audit + preprocess (7:00–8:00)

*(Configuration.)*

> Invariants are spelled out: unique = faces rows; one entry + one exit per visit; returning keeps the same ID.  
> **Count Audit** compares DB gallery vs log — here it’s **Consistent**.

*(Toggle preprocess ON briefly if time; else describe with Off.)*

> Optional **preprocess** is for the **face detector only** — dull/dark auto-gate, denoise, gamma, CLAHE.  
> Live Monitor and embeddings still use the **original** frame. Original / Enhanced compare appears when preprocess is on. I’ll leave it off for the fair run.

---

## 10. ARIA assistant (8:00–8:45)

*(Open ARIA — Ctrl/Cmd+K or pill.)*

> **ARIA** is a read-only Groq assistant. It can query visitors and dwell — it cannot mutate the tracker, DB, or settings, and we never send images or embeddings to the model.

*(Ask: “Who stayed longest?” or “How many unique visitors?”)*

> It answers from live tools — for example Visitor #N with dwell time — and I can open that card from the UI.

---

## 11. Close (8:45–9:30)

*(Back to Overview or README last line.)*

> To recap: VYRON does detect → track → recognize → auto-register, with **exactly one entry and exit per visit**, unique = `COUNT(faces)`, file/webcam/RTSP, Postgres + logs + crops, optional detector preprocess, and ARIA.  
> Repo, sample screenshots, and DOCX are in the README.  
> This project is a part of a hackathon run by **katomaran.com**.  
> Thanks for watching.

---

## Cut list if you must hit ~6 minutes

| Keep | Skip / shorten |
|---|---|
| Open + invariants + architecture | Long config.json scroll |
| Live Start + Live Events | Webcam second source |
| One visitor card (appearance OR embedding, not both deep) | Closest-identities deep dive |
| Event Logs OR System Log (one) | Both |
| Count audit Consistent | Preprocess toggle demo |
| One ARIA question | Suggested-prompt tour |

---

## On-screen checklist (before you hit Record)

- [ ] API + web running; System Online green  
- [ ] Local video selected; Clear faces **unchecked** (unless you want a fresh demo)  
- [ ] Someone/faces visible in feed within first 20s of Start  
- [ ] Count audit Consistent (or Verified badge)  
- [ ] ARIA key set if you demo chat  
- [ ] Mic level OK; close Slack/notifications  

---

## Short “elevator” version (90 seconds) — emergency only

> VYRON: YOLOv8 + InsightFace + ByteTrack visitor counter. One entry and one exit per visit; unique equals faces rows. Local video, webcam, RTSP into a React dashboard. Live IDs, gallery with clothing tags and embeddings, entry/exit crops, Postgres and events.log with a consistent count audit. Optional detector-only preprocess. ARIA answers questions read-only. Built for Katomaran — thanks.
