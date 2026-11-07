"""Generate DEMO data (clearly fake placeholder faces / random embeddings) so the dashboard can be
previewed without running the models.  Writes to demo/ only - never touches your real DB/logs.

    python scripts/seed_demo.py
    # Demo still uses a local SQLite file under demo/ for offline UI preview only.
    # Runtime API/tracker require DATABASE_URL (local Postgres).
"""
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.database import Database          # noqa: E402
from src.event_logger import EventLogger   # noqa: E402


def placeholder(face_id: int, size: int = 160) -> np.ndarray:
    rng = random.Random(face_id)
    img = np.full((160, 160, 3), (rng.randint(60, 140), rng.randint(60, 140), rng.randint(60, 140)), np.uint8)
    skin = (rng.randint(140, 200), rng.randint(150, 205), rng.randint(180, 235))
    cv2.circle(img, (80, 80), 52, skin, -1)
    cv2.circle(img, (62, 70), 6, (30, 30, 30), -1)
    cv2.circle(img, (98, 70), 6, (30, 30, 30), -1)
    cv2.ellipse(img, (80, 104), (18, 9), 0, 0, 180, (40, 40, 120), 3)
    cv2.putText(img, f"#{face_id}", (6, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img if size == 160 else cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def main():
    Path("demo").mkdir(exist_ok=True)
    db = Database("demo/visitors.db", reset=True)
    ev = EventLogger("demo/logs", "demo/logs/events.log")
    rng = random.Random(7)
    base = datetime.now().replace(microsecond=0) - timedelta(minutes=12)
    lines = []                                   # (timestamp, level, event, fields) -> events.log

    def log(ts, event, level="INFO", **f):
        lines.append((ts, level, event, " ".join(f"{k}={v}" for k, v in f.items())))

    unique = 0
    for fid in range(1, 19):
        nrng = np.random.default_rng(fid)
        identity = unit(nrng.normal(size=512))
        visits, t = [], rng.randint(0, 560)
        for _ in range(2 if rng.random() < 0.3 else 1):
            dur = rng.randint(20, 90)
            visits.append((t, dur))
            t += dur + rng.randint(40, 120)

        for n, (start, dur) in enumerate(visits):
            t_in, t_out = base + timedelta(seconds=start), base + timedelta(seconds=start + dur)
            track = fid * 10 + n
            log(t_in - timedelta(seconds=3), "TRACK_STARTED", track=track, frame=start * 25, conf="0.91")
            log(t_in - timedelta(seconds=2), "EMBEDDING_GENERATED", track=track, det_score="0.99", attempt=1)
            emb_ts = t_in - timedelta(seconds=1)
            closeup = ev.save_crop("face", fid, emb_ts, placeholder(fid, 224))
            if n == 0:
                emb = unit(identity + 0.25 * nrng.normal(size=512) / np.sqrt(512) * 4)
                unique += 1
                assert db.add_face(t_in.isoformat(timespec="milliseconds")) == fid
                db.add_embedding(fid, emb, emb_ts.isoformat(timespec="milliseconds"), closeup)
                db.set_face_image(fid, closeup)
                log(t_in - timedelta(seconds=2), "EMBEDDING_GENERATED", track=track, det_score="0.98", attempt=2)
                log(emb_ts, "FACE_REGISTERED", face_id=fid, track=track, unique_total=unique, closeup=closeup)
            else:
                emb = unit(identity + nrng.normal(size=512) / np.sqrt(512) * 0.9)
                sim = float(emb @ unit(identity))
                db.add_embedding(fid, emb, emb_ts.isoformat(timespec="milliseconds"), closeup)
                log(emb_ts, "FACE_RECOGNIZED", face_id=fid, track=track, similarity=f"{sim:.3f}")
            for kind, ts, note in (("entry", t_in, "new" if n == 0 else "re-entry"), ("exit", t_out, "lost")):
                path = ev.save_crop(kind, fid, ts, placeholder(fid))
                db.log_event(fid, kind, ts.isoformat(timespec="milliseconds"), path, track, int(ts.timestamp()) % 100000, note)
                if kind == "exit":
                    log(ts, "TRACK_ENDED", track=track, face_id=fid, reason=note)
                log(ts, kind.upper(), face_id=fid, track=track, note=note, image=path)

    lines.sort(key=lambda x: x[0])
    with open("demo/logs/events.log", "w", encoding="utf-8") as f:
        f.write(f"{base:%Y-%m-%d %H:%M:%S},000 | INFO    | SYSTEM_START         sources=1 skip=2 db_reset=True\n")
        for ts, level, event, rest in lines:
            f.write(f"{ts:%Y-%m-%d %H:%M:%S},{ts.microsecond // 1000:03d} | {level:<7} | {event:<20} {rest}\n")
    print("Demo DB:", db.unique_count(), "visitors ->", db.event_counts())
    db.close()


if __name__ == "__main__":
    main()
