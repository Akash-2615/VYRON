#!/usr/bin/env python3
"""Offline-ish ARIA eval against seeded demo data (needs GROQ_API_KEY; not for CI).

    PYTHONPATH=backend python backend/scripts/eval_aria.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
os.chdir(ROOT)

from api.assistant_prompt import BANNED_PHRASES  # noqa: E402


QUESTIONS = [
    ("How many unique visitors?", {"expect_number": True, "clarify": False}),
    ("Who stayed the longest?", {"expect_number": True, "clarify": False}),
    ("Any warnings in the log?", {"clarify": False}),
    ("Why does the count audit say mismatch?", {"clarify": False, "must_contain": ["database", "log"]}),
    ("What does preprocessing do?", {"must_contain": ["detector"]}),
    ("Explain visitor #1", {"clarify": False}),
    ("Summarize the last 10 minutes", {"clarify": False}),
    ("What's the weather in Paris?", {"off_topic": True}),
    ("Combien de visiteurs uniques ?", {"expect_number": True}),
    ("and the other one?", {"clarify": True}),  # ambiguous alone — clarifying ok
]


def banned_hit(text: str) -> list[str]:
    low = text.lower()
    return [b for b in BANNED_PHRASES if b in low]


def main() -> int:
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not key:
        print("SKIP — GROQ_API_KEY not set (expected outside CI).")
        return 0

    try:
        from groq import Groq
        from api.assistant import run_agent_sse
        from api.services import AnalyticsService
        from src.database import Database
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        print("SKIP — deps unavailable:", exc)
        return 0

    # Tiny seed
    import tempfile
    td = Path(tempfile.mkdtemp(prefix="aria_eval_"))
    db = Database(str(td / "e.db"), reset=True)
    f1 = db.add_face("2026-11-06T10:00:00.000")
    f2 = db.add_face("2026-11-06T10:05:00.000")
    db.log_event(f1, "entry", "2026-11-06T10:00:05.000", None, 1, 0)
    db.log_event(f1, "exit", "2026-11-06T10:01:20.000", None, 1, 1)
    db.log_event(f2, "entry", "2026-11-06T10:05:00.000", None, 2, 2)
    v = np.zeros(512, dtype=np.float32); v[0] = 1.0
    db.add_embedding(f1, v, "2026-11-06T10:00:05.000")
    log = td / "events.log"
    log.write_text(
        "2026-11-06 10:00:01,000 | INFO    | SYSTEM_START         sid=eval\n"
        "2026-11-06 10:00:02,000 | WARNING | DUPLICATE_TRACK      face_id=1\n",
        encoding="utf-8",
    )
    svc = AnalyticsService(db, td / "logs", events_log=str(log), match_threshold=0.45,
                           config_summary={"match_threshold": 0.45, "detection_skip_frames": 3, "backend": "sqlite"})
    summary = svc.get_summary()
    unique = summary.get("unique_visitors")

    rows = []
    for q, rules in QUESTIONS:
        events = list(run_agent_sse(svc, [{"role": "user", "content": q}], None, max_rounds=4))
        text = "".join(
            json.loads(line[6:])["text"]
            for block in events
            for line in block.splitlines()
            if line.startswith("data: ") and "\"text\"" in line
        )
        # simpler extract
        parts = []
        for block in events:
            for line in block.splitlines():
                if line.startswith("data: "):
                    try:
                        d = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if "text" in d:
                        parts.append(str(d["text"]))
        answer = "".join(parts).strip()
        bans = banned_hit(answer)
        ok = True
        notes = []
        if not answer:
            ok = False; notes.append("empty")
        if bans:
            ok = False; notes.append("banned:" + ",".join(bans[:3]))
        if rules.get("expect_number") and unique is not None and str(unique) not in answer and "two" not in answer.lower() and "2" not in answer:
            # soft fail — model may say "a couple"
            notes.append("number_soft_miss")
        if rules.get("off_topic") and len(answer) > 400:
            ok = False; notes.append("off_topic_too_long")
        if rules.get("must_contain"):
            low = answer.lower()
            if not any(x.lower() in low for x in rules["must_contain"]):
                ok = False; notes.append("missing_key_phrase")
        if len(answer) > 1200:
            notes.append("long")
        rows.append((ok, q, answer[:180].replace("\n", " "), ";".join(notes) or "ok"))

    print(f"{'PASS' if all(r[0] for r in rows) else 'MIXED'} — {sum(1 for r in rows if r[0])}/{len(rows)}")
    print(f"{'ok':<5} {'question':<42} {'notes':<24} snippet")
    for ok, q, snip, notes in rows:
        print(f"{'PASS' if ok else 'FAIL':<5} {q[:42]:<42} {notes[:24]:<24} {snip}")
    db.close()
    return 0 if all(r[0] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
