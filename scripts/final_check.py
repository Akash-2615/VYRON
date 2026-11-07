#!/usr/bin/env python3
"""End-of-run hygiene checks. Exit 1 on any FAIL.

    python scripts/final_check.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_ISO = re.compile(
    r"(?:^|[^\d])(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)"
)
_FIELD_TS = re.compile(
    r"\b(?:ts|last_seen|entry_ts|exit_ts)=(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)"
)
_SECRET = re.compile(
    r"(?i)("
    r"postgresql(\+[a-z0-9]+)?://[^\s\"']+|"
    r"\bpassword\s*[=:]\s*\S+|"
    r"\bDATABASE_URL\s*=\s*\S+|"
    r"\bTRACKER_DSN\s*=\s*\S+"
    r")"
)
_AWAY = re.compile(r"\bAWAY_[A-Z_]+\b")


def _pass(name: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"PASS  {name}{suffix}")


def _fail(name: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"FAIL  {name}{suffix}")


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    return p.returncode, out.strip()


def _parse_day(ts: str) -> date | None:
    raw = ts.strip().replace(" ", "T", 1)
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw[:26], fmt).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _latest_session_lines(log_text: str) -> tuple[str | None, list[str]]:
    """Return (session_id, lines belonging to the latest SYSTEM_START session)."""
    lines = log_text.splitlines()
    sid = None
    for ln in reversed(lines):
        if "SYSTEM_START" in ln and "sid=" in ln:
            for part in ln.split():
                if part.startswith("sid="):
                    sid = part.split("=", 1)[1]
                    break
            break
    if not sid:
        return None, []
    return sid, [ln for ln in lines if f"sid={sid}" in ln]


def check_count_py() -> bool:
    code, out = _run([sys.executable, "scripts/count.py", "--source", "both",
                      "--session", "latest"])
    print(out)
    if code == 0:
        _pass("count.py (both, latest)")
        return True
    _fail("count.py (both, latest)", f"exit={code}")
    return False


def check_verify_log() -> bool:
    code, out = _run([sys.executable, "scripts/verify_log.py", "--session", "latest"])
    print(out)
    if code == 0:
        _pass("verify_log.py")
        return True
    _fail("verify_log.py", f"exit={code}")
    return False


def check_dates(log_path: Path, db_path: Path) -> bool:
    today = date.today()
    window = timedelta(days=1)
    bad: list[str] = []

    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for ln in text.splitlines():
            if not any(k in ln for k in (" ENTRY", "| ENTRY", " EXIT", "| EXIT", "last_seen=")):
                continue
            if "AWAY_" in ln:
                continue
            for m in _FIELD_TS.finditer(ln):
                d = _parse_day(m.group("ts"))
                if d is None:
                    continue
                if abs((d - today).days) > window.days:
                    bad.append(f"log:{m.group('ts')}")
            # Also catch bare ISO in ENTRY/EXIT lines without field prefix.
            if "ENTRY" in ln or " EXIT" in ln or "| EXIT" in ln:
                for m in _ISO.finditer(ln):
                    d = _parse_day(m.group("ts"))
                    if d and abs((d - today).days) > window.days:
                        bad.append(f"log:{m.group('ts')}")

    if db_path.exists():
        try:
            from src.database import Database
            db = Database(str(db_path), backend="sqlite", reset=False)
            try:
                for row in db.fetch_dicts(
                    "SELECT event_type, timestamp FROM events "
                    "WHERE event_type IN ('entry','exit')"
                ):
                    d = _parse_day(str(row["timestamp"]))
                    if d and abs((d - today).days) > window.days:
                        bad.append(f"db_event:{row['timestamp']}")
                for row in db.fetch_dicts("SELECT face_id, last_seen FROM faces"):
                    d = _parse_day(str(row["last_seen"]))
                    if d and abs((d - today).days) > window.days:
                        bad.append(f"db_last_seen:{row['last_seen']}")
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001
            _fail("date_window", f"db_open_failed:{exc}")
            return False

    # Dedup while preserving order
    seen = set()
    uniq = []
    for b in bad:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    if uniq:
        _fail("date_window", f"{len(uniq)} stamp(s) >1 day from today: {uniq[:5]}")
        return False
    _pass("date_window", "ENTRY/EXIT/last_seen within ±1 day of today")
    return True


def check_system_start_stop(log_path: Path) -> bool:
    if not log_path.exists():
        _fail("system_start_stop", "events.log missing")
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    sid, lines = _latest_session_lines(text)
    if not sid:
        _fail("system_start_stop", "no SYSTEM_START found")
        return False
    starts = sum(1 for ln in lines if "SYSTEM_START" in ln)
    stops = sum(1 for ln in lines if "SYSTEM_STOP" in ln)
    if starts == 1 and stops == 1:
        _pass("system_start_stop", f"session={sid}")
        return True
    _fail("system_start_stop", f"session={sid} START={starts} STOP={stops}")
    return False


def check_no_away_in_events(log_path: Path) -> bool:
    if not log_path.exists():
        _fail("no_away_in_events", "events.log missing")
        return False
    hits = []
    for i, ln in enumerate(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        if _AWAY.search(ln):
            hits.append(i)
    if hits:
        _fail("no_away_in_events", f"{len(hits)} AWAY_* line(s), first line {hits[0]}")
        return False
    _pass("no_away_in_events")
    return True


def check_archive_dir(log_dir: Path) -> bool:
    archive = log_dir / "archive"
    if archive.is_dir():
        _pass("logs/archive_exists", str(archive))
        return True
    _fail("logs/archive_exists", f"missing {archive} (run python main.py --fresh once)")
    return False


def check_no_secrets(cfg_path: Path, log_dir: Path) -> bool:
    paths: list[Path] = []
    if cfg_path.exists():
        paths.append(cfg_path)
    if log_dir.exists():
        paths.extend(sorted(log_dir.rglob("*.log")))
    # Committed defaults / examples only — never scan .env
    for name in ("config.json", ".env.example"):
        p = ROOT / name
        if p.exists() and p not in paths:
            paths.append(p)

    bad: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # .env.example may document DATABASE_URL= as a commented template — allow comments.
        for i, ln in enumerate(text.splitlines(), 1):
            stripped = ln.strip()
            if stripped.startswith("#"):
                continue
            # Empty postgres object / placeholder keys are fine.
            if re.search(r'"postgres"\s*:\s*\{\s*\}', stripped):
                continue
            if _SECRET.search(stripped):
                # Allow commented DSN examples already skipped; flag real values.
                bad.append(f"{path.relative_to(ROOT)}:{i}")
    if bad:
        _fail("no_dsn_or_password", f"{len(bad)} hit(s): {bad[:5]}")
        return False
    _pass("no_dsn_or_password", "config + logs clean")
    return True


def main() -> int:
    try:
        from src.config import load_config
        st = load_config(str(ROOT / "config.json"))["storage"]
        log_path = ROOT / (st.get("events_log") or "logs/events.log")
        log_dir = ROOT / (st.get("log_dir") or "logs")
        db_path = ROOT / (st.get("db_path") or "data/visitors.db")
    except Exception:  # noqa: BLE001
        log_path = ROOT / "logs" / "events.log"
        log_dir = ROOT / "logs"
        db_path = ROOT / "data" / "visitors.db"

    print("=== final_check ===")
    results = [
        check_count_py(),
        check_verify_log(),
        check_dates(log_path, db_path),
        check_system_start_stop(log_path),
        check_no_away_in_events(log_path),
        check_archive_dir(log_dir),
        check_no_secrets(ROOT / "config.json", log_dir),
    ]
    failed = sum(1 for ok in results if not ok)
    print("---")
    if failed:
        print(f"RESULT: FAIL ({failed} check(s))")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
