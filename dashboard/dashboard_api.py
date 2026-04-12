"""
API for ingress/egress webhooks (used by counter.py) and dashboard stats.

Dev (dashboard): run API on 8000 and Vite on 5173 — Vite proxies /api to 8000.
  uvicorn dashboard_api:app --host 127.0.0.1 --port 8000
  npm run dev

Production UI: build then open http://127.0.0.1:8000/
  npm run build
  uvicorn dashboard_api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

WEEKDAY_NAMES_EN = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "counts.db"
DIST_DIR = ROOT / "dist"

_lock = threading.Lock()

EVENT_WEIGHTS: dict[str, tuple[int, int]] = {
    "ingress": (1, 0),
    "ingress_undo": (-1, 0),
    "egress": (0, 1),
    "egress_undo": (0, -1),
}


def _week_display_label(week_start: datetime) -> str:
    """Human-readable Mon–Sun UTC range plus ISO week, e.g. 'Apr 1–7 · W14'."""
    d0 = week_start.date()
    d1 = (week_start + timedelta(days=6)).date()
    iso = d0.isocalendar()
    wk = iso.week
    if d0.year == d1.year:
        if d0.month == d1.month:
            inner = f"{d0.strftime('%b')} {d0.day}–{d1.day}"
        else:
            inner = f"{d0.strftime('%b %d')} – {d1.strftime('%b %d')}"
    else:
        inner = f"{d0.strftime('%b %d, %Y')} – {d1.strftime('%b %d, %Y')}"
    return f"{inner} · W{wk:02d}"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock:
        conn = _connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                ts REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
        conn.commit()
        conn.close()


def _record(kind: str) -> None:
    if kind not in EVENT_WEIGHTS:
        return
    ts = datetime.now(timezone.utc).timestamp()
    with _lock:
        conn = _connect()
        conn.execute("INSERT INTO events (kind, ts) VALUES (?, ?)", (kind, ts))
        conn.commit()
        conn.close()


def _sum_between(conn: sqlite3.Connection, t0: float, t1: float) -> tuple[int, int]:
    cur = conn.execute(
        "SELECT kind FROM events WHERE ts >= ? AND ts < ?",
        (t0, t1),
    )
    tin, tout = 0, 0
    for row in cur:
        di, do = EVENT_WEIGHTS.get(row["kind"], (0, 0))
        tin += di
        tout += do
    return tin, tout


def _totals(conn: sqlite3.Connection) -> tuple[int, int]:
    cur = conn.execute("SELECT kind FROM events")
    tin, tout = 0, 0
    for row in cur:
        di, do = EVENT_WEIGHTS.get(row["kind"], (0, 0))
        tin += di
        tout += do
    return tin, tout


def _floor_hour_utc(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)


def _hourly_net_full_day(conn: sqlite3.Connection, day_start: datetime) -> list[int]:
    """Net inside change per UTC hour (in − out) for a full calendar day."""
    out: list[int] = []
    for h in range(24):
        t0 = day_start + timedelta(hours=h)
        t1 = t0 + timedelta(hours=1)
        tin, tout = _sum_between(conn, t0.timestamp(), t1.timestamp())
        out.append(tin - tout)
    return out


def _hourly_net_and_cumulative_today(
    conn: sqlite3.Connection, day_start: datetime, now: datetime
) -> tuple[list[int | None], list[int | None]]:
    """Partial-hour net for current hour; None for hours not yet started (UTC)."""
    ts_now = now.timestamp()
    hourly: list[int | None] = []
    cumulative: list[int | None] = []
    run = 0
    for h in range(24):
        t0 = day_start + timedelta(hours=h)
        t1 = t0 + timedelta(hours=1)
        t0f = t0.timestamp()
        t1f = t1.timestamp()
        if t0f >= ts_now:
            hourly.append(None)
            cumulative.append(None)
            continue
        end = min(t1f, ts_now)
        tin, tout = _sum_between(conn, t0f, end)
        delta = tin - tout
        hourly.append(delta)
        run += delta
        cumulative.append(run)
    return hourly, cumulative


def _baseline_same_weekday(
    conn: sqlite3.Connection,
    now: datetime,
    target_weekday: int,
    max_scan_days: int = 56,
    max_samples: int = 12,
) -> tuple[list[float], list[float], int]:
    """
    Average hourly net flow and its cumulative curve for the same weekday
    over recent past days (excluding today), up to max_samples matching days.
    """
    today_midnight = now.replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
    )
    samples: list[list[int]] = []
    for i in range(1, max_scan_days + 1):
        day_start = today_midnight - timedelta(days=i)
        if day_start.weekday() != target_weekday:
            continue
        samples.append(_hourly_net_full_day(conn, day_start))
        if len(samples) >= max_samples:
            break
    n = len(samples)
    if n == 0:
        z = [0.0] * 24
        return z, z, 0
    hourly_avg: list[float] = []
    for h in range(24):
        hourly_avg.append(
            round(sum(s[h] for s in samples) / n, 2)
        )
    cumulative: list[float] = []
    s = 0.0
    for v in hourly_avg:
        s += v
        cumulative.append(round(s, 2))
    return hourly_avg, cumulative, n


def build_stats() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    with _lock:
        conn = _connect()
        total_in, total_out = _totals(conn)

        end_h = _floor_hour_utc(now)
        start_h = end_h - timedelta(hours=23)
        hourly_labels: list[str] = []
        hourly_in: list[int] = []
        hourly_out: list[int] = []
        h = start_h
        for _ in range(24):
            t0 = h.timestamp()
            t1 = (h + timedelta(hours=1)).timestamp()
            di, do = _sum_between(conn, t0, t1)
            hourly_labels.append(h.strftime("%H:%M"))
            hourly_in.append(di)
            hourly_out.append(do)
            h += timedelta(hours=1)

        day_end = now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        daily_labels: list[str] = []
        daily_in: list[int] = []
        daily_out: list[int] = []
        for i in range(13, -1, -1):
            d0 = day_end - timedelta(days=i)
            d1 = d0 + timedelta(days=1)
            di, do = _sum_between(conn, d0.timestamp(), d1.timestamp())
            daily_labels.append(d0.strftime("%b %d"))
            daily_in.append(di)
            daily_out.append(do)

        monday_date = now.date() - timedelta(days=now.weekday())
        monday_start = datetime(
            monday_date.year, monday_date.month, monday_date.day, tzinfo=timezone.utc
        )
        weekly_labels: list[str] = []
        weekly_in: list[int] = []
        weekly_out: list[int] = []
        for i in range(7, -1, -1):
            week_start = monday_start - timedelta(weeks=i)
            week_end = week_start + timedelta(days=7)
            weekly_labels.append(_week_display_label(week_start))
            di, do = _sum_between(conn, week_start.timestamp(), week_end.timestamp())
            weekly_in.append(di)
            weekly_out.append(do)

        popular_cutoff = (now - timedelta(days=60)).timestamp()
        cur = conn.execute(
            "SELECT kind, ts FROM events WHERE ts >= ?", (popular_cutoff,)
        )
        hour_in = [0] * 24
        hour_out = [0] * 24
        for row in cur:
            di, do = EVENT_WEIGHTS.get(row["kind"], (0, 0))
            if di == 0 and do == 0:
                continue
            hr = int(datetime.fromtimestamp(row["ts"], tz=timezone.utc).strftime("%H"))
            hour_in[hr] += di
            hour_out[hr] += do

        today_start = now.replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
        labels_24h = [f"{h:02d}:00" for h in range(24)]
        today_hourly_net, today_cumulative = _hourly_net_and_cumulative_today(
            conn, today_start, now
        )
        wd = now.weekday()
        baseline_hour_avg, baseline_cum, baseline_n = _baseline_same_weekday(
            conn, now, wd
        )

        conn.close()

    popular_total = [hour_in[h] + hour_out[h] for h in range(24)]
    popular_labels = [
        "12a", "1a", "2a", "3a", "4a", "5a", "6a", "7a", "8a", "9a", "10a", "11a",
        "12p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "10p", "11p",
    ]

    return {
        "updated_at": now.isoformat(),
        "totals": {"in": total_in, "out": total_out},
        "hourly": {"labels": hourly_labels, "in": hourly_in, "out": hourly_out},
        "daily": {"labels": daily_labels, "in": daily_in, "out": daily_out},
        "weekly": {"labels": weekly_labels, "in": weekly_in, "out": weekly_out},
        "popular_hours": {
            "labels": popular_labels,
            "in": hour_in,
            "out": hour_out,
            "total": popular_total,
        },
        "occupancy": {
            "timezone_note": "UTC",
            "weekday_label": WEEKDAY_NAMES_EN[wd],
            "labels_24h": labels_24h,
            "today_hourly_net": today_hourly_net,
            "today_cumulative": today_cumulative,
            "baseline_sample_days": baseline_n,
            "baseline_hourly_net_avg": baseline_hour_avg,
            "baseline_cumulative": baseline_cum,
        },
    }


init_db()
app = FastAPI(title="TREVOR People Count API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ingress")
def post_ingress():
    _record("ingress")
    return {"ok": True}


@app.post("/ingress_undo")
def post_ingress_undo():
    _record("ingress_undo")
    return {"ok": True}


@app.post("/egress")
def post_egress():
    _record("egress")
    return {"ok": True}


@app.post("/egress_undo")
def post_egress_undo():
    _record("egress_undo")
    return {"ok": True}


@app.get("/api/stats")
def get_stats():
    return build_stats()


_assets_dir = DIST_DIR / "assets"
if _assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")


@app.get("/")
def serve_index():
    index = DIST_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {
        "message": "Dashboard API running. For the UI: npm run dev (Vite) or npm run build then reload.",
        "stats": "/api/stats",
    }


@app.get("/{path:path}")
def spa_fallback(path: str):
    if path.startswith("api") or path.startswith("."):
        raise HTTPException(404)
    candidate = (DIST_DIR / path).resolve()
    try:
        candidate.relative_to(DIST_DIR.resolve())
    except ValueError:
        raise HTTPException(404) from None
    if candidate.is_file():
        return FileResponse(candidate)
    index = DIST_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(404)
