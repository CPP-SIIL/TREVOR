from __future__ import annotations

import os
import sqlite3
import threading
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from counter import Counter
from nets import nn
from utils import util

warnings.filterwarnings("ignore")

WEEKDAY_NAMES_EN = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "counts.db"
DIST_DIR = ROOT / "dist"

_lock = threading.Lock()
frame_lock = threading.Lock()

latest_frame = None
latest_counts = {"ingress": 0, "egress": 0}

EVENT_WEIGHTS: dict[str, tuple[int, int]] = {
    "ingress": (1, 0),
    "ingress_undo": (-1, 0),
    "egress": (0, 1),
    "egress_undo": (0, -1),
}


def draw_line(image, x1, y1, x2, y2, index):
    w = 10
    h = 10
    color = (200, 0, 0)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)

    cv2.line(image, (x1, y1), (x1 + w, y1), color, 3)
    cv2.line(image, (x1, y1), (x1, y1 + h), color, 3)

    cv2.line(image, (x2, y1), (x2 - w, y1), color, 3)
    cv2.line(image, (x2, y1), (x2, y1 + h), color, 3)

    cv2.line(image, (x2, y2), (x2 - w, y2), color, 3)
    cv2.line(image, (x2, y2), (x2, y2 - h), color, 3)

    cv2.line(image, (x1, y2), (x1 + w, y2), color, 3)
    cv2.line(image, (x1, y2), (x1, y2 - h), color, 3)

    text = f"ID:{index}"
    cv2.putText(
        image,
        text,
        (x1, y1 - 2),
        0,
        0.5,
        (0, 255, 0),
        thickness=1,
        lineType=cv2.FILLED,
    )


def _week_display_label(week_start: datetime) -> str:
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
        hourly_avg.append(round(sum(s[h] for s in samples) / n, 2))
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
        cur = conn.execute("SELECT kind, ts FROM events WHERE ts >= ?", (popular_cutoff,))
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


def camera_loop():
    global latest_frame, latest_counts

    size = 640

    checkpoint = torch.load("./weights/v8_n.pt", map_location="cpu", weights_only=False)
    model = checkpoint["model"].float()
    model.eval()

    camera_source = os.getenv("CAMERA_SOURCE", "0")
    try:
        camera_source = int(camera_source)
    except ValueError:
        pass

    print(f"Using camera source: {camera_source}")

    if isinstance(camera_source, int):
        reader = cv2.VideoCapture(camera_source, cv2.CAP_AVFOUNDATION)
    else:
        reader = cv2.VideoCapture(camera_source)

    if not reader.isOpened():
        print(f"Error opening camera source: {camera_source}")
        return

    fps = int(reader.get(cv2.CAP_PROP_FPS)) or 30
    bytetrack = nn.BYTETracker(fps)

    success, frame = False, None
    for _ in range(20):
        success, frame = reader.read()
        if success and frame is not None:
            break
        time.sleep(0.1)

    if not success or frame is None:
        print("Could not read initial frame")
        reader.release()
        return

    height, width = frame.shape[:2]
    A = (width // 2, 0)
    B = (width // 2, height - 1)

    counter = Counter(A, B)

    while True:
        success, frame = reader.read()
        if not success or frame is None:
            time.sleep(0.05)
            continue

        boxes = []
        confidences = []
        object_classes = []

        image = frame.copy()
        shape = image.shape[:2]

        r = size / max(shape[0], shape[1])
        if r != 1:
            h, w = shape
            image = cv2.resize(
                image,
                dsize=(int(w * r), int(h * r)),
                interpolation=cv2.INTER_LINEAR,
            )

        h, w = image.shape[:2]
        image, ratio, pad = util.resize(image, size)
        shapes = shape, ((h / shape[0], w / shape[1]), pad)

        sample = image.transpose((2, 0, 1))[::-1]
        sample = np.ascontiguousarray(sample)
        sample = torch.from_numpy(sample).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            outputs = model(sample)

        outputs = util.non_max_suppression(outputs, 0.001, 0.7)

        for i, output in enumerate(outputs):
            if output is None or len(output) == 0:
                continue

            detections = output.clone()
            util.scale(
                detections[:, :4],
                sample[i].shape[1:],
                shapes[0],
                shapes[1],
            )
            detections = detections.cpu().numpy()

            for detection in detections:
                x1, y1, x2, y2 = list(map(int, detection[:4]))
                boxes.append([x1, y1, x2, y2])
                confidences.append(detection[4])
                object_classes.append(detection[5])

        if len(boxes) > 0:
            outputs = bytetrack.update(
                np.array(boxes),
                np.array(confidences),
                np.array(object_classes),
            )
        else:
            outputs = np.empty((0, 7))

        counter.update(outputs)

        if len(outputs) > 0:
            boxes = outputs[:, :4]
            identities = outputs[:, 4]
            object_classes = outputs[:, 6]

            for i, box in enumerate(boxes):
                if object_classes[i] != 0:
                    continue

                x1, y1, x2, y2 = list(map(int, box))
                index = int(identities[i]) if identities is not None else 0
                draw_line(frame, x1, y1, x2, y2, index)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        x_left = 10
        y0 = 30
        dy = 35

        cv2.putText(
            frame,
            f"Ingress: {counter.in_count}",
            (x_left, y0 + 0 * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            "---------->",
            (x_left, y0 + 1 * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        (text_width, _), _ = cv2.getTextSize(
            f"Egress: {counter.out_count}",
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            2,
        )

        x_right = width - text_width - 10

        cv2.putText(
            frame,
            f"Egress: {counter.out_count}",
            (x_right, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        (text_width_arrow, _), _ = cv2.getTextSize(
            "<----------",
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            2,
        )

        x_right_arrow = width - text_width_arrow - 10

        cv2.putText(
            frame,
            "<----------",
            (x_right_arrow, y0 + dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.line(frame, A, B, (0, 0, 255), 2)

        ok, jpeg = cv2.imencode(".jpg", frame)
        if ok:
            with frame_lock:
                latest_frame = jpeg.tobytes()
                latest_counts["ingress"] = int(counter.in_count)
                latest_counts["egress"] = int(counter.out_count)


def gen_frames():
    global latest_frame

    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


init_db()
app = FastAPI(title="TREVOR People Count API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    threading.Thread(target=camera_loop, daemon=True).start()


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


@app.get("/counts")
def get_counts():
    with frame_lock:
        return JSONResponse(content=latest_counts)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


_assets_dir = DIST_DIR / "assets"
if _assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")


@app.get("/")
def serve_index():
    index = DIST_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {
        "message": "Dashboard API running. For the UI: npm run dev or npm run build then reload.",
        "stats": "/api/stats",
        "counts": "/counts",
        "video": "/video_feed",
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