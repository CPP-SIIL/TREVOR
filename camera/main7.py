import warnings
import threading
import time

import cv2
import numpy as np
import torch

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

from counter import Counter
from nets import nn
from utils import util

warnings.filterwarnings("ignore")

app = FastAPI()

latest_frame = None
latest_counts = {"ingress": 0, "egress": 0}
lock = threading.Lock()


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
    cv2.putText(image, text, (x1, y1 - 2), 0, 0.5, (0, 255, 0), 1)


def camera_loop():
    global latest_frame, latest_counts

    size = 640

    checkpoint = torch.load("./weights/v8_n.pt", map_location="cpu", weights_only=False)
    model = checkpoint["model"].float()
    model.eval()

    reader = cv2.VideoCapture(0)

    if not reader.isOpened():
        print("Error opening camera")
        return

    fps = int(reader.get(cv2.CAP_PROP_FPS)) or 30
    bytetrack = nn.BYTETracker(fps)

    success, frame = reader.read()
    if not success:
        print("Could not read initial frame")
        return

    height, width = frame.shape[:2]

    A = (width // 2, 0)
    B = (width // 2, height - 1)

    counter = Counter(A, B)

    while True:
        success, frame = reader.read()
        if not success:
            time.sleep(0.05)
            continue

        boxes, confidences, object_classes = [], [], []

        image = frame.copy()
        shape = image.shape[:2]

        r = size / max(shape[0], shape[1])
        if r != 1:
            h, w = shape
            image = cv2.resize(image, (int(w * r), int(h * r)))

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
            if output is None:
                continue

            detections = output.clone()
            util.scale(detections[:, :4], sample[i].shape[1:], shapes[0], shapes[1])
            detections = detections.cpu().numpy()

            for d in detections:
                x1, y1, x2, y2 = map(int, d[:4])
                boxes.append([x1, y1, x2, y2])
                confidences.append(d[4])
                object_classes.append(d[5])

        if len(boxes) > 0:
            outputs = bytetrack.update(np.array(boxes), np.array(confidences), np.array(object_classes))
        else:
            outputs = np.empty((0, 7))

        counter.update(outputs)

        # 🔥 EXACT ORIGINAL VISUALS 🔥
        if len(outputs) > 0:
            boxes = outputs[:, :4]
            identities = outputs[:, 4]
            object_classes = outputs[:, 6]

            for i, box in enumerate(boxes):
                if object_classes[i] != 0:
                    continue

                x1, y1, x2, y2 = map(int, box)
                index = int(identities[i])
                draw_line(frame, x1, y1, x2, y2, index)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        x_left = 10
        y0 = 30
        dy = 35

        cv2.putText(frame, f"Ingress: {counter.in_count}", (x_left, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, "---------->", (x_left, y0 + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        (tw, _), _ = cv2.getTextSize(f"Egress: {counter.out_count}",
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        cv2.putText(frame, f"Egress: {counter.out_count}",
                    (width - tw - 10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, "<----------",
                    (width - 200, y0 + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.line(frame, A, B, (0, 0, 255), 2)

        # 🔁 Replace imshow with stream
        ok, jpeg = cv2.imencode(".jpg", frame)
        if ok:
            with lock:
                latest_frame = jpeg.tobytes()
                latest_counts["ingress"] = counter.in_count
                latest_counts["egress"] = counter.out_count


def gen_frames():
    while True:
        with lock:
            frame = latest_frame
        if frame:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'


@app.on_event("startup")
def start():
    threading.Thread(target=camera_loop, daemon=True).start()


@app.get("/video_feed")
def video():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/counts")
def counts():
    return JSONResponse(latest_counts)