import base64
import time
from collections import deque
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from eye_cropper import EyeCropper

# This API must be run from the Pupil-dilation directory.
app = FastAPI(title="Bright Minds Eye Tracking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BLINK_THRESHOLD = 3.2
MIN_FRAMES = 3
BASELINE_SECONDS = 30
BLINK_WINDOW_SECONDS = 60

RIGHT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]
LEFT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]


def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def blink_ratio(landmarks):
    r_h = distance(landmarks[RIGHT_EYE[0]], landmarks[RIGHT_EYE[8]])
    r_v = distance(landmarks[RIGHT_EYE[12]], landmarks[RIGHT_EYE[4]])
    l_v = distance(landmarks[LEFT_EYE[12]], landmarks[LEFT_EYE[4]])
    l_h = distance(landmarks[LEFT_EYE[0]], landmarks[LEFT_EYE[8]])

    if r_v == 0 or l_v == 0:
        return None

    return ((r_h / r_v) + (l_h / l_v)) / 2


class FrameRequest(BaseModel):
    image: str


class Tracker:
    def __init__(self):
        model_path = Path(__file__).resolve().parent / "face_landmarker.task"
        if not model_path.exists():
            raise RuntimeError(f"MediaPipe model not found: {model_path}")

        self.eye = EyeCropper()
        self.started = time.monotonic()
        self.baseline_started = None
        self.baseline_values = []
        self.baseline_mean = 0.0
        self.baseline_ready = False

        self.total_blinks = 0
        self.blink_timestamps = deque()
        self.observation_started = None
        self.observed_seconds = 0.0
        self.last_observation_time = None
        self.closed_frames = 0
        self.eye_closed = False
        self.ratio_history = deque(maxlen=5)
        self.lock = Lock()

    def process(self, frame):
        now = time.monotonic()
        iris_ratio = self.eye.process(frame)

        if iris_ratio is None or self.eye.last_landmarks is None:
            return {
                "face_detected": False,
                "status": "No face detected",
                "total_blinks": self.total_blinks,
                "blink_rate": None,
                "blink_rate_ready": False,
                "norm_pupil": None,
                "iris_eye_ratio": None,
                "baseline_ready": self.baseline_ready,
                "engagement_score": None,
            }

        landmarks_obj = self.eye.last_landmarks
        h, w = frame.shape[:2]
        landmarks = [(int(p.x * w), int(p.y * h)) for p in landmarks_obj]

        ratio = blink_ratio(landmarks)
        if ratio is not None:
            self.ratio_history.append(ratio)
            smooth_ratio = sum(self.ratio_history) / len(self.ratio_history)

            if smooth_ratio > BLINK_THRESHOLD:
                self.closed_frames += 1
            else:
                if self.closed_frames >= MIN_FRAMES and self.eye_closed:
                    self.total_blinks += 1
                    self.blink_timestamps.append(now)

                self.closed_frames = 0
                self.eye_closed = False

            if self.closed_frames >= MIN_FRAMES:
                self.eye_closed = True

        # Track valid observation time while a face is visible.
        if self.observation_started is None:
            self.observation_started = now

        if self.last_observation_time is None:
            self.observation_started = now
            self.observed_seconds = 0.0
        else:
            gap = now - self.last_observation_time

            if gap <= 1.0:
                self.observed_seconds += gap
            else:
                # Start a fresh measurement period after a tracking gap.
                self.observation_started = now
                self.observed_seconds = 0.0
                self.blink_timestamps.clear()
                self.closed_frames = 0
                self.eye_closed = False
                self.ratio_history.clear()

        self.last_observation_time = now

        # Keep only blink events from the rolling 60-second window.
        while (
            self.blink_timestamps
            and now - self.blink_timestamps[0] > BLINK_WINDOW_SECONDS
        ):
            self.blink_timestamps.popleft()

        blink_rate_ready = self.observed_seconds >= BLINK_WINDOW_SECONDS

        blink_rate = (
            len(self.blink_timestamps) * 60 / BLINK_WINDOW_SECONDS
            if blink_rate_ready
            else None
        )
        
        if self.baseline_started is None:
            self.baseline_started = now

        if not self.baseline_ready:
            self.baseline_values.append(float(iris_ratio))
            if now - self.baseline_started >= BASELINE_SECONDS:
                if self.baseline_values:
                    self.baseline_mean = float(np.mean(self.baseline_values))
                    self.baseline_ready = self.baseline_mean > 0

        norm_pupil = None
        if self.baseline_ready and self.baseline_mean > 0:
            norm_pupil = round(float(iris_ratio) / self.baseline_mean, 4)

        #blink_rate = self.total_blinks / max(now - self.started, 1e-6) * 60

        return {
            "face_detected": True,
            "status": "Tracking" if self.baseline_ready else "Calibrating baseline",
            "total_blinks": self.total_blinks,
            "blink_rate": round(blink_rate, 2) if blink_rate is not None else None,
            "blink_rate_ready": blink_rate_ready,
            "iris_eye_ratio": round(float(iris_ratio), 4),
            "norm_pupil": norm_pupil,
            "baseline_ready": self.baseline_ready,
            # No engagement model has been connected yet.
            "engagement_score": None,
        }


tracker = Tracker()


@app.get("/health")
def health():
    return {"status": "ok", "service": "eye-tracking-api"}


@app.post("/analyze-frame")
def analyze_frame(request: FrameRequest):
    encoded = request.image.split(",", 1)[-1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    with tracker.lock:
        return tracker.process(frame)
