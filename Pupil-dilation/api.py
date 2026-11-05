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

from functools import lru_cache
import os

from eye_cropper import EyeCropper
from live_i3d import LiveI3D

# This API must be run from the Pupil-dilation directory.
app = FastAPI(title="Bright Minds Eye Tracking API")

DEFAULT_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:8081",
    "http://127.0.0.1:8081",
    "http://localhost:8082",
    "http://127.0.0.1:8082",
}
ALLOWED_ORIGINS = sorted(
    DEFAULT_ORIGINS
    | {origin.strip() for origin in os.getenv("FRONTEND_ORIGINS", "").split(",") if origin.strip()}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

BLINK_CLOSED_EAR_THRESHOLD = 0.18
MIN_CLOSED_FRAMES = 2
MAX_CONTIGUOUS_GAP_SECONDS = 5.0
BASELINE_SECONDS = 30
BLINK_WINDOW_SECONDS = 60
GAZE_WINDOW_SECONDS = 10

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


def eye_aspect_ratio(landmarks):
    """Return the mean eye aspect ratio; it decreases as the eyelids close."""
    right_width = distance(landmarks[33], landmarks[133])
    left_width = distance(landmarks[362], landmarks[263])
    if right_width == 0 or left_width == 0:
        return None

    right_ear = (
        distance(landmarks[159], landmarks[145])
        + distance(landmarks[158], landmarks[153])
    ) / (2 * right_width)
    left_ear = (
        distance(landmarks[386], landmarks[374])
        + distance(landmarks[385], landmarks[380])
    ) / (2 * left_width)
    return (right_ear + left_ear) / 2


def estimate_screen_gaze(landmarks):
    """Estimate whether both irises are centered within their eye openings."""
    def eye_position(iris_index, corner_a, corner_b, upper, lower):
        corners = (landmarks[corner_a], landmarks[corner_b])
        x_min, x_max = sorted((corners[0][0], corners[1][0]))
        y_min, y_max = sorted((landmarks[upper][1], landmarks[lower][1]))
        if x_max <= x_min or y_max <= y_min:
            return None
        iris = landmarks[iris_index]
        return (
            (iris[0] - x_min) / (x_max - x_min),
            (iris[1] - y_min) / (y_max - y_min),
        )

    right = eye_position(468, 33, 133, 159, 145)
    left = eye_position(473, 362, 263, 386, 374)
    if right is None or left is None:
        return None

    gaze_x = (right[0] + left[0]) / 2
    gaze_y = (right[1] + left[1]) / 2
    toward_screen = (
        0.30 <= right[0] <= 0.70
        and 0.30 <= left[0] <= 0.70
        and 0.15 <= right[1] <= 0.85
        and 0.15 <= left[1] <= 0.85
    )
    return gaze_x, gaze_y, toward_screen


class FrameRequest(BaseModel):
    image: str
    session_id: str | None = None


class SessionRequest(BaseModel):
    session_id: str


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
        self.gaze_samples = deque()
        self.lock = Lock()

    def reset_session(self):
        """Clear all per-session tracking state without rebuilding the tracker."""
        self.started = time.monotonic()
        self.baseline_started = None
        self.baseline_values.clear()
        self.baseline_mean = 0.0
        self.baseline_ready = False
        self.total_blinks = 0
        self.blink_timestamps.clear()
        self.observation_started = None
        self.observed_seconds = 0.0
        self.last_observation_time = None
        self.closed_frames = 0
        self.eye_closed = False
        self.gaze_samples.clear()

    def process(self, frame):
        now = time.monotonic()
        iris_ratio = self.eye.process(frame)

        if iris_ratio is None or self.eye.last_landmarks is None:
            # Require a fresh continuous minute after face tracking is lost.
            self.observation_started = None
            self.observed_seconds = 0.0
            self.last_observation_time = None
            self.blink_timestamps.clear()
            self.closed_frames = 0
            self.eye_closed = False
            self.gaze_samples.clear()
            return {
                "face_detected": False,
                "status": "No face detected",
                "total_blinks": self.total_blinks,
                "blink_rate": None,
                "blink_rate_ready": False,
                "blink_window_count": 0,
                "blink_rate_elapsed_seconds": 0.0,
                "norm_pupil": None,
                "iris_eye_ratio": None,
                "baseline_ready": self.baseline_ready,
                "eye_contact_percent": None,
                "gaze_status": "No face detected",
                "engagement_score": None,
            }

        landmarks_obj = self.eye.last_landmarks
        h, w = frame.shape[:2]
        landmarks = [(int(p.x * w), int(p.y * h)) for p in landmarks_obj]

        ear = eye_aspect_ratio(landmarks)
        if ear is not None:
            if ear < BLINK_CLOSED_EAR_THRESHOLD:
                self.closed_frames += 1
                if self.closed_frames >= MIN_CLOSED_FRAMES:
                    self.eye_closed = True
            else:
                if self.eye_closed:
                    self.total_blinks += 1
                    self.blink_timestamps.append(now)
                self.closed_frames = 0
                self.eye_closed = False

        gaze = estimate_screen_gaze(landmarks)
        if gaze is not None:
            gaze_x, gaze_y, toward_screen = gaze
            self.gaze_samples.append((now, toward_screen))

        # Track valid observation time while a face is visible.
        if self.observation_started is None:
            self.observation_started = now

        if self.last_observation_time is None:
            self.observation_started = now
            self.observed_seconds = 0.0
        else:
            gap = now - self.last_observation_time

            if gap <= MAX_CONTIGUOUS_GAP_SECONDS:
                self.observed_seconds += gap
            else:
                # Ignore long processing/background gaps without resetting for
                # ordinary I3D inference pauses.
                self.observation_started = now
                self.observed_seconds = 0.0
                self.blink_timestamps.clear()
                self.closed_frames = 0
                self.eye_closed = False
                self.gaze_samples.clear()

        self.last_observation_time = now

        # Keep only blink events from the rolling 60-second window.
        while (
            self.blink_timestamps
            and now - self.blink_timestamps[0] > BLINK_WINDOW_SECONDS
        ):
            self.blink_timestamps.popleft()

        while self.gaze_samples and now - self.gaze_samples[0][0] > GAZE_WINDOW_SECONDS:
            self.gaze_samples.popleft()

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
            "blink_window_count": len(self.blink_timestamps),
            "blink_rate_elapsed_seconds": round(self.observed_seconds, 1),
            "eye_aspect_ratio": round(float(ear), 4) if ear is not None else None,
            "iris_eye_ratio": round(float(iris_ratio), 4),
            "norm_pupil": norm_pupil,
            "baseline_ready": self.baseline_ready,
            "eye_contact_percent": (
                round(100 * sum(int(sample[1]) for sample in self.gaze_samples) / len(self.gaze_samples), 1)
                if len(self.gaze_samples) >= 5
                else None
            ),
            "gaze_status": (
                "Looking toward screen" if gaze and gaze[2] else
                "Looking away" if gaze else "Gaze unavailable"
            ),
            "gaze_x": round(gaze_x, 3) if gaze else None,
            "gaze_y": round(gaze_y, 3) if gaze else None,
            "engagement_score": None,
        }


latest_engagement = None
active_session_id = None

tracker = Tracker()
live_i3d = LiveI3D()
W0P_CHECKPOINT = "fold_1__video14_person0__random__K20.pt"


def _reset_session_state(session_id):
    global active_session_id, latest_engagement
    tracker.reset_session()
    live_i3d.reset()
    latest_engagement = None
    active_session_id = session_id



@app.get("/health")
def health():
    return {"status": "ok", "service": "eye-tracking-api"}


@app.post("/start-session")
def start_session(request: SessionRequest):
    session_id = request.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id must not be empty")
    with tracker.lock:
        reset = session_id != active_session_id
        if reset:
            _reset_session_state(session_id)
    return {"status": "ok", "session_id": session_id, "reset": reset}


@app.post("/end-session")
def end_session(request: SessionRequest):
    global active_session_id
    with tracker.lock:
        ended = request.session_id == active_session_id
        if ended:
            _reset_session_state(None)
    return {"status": "ok", "ended": ended}


@app.post("/analyze-frame")
def analyze_frame(request: FrameRequest):
    global latest_engagement
    encoded = request.image.split(",", 1)[-1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    session_id = request.session_id or "default"
    with tracker.lock:
        # Older clients can still use /analyze-frame without /start-session;
        # initialize once on first frame, then reject stale session traffic.
        if active_session_id is None:
            _reset_session_state(session_id)
        elif session_id != active_session_id:
            raise HTTPException(
                status_code=409,
                detail="Tracking session is not active; call /start-session first",
            )
        result = tracker.process(frame)
        if not result["face_detected"]:
            live_i3d.reset()
            latest_engagement = None
            result.update({
                "engagement_status": "Waiting for face",
                "engagement_frames_collected": 0,
                "engagement_frames_required": live_i3d.frames_required,
                "engagement_label": None,
                "engagement_confidence": None,
            })
            return result

        try:
            embedding = live_i3d.process_frame(frame)
            if embedding is None:
                result.update({
                    "engagement_status": (
                        "Prediction available" if latest_engagement else
                        "Extracting I3D features" if live_i3d.inference_pending else
                        "Collecting clip frames"
                    ),
                    "engagement_frames_collected": live_i3d.frames_collected,
                    "engagement_frames_required": live_i3d.frames_required,
                    "engagement_label": latest_engagement.get("label") if latest_engagement else None,
                    "engagement_confidence": latest_engagement.get("confidence") if latest_engagement else None,
                    "engagement_class_index": latest_engagement.get("class_index") if latest_engagement else None,
                    "engagement_probabilities": latest_engagement.get("probabilities") if latest_engagement else None,
                    "engagement_model": latest_engagement.get("model") if latest_engagement else None,
                    "engagement_checkpoint": W0P_CHECKPOINT if latest_engagement else None,
                    "engagement_personalization_group": "video14_person0" if latest_engagement else None,
                })
            else:
                model = get_engagement_model(W0P_CHECKPOINT)
                prediction = model.predict(embedding)
                latest_engagement = prediction
                result.update({
                    "engagement_status": "Prediction available",
                    "engagement_frames_collected": live_i3d.frames_required,
                    "engagement_frames_required": live_i3d.frames_required,
                    "engagement_label": prediction.get("label"),
                    "engagement_confidence": prediction.get("confidence"),
                    "engagement_class_index": prediction.get("class_index"),
                    "engagement_probabilities": prediction.get("probabilities"),
                    "engagement_model": prediction.get("model"),
                    "engagement_checkpoint": W0P_CHECKPOINT,
                    "engagement_personalization_group": "video14_person0",
                })
        except Exception as exc:
            latest_engagement = None
            result.update({
                "engagement_status": "Inference error",
                "engagement_label": None,
                "engagement_confidence": None,
                "engagement_error": str(exc),
                "engagement_frames_collected": live_i3d.frames_collected,
                "engagement_frames_required": live_i3d.frames_required,
            })

    return result


# Separate endpoint for trained W0-P inference.
# Input must be a precomputed 1024-dimensional I3D embedding.
from model_inference import load_w0p


class EmbeddingRequest(BaseModel):
    embedding: list[float]


@lru_cache(maxsize=4)
def get_engagement_model(filename: str):
    try:
        return load_w0p(filename)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict-engagement")
def predict_engagement(request: EmbeddingRequest):
    if len(request.embedding) != 1024:
        raise HTTPException(
            status_code=422,
            detail="Expected a 1024-dimensional I3D embedding",
        )

    try:
        model = get_engagement_model(
            "fold_1__video14_person0__random__K20.pt"
        )
        return model.predict(request.embedding)
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))
