import cv2
import time
import numpy as np
from math import sqrt
from eye_cropper import EyeCropper
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# BLINK CONFIG
# =========================

COUNTER = 0
TOTAL_BLINKS = 0
EYE_CLOSED = False
BLINK_THRESHOLD = 3.2
MIN_FRAMES = 3
RATIO_HISTORY = []

START_TIME = time.time()
BLINK_RATE = 0.0

FONT = cv2.FONT_HERSHEY_SIMPLEX

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249,
            263, 466, 388, 387, 386, 385, 384, 398]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
             133, 173, 157, 158, 159, 160, 161, 246]

# =========================
# BLINK FUNCTIONS
# =========================

def euclideanDistance(p1, p2):
    return sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def blinkRatio(landmarks, right_indices, left_indices):
    r_h = euclideanDistance(landmarks[right_indices[0]], landmarks[right_indices[8]])
    r_v = euclideanDistance(landmarks[right_indices[12]], landmarks[right_indices[4]])

    l_v = euclideanDistance(landmarks[left_indices[12]], landmarks[left_indices[4]])
    l_h = euclideanDistance(landmarks[left_indices[0]], landmarks[left_indices[8]])

    if r_v == 0 or l_v == 0:
        return 0

    return ((r_h / r_v) + (l_h / l_v)) / 2

# =========================
# SETUP
# =========================

eye = EyeCropper()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

baseline = []
baseline_time = 30  # seconds
baseline_ready = False
baseline_mean = 0.0
start = time.time()

print("[INFO] Starting webcam")

# =========================
# MAIN LOOP
# =========================

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape

    # ---------- PUPIL ----------
    pupil_score = eye.process(frame)
    if pupil_score is None:
        continue

    now = time.time()

    # Baseline collection
    if not baseline_ready:
        baseline.append(pupil_score)

        if now - START_TIME >= baseline_time:
            baseline_mean = sum(baseline) / len(baseline)
            baseline_ready = True
            print("[INFO] Baseline collected")

        cv2.putText(frame, "Collecting baseline...", (30, 210), FONT, 0.7, (0, 0, 255), 2)
    
    if baseline_ready:
        norm_pupil = pupil_score / baseline_mean
    else:
        norm_pupil = None
    
    # ---------- BLINK ----------
    lm = eye.last_landmarks
    if lm is not None:
        landmarks = [(int(p.x * w), int(p.y * h)) for p in lm]

        eyes_ratio = blinkRatio(landmarks, RIGHT_EYE, LEFT_EYE)

        RATIO_HISTORY.append(eyes_ratio)
        if len(RATIO_HISTORY) > 5:
            RATIO_HISTORY.pop(0)
        eyes_ratio = sum(RATIO_HISTORY) / len(RATIO_HISTORY)

        if eyes_ratio > BLINK_THRESHOLD:
            COUNTER += 1
        else:
            if COUNTER >= MIN_FRAMES and EYE_CLOSED:
                TOTAL_BLINKS += 1
                elapsed = time.time() - START_TIME
                if elapsed > 0:
                    BLINK_RATE = (TOTAL_BLINKS / elapsed) * 60
            COUNTER = 0
            EYE_CLOSED = False

        if COUNTER >= MIN_FRAMES:
            EYE_CLOSED = True

        cv2.putText(frame, f'Ratio: {eyes_ratio:.2f}', (30, 50), FONT, 0.8, (255, 0, 0), 2)

    # ---------- OUTPUT ----------
    if baseline_ready:
        EyeFrameData = {
            "timestamp": now,
            "total_blinks": TOTAL_BLINKS,
            "blink_rate": BLINK_RATE,
            "norm_pupil": norm_pupil
        }
        print(EyeFrameData)

    cv2.putText(frame, f'Blinks: {TOTAL_BLINKS}', (30, 140), FONT, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f'Blink Rate: {BLINK_RATE:.2f} bpm', (30, 170), FONT, 0.8, (0, 255, 255), 2)

    cv2.imshow("Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
