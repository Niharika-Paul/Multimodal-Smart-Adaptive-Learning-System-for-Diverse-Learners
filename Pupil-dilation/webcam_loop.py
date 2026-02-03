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

LOCKED_FACE_CENTER = None
FACE_LOCKED = False
FACE_LOST_FRAMES = 0
MAX_FACE_LOST = 30  # ~1 second at 30 FPS
LOCKED_LANDMARKS = None
PRIMARY_USER_GONE = False

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

# ===========================
# Face Mesh Function
# ===========================
def draw_face_mesh(frame, lm, w, h):
    # draw points
    for p in lm:
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # draw connections (optional but nice)
    FACEMESH_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION
    for a, b in FACEMESH_CONNECTIONS:
        pt1 = int(lm[a].x * w), int(lm[a].y * h)
        pt2 = int(lm[b].x * w), int(lm[b].y * h)
        cv2.line(frame, pt1, pt2, (0, 200, 0), 1)

# =========================
# FACE CENTER FUNCTION
# =========================
def face_center(lm, w, h):
    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

# =========================
# FACE VISIBILITY FUNCTION
# =========================
def face_visibility(lm):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    visible = sum(0 <= x <= 1 and 0 <= y <= 1 for x, y in zip(xs, ys))
    return visible / len(lm)

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

    # Check if primary user has returned
    if PRIMARY_USER_GONE:
        if lm is not None:
            center = face_center(lm, w, h)
            # Check if detected face is close to where primary user was last seen
            dist = euclideanDistance(center, LOCKED_FACE_CENTER)
            
            if dist <= 120:  # Same person returned
                visibility = face_visibility(lm)
                if visibility >= 0.6:  # Good visibility
                    PRIMARY_USER_GONE = False
                    FACE_LOCKED = True
                    FACE_LOST_FRAMES = 0
                    LOCKED_LANDMARKS = lm
                    LOCKED_FACE_CENTER = center
                    print("[INFO] Primary user returned")
                    cv2.putText(frame, "Primary user returned!", (30, 240), FONT, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Primary user not present - waiting...",
                                (30, 240), FONT, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Primary user not present - different person detected",
                            (30, 240), FONT, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Primary user not present - no face detected",
                        (30, 240), FONT, 0.8, (0, 0, 255), 2)
        
        if PRIMARY_USER_GONE:  # Still gone after checks
            cv2.imshow("Liveness Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
    
    if lm is not None:
        center = face_center(lm, w, h)

        if not FACE_LOCKED and not PRIMARY_USER_GONE:
            # lock onto the FIRST face
            LOCKED_FACE_CENTER = center
            LOCKED_LANDMARKS = lm
            FACE_LOCKED = True
            FACE_LOST_FRAMES = 0
            print("[INFO] Primary face locked")

        else:
            # ---------- DISTANCE CHECK (FIRST) ----------
            # Check if this is still the same face BEFORE checking visibility
            # This prevents switching to a new person when the primary user moves to edge of frame
            dist = euclideanDistance(center, LOCKED_FACE_CENTER)

            if dist > 120:   # pixel threshold - this is a DIFFERENT person
                FACE_LOST_FRAMES += 1
                lm = LOCKED_LANDMARKS   # ignore new detections, freeze on primary user
                center = face_center(lm, w, h)
                cv2.putText(frame, "Different person detected - ignoring", (30, 240), FONT, 0.8, (0, 0, 255), 2)
                if FACE_LOST_FRAMES > MAX_FACE_LOST:
                    PRIMARY_USER_GONE = True
                    FACE_LOCKED = False
                    cv2.putText(frame, "Primary user gone", (30, 240), FONT, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Liveness Detection", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

            else:
                # ---------- VISIBILITY CHECK (SECOND) ----------
                # Distance is OK, now check if primary user is visible enough
                visibility = face_visibility(lm)

                if visibility < 0.6:
                    FACE_LOST_FRAMES += 1
                    lm = LOCKED_LANDMARKS  # freeze on primary user
                    center = face_center(lm, w, h)
                    cv2.putText(frame, "Primary user leaving frame", (30, 240), FONT, 0.8, (0, 0, 255), 2)
                    if FACE_LOST_FRAMES > MAX_FACE_LOST:
                        PRIMARY_USER_GONE = True
                        FACE_LOCKED = False
                        cv2.putText(frame, "Primary user gone", (30, 240), FONT, 0.8, (0, 0, 255), 2)
                        cv2.imshow("Liveness Detection", frame)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                        continue
                else:
                    # Same face, good visibility - update tracking
                    FACE_LOST_FRAMES = 0
                    LOCKED_LANDMARKS = lm
                    LOCKED_FACE_CENTER = center     # Update center position for smooth tracking

        landmarks = [(int(p.x * w), int(p.y * h)) for p in lm]
        
        draw_face_mesh(frame, lm, w, h)

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
