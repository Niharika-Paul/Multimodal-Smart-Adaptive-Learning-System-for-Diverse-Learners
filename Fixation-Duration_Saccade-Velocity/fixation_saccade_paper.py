
import cv2
import mediapipe as mp
import time
import math
from collections import deque

# MediaPipe setup (same family as blink module)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)


#STEP 2 — Choose gaze point (what are we tracking?)- use midpoint between eyes (very common, conservative).
# Eye corner landmarks (stable points)
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

def get_gaze_point(landmarks, frame_width, frame_height):
    left = landmarks[LEFT_EYE_OUTER]
    right = landmarks[RIGHT_EYE_OUTER]

    x = int((left.x + right.x) / 2 * frame_width)
    y = int((left.y + right.y) / 2 * frame_height)

    return x, y

# STEP 3 — Store gaze samples with timestamps
gaze_buffer = deque(maxlen=5)

# STEP 4 — SMOOTHING (I2MC / kollaR)
def smooth_gaze(buffer):
    xs = [p[0] for p in buffer]
    ys = [p[1] for p in buffer]
    ts = [p[2] for p in buffer]

    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
        ts[-1]
    )

# STEP 5 — Velocity computation (Salvucci & Goldberg, 2000)
def compute_velocity(p1, p2):
    x1, y1, t1 = p1
    x2, y2, t2 = p2

    dt = t2 - t1
    if dt == 0:
        return 0

    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist / dt  # pixels per second


# STEP 6 — Velocity threshold selection (webcam-safe)
VELOCITY_THRESHOLD = 80      # px/s
MIN_FIXATION_DURATION = 0.15 # seconds

# STEP 7 — Fixation state machine
in_fixation = False
fixation_start_time = None
fixation_durations = []

prev_smoothed = None  # IMPORTANT

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        x, y = get_gaze_point(landmarks, w, h)
        gaze_buffer.append((x, y, time.time()))

        # STEP 4 — smoothing (I2MC / kollaR)
        if len(gaze_buffer) >= 2:
            curr_smoothed = smooth_gaze(gaze_buffer)

            # STEP 5 — velocity (Salvucci & Goldberg, 2000)
            if prev_smoothed is not None:
                velocity = compute_velocity(prev_smoothed, curr_smoothed)

                # STEP 7 — fixation logic (I-VT)
                if velocity < VELOCITY_THRESHOLD:
                    if not in_fixation:
                        in_fixation = True
                        fixation_start_time = curr_smoothed[2]
                else:
                    if in_fixation:
                        duration = curr_smoothed[2] - fixation_start_time
                        if duration >= MIN_FIXATION_DURATION:
                            fixation_durations.append(duration * 1000)  # ms

                            # STEP 8 — output like dataset
                            print({
                                "timestamp": time.time(),
                                "Fixation_Duration": duration * 1000
                            })

                        in_fixation = False

            prev_smoothed = curr_smoothed

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
