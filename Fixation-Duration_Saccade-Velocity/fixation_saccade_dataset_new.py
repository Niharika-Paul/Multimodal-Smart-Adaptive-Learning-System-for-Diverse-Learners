import cv2
import mediapipe as mp
import time
import math
from collections import deque

# =========================
# ADDED: Global variables for primary user locking
# =========================
LOCKED_FACE_CENTER = None
FACE_LOCKED = False
FACE_LOST_FRAMES = 0
MAX_FACE_LOST = 30  # ~1 second at 30 FPS
LOCKED_LANDMARKS = None
PRIMARY_USER_GONE = False

FONT = cv2.FONT_HERSHEY_SIMPLEX

WINDOW_SIZE = 1.0  # seconds

window_start_time = time.time()

window_fixation_time = 0.0   # seconds
window_saccade_vels = []     # px/s


# MediaPipe setup (same family as blink module)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.6,
    
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
cv2.moveWindow("Webcam", 100, 100)  # Position at specific coordinates


#STEP 2 – Choose gaze point (what are we tracking?)- use midpoint between eyes (very common, conservative).
# Eye corner landmarks (stable points)
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

def get_gaze_point(landmarks, frame_width, frame_height):
    left = landmarks[LEFT_EYE_OUTER]
    right = landmarks[RIGHT_EYE_OUTER]

    x = int((left.x + right.x) / 2 * frame_width)
    y = int((left.y + right.y) / 2 * frame_height)

    return x, y

# STEP 3 – Store gaze samples with timestamps
gaze_buffer = deque(maxlen=5)

# STEP 4 – SMOOTHING (I2MC / kollaR)
def smooth_gaze(buffer):
    xs = [p[0] for p in buffer]
    ys = [p[1] for p in buffer]
    ts = [p[2] for p in buffer]

    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
        ts[-1]
    )

# STEP 5 – Velocity computation (Salvucci & Goldberg, 2000)
def compute_velocity(p1, p2):
    x1, y1, t1 = p1
    x2, y2, t2 = p2

    dt = t2 - t1
    if dt == 0:
        return 0

    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist / dt  # pixels per second


# STEP 6 – Velocity threshold selection (webcam-safe)
VELOCITY_THRESHOLD = 17     # px/s
#MIN_FIXATION_DURATION = 0.15 # seconds

# STEP 7 – Fixation state machine
in_fixation = False
fixation_start_time = None
#fixation_durations = []

# =========================
# ADDED: Face mesh drawing function
# =========================
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
# ADDED: Face center function
# =========================
def face_center(lm, w, h):
    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

# =========================
# ADDED: Face visibility function
# =========================
def face_visibility(lm):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    visible = sum(0 <= x <= 1 and 0 <= y <= 1 for x, y in zip(xs, ys))
    return visible / len(lm)

# =========================
# ADDED: Euclidean distance function
# =========================
def euclideanDistance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

prev_smoothed = None  # IMPORTANT

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Webcam frame not received")
        break

    now = time.time()
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # =========================
    # ADDED: Primary user locking logic
    # =========================
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        
        # Check if primary user has returned
        if PRIMARY_USER_GONE:
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
        
        if PRIMARY_USER_GONE:  # Still gone after checks
            cv2.imshow("Webcam", frame)
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
                        cv2.imshow("Webcam", frame)
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
                            cv2.imshow("Webcam", frame)
                            if cv2.waitKey(1) & 0xFF == 27:
                                break
                            continue
                    else:
                        # Same face, good visibility - update tracking
                        FACE_LOST_FRAMES = 0
                        LOCKED_LANDMARKS = lm
                        LOCKED_FACE_CENTER = center     # Update center position for smooth tracking

            # ADDED: Draw face mesh
            draw_face_mesh(frame, lm, w, h)

        # =========================
        # Original fixation/saccade code continues here
        # =========================
        #print("✅ Face detected")
        landmarks = results.multi_face_landmarks[0].landmark

        x, y = get_gaze_point(landmarks, w, h)
        gaze_buffer.append((x, y, now))

        # STEP 4 – smoothing (I2MC / kollaR)
        if len(gaze_buffer) >= 2:
            curr_smoothed = smooth_gaze(gaze_buffer)

            # STEP 5 – velocity (Salvucci & Goldberg, 2000)
            if prev_smoothed is not None:
                velocity = compute_velocity(prev_smoothed, curr_smoothed)
                current_time = curr_smoothed[2]

                 # ===== STEP A: VELOCITY LOGGING (TEMPORARY) =====
                #print(f"[VEL] {velocity:.2f} px/s")
                # ==============================================

                # ---- I-VT classification (Salvucci & Goldberg, 2000) ----
                if velocity < VELOCITY_THRESHOLD:
                    if not in_fixation:
                        in_fixation = True
                        fixation_start_time = current_time
                else:
                    # saccade
                    window_saccade_vels.append(velocity)

                    if in_fixation:
                        window_fixation_time += current_time - fixation_start_time
                        in_fixation = False
                # --------------------------------------------------------

                # ======== STEP C: WINDOW OUTPUT ========
                if now - window_start_time >= WINDOW_SIZE:
                    print("⏱ Window fired")

                    # close ongoing fixation at window boundary
                    if in_fixation:
                        window_fixation_time += now - fixation_start_time
                        fixation_start_time = now

                    fixation_ms = window_fixation_time * 1000
                    saccade_v = max(window_saccade_vels) if window_saccade_vels else 0.0

                    print({
                        "timestamp": now,
                        "Fixation_Duration": fixation_ms,
                        "Saccade_V": saccade_v
                    })

                    window_start_time = now
                    window_fixation_time = 0.0
                    window_saccade_vels.clear()
                # ======================================

            prev_smoothed = curr_smoothed
    """
    else:
        # ADDED: Handle no face detected
        if FACE_LOCKED and not PRIMARY_USER_GONE:
            FACE_LOST_FRAMES += 1
            if FACE_LOST_FRAMES > MAX_FACE_LOST:
                PRIMARY_USER_GONE = True
                FACE_LOCKED = False
                print("[INFO] Primary user gone - no face detected")
            cv2.putText(frame, "Primary user not present - no face detected",
                        (30, 240), FONT, 0.8, (0, 0, 255), 2)
    """

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()