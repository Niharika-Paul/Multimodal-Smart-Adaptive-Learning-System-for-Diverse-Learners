import cv2
import time
from eye_cropper import EyeCropper

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

eye = EyeCropper()

baseline = []
baseline_time = 30  # seconds
start = time.time()

print("[INFO] Starting webcam (Option 1: landmark pupil score)...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pupil_score = eye.process(frame)
    if pupil_score is None:
        continue

    now = time.time()

    # Collect baseline
    if now - start < baseline_time:
        baseline.append(pupil_score)
        continue

    baseline_mean = sum(baseline) / len(baseline)
    norm_pupil = pupil_score / baseline_mean

    EyeFrameData = {
        "timestamp": now,
        "pupil": norm_pupil
    }

    print(EyeFrameData)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
