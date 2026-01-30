import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

class EyeCropper:
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path="face_landmarker.task"
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def _euclidean(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _iris_diameter(self, landmarks, iris_idx, w, h):
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in iris_idx]
        center = np.mean(pts, axis=0)
        radii = [self._euclidean(center, p) for p in pts]
        return 2 * np.mean(radii)

    def process(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect(mp_image)
        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]

        left_eye_width = self._euclidean(
            (lm[LEFT_EYE[0]].x * w, lm[LEFT_EYE[0]].y * h),
            (lm[LEFT_EYE[1]].x * w, lm[LEFT_EYE[1]].y * h),
        )

        right_eye_width = self._euclidean(
            (lm[RIGHT_EYE[0]].x * w, lm[RIGHT_EYE[0]].y * h),
            (lm[RIGHT_EYE[1]].x * w, lm[RIGHT_EYE[1]].y * h),
        )

        if left_eye_width == 0 or right_eye_width == 0:
            return None

        left_iris = self._iris_diameter(lm, LEFT_IRIS, w, h)
        right_iris = self._iris_diameter(lm, RIGHT_IRIS, w, h)

        return (left_iris / left_eye_width + right_iris / right_eye_width) / 2
