"""
Face landmark detection using MediaPipe Face Mesh (468 landmarks).
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FaceDetectionResult:
    landmarks: list          # List of (x, y, z) tuples in pixel coords
    image_shape: tuple       # (height, width)
    face_bbox: tuple         # (x, y, w, h)
    annotated_image: Optional[np.ndarray] = field(default=None, repr=False)


class FaceDetector:
    """Detects facial landmarks using MediaPipe Face Mesh."""

    # Landmark index groups for each accessory region
    GLASSES_REGION    = [33, 133, 362, 263, 168, 6, 197, 195, 5]
    SUNGLASSES_REGION = [33, 133, 362, 263, 168, 6, 197, 195, 5]
    MASK_REGION       = [61, 291, 199, 200, 175, 152, 378, 379, 365]
    HAT_REGION        = [10, 338, 297, 332, 284, 251, 389, 356, 454]
    HEADBAND_REGION   = [10, 108, 67, 69, 104, 54, 21, 162, 127]
    EARRING_LEFT      = [234, 93, 132, 58, 172, 136, 150, 149, 176]
    EARRING_RIGHT     = [454, 323, 361, 288, 397, 365, 379, 378, 400]

    def __init__(self, max_faces: int = 1, min_confidence: float = 0.5):
        self._mp_mesh = mp.solutions.face_mesh
        self._drawing = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._mesh = self._mp_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
        )

    def detect(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        """
        Run face mesh detection on a BGR image.

        Returns FaceDetectionResult, or None if no face is found.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        face_lm = result.multi_face_landmarks[0]
        h, w = image.shape[:2]
        landmarks = [(lm.x * w, lm.y * h, lm.z) for lm in face_lm.landmark]

        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))

        annotated = image.copy()
        self._drawing.draw_landmarks(
            annotated,
            face_lm,
            self._mp_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._styles.get_default_face_mesh_tesselation_style(),
        )

        return FaceDetectionResult(
            landmarks=landmarks,
            image_shape=(h, w),
            face_bbox=bbox,
            annotated_image=annotated,
        )

    def crop_region(
        self,
        image: np.ndarray,
        landmark_indices: list,
        landmarks: list,
        padding: int = 12,
    ) -> np.ndarray:
        """Crop the image around the bounding box of the given landmark indices."""
        h, w = image.shape[:2]
        pts = [
            (int(landmarks[i][0]), int(landmarks[i][1]))
            for i in landmark_indices
            if i < len(landmarks)
        ]
        if not pts:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1 = max(0, min(xs) - padding)
        y1 = max(0, min(ys) - padding)
        x2 = min(w, max(xs) + padding)
        y2 = min(h, max(ys) + padding)
        crop = image[y1:y2, x1:x2]
        return crop if crop.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8)

    def close(self):
        self._mesh.close()
