"""
Heuristic accessory classification using facial landmark regions.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from pipeline.detector import FaceDetector


@dataclass
class AccessoryResult:
    name: str
    label: str
    detected: bool
    confidence: float
    region: Optional[np.ndarray] = field(default=None, repr=False)


# Display labels
LABELS = {
    "glasses":    "Glasses",
    "sunglasses": "Sunglasses",
    "face_mask":  "Face Mask",
    "hat":        "Hat / Cap",
    "headband":   "Headband",
    "earrings":   "Earrings",
}


class AccessoryClassifier:
    """
    Classifies face accessories by analysing landmark-bounded image regions.

    Each accessory uses a lightweight heuristic:
      - Glasses   : edge density in the eye region (frames create hard edges)
      - Sunglasses: darkness ratio + edge density (dark lens + frame edges)
      - Face mask : texture uniformity in lower face (masks are smooth/uniform)
      - Hat       : brightness discontinuity above forehead (hat brim/edge)
      - Headband  : mid-forehead band variance + edge cues
      - Earrings  : high edge density near ear lobes (small high-contrast objects)
    """

    ACCESSORIES = list(LABELS.keys())

    def __init__(self):
        self._detector = FaceDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, image: np.ndarray) -> Dict[str, AccessoryResult]:
        """
        Classify all accessories in a BGR image.

        Returns a dict: accessory_key -> AccessoryResult.
        """
        detection = self._detector.detect(image)
        if detection is None:
            return {
                k: AccessoryResult(name=k, label=LABELS[k], detected=False, confidence=0.0)
                for k in self.ACCESSORIES
            }

        lm = detection.landmarks
        results: Dict[str, AccessoryResult] = {}
        results["glasses"]    = self._glasses(image, lm)
        results["sunglasses"] = self._sunglasses(image, lm)
        results["face_mask"]  = self._face_mask(image, lm)
        results["hat"]        = self._hat(image, lm)
        results["headband"]   = self._headband(image, lm)
        results["earrings"]   = self._earrings(image, lm)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _crop(self, image, indices, landmarks, pad=14):
        return self._detector.crop_region(image, indices, landmarks, pad)

    @staticmethod
    def _gray(region: np.ndarray) -> np.ndarray:
        if region.ndim == 3:
            return cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return region

    @staticmethod
    def _edge_density(region: np.ndarray) -> float:
        gray = AccessoryClassifier._gray(region)
        if gray.size == 0:
            return 0.0
        edges = cv2.Canny(gray, 50, 150)
        return float(np.sum(edges > 0) / edges.size)

    @staticmethod
    def _darkness_ratio(region: np.ndarray, threshold: int = 60) -> float:
        gray = AccessoryClassifier._gray(region)
        if gray.size == 0:
            return 0.0
        return float(np.sum(gray < threshold) / gray.size)

    @staticmethod
    def _texture_variance(region: np.ndarray) -> float:
        gray = AccessoryClassifier._gray(region)
        return float(np.var(gray)) if gray.size > 0 else 0.0

    # --- Individual classifiers ---

    def _glasses(self, image, lm):
        region = self._crop(image, FaceDetector.GLASSES_REGION, lm, pad=22)
        ed = self._edge_density(region)
        conf = min(1.0, ed * 4.0)
        return AccessoryResult("glasses", LABELS["glasses"], ed > 0.08, round(conf, 3), region)

    def _sunglasses(self, image, lm):
        region = self._crop(image, FaceDetector.SUNGLASSES_REGION, lm, pad=22)
        darkness = self._darkness_ratio(region, threshold=55)
        ed = self._edge_density(region)
        conf = min(1.0, darkness * 1.6 + ed * 0.6)
        detected = darkness > 0.28 and ed > 0.04
        return AccessoryResult("sunglasses", LABELS["sunglasses"], detected, round(conf, 3), region)

    def _face_mask(self, image, lm):
        region = self._crop(image, FaceDetector.MASK_REGION, lm, pad=16)
        var = self._texture_variance(region)
        ed = self._edge_density(region)
        conf = min(1.0, max(0.0, (1.0 - var / 1800) * 0.65 + ed * 0.35))
        detected = var < 1200 and ed > 0.04
        return AccessoryResult("face_mask", LABELS["face_mask"], detected, round(conf, 3), region)

    def _hat(self, image, lm):
        region = self._crop(image, FaceDetector.HAT_REGION, lm, pad=32)
        gray = self._gray(region)
        if gray.size == 0:
            return AccessoryResult("hat", LABELS["hat"], False, 0.0)
        top = gray[: max(1, gray.shape[0] // 3), :]
        brightness_diff = abs(float(np.mean(top)) - float(np.mean(gray)))
        ed = self._edge_density(region)
        conf = min(1.0, ed * 2.5 + brightness_diff / 80)
        detected = ed > 0.09 and brightness_diff > 12
        return AccessoryResult("hat", LABELS["hat"], detected, round(conf, 3), region)

    def _headband(self, image, lm):
        region = self._crop(image, FaceDetector.HEADBAND_REGION, lm, pad=20)
        gray = self._gray(region)
        if gray.size == 0:
            return AccessoryResult("headband", LABELS["headband"], False, 0.0)
        mid = gray[gray.shape[0] // 4: gray.shape[0] * 3 // 4, :]
        mid_var = float(np.var(mid))
        ed = self._edge_density(region)
        conf = min(1.0, ed * 2.2 + mid_var / 2500)
        detected = ed > 0.07 and mid_var > 280
        return AccessoryResult("headband", LABELS["headband"], detected, round(conf, 3), region)

    def _earrings(self, image, lm):
        scores = []
        for indices in [FaceDetector.EARRING_LEFT, FaceDetector.EARRING_RIGHT]:
            region = self._crop(image, indices, lm, pad=10)
            scores.append(self._edge_density(region))
        avg_ed = float(np.mean(scores)) if scores else 0.0
        conf = min(1.0, avg_ed * 4.5)
        return AccessoryResult("earrings", LABELS["earrings"], avg_ed > 0.09, round(conf, 3))

    def close(self):
        self._detector.close()
