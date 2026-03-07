"""
Annotate images with detected accessory results.
"""

import cv2
import numpy as np
from typing import Dict
from pipeline.classifier import AccessoryResult


# BGR colours per accessory
COLOURS = {
    "glasses":    (255, 200,   0),
    "sunglasses": ( 50,  50, 200),
    "face_mask":  (  0, 200, 100),
    "hat":        (200,  50, 200),
    "headband":   (  0, 180, 255),
    "earrings":   (255,  80,  80),
}


def annotate_image(
    image: np.ndarray,
    results: Dict[str, AccessoryResult],
    face_bbox: tuple = None,
) -> np.ndarray:
    """
    Draw accessory detection labels onto the image.

    Args:
        image:      BGR image.
        results:    Output of AccessoryClassifier.classify().
        face_bbox:  Optional (x, y, w, h) to draw a face bounding box.

    Returns:
        Annotated BGR image.
    """
    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    if face_bbox:
        x, y, w, h = face_bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (200, 200, 200), 2)

    detected = [(k, v) for k, v in results.items() if v.detected]
    others   = [(k, v) for k, v in results.items() if not v.detected]

    for i, (key, res) in enumerate(detected + others):
        colour = COLOURS.get(key, (255, 255, 255)) if res.detected else (110, 110, 110)
        y_pos = 28 + i * 30
        text = (
            f"[+] {res.label}  {res.confidence:.0%}"
            if res.detected
            else f"[ ] {res.label}"
        )
        cv2.putText(out, text, (11, y_pos + 1), font, 0.62, (0, 0, 0), 2)
        cv2.putText(out, text, (10, y_pos),     font, 0.62, colour,     2)

    return out


def build_summary_table(results: Dict[str, AccessoryResult]) -> list:
    """Return a list of dicts suitable for a Streamlit/pandas dataframe."""
    return [
        {
            "Accessory":  res.label,
            "Detected":   "Yes" if res.detected else "No",
            "Confidence": f"{res.confidence:.1%}",
        }
        for res in results.values()
    ]
