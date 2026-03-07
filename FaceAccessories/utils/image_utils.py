"""
Image loading and preprocessing utilities.
"""

import cv2
import numpy as np
from PIL import Image
import io


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Load an image from raw bytes and return a BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure the file is a valid JPG/PNG/WEBP.")
    return img


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to an OpenCV BGR numpy array."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr_image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR numpy array to a PIL Image (RGB)."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_keep_aspect(image: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Resize image so that its longest side is at most max_side pixels."""
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
