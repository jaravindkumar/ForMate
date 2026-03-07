"""
Face Accessories Detection — Streamlit App
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from PIL import Image

from utils.image_utils import pil_to_bgr, bgr_to_pil, resize_keep_aspect
from pipeline.classifier import AccessoryClassifier
from pipeline.annotator import annotate_image, build_summary_table


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Face Accessories Detection",
    page_icon="👓",
    layout="wide",
)

st.title("👓 Face Accessories Detection")
st.markdown(
    "Upload a photo to detect face accessories: **glasses, sunglasses, face mask, "
    "hat, headband, and earrings**."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    max_side = st.slider("Max image size (px)", 400, 1200, 800, step=100)
    show_mesh = st.checkbox("Show face mesh overlay", value=True)
    detection_confidence = st.slider("Detection confidence", 0.1, 1.0, 0.5, step=0.05)
    st.divider()
    st.markdown("**Supported accessories**")
    for acc in ["Glasses", "Sunglasses", "Face Mask", "Hat / Cap", "Headband", "Earrings"]:
        st.markdown(f"- {acc}")

# ── File upload ───────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Portrait or close-up photos work best.",
)

if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

# ── Process ───────────────────────────────────────────────────────────────────

with st.spinner("Detecting face accessories…"):
    pil_img = Image.open(uploaded).convert("RGB")
    bgr_img = pil_to_bgr(pil_img)
    bgr_img = resize_keep_aspect(bgr_img, max_side)

    classifier = AccessoryClassifier()
    results = classifier.classify(bgr_img)
    classifier.close()

    from pipeline.detector import FaceDetector
    detector = FaceDetector(min_confidence=detection_confidence)
    detection = detector.detect(bgr_img)
    detector.close()

    face_bbox = detection.face_bbox if detection else None
    source_img = detection.annotated_image if (detection and show_mesh) else bgr_img
    annotated = annotate_image(source_img, results, face_bbox)

# ── Results ───────────────────────────────────────────────────────────────────

col_img, col_results = st.columns([1.4, 1])

with col_img:
    st.subheader("Result")
    st.image(bgr_to_pil(annotated), use_container_width=True)

with col_results:
    st.subheader("Detected Accessories")
    if face_bbox is None:
        st.warning("No face detected. Try a clearer, front-facing photo.")
    else:
        detected_names = [r.label for r in results.values() if r.detected]
        if detected_names:
            st.success(", ".join(detected_names))
        else:
            st.info("No accessories detected.")

        st.divider()
        df = pd.DataFrame(build_summary_table(results))
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Confidence Scores")
        chart_data = {r.label: r.confidence for r in results.values()}
        st.bar_chart(chart_data)
