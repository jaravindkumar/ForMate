"""
FaceMesh3D — Streamlit app
Camera selfie → DECA 3D face reconstruction → interactive Plotly mesh
"""

import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── DECA path ──────────────────────────────────────────────────────────────────
DECA_DIR = Path(__file__).parent / "DECA"
if DECA_DIR.exists() and str(DECA_DIR) not in sys.path:
    sys.path.insert(0, str(DECA_DIR))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceMesh 3D",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS tweaks ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* tighten top padding */
  .block-container { padding-top: 1.5rem !important; }
  /* download button full-width */
  div[data-testid="stDownloadButton"] > button { width: 100%; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def is_deca_available() -> bool:
    return (DECA_DIR / "decalib" / "deca.py").exists()


@st.cache_resource(show_spinner="Loading DECA model (first run ~30 s)…")
def load_deca():
    """Load and cache DECA + device across reruns."""
    import torch
    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg

    deca_cfg.model.use_tex = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    deca = DECA(config=deca_cfg, device=device)
    return deca, device


def run_reconstruction(image_pil: Image.Image, deca, device: str) -> dict:
    """
    Run DECA on a PIL image.

    Returns dict with keys:
        verts  [N, 3] float32
        faces  [F, 3] int64
        views  {"Shape": PIL, "Detail": PIL, "Textured": PIL}
    Raises ValueError if no face detected.
    """
    import torch
    from decalib.datasets.datasets import TestData

    # Write to temp file so TestData can handle preprocessing
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image_pil.save(tmp.name, quality=95)
        tmp_path = tmp.name

    try:
        testdata = TestData(tmp_path, iscrop=True, face_detector="fan")
        if len(testdata) == 0:
            raise ValueError(
                "No face detected. Use a clear, front-facing photo with good lighting."
            )

        data   = testdata[0]
        images = data["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            codedict          = deca.encode(images)
            opdict, visdict   = deca.decode(codedict)

        verts = opdict["verts"][0].cpu().numpy().astype(np.float32)   # [5023, 3]
        faces = deca.flame.faces_tensor.cpu().numpy().astype(np.int64) # [F,    3]

        views = {}
        label_map = {
            "shape_images":        "Shape",
            "shape_detail_images": "Detail",
            "tex_images":          "Textured",
        }
        for key, label in label_map.items():
            if key in visdict:
                arr = visdict[key][0].permute(1, 2, 0).cpu().numpy()
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
                views[label] = Image.fromarray(arr)

        return {"verts": verts, "faces": faces, "views": views}

    finally:
        os.unlink(tmp_path)


def make_plotly_mesh(verts: np.ndarray, faces: np.ndarray) -> go.Figure:
    """Build an interactive Plotly 3D mesh figure."""
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color="#c89b7b",
        opacity=1.0,
        flatshading=False,
        lighting=dict(
            ambient=0.45,
            diffuse=0.85,
            specular=0.25,
            roughness=0.6,
            fresnel=0.15,
        ),
        lightposition=dict(x=2, y=4, z=5),
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="#0d1020",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(
                up=dict(x=0, y=1, z=0),
                eye=dict(x=0, y=0, z=2),
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=480,
    )
    return fig


def export_obj(verts: np.ndarray, faces: np.ndarray) -> bytes:
    """Minimal OBJ exporter — no external deps required."""
    lines = ["# FaceMesh3D export", "# DECA / FLAME model", ""]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    lines.append("")
    for f in faces:            # OBJ uses 1-based indices
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
    return "\n".join(lines).encode()


# ── App layout ─────────────────────────────────────────────────────────────────

st.title("⬡ FaceMesh 3D")
st.caption("Single-photo 3D face reconstruction · **DECA** · FLAME model")

# ── Setup guard ────────────────────────────────────────────────────────────────
if not is_deca_available():
    st.error(
        "**DECA is not set up.**\n\n"
        "Run the one-time setup script:\n"
        "```bash\nbash scripts/setup_deca.sh\n```\n"
        "Then download FLAME 2020 from https://flame.is.tue.mpg.de/ "
        "and place the files in `DECA/data/`."
    )
    st.stop()

# ── Load model ─────────────────────────────────────────────────────────────────
deca, device = load_deca()
st.success(f"DECA ready · running on **{device.upper()}**", icon="✅")
st.divider()

# ── Step 1: Camera ─────────────────────────────────────────────────────────────
st.subheader("Step 1 — Take a selfie")
st.caption("Look straight at the camera · good lighting · no glasses or obstructions")

img_file = st.camera_input(label="📷 Selfie", label_visibility="collapsed")

if not img_file:
    st.stop()

image_pil = Image.open(img_file).convert("RGB")

# ── Step 2: Confirm ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Step 2 — Confirm & reconstruct")

col_photo, col_btn = st.columns([1, 2], gap="large")
with col_photo:
    st.image(image_pil, caption="Your selfie", use_container_width=True)

with col_btn:
    st.markdown(
        "Click the button below to run **DECA** — a deep learning model "
        "that fits a parametric 3D FLAME face model to your photo.\n\n"
        "- Detects facial landmarks (FAN detector)\n"
        "- Predicts shape, expression, pose, texture & lighting\n"
        "- Outputs an interactive 3D mesh (~5 000 vertices)\n\n"
        "⏱ ~10–30 s on CPU, ~2–5 s on GPU"
    )
    build_btn = st.button(
        "✨ Build 3D Mesh",
        type="primary",
        use_container_width=True,
    )

# ── Step 3: Reconstruct ────────────────────────────────────────────────────────
if build_btn or "result" in st.session_state:

    if build_btn:
        # Clear previous result
        st.session_state.pop("result", None)

        progress = st.progress(0, text="Detecting face…")
        progress.progress(20, text="Running encoder…")

        try:
            result = run_reconstruction(image_pil, deca, device)
            st.session_state["result"] = result
        except ValueError as exc:
            st.error(f"❌ {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"❌ Reconstruction failed: {exc}")
            st.stop()

        progress.progress(100, text="Done!")
        progress.empty()

    # ── Step 4: Results ────────────────────────────────────────────────────────
    result = st.session_state["result"]
    st.divider()
    st.subheader("3D Face Mesh")

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("**Interactive 3D viewer** — drag to rotate · scroll to zoom")
        fig = make_plotly_mesh(result["verts"], result["faces"])
        st.plotly_chart(fig, use_container_width=True)

        # Download OBJ
        obj_bytes = export_obj(result["verts"], result["faces"])
        st.download_button(
            label="⬇ Download OBJ",
            data=obj_bytes,
            file_name="face_mesh.obj",
            mime="text/plain",
            use_container_width=True,
        )

    with right:
        views = result.get("views", {})
        if views:
            tab_names = list(views.keys())
            tabs = st.tabs(tab_names)
            for tab, name in zip(tabs, tab_names):
                with tab:
                    # Convert PIL to bytes for st.image
                    buf = io.BytesIO()
                    views[name].save(buf, format="PNG")
                    st.image(buf.getvalue(), caption=f"{name} view", use_container_width=True)

        st.markdown("**Original (cropped input)**")
        st.image(image_pil, use_container_width=True)

    st.divider()
    if st.button("🔄 New Selfie", use_container_width=True):
        st.session_state.pop("result", None)
        st.rerun()
