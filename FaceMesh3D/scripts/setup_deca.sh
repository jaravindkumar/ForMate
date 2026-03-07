#!/usr/bin/env bash
# =============================================================================
# FaceMesh3D — DECA Setup Script
# Run this once before starting the app.
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DECA_DIR="$ROOT_DIR/DECA"

echo "============================================================"
echo "  FaceMesh3D — DECA Setup"
echo "============================================================"

# ── 1. Clone DECA ─────────────────────────────────────────────────────────────
if [ ! -d "$DECA_DIR" ]; then
  echo "[1/4] Cloning DECA repository…"
  git clone https://github.com/YadiraF/DECA.git "$DECA_DIR"
else
  echo "[1/4] DECA directory already exists — skipping clone."
fi

# ── 2. Install Python dependencies ────────────────────────────────────────────
echo "[2/4] Installing Python dependencies…"
pip install -r "$ROOT_DIR/requirements.txt" --quiet

# ── 3. Download DECA pretrained model ─────────────────────────────────────────
DECA_MODEL_DIR="$DECA_DIR/data"
mkdir -p "$DECA_MODEL_DIR"

DECA_MODEL="$DECA_MODEL_DIR/deca_model.tar"
if [ ! -f "$DECA_MODEL" ]; then
  echo "[3/4] Downloading DECA pretrained model (~300 MB)…"
  # Official Google Drive link for DECA model
  # If this fails, manually download from: https://github.com/YadiraF/DECA
  pip install gdown --quiet
  gdown "https://drive.google.com/uc?id=1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje" \
        -O "$DECA_MODEL"
else
  echo "[3/4] DECA model already downloaded — skipping."
fi

# ── 4. FLAME model (requires manual registration) ─────────────────────────────
echo ""
echo "============================================================"
echo "  STEP 4 — FLAME model (manual download required)"
echo "============================================================"
echo ""
echo "  FLAME 2020 requires a free registration at:"
echo "  https://flame.is.tue.mpg.de/"
echo ""
echo "  After registering, download 'FLAME 2020' and place these"
echo "  files into: $DECA_MODEL_DIR/"
echo ""
echo "    generic_model.pkl   ← FLAME model"
echo "    FLAME_texture.npz   ← texture space"
echo ""
echo "  Also download the DECA data bundle:"
echo "  https://github.com/YadiraF/DECA  (see Data section in README)"
echo "  Unzip into: $DECA_MODEL_DIR/"
echo ""
echo "  Required files in $DECA_MODEL_DIR/:"
echo "    generic_model.pkl"
echo "    FLAME_texture.npz"
echo "    deca_model.tar"
echo "    fixed_displacement_256.npy"
echo "    head_template.obj"
echo "    landmark_embedding.npy"
echo "    mean_texture.jpg"
echo "    texture_data_256.npy"
echo "    uv_face_eye_mask.png"
echo "    uv_face_mask.png"
echo "============================================================"
echo ""

# Check if FLAME files exist
if [ -f "$DECA_MODEL_DIR/generic_model.pkl" ] && \
   [ -f "$DECA_MODEL_DIR/FLAME_texture.npz" ]; then
  echo "✅ FLAME model files found!"
else
  echo "⚠️  FLAME model files NOT found — please download them manually."
  echo "   The app will not work until these files are in place."
fi

echo ""
echo "============================================================"
echo "  Setup complete. Start the app with:"
echo "  cd $ROOT_DIR && uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo "  Then open: http://localhost:8000"
echo "============================================================"
