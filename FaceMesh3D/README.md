# FaceMesh 3D

A web app that lets you take a selfie with your phone's front camera and converts it into an interactive 3D face mesh using **DECA** (Detailed Expression Capture and Animation).

## Demo Flow

```
📷 Front camera opens  →  Take selfie  →  DECA reconstructs FLAME 3D mesh
→  Interactive Three.js viewer  +  Rendered shape/texture views  →  Download OBJ
```

## Architecture

```
FaceMesh3D/
├── streamlit_app.py      # ★ Streamlit entry point (camera + Plotly 3D)
├── .streamlit/
│   └── config.toml       # Dark theme + server settings
├── app/
│   ├── main.py           # FastAPI backend (alternative to Streamlit)
│   ├── deca_runner.py    # DECA wrapper (preprocessing + reconstruction)
│   └── static/           # HTML/CSS/JS + Three.js viewer
├── scripts/
│   └── setup_deca.sh     # One-time setup: clones DECA, downloads model
├── packages.txt          # System deps for Streamlit Cloud
├── requirements.txt
└── README.md
```

## Tech Stack

| Layer            | Technology                          |
|------------------|-------------------------------------|
| UI               | **Streamlit** (camera, layout, tabs)|
| 3D Viewer        | **Plotly** `go.Mesh3d` (interactive)|
| 3D Reconstruction| **DECA** (FLAME model)              |
| Face Detection   | face-alignment (FAN detector)       |
| Deep Learning    | PyTorch                             |
| Alt. Backend     | FastAPI + Three.js (also included)  |

## Setup

### Requirements
- Python 3.10+
- ~4 GB disk space (DECA model + FLAME files)
- NVIDIA GPU recommended (CPU works, ~30 s/image)

### Step 1 — Clone this repo and install

```bash
git clone https://github.com/jaravindkumar/ForMate
cd ForMate/FaceMesh3D
pip install -r requirements.txt
```

### Step 2 — Run setup script

```bash
bash scripts/setup_deca.sh
```

This will:
- Clone the DECA GitHub repository into `FaceMesh3D/DECA/`
- Download the DECA pretrained model (~300 MB)
- Print instructions for the FLAME model

### Step 3 — Download FLAME model (manual, free)

1. Register at **https://flame.is.tue.mpg.de/** (free)
2. Download **FLAME 2020**
3. Place `generic_model.pkl` and `FLAME_texture.npz` into `DECA/data/`

Also download the DECA data bundle from the [DECA GitHub page](https://github.com/YadiraF/DECA) and unzip into `DECA/data/`.

Required files in `DECA/data/`:
```
generic_model.pkl
FLAME_texture.npz
deca_model.tar
fixed_displacement_256.npy
head_template.obj
landmark_embedding.npy
mean_texture.jpg
texture_data_256.npy
uv_face_eye_mask.png
uv_face_mask.png
```

### Step 4 — Start the app

**Streamlit (recommended):**
```bash
cd FaceMesh3D
streamlit run streamlit_app.py
```
Open **http://localhost:8501** in your browser.

**FastAPI alternative:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000
```

> **Mobile camera note:** Camera access requires HTTPS outside `localhost`. Use `ngrok` (see below) to test on your phone.

## Streamlit Community Cloud

> ⚠️ **Important:** DECA requires ~2–4 GB RAM + large model files (~1 GB) that cannot be stored in a GitHub repo. Streamlit Community Cloud's free tier (1 GB RAM) will **not** be enough.
>
> **Recommended alternatives:**
> - Deploy on a cloud VM (AWS EC2, GCP, Hetzner) with ≥8 GB RAM
> - Use [Streamlit on Hugging Face Spaces](https://huggingface.co/spaces) (GPU Spaces work well)
> - Run locally and expose via ngrok

### Hugging Face Spaces (GPU)
1. Create a new Space → SDK: Streamlit → Hardware: GPU (T4 small)
2. Add `DECA/` as a Git LFS tracked folder or download models at startup
3. Set `HF_TOKEN` secret if downloading gated models
4. Push — Spaces will run `streamlit run streamlit_app.py` automatically

## Usage

1. **Take selfie** — Streamlit's built-in camera widget opens your front cam
2. Click **Build 3D Mesh** — DECA fits the FLAME model (~10–30 s on CPU)
3. **Rotate/zoom** the interactive Plotly 3D mesh
4. Switch between **Shape / Detail / Textured** rendered view tabs
5. Click **Download OBJ** to save the mesh file

## Mobile Access via ngrok

```bash
# Install ngrok (https://ngrok.com)
streamlit run streamlit_app.py &
ngrok http 8501
# Open the https:// URL on your phone
```

## What DECA Outputs

| Output           | Description                              |
|------------------|------------------------------------------|
| `face_mesh.obj`  | 3D FLAME mesh (5023 vertices)            |
| `face_mesh.mtl`  | Material file referencing texture        |
| `face_mesh.jpg`  | Predicted UV texture                     |
| `shape_images`   | Rendered grey-shaded shape               |
| `shape_detail_images` | Detailed geometry (wrinkles etc.) |
| `tex_images`     | Textured rendering                       |

## License

MIT License

DECA and FLAME are subject to their own licenses:
- DECA: https://github.com/YadiraF/DECA/blob/master/LICENSE
- FLAME: https://flame.is.tue.mpg.de/
