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
├── app/
│   ├── main.py           # FastAPI backend — receives image, runs DECA, serves files
│   ├── deca_runner.py    # DECA wrapper (preprocessing + reconstruction)
│   └── static/
│       ├── index.html    # Single-page UI
│       ├── style.css     # Dark-theme responsive styles
│       └── app.js        # Camera API + fetch + Three.js OBJ viewer
├── scripts/
│   └── setup_deca.sh     # One-time setup: clones DECA, downloads model
├── outputs/              # Per-session reconstruction results (auto-created)
├── requirements.txt
└── README.md
```

## Tech Stack

| Layer            | Technology                        |
|------------------|-----------------------------------|
| Backend          | FastAPI + Uvicorn                 |
| 3D Reconstruction| DECA (FLAME model)                |
| Face Detection   | face-alignment (FAN detector)     |
| Deep Learning    | PyTorch                           |
| Frontend         | Vanilla HTML + CSS + JS           |
| 3D Viewer        | Three.js (OBJLoader, OrbitControls) |
| Camera           | WebRTC `getUserMedia` API         |

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

```bash
cd FaceMesh3D
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

> **Note:** Camera requires a secure context. On mobile, either use `localhost` or serve over HTTPS (e.g. via `ngrok`).

## Usage

1. Click **Start Camera** — your front camera opens
2. Click **Take Selfie** — 3-second countdown, then capture
3. Click **Build 3D Mesh** — DECA reconstructs your face (~10–30 s)
4. Explore the **interactive 3D viewer** — drag to rotate, scroll to zoom
5. Switch between **Shape / Detail / Textured** rendered views
6. Click **Download OBJ** to save the mesh

## Mobile Access via ngrok

```bash
# Install ngrok (https://ngrok.com)
ngrok http 8000
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
