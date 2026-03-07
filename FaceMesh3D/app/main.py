"""
FaceMesh3D — FastAPI backend
Receives a selfie image, runs DECA 3D reconstruction, serves mesh + views.
"""

import asyncio
import base64
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
STATIC_DIR  = BASE_DIR / "static"
OUTPUTS_DIR = BASE_DIR.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="FaceMesh3D", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static",  StaticFiles(directory=str(STATIC_DIR)),  name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# DECA is CPU/GPU-intensive — one worker at a time
_executor = ThreadPoolExecutor(max_workers=1)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    from deca_runner import is_deca_available
    return {"status": "ok", "deca_ready": is_deca_available()}


@app.post("/api/reconstruct")
async def reconstruct(file: UploadFile = File(...)):
    """
    Accepts a face image, runs DECA 3D reconstruction.

    Returns JSON with:
      - session_id
      - obj_url, mtl_url, texture_url  (for Three.js loader)
      - views: [{name, data}]           (base64-encoded rendered images)
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    session_id = str(uuid.uuid4())
    session_dir = OUTPUTS_DIR / session_id
    session_dir.mkdir()

    # Save upload
    img_path = session_dir / "input.jpg"
    content = await file.read()
    img_path.write_bytes(content)

    # Run reconstruction in thread pool (blocking)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            _blocking_reconstruct,
            str(img_path),
            str(session_dir),
        )
    except ValueError as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {exc}")

    # Encode rendered view images as base64 for inline display
    views = []
    for view in result["views"]:
        view_path = session_dir / view["file"]
        if view_path.exists():
            b64 = base64.b64encode(view_path.read_bytes()).decode()
            views.append({"name": view["name"], "data": f"data:image/png;base64,{b64}"})

    base_url = f"/outputs/{session_id}"
    return JSONResponse({
        "session_id":   session_id,
        "obj_url":      f"{base_url}/{result['obj_file']}",
        "mtl_url":      f"{base_url}/{result['mtl_file']}",
        "texture_url":  f"{base_url}/{result['texture_file']}",
        "input_url":    f"{base_url}/input.jpg",
        "views":        views,
    })


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blocking_reconstruct(image_path: str, output_dir: str) -> dict:
    """Called in the thread-pool executor (not async)."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from deca_runner import reconstruct
    return reconstruct(image_path, output_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
