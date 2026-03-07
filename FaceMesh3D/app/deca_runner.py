"""
DECA 3D face reconstruction wrapper.

DECA: Detailed Expression Capture and Animation
Paper: https://arxiv.org/abs/2012.04012
Repo:  https://github.com/YadiraF/DECA
"""

import sys
import os
import numpy as np
from pathlib import Path
from PIL import Image as PILImage

# DECA is cloned into FaceMesh3D/DECA/ by the setup script
DECA_DIR = Path(__file__).parent.parent / "DECA"


def is_deca_available() -> bool:
    return (DECA_DIR / "decalib" / "deca.py").exists()


def _add_deca_to_path():
    deca_str = str(DECA_DIR)
    if deca_str not in sys.path:
        sys.path.insert(0, deca_str)


def reconstruct(image_path: str, output_dir: str) -> dict:
    """
    Run DECA 3D face reconstruction on a single image.

    Args:
        image_path: Path to the input face image (JPG/PNG).
        output_dir: Directory where outputs will be saved.

    Returns:
        {
          "obj_file":     "face_mesh.obj"  (relative to output_dir),
          "mtl_file":     "face_mesh.mtl",
          "texture_file": "face_mesh.jpg",
          "views": [
              {"name": "Shape",    "file": "shape_images.png"},
              {"name": "Detail",   "file": "shape_detail_images.png"},
              {"name": "Textured", "file": "tex_images.png"},
          ]
        }

    Raises:
        RuntimeError: DECA is not set up.
        ValueError:   No face detected in the image.
    """
    if not is_deca_available():
        raise RuntimeError(
            "DECA is not set up. Please run scripts/setup_deca.sh first."
        )

    os.makedirs(output_dir, exist_ok=True)
    _add_deca_to_path()

    import torch
    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg
    from decalib.datasets.datasets import TestData

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Enable texture prediction
    deca_cfg.model.use_tex = True

    deca = DECA(config=deca_cfg, device=device)

    # TestData handles face detection + crop to 224×224
    testdata = TestData(image_path, iscrop=True, face_detector="fan")
    if len(testdata) == 0:
        raise ValueError(
            "No face detected in the image. "
            "Please use a clear, front-facing photo with good lighting."
        )

    data = testdata[0]
    images = data["image"].unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.no_grad():
        codedict = deca.encode(images)
        opdict, visdict = deca.decode(codedict)

    # ── Save OBJ mesh (also writes .mtl + .jpg texture) ──────────────────────
    mesh_name = "face_mesh"
    obj_path = os.path.join(output_dir, f"{mesh_name}.obj")
    deca.save_obj(obj_path, opdict)

    # ── Save rendered view images ─────────────────────────────────────────────
    view_keys = {
        "shape_images":        "Shape",
        "shape_detail_images": "Detail",
        "tex_images":          "Textured",
    }
    views = []
    for key, display_name in view_keys.items():
        if key not in visdict:
            continue
        img_tensor = visdict[key][0]                            # [3, H, W], [0,1]
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        filename = f"{key}.png"
        PILImage.fromarray(img_np).save(os.path.join(output_dir, filename))
        views.append({"name": display_name, "file": filename})

    # ── Also save the cropped input ───────────────────────────────────────────
    if "inputs" in visdict:
        inp = visdict["inputs"][0].permute(1, 2, 0).cpu().numpy()
        inp = (inp * 255).clip(0, 255).astype(np.uint8)
        PILImage.fromarray(inp).save(os.path.join(output_dir, "cropped_input.png"))

    return {
        "obj_file":     f"{mesh_name}.obj",
        "mtl_file":     f"{mesh_name}.mtl",
        "texture_file": f"{mesh_name}.jpg",
        "views":        views,
    }
