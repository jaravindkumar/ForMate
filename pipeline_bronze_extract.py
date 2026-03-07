"""
pipeline_bronze_extract.py
──────────────────────────
Primary model  : MediaPipe PoseLandmarker (Tasks API) — works on mediapipe 0.10+
Fallback       : YOLOv8n-pose if ultralytics available
Output         : JSONL with 33 BlazePose-indexed keypoints per frame
"""

import os, json, time, shutil, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

KEY_IDS = [11, 12, 23, 24, 25, 26, 27, 28, 15, 16]

YOLO17_TO_BP33 = {
    0:0, 1:2, 2:5, 3:7, 4:8,
    5:11, 6:12, 7:13, 8:14, 9:15, 10:16,
    11:23, 12:24, 13:25, 14:26, 15:27, 16:28,
}

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_NAME = "pose_landmarker_lite.task"


def _get_model_path(root: Path) -> Path:
    model_path = root / MODEL_NAME
    if not model_path.exists():
        print(f"[bronze] Downloading pose model to {model_path}...")
        root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
        print(f"[bronze] Downloaded ({model_path.stat().st_size // 1024} KB)")
    return model_path


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def make_session_id(exercise):
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + exercise


def _run_mediapipe_tasks(cap, fps, width, height, model_path, out_jsonl, session_id):
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
    from mediapipe.tasks.python.core.base_options import BaseOptions
    import mediapipe as mp

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    frame_idx = detected_frames = 0
    with PoseLandmarker.create_from_options(options) as landmarker, \
         out_jsonl.open("w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(frame_idx * 1000 / fps))
            pose_detected = bool(result.pose_landmarks)
            lms = [{"id":i,"x":0.0,"y":0.0,"z":0.0,"vis":0.0} for i in range(33)]
            if pose_detected:
                detected_frames += 1
                for i, lm in enumerate(result.pose_landmarks[0]):
                    if i < 33:
                        lms[i] = {"id":i,"x":float(lm.x),"y":float(lm.y),
                                  "z":float(lm.z),"vis":float(lm.visibility)}
            f.write(json.dumps({"session_id":session_id,"frame_idx":frame_idx,
                "t_sec":float(frame_idx/fps),"pose_detected":pose_detected,
                "landmarks":lms}) + "\n")
            frame_idx += 1
    return frame_idx, detected_frames


def _try_load_yolo():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n-pose.pt")
        print("[bronze] YOLOv8n-pose loaded")
        return model
    except Exception as e:
        print(f"[bronze] YOLOv8 unavailable ({e}) — using MediaPipe Tasks API")
        return None


def _run_yolo(cap, fps, width, height, yolo_model, out_jsonl, session_id):
    frame_idx = detected_frames = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok: break
            results = yolo_model(frame, verbose=False)
            lms = [{"id":i,"x":0.0,"y":0.0,"z":0.0,"vis":0.0} for i in range(33)]
            pose_detected = False
            r = results[0]
            if r.keypoints is not None:
                kps = r.keypoints.data
                if kps is not None and kps.shape[0] > 0:
                    best = int(kps[:,:,2].mean(dim=1).argmax())
                    kp = kps[best]
                    for yi, bi in YOLO17_TO_BP33.items():
                        x,y,c = float(kp[yi,0]),float(kp[yi,1]),float(kp[yi,2])
                        lms[bi] = {"id":bi,"x":x/max(width,1),"y":y/max(height,1),"z":0.0,"vis":c}
                        if c > 0.3: pose_detected = True
            if pose_detected: detected_frames += 1
            f.write(json.dumps({"session_id":session_id,"frame_idx":frame_idx,
                "t_sec":float(frame_idx/fps),"pose_detected":pose_detected,
                "landmarks":lms}) + "\n")
            frame_idx += 1
    return frame_idx, detected_frames


def extract_bronze(video_path, out_root="pipeline/bronze", exercise="deadlift",
                   camera_view="front_oblique", write_overlay=False, model_dir=None):
    video_path = str(os.path.abspath(video_path))
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    session_id = make_session_id(exercise)
    out_dir = Path(out_root) / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    input_copy = out_dir / "input.mp4"
    shutil.copy2(video_path, str(input_copy))

    cap = cv2.VideoCapture(str(input_copy))
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps        = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)  or 0)
    print(f"[bronze] Video: {width}x{height} @ {fps:.1f}fps  ~{num_frames} frames")

    out_jsonl = out_dir / "keypoints.jsonl"
    t0 = time.time()

    yolo_model = _try_load_yolo()
    if yolo_model:
        model_name = "yolov8n-pose"
        frame_idx, detected_frames = _run_yolo(cap, fps, width, height, yolo_model, out_jsonl, session_id)
    else:
        model_name = "mediapipe-pose-landmarker-lite"
        m_dir = Path(model_dir) if model_dir else Path(out_root).parent.parent
        model_path = _get_model_path(m_dir)
        frame_idx, detected_frames = _run_mediapipe_tasks(cap, fps, width, height, model_path, out_jsonl, session_id)

    cap.release()

    meta = {"session_id":session_id,"exercise":exercise,"camera_view":camera_view,
            "input_video":str(input_copy),"source_video_path":video_path,
            "fps":fps,"width":width,"height":height,"num_frames":num_frames,
            "created_utc":utc_now_iso(),"model":model_name}
    (out_dir/"meta.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")

    summary = {"session_id":session_id,"model_used":model_name,
               "frames_processed":frame_idx,"pose_detected_frames":detected_frames,
               "pose_detected_ratio":float(detected_frames/max(frame_idx,1)),
               "elapsed_sec":float(time.time()-t0),
               "outputs":{"session_dir":str(out_dir),"meta":str(out_dir/"meta.json"),
                          "input_copy":str(input_copy),"keypoints_jsonl":str(out_jsonl),
                          "overlay_mp4":None}}
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"[bronze] Done: {frame_idx} frames, {detected_frames} detected")
    return session_id


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",       required=True)
    ap.add_argument("--exercise",    default="deadlift", choices=["deadlift","squat"])
    ap.add_argument("--out_root",    default="pipeline/bronze")
    ap.add_argument("--camera_view", default="front_oblique")
    ap.add_argument("--overlay",     action="store_true")
    ap.add_argument("--model_dir",   default=None)
    args = ap.parse_args()
    extract_bronze(video_path=args.video, out_root=args.out_root,
                   exercise=args.exercise, camera_view=args.camera_view,
                   write_overlay=args.overlay, model_dir=args.model_dir)
