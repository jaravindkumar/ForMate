"""
pipeline_bronze_extract.py — MediaPipe Tasks API (mediapipe >= 0.10)
"""
import os, json, time, shutil, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_NAME = "pose_landmarker_lite.task"

YOLO17_TO_BP33 = {0:0,1:2,2:5,3:7,4:8,5:11,6:12,7:13,8:14,9:15,10:16,11:23,12:24,13:25,14:26,15:27,16:28}


def _find_model():
    here = Path(__file__).resolve().parent
    for p in [here/MODEL_NAME, here.parent/MODEL_NAME, Path.cwd()/MODEL_NAME, Path("/tmp")/MODEL_NAME]:
        if p.exists() and p.stat().st_size > 100_000:
            print(f"[bronze] Using cached model: {p}")
            return p
    return None

def _download_model():
    dst = Path("/tmp") / MODEL_NAME
    if dst.exists() and dst.stat().st_size > 100_000:
        return dst
    print(f"[bronze] Downloading model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, str(dst))
        print(f"[bronze] Downloaded ({dst.stat().st_size//1024} KB)")
        return dst
    except Exception as e:
        raise RuntimeError(
            f"Cannot download pose model: {e}\n"
            f"Add pose_landmarker_lite.task to your repo root OR ensure network access."
        )

def _get_model_path():
    return _find_model() or _download_model()

def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def make_session_id(exercise):
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + exercise


def _run_mediapipe(cap, fps, model_path, out_jsonl, session_id):
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
    from mediapipe.tasks.python.core.base_options import BaseOptions
    import mediapipe as mp

    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    frame_idx = detected = 0
    with PoseLandmarker.create_from_options(opts) as lmk, out_jsonl.open("w") as f:
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = lmk.detect_for_video(img, int(frame_idx * 1000 / fps))
            found = bool(res.pose_landmarks)
            lms = [{"id":i,"x":0.0,"y":0.0,"z":0.0,"vis":0.0} for i in range(33)]
            if found:
                detected += 1
                for i, lm in enumerate(res.pose_landmarks[0]):
                    if i < 33:
                        lms[i] = {"id":i,"x":float(lm.x),"y":float(lm.y),
                                  "z":float(lm.z),"vis":float(lm.visibility)}
            f.write(json.dumps({"session_id":session_id,"frame_idx":frame_idx,
                "t_sec":float(frame_idx/fps),"pose_detected":found,"landmarks":lms})+"\n")
            frame_idx += 1
    return frame_idx, detected


def _try_yolo():
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n-pose.pt")
    except:
        return None

def _run_yolo(cap, fps, width, height, model, out_jsonl, session_id):
    frame_idx = detected = 0
    with out_jsonl.open("w") as f:
        while True:
            ok, frame = cap.read()
            if not ok: break
            res = model(frame, verbose=False)
            lms = [{"id":i,"x":0.0,"y":0.0,"z":0.0,"vis":0.0} for i in range(33)]
            found = False
            r = res[0]
            if r.keypoints is not None:
                kps = r.keypoints.data
                if kps is not None and kps.shape[0] > 0:
                    best = int(kps[:,:,2].mean(dim=1).argmax())
                    kp = kps[best]
                    for yi,bi in YOLO17_TO_BP33.items():
                        x,y,c = float(kp[yi,0]),float(kp[yi,1]),float(kp[yi,2])
                        lms[bi] = {"id":bi,"x":x/max(width,1),"y":y/max(height,1),"z":0.0,"vis":c}
                        if c > 0.3: found = True
            if found: detected += 1
            f.write(json.dumps({"session_id":session_id,"frame_idx":frame_idx,
                "t_sec":float(frame_idx/fps),"pose_detected":found,"landmarks":lms})+"\n")
            frame_idx += 1
    return frame_idx, detected


def _open_video(path):
    for backend in [cv2.CAP_ANY, cv2.CAP_FFMPEG]:
        cap = cv2.VideoCapture(path, backend)
        if cap.isOpened():
            # Read one frame to verify
            ok, _ = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if ok:
                return cap
        cap.release()
    raise RuntimeError(
        f"Cannot open video: {path} ({Path(path).stat().st_size} bytes)\n"
        f"Supported formats: .mp4, .mov, .avi — try re-encoding as mp4."
    )


def extract_bronze(video_path, out_root="pipeline/bronze", exercise="deadlift",
                   camera_view="front_oblique", write_overlay=False, model_dir=None):
    video_path = str(os.path.abspath(video_path))
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    print(f"[bronze] Input: {video_path} ({Path(video_path).stat().st_size} bytes)")

    session_id = make_session_id(exercise)
    out_dir = Path(out_root) / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_ext = Path(video_path).suffix or ".mp4"
    input_copy = out_dir / f"input{orig_ext}"
    shutil.copy2(video_path, str(input_copy))

    cap = _open_video(str(input_copy))
    fps    = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    nframes= int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[bronze] {width}x{height} @ {fps:.1f}fps ~{nframes} frames")

    out_jsonl = out_dir / "keypoints.jsonl"
    t0 = time.time()

    yolo = _try_yolo()
    if yolo:
        model_name = "yolov8n-pose"
        frame_idx, detected = _run_yolo(cap, fps, width, height, yolo, out_jsonl, session_id)
    else:
        model_name = "mediapipe-pose-landmarker-lite"
        model_path = _get_model_path()
        frame_idx, detected = _run_mediapipe(cap, fps, model_path, out_jsonl, session_id)
    cap.release()
    elapsed = time.time() - t0

    if frame_idx == 0:
        raise RuntimeError("No frames read — file may be corrupt or unsupported format.")
    print(f"[bronze] Done: {frame_idx} frames, {detected} detected ({elapsed:.1f}s)")

    meta = {"session_id":session_id,"exercise":exercise,"camera_view":camera_view,
            "input_video":str(input_copy),"source_video_path":video_path,
            "fps":fps,"width":width,"height":height,"num_frames":nframes,
            "created_utc":utc_now_iso(),"model":model_name}
    (out_dir/"meta.json").write_text(json.dumps(meta,indent=2))

    summary = {"session_id":session_id,"model_used":model_name,
               "frames_processed":frame_idx,"pose_detected_frames":detected,
               "pose_detected_ratio":float(detected/max(frame_idx,1)),
               "elapsed_sec":float(elapsed),
               "outputs":{"session_dir":str(out_dir),"meta":str(out_dir/"meta.json"),
                          "input_copy":str(input_copy),"keypoints_jsonl":str(out_jsonl),
                          "overlay_mp4":None}}
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2))
    return session_id


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--exercise", default="deadlift")
    ap.add_argument("--out_root", default="pipeline/bronze")
    ap.add_argument("--camera_view", default="front_oblique")
    args = ap.parse_args()
    extract_bronze(args.video, args.out_root, args.exercise, args.camera_view)
