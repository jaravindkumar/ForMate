"""
pipeline_bronze_extract.py
──────────────────────────
Primary model  : YOLOv8-pose  (ultralytics) — best accuracy, occlusion handling
Fallback model : MediaPipe BlazePose complexity=2 — if YOLO unavailable
Output format  : identical JSONL regardless of which model ran
                 landmarks use BlazePose 33-point indexing so silver/gold
                 pipeline is unchanged
"""

import os
import json
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# ── BlazePose landmark index names (33 points) ────────────────────
# YOLO-pose returns 17 COCO keypoints — we map them to the
# 33-point BlazePose indices that silver/gold expect.
YOLO17_TO_BP33 = {
    0:  0,   # nose
    1:  2,   # left_eye
    2:  5,   # right_eye
    3:  7,   # left_ear
    4:  8,   # right_ear
    5:  11,  # left_shoulder
    6:  12,  # right_shoulder
    7:  13,  # left_elbow
    8:  14,  # right_elbow
    9:  15,  # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
}

KEY_IDS = [11, 12, 23, 24, 25, 26, 27, 28, 15, 16]

SKELETON_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
]


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")


def make_session_id(exercise):
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + exercise


def _try_load_yolo():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n-pose.pt")
        print("[bronze] YOLOv8n-pose loaded")
        return model
    except Exception as e:
        print(f"[bronze] YOLOv8 unavailable ({e}) — using MediaPipe BlazePose complexity=2")
        return None


def _yolo_to_landmarks(result, width, height):
    lms = [{"id":i,"x":0.0,"y":0.0,"z":0.0,"vis":0.0} for i in range(33)]
    if result.keypoints is None:
        return lms, False
    kps = result.keypoints.data
    if kps is None or kps.shape[0] == 0:
        return lms, False
    mean_conf = kps[:,:,2].mean(dim=1)
    best = int(mean_conf.argmax())
    kp = kps[best]
    detected = False
    for yi, bi in YOLO17_TO_BP33.items():
        x, y, c = float(kp[yi,0]), float(kp[yi,1]), float(kp[yi,2])
        lms[bi] = {"id":bi,"x":x/max(width,1),"y":y/max(height,1),"z":0.0,"vis":c}
        if c > 0.3:
            detected = True
    return lms, detected


def _draw_yolo_overlay(vis, lms, width, height):
    pts = {}
    for lm in lms:
        if lm["vis"] > 0.3:
            pts[lm["id"]] = (int(lm["x"]*width), int(lm["y"]*height))
    for a,b in SKELETON_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(vis, pts[a], pts[b], (0,200,255), 2)
    for kid in KEY_IDS:
        if kid in pts:
            cv2.circle(vis, pts[kid], 8, (0,255,255), -1)


def extract_bronze(
    video_path,
    out_root="pipeline/bronze",
    exercise="deadlift",
    camera_view="front_oblique",
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    write_overlay=False,
):
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    session_id = make_session_id(exercise)
    out_dir = Path(out_root) / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    input_copy = out_dir / "input.mp4"
    shutil.copy2(video_path, input_copy)

    cap = cv2.VideoCapture(str(input_copy))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_copy}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)  or 0)

    yolo_model = _try_load_yolo()
    model_name = "yolov8n-pose" if yolo_model else f"mediapipe-blazepose-c{model_complexity}"

    meta = {
        "session_id": session_id, "exercise": exercise, "camera_view": camera_view,
        "input_video": str(input_copy), "source_video_path": video_path,
        "fps": float(fps), "width": width, "height": height, "num_frames": num_frames,
        "created_utc": utc_now_iso(), "model": model_name,
        "mediapipe": {"model_complexity": model_complexity,
                      "min_detection_confidence": min_detection_confidence,
                      "min_tracking_confidence": min_tracking_confidence},
    }
    (out_dir/"meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    out_jsonl    = out_dir / "keypoints.jsonl"
    overlay_path = out_dir / "overlay.mp4"
    writer = None

    if write_overlay:
        for fc in ("avc1","mp4v"):
            writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*fc),
                                     float(fps), (width,height))
            if writer.isOpened(): break
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter")

    t0 = time.time()
    frame_idx = detected_frames = 0

    # ── YOLOv8-pose path ─────────────────────────────────────────
    if yolo_model:
        with out_jsonl.open("w", encoding="utf-8") as f:
            while True:
                ok, frame = cap.read()
                if not ok: break
                results = yolo_model(frame, verbose=False)
                lms, pose_detected = _yolo_to_landmarks(results[0], width, height)
                if pose_detected: detected_frames += 1
                f.write(json.dumps({
                    "session_id": session_id, "frame_idx": frame_idx,
                    "t_sec": float(frame_idx/fps),
                    "pose_detected": bool(pose_detected), "landmarks": lms,
                }) + "\n")
                if writer:
                    vis = frame.copy()
                    if pose_detected: _draw_yolo_overlay(vis, lms, width, height)
                    writer.write(vis)
                frame_idx += 1

    # ── MediaPipe BlazePose fallback path ─────────────────────────
    else:
        import mediapipe as mp
        mp_pose  = mp.solutions.pose
        mp_draw  = mp.solutions.drawing_utils
        mp_style = mp.solutions.drawing_styles

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,   # 2 = highest accuracy
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ) as pose, out_jsonl.open("w", encoding="utf-8") as f:
            while True:
                ok, frame = cap.read()
                if not ok: break
                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pose_detected = res.pose_landmarks is not None
                if pose_detected:
                    detected_frames += 1
                    lms = [{"id":i,"x":float(lm.x),"y":float(lm.y),
                             "z":float(lm.z),"vis":float(lm.visibility)}
                            for i,lm in enumerate(res.pose_landmarks.landmark)]
                else:
                    lms = []
                f.write(json.dumps({
                    "session_id": session_id, "frame_idx": frame_idx,
                    "t_sec": float(frame_idx/fps),
                    "pose_detected": bool(pose_detected), "landmarks": lms,
                }) + "\n")
                if writer:
                    vis = frame.copy()
                    if res.pose_landmarks:
                        mp_draw.draw_landmarks(vis, res.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_style.get_default_pose_landmarks_style())
                        h,w = vis.shape[:2]
                        for idx in KEY_IDS:
                            lm = res.pose_landmarks.landmark[idx]
                            cv2.circle(vis,(int(lm.x*w),int(lm.y*h)),8,(0,255,255),-1)
                    writer.write(vis)
                frame_idx += 1

    cap.release()
    if writer: writer.release()

    summary = {
        "session_id": session_id, "model_used": model_name,
        "frames_processed": frame_idx, "pose_detected_frames": detected_frames,
        "pose_detected_ratio": float(detected_frames/max(frame_idx,1)),
        "elapsed_sec": float(time.time()-t0),
        "outputs": {
            "session_dir": str(out_dir), "meta": str(out_dir/"meta.json"),
            "input_copy": str(input_copy), "keypoints_jsonl": str(out_jsonl),
            "overlay_mp4": str(overlay_path) if write_overlay else None,
        },
    }
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    return session_id


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",            required=True)
    ap.add_argument("--exercise",         default="deadlift", choices=["deadlift","squat"])
    ap.add_argument("--out_root",         default="pipeline/bronze")
    ap.add_argument("--camera_view",      default="front_oblique")
    ap.add_argument("--model_complexity", type=int,   default=2)
    ap.add_argument("--min_det",          type=float, default=0.5)
    ap.add_argument("--min_track",        type=float, default=0.5)
    ap.add_argument("--overlay",          action="store_true")
    args = ap.parse_args()
    extract_bronze(
        video_path=args.video, out_root=args.out_root,
        exercise=args.exercise, camera_view=args.camera_view,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_track,
        write_overlay=args.overlay,
    )
