import os
import json
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_session_id(exercise: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{exercise}"


def extract_bronze(
    video_path: str,
    out_root: str = "pipeline/bronze",
    exercise: str = "deadlift",
    camera_view: str = "front_oblique",
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    write_overlay: bool = False,
) -> str:
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

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    meta = {
        "session_id": session_id,
        "exercise": exercise,
        "camera_view": camera_view,
        "input_video": str(input_copy),
        "source_video_path": video_path,
        "fps": float(fps),
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "created_utc": utc_now_iso(),
        "mediapipe": {
            "model_complexity": model_complexity,
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    out_jsonl = out_dir / "keypoints.jsonl"

    overlay_path = out_dir / "overlay.mp4"
    writer = None
    if write_overlay:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(overlay_path), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            # Fallback to mp4v
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(overlay_path), fourcc, float(fps), (width, height))
            if not writer.isOpened():
                raise RuntimeError("Failed to open VideoWriter for overlay.mp4")

    t0 = time.time()
    frame_idx = 0
    detected_frames = 0

    key_ids = [11, 12, 23, 24, 25, 26, 27, 28, 15, 16]

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose, out_jsonl.open("w", encoding="utf-8") as f:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            pose_detected = res.pose_landmarks is not None
            if pose_detected:
                detected_frames += 1
                lms = [
                    {"id": i, "x": float(lm.x), "y": float(lm.y), "z": float(lm.z), "vis": float(lm.visibility)}
                    for i, lm in enumerate(res.pose_landmarks.landmark)
                ]
            else:
                lms = []

            rec = {
                "session_id": session_id,
                "frame_idx": frame_idx,
                "t_sec": float(frame_idx / fps),
                "pose_detected": bool(pose_detected),
                "landmarks": lms,
            }
            f.write(json.dumps(rec) + "\n")

            if writer is not None:
                vis = frame.copy()
                if res.pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        vis,
                        res.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                    h, w = vis.shape[:2]
                    for idx in key_ids:
                        lm = res.pose_landmarks.landmark[idx]
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(vis, (x, y), 8, (0, 255, 255), -1)
                        # Removed joint number text overlay

                writer.write(vis)

            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.time() - t0
    summary = {
        "session_id": session_id,
        "frames_processed": frame_idx,
        "pose_detected_frames": detected_frames,
        "pose_detected_ratio": float(detected_frames / max(frame_idx, 1)),
        "elapsed_sec": float(elapsed),
        "outputs": {
            "session_dir": str(out_dir),
            "meta": str(out_dir / "meta.json"),
            "input_copy": str(input_copy),
            "keypoints_jsonl": str(out_jsonl),
            "overlay_mp4": str(overlay_path) if write_overlay else None,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return session_id


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--exercise", default="deadlift", choices=["deadlift", "squat"])
    ap.add_argument("--out_root", default="pipeline/bronze")
    ap.add_argument("--camera_view", default="front_oblique")
    ap.add_argument("--model_complexity", type=int, default=1)
    ap.add_argument("--min_det", type=float, default=0.5)
    ap.add_argument("--min_track", type=float, default=0.5)
    ap.add_argument("--overlay", action="store_true")
    args = ap.parse_args()

    extract_bronze(
        video_path=args.video,
        out_root=args.out_root,
        exercise=args.exercise,
        camera_view=args.camera_view,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_track,
        write_overlay=args.overlay,
    )
