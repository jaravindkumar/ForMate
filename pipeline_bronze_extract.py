import os
import json
import time
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_session_id(exercise: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{exercise}"


# ── THRESHOLD CHECKS ─────────────────────────────────────────────
def _angle(ax, ay, bx, by, cx, cy):
    """Angle at B in triangle ABC (degrees)."""
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    dot  = v1x*v2x + v1y*v2y
    mag  = math.sqrt((v1x**2+v1y**2)*(v2x**2+v2y**2)) + 1e-9
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag))))


def check_thresholds(lms, exercise, w, h):
    """
    lms: list of mediapipe landmark objects (normalised 0-1).
    Returns dict: key -> ("ok"|"warn"|"bad", label)
    Uses pixel coords internally for consistent threshold values.
    """
    def px(i): return lms[i].x * w
    def py(i): return lms[i].y * h
    def vis(i): return lms[i].visibility

    flags = {}
    L = mp_pose.PoseLandmark

    if exercise == "deadlift":
        # 1. Back angle: shoulder-hip-knee
        if all(vis(i) > 0.4 for i in [L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_KNEE]):
            ang = _angle(px(L.LEFT_SHOULDER), py(L.LEFT_SHOULDER),
                         px(L.LEFT_HIP),      py(L.LEFT_HIP),
                         px(L.LEFT_KNEE),     py(L.LEFT_KNEE))
            if ang >= 145:      flags["back"]  = ("ok",   "Back OK")
            elif ang >= 115:    flags["back"]  = ("warn", "Back rounding")
            else:               flags["back"]  = ("bad",  "Severe back round!")

        # 2. Bar drift: shoulder should stay over hip
        if all(vis(i) > 0.4 for i in [L.LEFT_SHOULDER, L.LEFT_HIP]):
            drift = abs(px(L.LEFT_SHOULDER) - px(L.LEFT_HIP))
            if drift < w*0.08:      flags["drift"] = ("ok",   "Bar path OK")
            elif drift < w*0.15:    flags["drift"] = ("warn", "Bar drifting")
            else:                   flags["drift"] = ("bad",  "Bar too far!")

        # 3. Hip hinge depth
        if all(vis(i) > 0.4 for i in [L.LEFT_HIP, L.LEFT_KNEE]):
            diff = py(L.LEFT_KNEE) - py(L.LEFT_HIP)
            flags["hinge"] = ("ok", "Good hinge") if diff > h*0.04 else ("warn", "Hinge deeper")

    else:  # squat
        # 1. Knee cave: knee X vs hip X
        if all(vis(i) > 0.4 for i in [L.LEFT_HIP, L.LEFT_KNEE]):
            cave = px(L.LEFT_KNEE) - px(L.LEFT_HIP)
            if cave >= -w*0.02:     flags["knee"] = ("ok",   "Knees OK")
            elif cave >= -w*0.06:   flags["knee"] = ("warn", "Knee caving")
            else:                   flags["knee"] = ("bad",  "Knee cave!")

        # 2. Squat depth: hip Y vs knee Y
        if all(vis(i) > 0.4 for i in [L.LEFT_HIP, L.LEFT_KNEE]):
            depth = py(L.LEFT_HIP) - py(L.LEFT_KNEE)
            if depth >= h*0.02:     flags["depth"] = ("ok",   "Good depth")
            elif depth >= -h*0.05:  flags["depth"] = ("warn", "Go deeper")
            else:                   flags["depth"] = ("bad",  "Too shallow")

        # 3. Forward lean: shoulder over hip
        if all(vis(i) > 0.4 for i in [L.LEFT_SHOULDER, L.LEFT_HIP]):
            lean = abs(px(L.LEFT_SHOULDER) - px(L.LEFT_HIP))
            if lean < w*0.08:       flags["lean"] = ("ok",   "Upright OK")
            elif lean < w*0.15:     flags["lean"] = ("warn", "Leaning forward")
            else:                   flags["lean"] = ("bad",  "Too much lean!")

    return flags


# ── JOINT → COLOUR MAPPING ────────────────────────────────────────
KEY_JOINTS = {
    "back":  [11, 12, 23, 24],   # shoulders + hips
    "drift": [11, 12],            # shoulders
    "hinge": [23, 24],            # hips
    "knee":  [25, 26],            # knees
    "depth": [23, 24, 25, 26],   # hips + knees
    "lean":  [11, 12, 23, 24],   # shoulders + hips
}

COLOR_OK   = (0, 230, 80)     # green  BGR
COLOR_WARN = (0, 160, 245)    # orange BGR
COLOR_BAD  = (50,  50, 255)   # red    BGR
COLOR_CONN = (200, 200, 200)  # connection lines

CONNECTIONS = [
    (11,12),(11,13),(12,14),(13,15),(14,16),   # upper body
    (11,23),(12,24),(23,24),                    # torso
    (23,25),(24,26),(25,27),(26,28),            # legs
]


def joint_color(idx, flags):
    worst = "ok"
    for key, joints in KEY_JOINTS.items():
        if key not in flags: continue
        if idx in joints:
            st = flags[key][0]
            if st == "bad":
                return COLOR_BAD
            if st == "warn" and worst == "ok":
                worst = "warn"
    return COLOR_WARN if worst == "warn" else COLOR_OK


def draw_threshold_skeleton(frame, lms, flags, w, h):
    """Draw colour-coded skeleton + flag labels onto frame (BGR in-place)."""

    # Connection colour = worst global status
    statuses = [f[0] for f in flags.values()]
    if "bad"  in statuses: conn_col = COLOR_BAD
    elif "warn" in statuses: conn_col = COLOR_WARN
    else:                   conn_col = COLOR_OK

    # Draw connections
    for a, b in CONNECTIONS:
        lm_a, lm_b = lms[a], lms[b]
        if lm_a.visibility < 0.35 or lm_b.visibility < 0.35: continue
        ax, ay = int(lm_a.x * w), int(lm_a.y * h)
        bx, by = int(lm_b.x * w), int(lm_b.y * h)
        cv2.line(frame, (ax, ay), (bx, by), conn_col, 2, cv2.LINE_AA)

    # Draw joints
    for i, lm in enumerate(lms):
        if lm.visibility < 0.35: continue
        px_x, px_y = int(lm.x * w), int(lm.y * h)
        col = joint_color(i, flags)
        cv2.circle(frame, (px_x, px_y), 6, col,       -1, cv2.LINE_AA)
        cv2.circle(frame, (px_x, px_y), 6, (20,20,20), 1, cv2.LINE_AA)

    # Draw flag labels — only warn/bad
    y_off = 32
    for key, (st, label) in flags.items():
        if st == "ok": continue
        col = COLOR_BAD if st == "bad" else COLOR_WARN
        # Dark shadow for readability
        cv2.putText(frame, label, (14, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (14, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col,     1, cv2.LINE_AA)
        y_off += 26

    return frame


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
                    lms    = res.pose_landmarks.landmark
                    flags  = check_thresholds(lms, exercise, width, height)
                    draw_threshold_skeleton(vis, lms, flags, width, height)
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
