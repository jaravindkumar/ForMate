"""
pipeline_bronze_extract.py — MediaPipe pose extraction + overlay generation
"""
import os, json, time, shutil, urllib.request, subprocess, math
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_NAME = "pose_landmarker_lite.task"

# MediaPipe landmark indices
MP_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),   # arms (shoulder to wrist only)
    (11,23),(12,24),(23,24),                    # torso
    (23,25),(25,27),(24,26),(26,28),            # legs
    # feet omitted — noisy at distance
    # face landmarks (0-10) omitted
    # finger joints (17-22) omitted
]

# Colour palette
COL_OK    = (56, 189, 96)    # green  BGR
COL_WARN  = (60, 165, 248)   # amber  BGR
COL_BAD   = (68,  68, 239)   # red    BGR
COL_BONE  = (200, 200, 200)  # grey

def _find_model():
    here = Path(__file__).resolve().parent
    for p in [here/MODEL_NAME, here.parent/MODEL_NAME, Path.cwd()/MODEL_NAME, Path("/tmp")/MODEL_NAME]:
        if p.exists() and p.stat().st_size > 100_000:
            return p
    return None

def _download_model():
    dst = Path("/tmp") / MODEL_NAME
    if dst.exists() and dst.stat().st_size > 100_000:
        return dst
    print("[bronze] Downloading pose model…")
    urllib.request.urlretrieve(MODEL_URL, str(dst))
    return dst

def _get_model_path():
    return _find_model() or _download_model()

def _ensure_readable(video_path):
    path = str(video_path)
    def cv_ok(p):
        try:
            cap = cv2.VideoCapture(p, cv2.CAP_FFMPEG)
            ok, _ = cap.read(); cap.release(); return ok
        except: return False

    if cv_ok(path):
        return path

    print(f"[bronze] Converting {Path(path).name} via ffmpeg…")
    out = Path("/tmp") / "formate_converted.mp4"
    ret = subprocess.run([
        "ffmpeg","-y","-i",path,
        "-c:v","libx264","-preset","ultrafast","-pix_fmt","yuv420p","-an",
        str(out)], capture_output=True, timeout=300)
    if ret.returncode == 0 and cv_ok(str(out)):
        print(f"[bronze] Converted OK ({out.stat().st_size//1024} KB)")
        return str(out)

    diag = ret.stderr.decode()[-800:]
    raise RuntimeError(f"Cannot read video.\nffmpeg: {diag}")


# ── Per-frame form checks (returns list of (joint_idx, severity)) ──
def _check_form(lms, exercise, W, H):
    """
    Returns dict: landmark_index -> 'ok'|'warn'|'bad'
    Uses MediaPipe normalised coords (0-1).
    """
    if not lms:
        return {}

    def pt(idx):
        lm = lms.get(idx)
        if lm is None: return None
        return lm["x"], lm["y"], lm.get("vis", 0)

    def angle(ax,ay, bx,by, cx,cy):
        v1x,v1y = ax-bx, ay-by
        v2x,v2y = cx-bx, cy-by
        dot = v1x*v2x + v1y*v2y
        mag = (math.sqrt(v1x**2+v1y**2)+1e-9)*(math.sqrt(v2x**2+v2y**2)+1e-9)
        return math.degrees(math.acos(max(-1,min(1,dot/mag))))

    flags = {}  # landmark_idx -> severity

    def flag(idxs, sev):
        for i in idxs:
            flags[i] = sev

    ls = pt(11); rs = pt(12)
    lh = pt(23); rh = pt(24)
    lk = pt(25); rk = pt(26)
    la = pt(27); ra = pt(28)
    lw = pt(15); rw = pt(16)

    if exercise == "deadlift" or exercise in ("romanian_deadlift","dumbbell_deadlift"):
        # Back angle: shoulder-hip-knee — warn <120 bad <100
        if ls and lh and lk and ls[2]>.3 and lh[2]>.3 and lk[2]>.3:
            a = angle(ls[0],ls[1], lh[0],lh[1], lk[0],lk[1])
            sev = "ok" if a>=120 else ("warn" if a>=100 else "bad")
            flag([11,12,23,24], sev)
        # Knee cave: knees closer than feet
        if lk and rk and la and ra:
            knee_w = abs(lk[0]-rk[0])
            foot_w = abs(la[0]-ra[0])
            if knee_w < foot_w * 0.7:
                flag([25,26,27,28], "warn")

    elif exercise == "squat" or "squat" in exercise:
        # Knee over toe depth
        if lk and la and lk[2]>.3 and la[2]>.3:
            # knee forward of ankle by more than 30% of leg length
            leg_len = abs(lk[1] - la[1]) + 1e-9
            forward = lk[0] - la[0]
            ratio = forward / leg_len
            sev = "ok" if ratio < 0.25 else ("warn" if ratio < 0.45 else "bad")
            flag([25,26,27,28], sev)
        # Depth: hip below knee
        if lh and lk and lh[2]>.3 and lk[2]>.3:
            if lh[1] >= lk[1]:  # hip Y >= knee Y = parallel or below
                flag([23,24], "ok")
            else:
                flag([23,24], "warn")

    elif "press" in exercise or "shoulder" in exercise:
        # Wrist above elbow in press
        le = pt(13); re = pt(14)
        if lw and le and lw[2]>.3 and le[2]>.3:
            sev = "ok" if lw[1] < le[1] else "warn"
            flag([13,14,15,16], sev)

    elif "row" in exercise or "curl" in exercise:
        # Elbow tracking: elbow stays close to torso
        le = pt(13); re = pt(14)
        if le and ls and le[2]>.3 and ls[2]>.3:
            dist = abs(le[0] - ls[0])
            sev = "ok" if dist < 0.15 else ("warn" if dist < 0.25 else "bad")
            flag([13,14,15,16], sev)

    return flags


# ── Draw skeleton on frame ─────────────────────────────────────────
def _draw_skeleton(frame, lms, flags, W, H):
    if not lms:
        return frame
    frame = frame.copy()

    def px(idx):
        lm = lms.get(idx)
        if lm is None or lm.get("vis",0) < 0.2: return None
        return int(lm["x"]*W), int(lm["y"]*H)

    # Draw bones
    for a, b in MP_CONNECTIONS:
        pa, pb = px(a), px(b)
        if pa and pb:
            cv2.line(frame, pa, pb, COL_BONE, 2, cv2.LINE_AA)

    # Draw joints coloured by flag severity
    # Skip face (0-10) and finger/palm (17-22, 29-32) landmarks
    BODY_LANDMARKS = list(range(11, 17)) + list(range(23, 29))
    for idx in BODY_LANDMARKS:
        p = px(idx)
        if p is None: continue
        sev = flags.get(idx, "ok")
        col = COL_OK if sev == "ok" else (COL_WARN if sev == "warn" else COL_BAD)
        cv2.circle(frame, p, 6, col, -1, cv2.LINE_AA)
        cv2.circle(frame, p, 7, (0,0,0), 1, cv2.LINE_AA)

    return frame


# ── Main pose runner ───────────────────────────────────────────────
def _run_mediapipe(video_path, fps, model_path, out_jsonl, session_id,
                   exercise, overlay_path=None):
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
    from mediapipe.tasks.python.core.base_options import BaseOptions
    import mediapipe as mp

    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )

    cap   = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps_v = float(cap.get(cv2.CAP_PROP_FPS) or fps)

    # Video writer for overlay
    writer = None
    if overlay_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_path), fourcc, fps_v, (W, H))

    frame_idx = detected = 0
    with PoseLandmarker.create_from_options(opts) as lmk, open(out_jsonl, "w") as f:
        while True:
            ok, frame = cap.read()
            if not ok: break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(frame_idx * 1000 / max(fps_v, 1))
            res   = lmk.detect_for_video(img, ts_ms)

            found = bool(res.pose_landmarks)
            lms   = [{"id":i,"x":0.,"y":0.,"z":0.,"vis":0.} for i in range(33)]
            lms_d = {}

            if found:
                detected += 1
                for i, lm in enumerate(res.pose_landmarks[0]):
                    if i < 33:
                        lms[i]   = {"id":i,"x":float(lm.x),"y":float(lm.y),
                                    "z":float(lm.z),"vis":float(lm.visibility)}
                        lms_d[i] = lms[i]

            f.write(json.dumps({
                "session_id": session_id, "frame_idx": frame_idx,
                "t_sec": float(frame_idx/max(fps_v,1)),
                "pose_detected": found, "landmarks": lms
            }) + "\n")

            # Write overlay frame
            if writer:
                if found:
                    flags = _check_form(lms_d, exercise, W, H)
                    frame = _draw_skeleton(frame, lms_d, flags, W, H)
                    # Frame counter HUD
                    cv2.rectangle(frame, (8,8), (180,32), (0,0,0), -1)
                    cv2.putText(frame, f"Frame {frame_idx} | Pose OK", (12,26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                else:
                    cv2.rectangle(frame, (8,8), (200,32), (0,0,0), -1)
                    cv2.putText(frame, f"Frame {frame_idx} | No pose", (12,26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
                writer.write(frame)

            frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    return frame_idx, detected


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def make_session_id(exercise):
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + exercise


# ── Entry point ────────────────────────────────────────────────────
def extract_bronze(video_path, out_root="pipeline/bronze", exercise="deadlift",
                   camera_view="front_oblique", write_overlay=True, model_dir=None):

    video_path = str(os.path.abspath(video_path))
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    sz = Path(video_path).stat().st_size
    if sz < 1000:
        raise RuntimeError(f"Video too small ({sz} bytes)")
    print(f"[bronze] Input: {video_path} ({sz} bytes)")

    readable = _ensure_readable(video_path)

    cap     = cv2.VideoCapture(str(readable), cv2.CAP_FFMPEG)
    fps     = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)  or 0)
    cap.release()
    print(f"[bronze] {W}x{H} @ {fps:.1f}fps ~{nframes} frames")

    session_id = make_session_id(exercise)
    out_dir    = Path(out_root) / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    input_copy = out_dir / "input.mp4"
    shutil.copy2(readable, str(input_copy))

    out_jsonl    = out_dir / "keypoints.jsonl"
    overlay_path = (out_dir / "overlay.mp4") if write_overlay else None

    t0 = time.time()
    model_path = _get_model_path()
    frame_idx, detected = _run_mediapipe(
        str(input_copy), fps, model_path, out_jsonl, session_id,
        exercise, overlay_path=overlay_path
    )
    elapsed = time.time() - t0

    if frame_idx == 0:
        raise RuntimeError("No frames extracted from video.")

    # Convert overlay to H264 for browser compatibility
    final_overlay = None
    if overlay_path and overlay_path.exists() and overlay_path.stat().st_size > 1000:
        h264_overlay = out_dir / "overlay_h264.mp4"
        ret = subprocess.run([
            "ffmpeg","-y","-i", str(overlay_path),
            "-c:v","libx264","-preset","fast","-pix_fmt","yuv420p",
            str(h264_overlay)], capture_output=True, timeout=300)
        if ret.returncode == 0 and h264_overlay.exists():
            final_overlay = str(h264_overlay)
            overlay_path.unlink(missing_ok=True)
        else:
            final_overlay = str(overlay_path)
    
    print(f"[bronze] {frame_idx} frames, {detected} pose detected ({elapsed:.1f}s)")

    meta = {
        "session_id": session_id, "exercise": exercise, "camera_view": camera_view,
        "input_video": str(input_copy), "fps": fps, "width": W, "height": H,
        "num_frames": nframes, "created_utc": utc_now_iso(), "model": "mediapipe-lite"
    }
    (out_dir/"meta.json").write_text(json.dumps(meta, indent=2))

    summary = {
        "session_id": session_id, "model_used": "mediapipe-lite",
        "frames_processed": frame_idx, "pose_detected_frames": detected,
        "pose_detected_ratio": float(detected/max(frame_idx,1)),
        "elapsed_sec": float(elapsed),
        "outputs": {
            "session_dir": str(out_dir), "meta": str(out_dir/"meta.json"),
            "input_copy": str(input_copy), "keypoints_jsonl": str(out_jsonl),
            "overlay_mp4": final_overlay
        }
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    return session_id
