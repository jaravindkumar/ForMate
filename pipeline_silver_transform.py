import json
from pathlib import Path
import numpy as np
import pandas as pd


# --------- utilities ---------

POSE_IDS = {
    "l_shoulder": 11, "r_shoulder": 12,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
    "l_wrist": 15, "r_wrist": 16,
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_medfilt(x: np.ndarray, k: int = 5) -> np.ndarray:
    # simple rolling median using pandas to avoid scipy dependency
    s = pd.Series(x)
    return s.rolling(k, center=True, min_periods=1).median().to_numpy()


def segment_reps(df: pd.DataFrame, exercise: str) -> pd.DataFrame:
    """
    Unified rep segmenter using a state machine on smoothed hip Y signal.

    MediaPipe Y coords: 0 = top of image, 1 = bottom.
    - Deadlift: standing (hip Y LOW) -> hinge down (hip Y HIGH) -> lockout (hip Y LOW)
                Signal rises at the start of a pull, falls at lockout.
    - Squat:    standing (hip Y LOW) -> squat down (hip Y HIGH) -> stand up (hip Y LOW)
                Same signal shape — hip drops, then rises back.

    State machine per rep:
      IDLE -> DESCENDING (hip drops > threshold from baseline)
           -> AT_BOTTOM  (hip stopped dropping, near peak)
           -> ASCENDING  (hip rising back up)
           -> COMPLETE   (hip returned within threshold of baseline) -> new rep

    This is far more robust than peak/trough detection because:
    - It uses a dynamic baseline (rolling window of recent "standing" frames)
    - Thresholds are adaptive (% of person's standing hip height)
    - It ignores noise/wobble that doesn't reach the descent threshold
    """
    df = df.copy()

    # ── Signal: mean hip Y, smoothed ──────────────────────────────
    hip_y_raw = (df["l_hip_y"].ffill().bfill() +
                 df["r_hip_y"].ffill().bfill()) / 2.0

    # Smooth with a longer window to remove MediaPipe jitter
    sig = hip_y_raw.rolling(9, center=True, min_periods=1).mean().to_numpy()

    n   = len(sig)
    dt  = float(np.median(np.diff(df["t_sec"].to_numpy()))) if n > 5 else 1.0/30.0
    fps = 1.0 / dt if dt > 0 else 30.0

    # ── Adaptive thresholds ────────────────────────────────────────
    # Standing baseline = median of lowest 30% of hip Y values
    standing_y  = np.nanpercentile(sig, 25)   # hip high on screen = low Y
    bottom_y    = np.nanpercentile(sig, 75)   # hip low on screen  = high Y
    total_range = bottom_y - standing_y

    if total_range < 0.03:
        # Not enough movement detected — return empty
        df["rep_id"]     = -1
        df["phase"]      = "unknown"
        df["seg_signal"] = sig
        df["peak"]       = False
        df["trough"]     = False
        return df

    # Descent threshold: hip must drop at least 35% of total range
    descent_thr = standing_y + total_range * 0.35
    # Bottom threshold: hip must reach at least 65% of total range
    bottom_thr  = standing_y + total_range * 0.65
    # Return threshold: hip must return within 30% of standing
    return_thr  = standing_y + total_range * 0.30

    # Min rep duration: 0.8 seconds
    min_rep_frames = max(int(0.8 * fps), 8)

    # ── State machine ──────────────────────────────────────────────
    IDLE, DESCENDING, AT_BOTTOM, ASCENDING = 0, 1, 2, 3

    state       = IDLE
    rep_start   = 0
    rep_peak_i  = 0
    rep_counter = 0

    rep_id  = np.full(n, -1, dtype=int)
    phase   = np.array(["unknown"] * n, dtype=object)
    peaks   = []
    troughs = []

    for i in range(n):
        v = sig[i]
        if np.isnan(v):
            continue

        if state == IDLE:
            # Wait for meaningful descent
            if v > descent_thr:
                state     = DESCENDING
                rep_start = i
                rep_peak_i = i

        elif state == DESCENDING:
            if v > sig[rep_peak_i]: rep_peak_i = i  # track deepest point

            if v >= bottom_thr:
                # Reached sufficient depth
                state = AT_BOTTOM
                peaks.append(rep_peak_i)

            elif v < return_thr and (i - rep_start) < min_rep_frames:
                # Came back up too fast without reaching depth — false start
                state = IDLE

        elif state == AT_BOTTOM:
            if v > sig[rep_peak_i]: rep_peak_i = i
            # Wait for ascent
            if v < descent_thr:
                state = ASCENDING

        elif state == ASCENDING:
            if v <= return_thr and (i - rep_start) >= min_rep_frames:
                # Rep complete — assign labels
                rep_end = i
                troughs.append(rep_start)
                troughs.append(rep_end)

                for j in range(rep_start, rep_end + 1):
                    rep_id[j] = rep_counter
                    if j < rep_peak_i:
                        phase[j] = "descent" if exercise == "squat" else "pull"
                    else:
                        phase[j] = "ascent"  if exercise == "squat" else "descent"

                rep_counter += 1
                state = IDLE
                # Brief cooldown
                i_skip = min(i + int(0.3 * fps), n - 1)
                rep_start = i_skip

            elif v > sig[rep_peak_i]:
                # Still going deeper — back to at_bottom
                rep_peak_i = i
                state = AT_BOTTOM

    df["rep_id"]     = rep_id
    df["phase"]      = phase
    df["seg_signal"] = sig

    df["peak"]   = False
    df["trough"] = False
    if peaks:
        valid_peaks = [p for p in peaks if 0 <= p < n]
        if valid_peaks:
            df.loc[df.index[valid_peaks], "peak"] = True
    if troughs:
        valid_troughs = [t for t in troughs if 0 <= t < n]
        if valid_troughs:
            df.loc[df.index[valid_troughs], "trough"] = True

    return df


# --------- main pipeline ---------

def run_silver(session_dir: str) -> dict:
    session_dir = Path(session_dir)
    meta_path = session_dir / "meta.json"
    keypoints_path = session_dir / "keypoints.jsonl"
    if not meta_path.exists() or not keypoints_path.exists():
        raise FileNotFoundError(f"Missing bronze outputs in: {session_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    session_id = meta["session_id"]
    exercise = meta.get("exercise", "deadlift")
    fps = float(meta.get("fps", 30.0))

    rows = []
    for rec in read_jsonl(keypoints_path):
        if not rec.get("pose_detected", False):
            continue
        lms = {lm["id"]: lm for lm in rec.get("landmarks", [])}
        row = {"frame_idx": rec["frame_idx"], "t_sec": rec["t_sec"]}

        # store x,y,vis for key joints (normalised coords from mediapipe)
        for name, idx in POSE_IDS.items():
            lm = lms.get(idx)
            if lm is None:
                row[f"{name}_x"] = np.nan
                row[f"{name}_y"] = np.nan
                row[f"{name}_vis"] = 0.0
            else:
                row[f"{name}_x"] = float(lm["x"])
                row[f"{name}_y"] = float(lm["y"])
                row[f"{name}_vis"] = float(lm.get("vis", 0.0))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No pose-detected frames in keypoints.jsonl")

    # simple smoothing
    for name in POSE_IDS.keys():
        df[f"{name}_x"] = safe_medfilt(df[f"{name}_x"].to_numpy(), k=5)
        df[f"{name}_y"] = safe_medfilt(df[f"{name}_y"].to_numpy(), k=5)

    # create left/right convenience columns in the format expected by Gold
    df["l_shoulder_x_n"] = df["l_shoulder_x"]
    df["l_shoulder_y_n"] = df["l_shoulder_y"]
    df["r_shoulder_x_n"] = df["r_shoulder_x"]
    df["r_shoulder_y_n"] = df["r_shoulder_y"]

    df["l_hip_x_n"] = df["l_hip_x"]
    df["l_hip_y_n"] = df["l_hip_y"]
    df["r_hip_x_n"] = df["r_hip_x"]
    df["r_hip_y_n"] = df["r_hip_y"]

    df["l_knee_x_n"] = df["l_knee_x"]
    df["l_knee_y_n"] = df["l_knee_y"]
    df["r_knee_x_n"] = df["r_knee_x"]
    df["r_knee_y_n"] = df["r_knee_y"]

    df["l_ankle_x_n"] = df["l_ankle_x"]
    df["l_ankle_y_n"] = df["l_ankle_y"]
    df["r_ankle_x_n"] = df["r_ankle_x"]
    df["r_ankle_y_n"] = df["r_ankle_y"]

    df["l_wrist_x_n"] = df["l_wrist_x"]
    df["l_wrist_y_n"] = df["l_wrist_y"]
    df["r_wrist_x_n"] = df["r_wrist_x"]
    df["r_wrist_y_n"] = df["r_wrist_y"]

    # Segment reps
    df = segment_reps(df, exercise)

    # reps table
    reps = (
        df[df["rep_id"] >= 0]
        .groupby("rep_id")
        .agg(t_start=("t_sec", "min"), t_end=("t_sec", "max"), frames=("frame_idx", "count"))
        .reset_index()
    )

    silver_dir = Path("pipeline") / "silver" / session_id
    silver_dir.mkdir(parents=True, exist_ok=True)

    clean_path = silver_dir / "clean.parquet"
    reps_path = silver_dir / "reps.parquet"
    df.to_parquet(clean_path, index=False)
    reps.to_parquet(reps_path, index=False)

    summary = {
        "session_id": session_id,
        "silver_dir": str(silver_dir),
        "outputs": {"clean": str(clean_path), "reps": str(reps_path)},
        "num_reps_detected": int(reps["rep_id"].nunique()) if not reps.empty else 0,
    }
    (silver_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--session_dir", required=True)
    args = ap.parse_args()
    run_silver(args.session_dir)
