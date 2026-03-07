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


def segment_deadlift_reps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deadlift rep = bottom -> lockout -> bottom
    Segment using trough -> peak -> trough on a smoothed signal.
    """
    df = df.copy()

    mid_sh_y = (df["l_shoulder_y_n"] + df["r_shoulder_y_n"]) / 2.0
    sig = (-mid_sh_y).rolling(11, center=True, min_periods=1).mean()

    values = sig.to_numpy()
    dt = np.median(np.diff(df["t_sec"])) if len(df) > 5 else (1.0 / 30.0)
    fps = 1.0 / dt if dt > 0 else 30.0

    min_sep_trough = int(1.0 * fps)
    min_sep_peak = int(1.0 * fps)

    peak_thr = np.nanpercentile(values, 80)
    trough_thr = np.nanpercentile(values, 40)

    troughs = []
    last = -10**9
    for i in range(2, len(values) - 2):
        if i - last < min_sep_trough:
            continue
        v = values[i]
        if np.isnan(v):
            continue
        if v < values[i - 1] and v < values[i + 1] and v < trough_thr:
            troughs.append(i)
            last = i

    peaks = []
    last = -10**9
    for i in range(2, len(values) - 2):
        if i - last < min_sep_peak:
            continue
        v = values[i]
        if np.isnan(v):
            continue
        if v > values[i - 1] and v > values[i + 1] and v > peak_thr:
            peaks.append(i)
            last = i

    rep_id = np.full(len(df), -1, dtype=int)
    phase = np.array(["unknown"] * len(df), dtype=object)

    if len(troughs) < 2:
        df["rep_id"] = rep_id
        df["phase"] = phase
        df["seg_signal"] = sig
        df["peak"] = False
        df["trough"] = False
        return df

    rep_counter = 0
    for k in range(len(troughs) - 1):
        a, b = troughs[k], troughs[k + 1]
        inner_peaks = [p for p in peaks if a < p < b]
        if not inner_peaks:
            continue

        p = max(inner_peaks, key=lambda idx: values[idx])
        rep_id[a:b + 1] = rep_counter

        t_a = float(df.iloc[a]["t_sec"])
        t_setup_end = t_a + 0.2

        for idx in range(a, b + 1):
            t = float(df.iloc[idx]["t_sec"])
            if t <= t_setup_end:
                phase[idx] = "setup"
            elif idx <= p:
                phase[idx] = "pull"
            else:
                phase[idx] = "descent"

        rep_counter += 1

    df["rep_id"] = rep_id
    df["phase"] = phase
    df["seg_signal"] = sig

    df["peak"] = False
    df["trough"] = False
    if peaks:
        df.loc[df.index[peaks], "peak"] = True
    if troughs:
        df.loc[df.index[troughs], "trough"] = True

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
    if exercise == "deadlift":
        df = segment_deadlift_reps(df)
    else:
        # placeholder for squat segmentation later
        df["rep_id"] = -1
        df["phase"] = "unknown"
        df["seg_signal"] = 0.0
        df["peak"] = False
        df["trough"] = False

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
