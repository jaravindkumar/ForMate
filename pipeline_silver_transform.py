"""
pipeline_silver_transform.py — robust rep segmentation
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

POSE_IDS = {
    "l_shoulder": 11, "r_shoulder": 12,
    "l_hip":      23, "r_hip":      24,
    "l_knee":     25, "r_knee":     26,
    "l_ankle":    27, "r_ankle":    28,
    "l_wrist":    15, "r_wrist":    16,
}

HINGE_EXERCISES = {
    "deadlift", "romanian_deadlift", "dumbbell_deadlift",
    "bent_over_row", "single_arm_row", "dumbbell_swing"
}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def medfilt(x, k=7):
    return pd.Series(x).rolling(k, center=True, min_periods=1).median().to_numpy()


def find_extrema(sig, min_sep, threshold, mode="max"):
    found, last = [], -10**9
    for i in range(1, len(sig) - 1):
        if np.isnan(sig[i]) or i - last < min_sep:
            continue
        if mode == "max" and sig[i] >= sig[i-1] and sig[i] >= sig[i+1] and sig[i] >= threshold:
            found.append(i); last = i
        elif mode == "min" and sig[i] <= sig[i-1] and sig[i] <= sig[i+1] and sig[i] <= threshold:
            found.append(i); last = i
    return found


def segment_reps(df, exercise):
    df = df.copy()

    if exercise in HINGE_EXERCISES:
        raw = (df["l_shoulder_y_n"].values + df["r_shoulder_y_n"].values) / 2.0
        sig = medfilt(-raw, k=9)
        phase_up, phase_down = "pull", "descent"
    else:
        raw = (df["l_hip_y_n"].values + df["r_hip_y_n"].values) / 2.0
        sig = medfilt(raw, k=9)
        phase_up, phase_down = "descent", "ascent"

    diffs = np.diff(df["t_sec"].values)
    valid_diffs = diffs[diffs > 0]
    dt  = float(np.median(valid_diffs)) if len(valid_diffs) else 1/30
    fps = min(max(1/dt, 5), 120)

    p10, p90 = np.nanpercentile(sig, 10), np.nanpercentile(sig, 90)
    sig_range = p90 - p10
    sig_mid   = (p10 + p90) / 2

    df["seg_signal"] = sig

    if sig_range < 0.02:
        print(f"[silver] range {sig_range:.4f} too small")
        df["rep_id"] = -1; df["phase"] = "unknown"
        return df

    min_sep    = max(3, int(0.55 * fps))
    peak_thr   = sig_mid + sig_range * 0.20
    valley_thr = sig_mid - sig_range * 0.20

    peaks   = find_extrema(sig, min_sep, peak_thr,   "max")
    valleys = find_extrema(sig, min_sep, valley_thr, "min")
    print(f"[silver] fps={fps:.1f} range={sig_range:.4f} peaks={len(peaks)} valleys={len(valleys)}")

    rep_id = np.full(len(df), -1, dtype=int)
    phase  = np.full(len(df), "unknown", dtype=object)

    if not peaks or not valleys:
        df["rep_id"] = rep_id; df["phase"] = phase
        return df

    events = sorted([(i,"valley") for i in valleys] + [(i,"peak") for i in peaks])
    rc = 0
    i  = 0
    while i < len(events):
        if events[i][1] != "valley":
            i += 1; continue
        pk = next((j for j in range(i+1, len(events)) if events[j][1]=="peak"), None)
        if pk is None: break
        vl = next((j for j in range(pk+1, len(events)) if events[j][1]=="valley"), None)
        if vl is None: break
        a, p, c = events[i][0], events[pk][0], events[vl][0]
        rep_id[a:c+1] = rc
        for k in range(a, c+1):
            if exercise in HINGE_EXERCISES:
                phase[k] = "setup" if k < a + max(2, int(0.1*(c-a))) else ("pull" if k <= p else "descent")
            else:
                phase[k] = phase_up if k <= p else phase_down
        rc += 1
        i = vl

    print(f"[silver] segmented {rc} reps")
    df["rep_id"] = rep_id
    df["phase"]  = phase
    return df


def run_silver(session_dir):
    session_dir = Path(session_dir)
    meta        = json.loads((session_dir/"meta.json").read_text())
    session_id  = meta["session_id"]
    exercise    = meta.get("exercise", "deadlift")

    rows = []
    for rec in read_jsonl(session_dir/"keypoints.jsonl"):
        if not rec.get("pose_detected", False): continue
        lms = {lm["id"]: lm for lm in rec.get("landmarks", [])}
        row = {"frame_idx": rec["frame_idx"], "t_sec": rec["t_sec"]}
        for name, idx in POSE_IDS.items():
            lm = lms.get(idx)
            if lm and lm.get("vis", 0) > 0.05:
                row[f"{name}_x"] = float(lm["x"])
                row[f"{name}_y"] = float(lm["y"])
                row[f"{name}_vis"] = float(lm.get("vis", 0))
            else:
                row[f"{name}_x"] = row[f"{name}_y"] = np.nan
                row[f"{name}_vis"] = 0.0
        rows.append(row)

    if not rows:
        raise RuntimeError("No pose-detected frames in keypoints.jsonl")

    df = pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)
    for name in POSE_IDS:
        df[f"{name}_x"] = medfilt(df[f"{name}_x"].values)
        df[f"{name}_y"] = medfilt(df[f"{name}_y"].values)
        df[f"{name}_x_n"] = df[f"{name}_x"]
        df[f"{name}_y_n"] = df[f"{name}_y"]

    df = segment_reps(df, exercise)

    valid = df[df["rep_id"] >= 0]
    reps  = (valid.groupby("rep_id")
             .agg(t_start=("t_sec","min"), t_end=("t_sec","max"), frames=("frame_idx","count"))
             .reset_index()) if not valid.empty else pd.DataFrame(columns=["rep_id","t_start","t_end","frames"])

    silver_dir = Path("pipeline")/"silver"/session_id
    silver_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(silver_dir/"clean.csv", index=False)
    reps.to_csv(silver_dir/"reps.csv", index=False)

    num_reps = int(reps["rep_id"].nunique()) if not reps.empty else 0
    summary  = {
        "session_id": session_id, "silver_dir": str(silver_dir),
        "outputs": {"clean": str(silver_dir/"clean.csv"), "reps": str(silver_dir/"reps.csv")},
        "num_reps_detected": num_reps,
    }
    (silver_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"reps": num_reps, "exercise": exercise}))
    return summary
