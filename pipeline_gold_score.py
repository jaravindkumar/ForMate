import json
from pathlib import Path

import numpy as np
import pandas as pd


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def score_from_std(std_val: float, good: float, bad: float) -> float:
    # lower std is better
    if std_val <= good:
        return 100.0
    if std_val >= bad:
        return 0.0
    t = (std_val - good) / (bad - good)
    return 100.0 * (1.0 - t)


def cv(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    m = np.mean(x)
    if m <= 1e-9:
        return 0.0
    return float(np.std(x) / m)


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    # angle between vectors in degrees
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    c = float(np.dot(a, b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def compute_rep_metrics(df: pd.DataFrame, reps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rep_id in sorted(reps["rep_id"].unique()):
        sub = df[df["rep_id"] == rep_id].copy()
        if sub.empty:
            continue

        pull = sub[sub["phase"] == "pull"]
        descent = sub[sub["phase"] == "descent"]

        t_pull = float(pull["t_sec"].max() - pull["t_sec"].min()) if not pull.empty else np.nan
        t_desc = float(descent["t_sec"].max() - descent["t_sec"].min()) if not descent.empty else np.nan

        # hinge proxy: compare hip movement vs knee movement during pull
        # using y (normalised). smaller y means higher on screen.
        if not pull.empty:
            hip_y0 = float(((pull["l_hip_y_n"] + pull["r_hip_y_n"]) / 2.0).iloc[0])
            hip_y1 = float(((pull["l_hip_y_n"] + pull["r_hip_y_n"]) / 2.0).iloc[-1])
            knee_y0 = float(((pull["l_knee_y_n"] + pull["r_knee_y_n"]) / 2.0).iloc[0])
            knee_y1 = float(((pull["l_knee_y_n"] + pull["r_knee_y_n"]) / 2.0).iloc[-1])

            d_hip = abs(hip_y1 - hip_y0)
            d_knee = abs(knee_y1 - knee_y0)
            hinge_ratio = float(d_hip / (d_knee + 1e-9))
        else:
            hinge_ratio = np.nan

        # trunk control proxy: angle change of shoulder-hip vector (2D)
        mid_sh = np.column_stack([
            (sub["l_shoulder_x_n"] + sub["r_shoulder_x_n"]) / 2.0,
            (sub["l_shoulder_y_n"] + sub["r_shoulder_y_n"]) / 2.0
        ])
        mid_hip = np.column_stack([
            (sub["l_hip_x_n"] + sub["r_hip_x_n"]) / 2.0,
            (sub["l_hip_y_n"] + sub["r_hip_y_n"]) / 2.0
        ])
        vec = mid_sh - mid_hip
        ref = np.array([0.0, -1.0])  # vertical up
        ang = np.array([angle_deg(v, ref) for v in vec])
        trunk_p95 = float(np.nanpercentile(ang, 95))

        # symmetry proxy
        shoulder_sym = float(np.nanmedian(np.abs(sub["l_shoulder_x_n"] - sub["r_shoulder_x_n"])))
        knee_sym = float(np.nanmedian(np.abs(sub["l_knee_x_n"] - sub["r_knee_x_n"])))

        rows.append({
            "rep_id": int(rep_id),
            "t_pull_s": t_pull,
            "t_descent_s": t_desc,
            "hinge_ratio": hinge_ratio,
            "trunk_p95_deg": trunk_p95,
            "shoulder_sym": shoulder_sym,
            "knee_sym": knee_sym,
        })

    return pd.DataFrame(rows)


def run_gold(session_id: str, exercise: str = "deadlift") -> dict:
    silver_dir = Path("pipeline") / "silver" / session_id
    clean_path = silver_dir / "clean.parquet"
    reps_path = silver_dir / "reps.parquet"

    if not clean_path.exists() or not reps_path.exists():
        raise FileNotFoundError(f"Missing Silver outputs in {silver_dir}")

    df = pd.read_parquet(clean_path)
    reps = pd.read_parquet(reps_path)

    # Add trunk angle
    mid_sh = np.column_stack([
        (df["l_shoulder_x_n"] + df["r_shoulder_x_n"]) / 2.0,
        (df["l_shoulder_y_n"] + df["r_shoulder_y_n"]) / 2.0
    ])
    mid_hip = np.column_stack([
        (df["l_hip_x_n"] + df["r_hip_x_n"]) / 2.0,
        (df["l_hip_y_n"] + df["r_hip_y_n"]) / 2.0
    ])
    vec = mid_sh - mid_hip
    ref = np.array([0.0, -1.0])  # vertical up
    df["trunk_angle_deg"] = np.array([angle_deg(v, ref) for v in vec])

    # Keep only valid reps
    reps_valid = reps.copy()
    reps_valid = reps_valid[reps_valid["frames"] >= 5].reset_index(drop=True)

    # Rep metrics table
    rep_metrics = compute_rep_metrics(df, reps_valid)

    # Aggregate metrics
    tempo_pull_med = float(np.nanmedian(rep_metrics["t_pull_s"])) if not rep_metrics.empty else np.nan
    tempo_desc_med = float(np.nanmedian(rep_metrics["t_descent_s"])) if not rep_metrics.empty else np.nan
    tempo_pull_cv = cv(rep_metrics["t_pull_s"].to_numpy()) if not rep_metrics.empty else 0.0

    hinge_ratio_med = float(np.nanmedian(rep_metrics["hinge_ratio"])) if not rep_metrics.empty else np.nan
    trunk_p95 = float(np.nanmedian(rep_metrics["trunk_p95_deg"])) if not rep_metrics.empty else np.nan
    shoulder_sym_med = float(np.nanmedian(rep_metrics["shoulder_sym"])) if not rep_metrics.empty else np.nan
    knee_sym_med = float(np.nanmedian(rep_metrics["knee_sym"])) if not rep_metrics.empty else np.nan

    # Setup consistency proxy: ankle x variation during setup frames
    setup = df[df["phase"] == "setup"]
    if setup.empty:
        setup_x_std = float(np.nanstd((df["l_ankle_x_n"] + df["r_ankle_x_n"]) / 2.0))
    else:
        setup_x_std = float(np.nanstd((setup["l_ankle_x_n"] + setup["r_ankle_x_n"]) / 2.0))

    # Scores (simple, explainable)
    setup_consistency = score_from_std(setup_x_std, good=0.01, bad=0.08)

    # hinge score from hinge ratio: > 1 is hip dominant, < 1 is knee dominant
    if not np.isfinite(hinge_ratio_med):
        hinge_quality = 50.0
    else:
        hinge_quality = 100.0 * clamp01((hinge_ratio_med - 0.6) / (1.2 - 0.6))  # maps 0.6->0, 1.2->1
        hinge_quality = float(max(0.0, min(100.0, hinge_quality)))

    # symmetry score: smaller diffs better
    sym_raw = 0.5 * shoulder_sym_med + 0.5 * knee_sym_med
    symmetry = score_from_std(sym_raw, good=0.01, bad=0.08)

    # trunk control score: smaller angle means more stable
    trunk_control = score_from_std(trunk_p95, good=10.0, bad=35.0) if np.isfinite(trunk_p95) else 50.0

    # tempo score: lower CV is better
    tempo_consistency = float(100.0 * clamp01((0.30 - tempo_pull_cv) / 0.30))

    overall = float(
        0.20 * setup_consistency
        + 0.25 * hinge_quality
        + 0.20 * symmetry
        + 0.20 * trunk_control
        + 0.15 * tempo_consistency
    )

    flags = []
    if hinge_quality < 60:
        flags.append({
            "code": "knee_dominant_pull",
            "severity": "amber",
            "message": "This looks knee dominant. Try pushing hips back more during setup and pull."
        })
    if trunk_control < 60:
        flags.append({
            "code": "trunk_instability",
            "severity": "amber",
            "message": "Your torso angle changes a lot. Focus on bracing and keeping your chest proud."
        })
    if tempo_consistency < 60:
        flags.append({
            "code": "tempo_variability",
            "severity": "amber",
            "message": "Your pull speed varies between reps. Aim for a steadier pull each rep."
        })

    # Identify problematic frames
    issues = []
    if trunk_control < 60 and np.isfinite(trunk_p95):
        # Frames where trunk angle > 25 deg
        bad_frames = df[df["trunk_angle_deg"] > 25]["frame_idx"].tolist()
        if bad_frames:
            issues.append({
                "type": "trunk_instability",
                "frames": bad_frames[:5],  # limit to 5 snapshots
                "description": "Frames where torso angle exceeds 25 degrees, indicating instability."
            })

    if hinge_quality < 60 and not rep_metrics.empty:
        # For reps with low hinge_ratio, find frames in pull phase
        bad_reps = rep_metrics[rep_metrics["hinge_ratio"] < 0.8]["rep_id"].tolist()
        bad_frames = []
        for rep_id in bad_reps[:2]:  # limit reps
            pull_frames = df[(df["rep_id"] == rep_id) & (df["phase"] == "pull")]["frame_idx"].tolist()
            bad_frames.extend(pull_frames[:3])  # 3 frames per rep
        if bad_frames:
            issues.append({
                "type": "knee_dominant_pull",
                "frames": bad_frames,
                "description": "Frames during pull phase of reps with knee-dominant movement."
            })

    gold_dir = Path("pipeline") / "gold" / session_id
    gold_dir.mkdir(parents=True, exist_ok=True)

    rep_metrics_path = gold_dir / "metrics_reps.parquet"
    rep_metrics.to_parquet(rep_metrics_path, index=False)

    summary = {
        "session_id": session_id,
        "exercise": exercise,
        "reps": int(reps_valid["rep_id"].nunique()) if not reps_valid.empty else 0,
        "tempo_pull_med_s": tempo_pull_med,
        "tempo_descent_med_s": tempo_desc_med,
        "tempo_pull_cv": tempo_pull_cv,
        "setup_x_std": setup_x_std,
        "trunk_lean_p95_deg": trunk_p95,
        "hinge_ratio_med": hinge_ratio_med,
        "shoulder_sym_med": shoulder_sym_med,
        "knee_sym_med": knee_sym_med,
        "scores": {
            "setup_consistency": setup_consistency,
            "hinge_quality": hinge_quality,
            "symmetry": symmetry,
            "trunk_control": trunk_control,
            "tempo_consistency": tempo_consistency,
            "overall": overall,
        },
        "flags": flags,
        "issues": issues,
        "confidence": 1.0,
    }

    (gold_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Helpful CLI print
    print(json.dumps({
        "gold_dir": str(gold_dir),
        "reps": summary["reps"],
        "overall_score": summary["scores"]["overall"],
        "num_flags": len(flags),
        "outputs": {
            "summary": str(gold_dir / "summary.json"),
            "metrics_reps": str(rep_metrics_path),
        }
    }, indent=2))

    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--session_id", required=True)
    ap.add_argument("--exercise", default="deadlift", choices=["deadlift", "squat"])
    args = ap.parse_args()

    run_gold(session_id=args.session_id, exercise=args.exercise)
