import json
import time
import tempfile
from pathlib import Path
import subprocess
import sys

import streamlit as st
import pandas as pd
import cv2


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


# ------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "tmp_video_path" not in st.session_state:
    st.session_state.tmp_video_path = None
if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def run_cmd(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def latest_session_dir(root: Path, after_ts: float) -> Path:
    if not root.exists():
        raise RuntimeError(f"Directory does not exist: {root}")

    candidates = []
    for p in root.iterdir():
        if p.is_dir() and p.stat().st_mtime >= after_ts - 1.0:
            candidates.append((p.stat().st_mtime, p))

    if candidates:
        return sorted(candidates, key=lambda x: x[0])[-1][1]

    all_dirs = [(p.stat().st_mtime, p) for p in root.iterdir() if p.is_dir()]
    if not all_dirs:
        raise RuntimeError("No session directories found.")
    return sorted(all_dirs, key=lambda x: x[0])[-1][1]


def coach_summary(summary: dict) -> dict:
    s = summary.get("scores", {})
    flags = summary.get("flags", [])

    positives = []
    improvements = []
    next_steps = []

    if s.get("hinge_quality", 0) >= 90:
        positives.append("Strong hip hinge. You are using the right muscles for a deadlift.")
    if s.get("symmetry", 0) >= 90:
        positives.append("Good left to right balance.")
    if s.get("trunk_control", 0) >= 90:
        positives.append("Solid torso control. Nice bracing.")
    if s.get("setup_consistency", 0) >= 85:
        positives.append("Consistent setup across reps.")

    if s.get("tempo_consistency", 100) < 85:
        improvements.append("Your pull speed varies between reps. Aim for a steadier pull each rep.")
    if s.get("setup_consistency", 100) < 80:
        improvements.append("Your setup position changes across reps. Reset feet and brace the same way each time.")

    for f in flags[:2]:
        improvements.append(f.get("message", "Form issue detected."))

    if s.get("tempo_consistency", 100) < 85:
        next_steps.append("Try a 1 second pause at the bottom, then pull smoothly with the same speed each rep.")
    next_steps.append("Keep the bar close and brace hard before each pull.")

    if not positives:
        positives = ["Good effort. The system tracked you reliably."]
    if not improvements:
        improvements = ["No major issues detected. Keep it up."]

    return {
        "headline": f"Overall score: {s.get('overall', 0):.1f}/100",
        "positives": positives[:2],
        "improvements": improvements[:2],
        "next_steps": next_steps[:2],
    }


def generate_llm_report(exercise, gold_summary, issues, gold_dir):
    """Generate LLM coaching report. Returns the report text."""
    try:
        import os
        import requests

        openai_api_key = os.getenv("OPENAI_API_KEY")
        hf_api_key = os.getenv("HF_API_KEY")
        together_api_key = os.getenv("TOGETHER_API_KEY")

        issue_text = ', '.join([i['type'] for i in issues]) if issues else 'None detected'

        prompt = f"""Generate a concise one-page coaching report for a {exercise} session.

Session summary:
- Overall score: {gold_summary['scores']['overall']:.1f}/100
- Reps: {gold_summary['reps']}
- Key issues: {issue_text}

Provide personalized advice on how to improve form, focusing on the identified issues. Keep it encouraging and actionable."""

        if openai_api_key:
            import openai
            client = openai.OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content

        elif hf_api_key:
            headers = {"Authorization": f"Bearer {hf_api_key}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            response = requests.post(
                "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
                headers=headers,
                json=payload
            )
            if response.status_code == 200:
                return response.json()[0]["generated_text"]
            else:
                raise Exception(f"Hugging Face API error: {response.status_code}")

        elif together_api_key:
            headers = {
                "Authorization": f"Bearer {together_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.7
            }
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Together AI API error: {response.status_code}")

        else:
            # Local Ollama (completely free)
            import openai
            client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            response = client.chat.completions.create(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content

    except Exception:
        # Fallback report when no LLM is available
        report = "AI Coaching Report (Generated without LLM)\n\n"
        report += f"Session Summary: {exercise.capitalize()} with {gold_summary['reps']} reps, overall score {gold_summary['scores']['overall']:.1f}/100.\n\n"
        if issues:
            report += "Key Issues Identified:\n"
            for issue in issues:
                report += f"- {issue['type'].replace('_', ' ').title()}: {issue['description']}\n"
        else:
            report += "No major form issues were detected.\n"
        report += "\nRecommendations:\n"
        report += "- Focus on maintaining proper form throughout the movement.\n"
        report += "- Practice with lighter weights to perfect technique.\n"
        report += "- Consider filming from different angles for better self-assessment.\n"
        report += "\nKeep up the good work and stay consistent!"
        return report


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

st.set_page_config(
    page_title="ForMate - Exercise Form Analysis",
    layout="wide",
    page_icon="🏋️",
    initial_sidebar_state="expanded"
)

# Header
st.title("🏋️ ForMate - AI Exercise Form Analysis")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    exercise = st.selectbox("Exercise Type", ["deadlift", "squat"], index=0, help="Select the exercise you're analyzing")
    camera_view = st.selectbox("Camera Angle", ["front_oblique", "side"], index=0, help="Camera perspective used for filming")

    st.markdown("---")
    st.markdown("**Pipeline Overview:**")
    st.markdown("1. **Bronze**: Pose detection & overlay")
    st.markdown("2. **Silver**: Rep segmentation & cleaning")
    st.markdown("3. **Gold**: Metrics & AI coaching")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📹 Upload Video")
    uploaded = st.file_uploader(
        "Choose a video file (.mp4, .mov, .m4v)",
        type=["mp4", "mov", "m4v"],
        help="Upload your exercise video for analysis"
    )

    if uploaded:
        st.success(f"✅ {uploaded.name} uploaded successfully!")
        st.caption(f"Size: {uploaded.size / (1024*1024):.1f} MB")

with col2:
    if uploaded:
        # Save to temp file only once per upload (avoid re-creating on every rerun)
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.uploaded_file_id != file_id:
            st.session_state.uploaded_file_id = file_id
            st.session_state.analysis_results = None  # reset results for new file
            tmp_dir = Path(tempfile.mkdtemp())
            tmp_video = tmp_dir / uploaded.name
            tmp_video.write_bytes(uploaded.read())
            st.session_state.tmp_video_path = str(tmp_video)

        st.subheader("🎬 Preview")
        st.video(st.session_state.tmp_video_path)
    else:
        # File was removed — reset all state
        st.session_state.uploaded_file_id = None
        st.session_state.tmp_video_path = None
        st.session_state.analysis_results = None
        st.info("👆 Upload a video to get started")

# Analysis section
if uploaded:

    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        st.markdown("---")

        tmp_video = Path(st.session_state.tmp_video_path)

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # -------------------------
        # BRONZE
        # -------------------------
        status_text.text("🔍 Step 1/3: Extracting pose data...")
        progress_bar.progress(10)

        st.info("Running Bronze (pose extraction + overlay)...")
        t0 = time.time()

        bronze_cmd = [
            PY, "pipeline_bronze_extract.py",
            "--video", str(tmp_video),
            "--exercise", exercise,
            "--camera_view", camera_view,
            "--overlay",
        ]
        p = run_cmd(bronze_cmd, cwd=ROOT)
        if p.returncode != 0:
            st.error("Bronze failed")
            st.code(p.stderr if p.stderr else p.stdout)
            st.stop()

        bronze_root = ROOT / "pipeline" / "bronze"
        bronze_session = latest_session_dir(bronze_root, after_ts=t0)
        bronze_summary = read_json(bronze_session / "summary.json")

        session_id = bronze_summary["session_id"]
        session_dir = bronze_summary["outputs"]["session_dir"]

        progress_bar.progress(40)
        status_text.text("✅ Pose extraction complete!")

        # -------------------------
        # SILVER
        # -------------------------
        status_text.text("🔄 Step 2/3: Analyzing movement patterns...")
        progress_bar.progress(50)

        st.info("Running Silver (cleaning + rep segmentation)...")

        silver_cmd = [
            PY, "pipeline_silver_transform.py",
            "--session_dir", session_dir,
        ]
        p = run_cmd(silver_cmd, cwd=ROOT)
        if p.returncode != 0:
            st.error("Silver failed")
            st.code(p.stderr if p.stderr else p.stdout)
            st.stop()

        silver_dir = ROOT / "pipeline" / "silver" / session_id
        silver_summary = read_json(silver_dir / "summary.json")
        num_reps = silver_summary["num_reps_detected"]

        progress_bar.progress(70)
        status_text.text("✅ Movement analysis complete!")

        # -------------------------
        # GOLD
        # -------------------------
        status_text.text("🧠 Step 3/3: Generating AI coaching report...")
        progress_bar.progress(80)

        st.info("Running Gold (metrics + scoring)...")

        gold_cmd = [
            PY, "pipeline_gold_score.py",
            "--session_id", session_id,
            "--exercise", exercise,
        ]
        p = run_cmd(gold_cmd, cwd=ROOT)
        if p.returncode != 0:
            st.error("Gold failed")
            st.code(p.stderr if p.stderr else p.stdout)
            st.stop()

        gold_dir = ROOT / "pipeline" / "gold" / session_id
        gold_summary = read_json(gold_dir / "summary.json")
        rep_df = pd.read_parquet(gold_dir / "metrics_reps.parquet")

        # -------------------------
        # Generate snapshots + LLM report (once, during analysis)
        # -------------------------
        issues = gold_summary.get("issues", [])
        snapshot_paths = {}

        if issues:
            bronze_session_dir = ROOT / "pipeline" / "bronze" / session_id
            bronze_meta = read_json(bronze_session_dir / "meta.json")
            video_path = bronze_meta["input_video"]

            snapshots_dir = gold_dir / "snapshots"
            snapshots_dir.mkdir(exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            for issue in issues:
                issue_type = issue["type"]
                for frame_idx in issue["frames"][:3]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        img_path = snapshots_dir / f"{issue_type}_{frame_idx}.jpg"
                        cv2.imwrite(str(img_path), frame)
                        snapshot_paths[f"{issue_type}_{frame_idx}"] = str(img_path)
            cap.release()

        llm_report = generate_llm_report(exercise, gold_summary, issues, gold_dir)
        (gold_dir / "llm_report.txt").write_text(llm_report, encoding="utf-8")

        progress_bar.progress(100)
        status_text.text("🎉 Analysis complete!")

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Store all results in session state so they persist across reruns
        st.session_state.analysis_results = {
            "session_id": session_id,
            "bronze_summary": bronze_summary,
            "gold_summary": gold_summary,
            "gold_dir": str(gold_dir),
            "num_reps": num_reps,
            "rep_df": rep_df,
            "issues": issues,
            "snapshot_paths": snapshot_paths,
            "llm_report": llm_report,
        }

        st.balloons()

    # -------------------------
    # RESULTS DISPLAY (reads from session state — survives reruns)
    # -------------------------
    if st.session_state.analysis_results is not None:
        r = st.session_state.analysis_results
        session_id = r["session_id"]
        bronze_summary = r["bronze_summary"]
        gold_summary = r["gold_summary"]
        gold_dir = Path(r["gold_dir"])
        num_reps = r["num_reps"]
        rep_df = r["rep_df"]
        issues = r["issues"]
        snapshot_paths = r["snapshot_paths"]
        llm_report = r["llm_report"]

        st.success("🎯 Analysis Complete!")
        st.markdown("---")

        # Top metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("🏆 Overall Score", f'{gold_summary["scores"]["overall"]:.1f}/100')
        with m2:
            st.metric("🔢 Reps Detected", gold_summary.get("reps", num_reps))
        with m3:
            session_label = session_id.split('_')[1][:6] if '_' in session_id else session_id[:6]
            st.metric("⏱️ Session ID", session_label)
        with m4:
            st.metric("🎯 Confidence", f'{gold_summary.get("confidence", 1.0):.0%}')

        st.markdown("---")

        # Main content columns
        left_col, right_col = st.columns([1, 1])

        with left_col:
            # Video section
            st.subheader("📹 Analysis Results")

            # Overlay video
            overlay_rel = bronze_summary["outputs"].get("overlay_mp4", None)
            overlay_path = Path(overlay_rel) if overlay_rel else None

            if overlay_path and overlay_path.exists():
                st.video(str(overlay_path))
                st.caption("Pose tracking overlay with key joint highlights")
            else:
                st.warning("⚠️ Overlay video not found")

            # Coach summary
            st.subheader("🎯 Coach Summary")
            cs = coach_summary(gold_summary)

            # Score headline
            st.markdown(f"### {cs['headline']}")

            # Positives
            if cs["positives"]:
                st.markdown("**✅ What you did well:**")
                for msg in cs["positives"]:
                    st.success(msg)

            # Improvements
            if cs["improvements"]:
                st.markdown("**🔧 Areas to improve:**")
                for msg in cs["improvements"]:
                    st.warning(msg)

            # Next steps
            if cs["next_steps"]:
                st.markdown("**🎯 Next set focus:**")
                for msg in cs["next_steps"]:
                    st.info(msg)

        with right_col:
            # AI Coaching Report
            st.subheader("🤖 AI Coaching Report")
            st.markdown(llm_report)

            # Form issue snapshots
            if issues:
                st.subheader("📸 Form Issue Snapshots")
                for issue in issues:
                    with st.expander(f"🔍 {issue['type'].replace('_', ' ').title()}", expanded=True):
                        st.write(issue['description'])
                        cols = st.columns(min(len(issue["frames"]), 3))
                        for i, frame_idx in enumerate(issue["frames"][:3]):
                            key = f"{issue['type']}_{frame_idx}"
                            if key in snapshot_paths:
                                img_path = Path(snapshot_paths[key])
                                if img_path.exists():
                                    cols[i].image(str(img_path), caption=f"Frame {frame_idx}", width=200)

            # Detailed metrics
            st.subheader("📊 Detailed Metrics")

            # Score breakdown
            with st.expander("Score Breakdown", expanded=False):
                st.json(gold_summary["scores"])

            # Per rep metrics
            with st.expander("Per-Rep Metrics", expanded=False):
                st.dataframe(rep_df, use_container_width=True)

            # Flags
            if gold_summary.get("flags"):
                st.subheader("⚠️ Form Flags")
                for f in gold_summary["flags"]:
                    st.warning(f'**{f.get("severity","").upper()}**: {f.get("message","")}')

        # Footer
        st.markdown("---")
        st.caption(f"Session: {session_id} | Output directory: {gold_dir}")

else:
    # Landing page when no video uploaded
    st.markdown("---")
    st.markdown("""
    ## How it works:

    1. **Upload** your exercise video
    2. **Configure** exercise type and camera angle
    3. **Click "Run Analysis"** to process
    4. **Get** AI-powered coaching feedback with visual snapshots

    ## Features:

    - 🤖 **AI Coaching**: Personalized feedback powered by local LLM
    - 📹 **Visual Analysis**: Pose tracking with key joint highlights
    - 📊 **Detailed Metrics**: Comprehensive form scoring
    - 🔍 **Issue Snapshots**: Visual examples of form problems
    - 📱 **Responsive**: Works on desktop and mobile
    """)

    st.info("👆 Upload a video above to get started!")
