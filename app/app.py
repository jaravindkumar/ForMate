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
        st.subheader("🎬 Preview")
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_video = tmp_dir / uploaded.name
        tmp_video.write_bytes(uploaded.read())
        st.video(str(tmp_video))
    else:
        st.info("👆 Upload a video to get started")

# Analysis section
if uploaded:

    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        st.markdown("---")

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

        progress_bar.progress(100)
        status_text.text("🎉 Analysis complete!")

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        st.success("🎯 Analysis Complete!")
        st.balloons()

        # -------------------------
        # RESULTS DISPLAY
        # -------------------------
        st.markdown("---")

        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏆 Overall Score", f'{gold_summary["scores"]["overall"]:.1f}/100')
        with col2:
            st.metric("🔢 Reps Detected", gold_summary.get("reps", num_reps))
        with col3:
            st.metric("⏱️ Session ID", session_id.split('_')[1][:6])
        with col4:
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
            # AI Report section
            issues = gold_summary.get("issues", [])
            if issues:
                st.info("Generating snapshots and AI coaching report...")

                # Get video path from bronze
                bronze_session_dir = ROOT / "pipeline" / "bronze" / session_id
                bronze_meta = read_json(bronze_session_dir / "meta.json")
                video_path = bronze_meta["input_video"]

                # Create snapshots dir
                snapshots_dir = gold_dir / "snapshots"
                snapshots_dir.mkdir(exist_ok=True)

                # Extract frames
                cap = cv2.VideoCapture(video_path)
                for issue in issues:
                    issue_type = issue["type"]
                    for frame_idx in issue["frames"][:3]:  # limit to 3 per issue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if ret:
                            img_path = snapshots_dir / f"{issue_type}_{frame_idx}.jpg"
                            cv2.imwrite(str(img_path), frame)
                cap.release()

            # LLM report
            try:
                import os
                import requests

                # Determine which LLM provider to use
                def get_secret(key):
                    try:
                        val = st.secrets[key]
                        if val:
                            return str(val).strip()
                    except Exception:
                        pass
                    return os.getenv(key, "").strip()

                google_api_key = get_secret("GOOGLE_API_KEY")
                groq_api_key = get_secret("GROQ_API_KEY")
                openai_api_key = get_secret("OPENAI_API_KEY")
                together_api_key = get_secret("TOGETHER_API_KEY")
                hf_api_key = get_secret("HF_API_KEY")

                llm_report = None

                if google_api_key:
                    # Google Gemini 2.0 Flash — free
                    response = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={google_api_key}",
                        headers={"Content-Type": "application/json"},
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"maxOutputTokens": 600, "temperature": 0.7}
                        },
                        timeout=30
                    )
                    if response.status_code == 200:
                        llm_report = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        raise Exception(f"Gemini error: {response.status_code} - {response.text}")

                elif groq_api_key:
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
                        json={"model": "llama3-8b-8192", "messages": [{"role": "user", "content": prompt}], "max_tokens": 600},
                        timeout=30
                    )
                    if response.status_code == 200:
                        llm_report = response.json()["choices"][0]["message"]["content"]
                    else:
                        raise Exception(f"Groq error: {response.status_code} - {response.text}")

                elif openai_api_key:
                    import openai
                    client = openai.OpenAI(api_key=openai_api_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=600
                    )
                    llm_report = resp.choices[0].message.content

                elif together_api_key:
                    response = requests.post(
                        "https://api.together.xyz/v1/chat/completions",
                        headers={"Authorization": f"Bearer {together_api_key}", "Content-Type": "application/json"},
                        json={"model": "meta-llama/Llama-3-8b-chat-hf", "messages": [{"role": "user", "content": prompt}], "max_tokens": 600},
                        timeout=30
                    )
                    if response.status_code == 200:
                        llm_report = response.json()["choices"][0]["message"]["content"]
                    else:
                        raise Exception(f"Together AI error: {response.status_code} - {response.text}")

                elif hf_api_key:
                    response = requests.post(
                        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
                        headers={"Authorization": f"Bearer {hf_api_key}"},
                        json={"inputs": f"<s>[INST] {prompt} [/INST]", "parameters": {"max_new_tokens": 500, "return_full_text": False}},
                        timeout=60
                    )
                    if response.status_code == 200:
                        llm_report = response.json()[0].get("generated_text", "").strip()
                    else:
                        raise Exception(f"HuggingFace error: {response.status_code} - {response.text}")

                else:
                    raise Exception("No API key found. Add GOOGLE_API_KEY to Streamlit secrets. Get a free key at aistudio.google.com")

                # Save
                (gold_dir / "llm_report.txt").write_text(llm_report, encoding="utf-8")

            except Exception as e:
                st.error(f"LLM Error: {e}")
                llm_report = f"AI Coaching Report (Generated without LLM)\n\n"
                llm_report += f"Session Summary: {exercise.capitalize()} with {gold_summary['reps']} reps, overall score {gold_summary['scores']['overall']:.1f}/100.\n\n"
                llm_report += "Key Issues Identified:\n"
                for issue in issues:
                    llm_report += f"- {issue['type'].replace('_', ' ').title()}: {issue['description']}\n"
                llm_report += "\nRecommendations:\n"
                llm_report += "- Focus on maintaining proper form throughout the movement.\n"
                llm_report += "- Practice with lighter weights to perfect technique.\n"
                llm_report += "- Consider filming from different angles for better self-assessment.\n"
                llm_report += "\nKeep up the good work and stay consistent!"

            # Display LLM report
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
                            img_path = snapshots_dir / f"{issue['type']}_{frame_idx}.jpg"
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
