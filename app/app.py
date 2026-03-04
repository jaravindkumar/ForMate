import json
import os
import time
import tempfile
import requests
from pathlib import Path
import subprocess
import sys

import streamlit as st
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parents[1]
PY   = sys.executable

st.set_page_config(page_title="ForMate", layout="wide", page_icon="F", initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;700;800;900&family=Barlow:wght@300;400;500&display=swap');
:root{--lime:#C8F400;--carbon:#0C0C0C;--panel:#111;--b1:#1E1E1E;--b2:#2A2A2A;--txt:#F2F2F2;--mut:#555;--mut2:#2E2E2E;--warn:#FF6B35;--good:#4ADE80;}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[class*="css"]{font-family:'Barlow',sans-serif;background:var(--carbon)!important;color:var(--txt)!important;}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden!important;display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stSidebar"]{display:none!important;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-track{background:var(--carbon);}::-webkit-scrollbar-thumb{background:var(--lime);border-radius:2px;}
.wrap{max-width:1280px;margin:0 auto;padding:0 1.5rem 4rem;}
.nav{display:flex;align-items:center;justify-content:space-between;padding:1.4rem 0 1rem;border-bottom:1px solid var(--b1);margin-bottom:2rem;}
.logo{font-family:'Barlow Condensed',sans-serif;font-weight:900;font-size:1.8rem;letter-spacing:.04em;text-transform:uppercase;color:var(--txt);}
.logo span{color:var(--lime);}
.tag{font-size:.65rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--mut);border:1px solid var(--b2);padding:.28rem .7rem;border-radius:20px;}
.hero{padding:1.5rem 0 2.5rem;position:relative;overflow:hidden;}
.hero-kicker{font-size:.7rem;font-weight:700;letter-spacing:.22em;text-transform:uppercase;color:var(--lime);margin-bottom:.75rem;}
.hero-h{font-family:'Barlow Condensed',sans-serif;font-weight:900;font-size:clamp(3rem,9vw,7rem);line-height:.92;letter-spacing:-.01em;text-transform:uppercase;color:var(--txt);}
.hero-h em{color:var(--lime);font-style:normal;}
.hero-sub{font-size:clamp(.85rem,1.5vw,1rem);color:var(--mut);margin-top:1.2rem;max-width:540px;line-height:1.6;font-weight:300;}
.hero-num{position:absolute;right:2rem;bottom:1rem;font-family:'Barlow Condensed',sans-serif;font-size:9rem;font-weight:900;color:rgba(200,244,0,.04);letter-spacing:-.05em;user-select:none;line-height:1;}
.panel{background:var(--panel);border:1px solid var(--b1);border-radius:16px;padding:1.5rem;position:relative;overflow:hidden;margin-bottom:1rem;}
.panel::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--lime),transparent);opacity:.5;}
.plabel{font-size:.63rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--lime);margin-bottom:1rem;}
.stSelectbox>div>div{background:#1A1A1A!important;border:1px solid var(--b2)!important;border-radius:10px!important;color:var(--txt)!important;}
.stSelectbox label{display:none!important;}
[data-testid="stFileUploader"]{background:#141414!important;border:1.5px dashed var(--b2)!important;border-radius:12px!important;transition:border-color .25s!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--lime)!important;}
.stButton>button{font-family:'Barlow Condensed',sans-serif!important;font-weight:800!important;font-size:1.1rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;background:var(--lime)!important;color:var(--carbon)!important;border:none!important;border-radius:10px!important;padding:.9rem 2rem!important;width:100%!important;transition:all .2s ease!important;}
.stButton>button:hover{background:#d9ff1a!important;box-shadow:0 0 30px rgba(200,244,0,.25)!important;transform:translateY(-2px)!important;}
.stProgress>div{background:var(--b1)!important;border-radius:4px!important;height:4px!important;}
.stProgress>div>div{background:var(--lime)!important;border-radius:4px!important;}
.score-banner{display:flex;align-items:stretch;background:var(--b1);border-radius:16px;overflow:hidden;margin:1.5rem 0;}
.score-box{background:var(--lime);padding:2rem 2.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;min-width:155px;}
.score-big{font-family:'Barlow Condensed',sans-serif;font-size:5rem;font-weight:900;color:var(--carbon);line-height:1;letter-spacing:-.03em;}
.score-den{font-family:'Barlow Condensed',sans-serif;font-size:.9rem;font-weight:700;color:rgba(12,12,12,.45);letter-spacing:.1em;text-transform:uppercase;}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(95px,1fr));flex:1;background:var(--panel);}
.stat{padding:1.25rem 1rem;border-right:1px solid var(--b1);display:flex;flex-direction:column;justify-content:center;}
.stat:last-child{border-right:none;}
.stat-val{font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:700;color:var(--txt);line-height:1;}
.stat-val.g{color:var(--good);}.stat-val.w{color:var(--warn);}
.stat-key{font-size:.6rem;font-weight:600;letter-spacing:.15em;text-transform:uppercase;color:var(--mut);margin-top:.3rem;}
.sh{display:flex;align-items:center;gap:.6rem;margin-bottom:.9rem;}
.sh-bar{width:3px;height:18px;background:var(--lime);border-radius:2px;flex-shrink:0;}
.sh-txt{font-family:'Barlow Condensed',sans-serif;font-size:.78rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--mut);}
.ci{display:flex;gap:.75rem;align-items:flex-start;padding:.85rem 1rem;border-radius:10px;margin-bottom:.55rem;font-size:.88rem;line-height:1.5;}
.ci.pos{background:rgba(74,222,128,.07);border:1px solid rgba(74,222,128,.15);color:#a3f0b8;}
.ci.imp{background:rgba(255,107,53,.07);border:1px solid rgba(255,107,53,.15);color:#ffc4ac;}
.ci.foc{background:rgba(200,244,0,.06);border:1px solid rgba(200,244,0,.12);color:#e8ff80;}
.ci-icon{font-size:.95rem;flex-shrink:0;margin-top:.1rem;}
.report{background:#0E0E0E;border:1px solid var(--b1);border-radius:12px;padding:1.5rem;font-size:.87rem;line-height:1.8;color:#888;white-space:pre-wrap;}
.report::before{content:'AI COACHING REPORT';display:block;font-size:.58rem;font-weight:700;letter-spacing:.2em;color:var(--lime);margin-bottom:.75rem;}
.flag{display:flex;align-items:flex-start;gap:.6rem;padding:.7rem 1rem;border-radius:8px;background:rgba(255,107,53,.06);border:1px solid rgba(255,107,53,.15);font-size:.83rem;color:#ffc4ac;margin-bottom:.45rem;line-height:1.5;}
.flag-dot{width:6px;height:6px;border-radius:50%;background:var(--warn);flex-shrink:0;margin-top:.45rem;}
.bar-wrap{margin-bottom:.85rem;}
.bar-hd{display:flex;justify-content:space-between;margin-bottom:.28rem;}
.bar-name{font-size:.73rem;font-weight:500;color:var(--mut);letter-spacing:.03em;}
.bar-val{font-family:'Barlow Condensed',sans-serif;font-size:.82rem;font-weight:700;color:var(--txt);}
.bar-track{height:4px;background:var(--b2);border-radius:2px;overflow:hidden;}
.bar-fill{height:100%;border-radius:2px;background:var(--lime);}
.bar-fill.mid{background:#f0b429;}.bar-fill.low{background:var(--warn);}
.fm-status{font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--lime);}
.empty{padding:4rem 1rem;text-align:center;}
.empty-icon{font-size:3.5rem;margin-bottom:1rem;}
.empty-title{font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:700;text-transform:uppercase;color:var(--mut);letter-spacing:.05em;}
.empty-sub{font-size:.83rem;color:var(--mut2);margin-top:.5rem;}
.div{border:none;border-top:1px solid var(--b1);margin:1.25rem 0;}
.footer{text-align:center;padding:2rem 0 1rem;font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:var(--mut2);}
.streamlit-expanderHeader{background:var(--panel)!important;border:1px solid var(--b1)!important;border-radius:10px!important;font-family:'Barlow Condensed',sans-serif!important;font-weight:700!important;letter-spacing:.06em!important;color:var(--txt)!important;text-transform:uppercase!important;font-size:.83rem!important;}
.file-ok{margin-top:.75rem;padding:.7rem 1rem;background:rgba(200,244,0,.06);border-radius:8px;border:1px solid rgba(200,244,0,.2);font-size:.8rem;color:#c8f400;}
.cfg-info{margin-top:.75rem;padding:.7rem 1rem;background:#1A1A1A;border-radius:8px;border:1px solid #222;font-size:.78rem;color:#555;}
.cfg-ex{color:var(--lime);font-weight:600;}
.overlay-na{padding:2rem;text-align:center;color:#2a2a2a;background:#0e0e0e;border-radius:12px;font-size:.85rem;}
@media(max-width:768px){
  .wrap{padding:0 1rem 3rem;}
  .hero-h{font-size:clamp(2.4rem,11vw,4rem);}
  .score-banner{flex-direction:column;}
  .score-box{padding:1.5rem;flex-direction:row;gap:1rem;min-width:unset;}
  .score-big{font-size:3.5rem;}
  .stat-grid{grid-template-columns:repeat(3,1fr);}
  .stat{border-right:1px solid var(--b1);border-bottom:1px solid var(--b1);}
  .hero-num{display:none;}
}
@media(max-width:480px){
  .stat-grid{grid-template-columns:repeat(2,1fr);}
  .score-big{font-size:3rem;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── helpers ──────────────────────────────────

def run_cmd(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def latest_session_dir(root, after_ts):
    root = Path(root)
    if not root.exists():
        raise RuntimeError(f"Missing: {root}")
    cands = [(p.stat().st_mtime, p) for p in root.iterdir()
             if p.is_dir() and p.stat().st_mtime >= after_ts - 1.0]
    if cands:
        return sorted(cands)[-1][1]
    all_d = [(p.stat().st_mtime, p) for p in root.iterdir() if p.is_dir()]
    if not all_d:
        raise RuntimeError("No session dirs found.")
    return sorted(all_d)[-1][1]

def get_secret(key):
    try:
        v = st.secrets[key]
        if v: return str(v).strip()
    except Exception:
        pass
    return os.getenv(key, "").strip()

def call_llm(prompt):
    gk = get_secret("GROQ_API_KEY")
    ok = get_secret("OPENAI_API_KEY")
    tk = get_secret("TOGETHER_API_KEY")
    hk = get_secret("HF_API_KEY")
    if gk:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": "Bearer " + gk, "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": prompt}], "max_tokens": 600},
            timeout=30)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        raise Exception("Groq " + str(r.status_code) + ": " + r.text)
    if ok:
        import openai
        c = openai.OpenAI(api_key=ok)
        return c.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600).choices[0].message.content
    if tk:
        r = requests.post("https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": "Bearer " + tk, "Content-Type": "application/json"},
            json={"model": "meta-llama/Llama-3-8b-chat-hf",
                  "messages": [{"role": "user", "content": prompt}], "max_tokens": 600},
            timeout=30)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        raise Exception("Together " + str(r.status_code))
    if hk:
        r = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            headers={"Authorization": "Bearer " + hk},
            json={"inputs": "<s>[INST] " + prompt + " [/INST]",
                  "parameters": {"max_new_tokens": 500, "return_full_text": False}},
            timeout=60)
        if r.status_code == 200:
            return r.json()[0].get("generated_text", "").strip()
        raise Exception("HF " + str(r.status_code))
    raise Exception("No API key found. Add GROQ_API_KEY to Streamlit secrets.")

def safe_score(v):
    import math
    if v is None: return 0.0
    try:
        f = float(v)
        return 0.0 if math.isnan(f) or math.isinf(f) else f
    except Exception:
        return 0.0

def sc(scores, key):
    return str(int(safe_score(scores.get(key, 0))))

def sc_cls(scores, key):
    return "g" if safe_score(scores.get(key, 0)) >= 75 else "w"

def bar_cls(v):
    v = safe_score(v)
    if v >= 75: return ""
    if v >= 50: return "mid"
    return "low"

def coach_feedback(summary):
    s = summary.get("scores", {})
    flags = summary.get("flags", [])
    pos, imp, nxt = [], [], []
    if s.get("hinge_quality",    0) >= 80: pos.append("Strong hip hinge. Firing the right muscles.")
    if s.get("symmetry",         0) >= 80: pos.append("Excellent left-right balance throughout.")
    if s.get("trunk_control",    0) >= 80: pos.append("Solid torso control and bracing.")
    if s.get("setup_consistency",0) >= 80: pos.append("Consistent setup across all reps.")
    if s.get("tempo_consistency",100) < 75: imp.append("Pull speed varies between reps. Aim for a steady cadence.")
    if s.get("setup_consistency",100) < 70: imp.append("Foot position shifts rep to rep. Reset identically every time.")
    for f in flags[:2]: imp.append(f.get("message", "Form issue detected."))
    if s.get("tempo_consistency",100) < 75: nxt.append("Pause 1 sec at the bottom, then drive with consistent speed.")
    nxt.append("Keep bar close and brace your core before every pull.")
    if not pos: pos = ["Good effort. Form was trackable throughout."]
    if not imp: imp = ["No major form issues detected. Keep pushing."]
    return pos[:2], imp[:2], nxt[:2]


# ── page ─────────────────────────────────────

st.markdown('<div class="wrap">', unsafe_allow_html=True)

st.markdown(
    '<nav class="nav">'
    '<div class="logo">For<span>Mate</span></div>'
    '<div class="tag">MVP &middot; AI Form Coach</div>'
    '</nav>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero">'
    '<p class="hero-kicker">Computer Vision &middot; Pose Analysis &middot; AI Coaching</p>'
    '<h1 class="hero-h">Analyse<br><em>Your Form.</em><br>Train Smarter.</h1>'
    '<p class="hero-sub">Upload your workout video and get instant AI-powered form analysis,'
    ' rep detection, and personalised coaching feedback.</p>'
    '<div class="hero-num">01</div>'
    '</div>',
    unsafe_allow_html=True
)

# config + upload
col_cfg, col_up = st.columns([1, 1], gap="medium")

with col_cfg:
    st.markdown('<div class="panel"><p class="plabel">Session Setup</p></div>', unsafe_allow_html=True)
    exercise    = st.selectbox("Exercise", ["deadlift", "squat"])
    camera_view = st.selectbox("Camera Angle", ["front_oblique", "side"])
    cam_label   = camera_view.replace("_", " ").title()
    ex_upper    = exercise.upper()
    st.markdown(
        '<div class="cfg-info">'
        '<span class="cfg-ex">' + ex_upper + '</span>'
        ' &nbsp;&middot;&nbsp; ' + cam_label +
        '</div>',
        unsafe_allow_html=True
    )

with col_up:
    st.markdown('<div class="panel"><p class="plabel">Upload Video</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["mp4", "mov", "m4v"], label_visibility="collapsed")
    if uploaded:
        fname = uploaded.name
        fmb   = round(uploaded.size / (1024 * 1024), 1)
        st.markdown(
            '<div class="file-ok">&#10003; &nbsp;<strong>' + fname + '</strong>'
            ' &nbsp;&middot;&nbsp; ' + str(fmb) + ' MB</div>',
            unsafe_allow_html=True
        )

# preview + run
if uploaded:
    tmp_dir   = Path(tempfile.mkdtemp())
    tmp_video = tmp_dir / uploaded.name
    tmp_video.write_bytes(uploaded.read())

    pc, bc = st.columns([3, 1], gap="medium")
    with pc:
        st.markdown('<div class="panel"><p class="plabel">Video Preview</p></div>', unsafe_allow_html=True)
        st.video(str(tmp_video))
    with bc:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        run_clicked = st.button("ANALYSE FORM", type="primary", use_container_width=True)
else:
    st.markdown(
        '<div class="empty">'
        '<div class="empty-icon">🏋️</div>'
        '<p class="empty-title">Ready to Analyse</p>'
        '<p class="empty-sub">Upload an MP4, MOV or M4V &middot; Max 200 MB</p>'
        '</div>',
        unsafe_allow_html=True
    )
    run_clicked = False


# ── pipeline ─────────────────────────────────

if uploaded and run_clicked:
    st.markdown('<hr class="div"/>', unsafe_allow_html=True)
    prog = st.progress(0)
    stat = st.empty()

    # Bronze
    stat.markdown('<p class="fm-status">Extracting Pose Data...</p>', unsafe_allow_html=True)
    prog.progress(8)
    t0 = time.time()
    p  = run_cmd([PY, "pipeline_bronze_extract.py",
                  "--video", str(tmp_video),
                  "--exercise", exercise,
                  "--camera_view", camera_view,
                  "--overlay"], cwd=ROOT)
    if p.returncode != 0:
        st.error("Pose extraction failed.")
        with st.expander("Error details"):
            st.code(p.stderr or p.stdout)
        st.stop()

    b_sess     = latest_session_dir(ROOT / "pipeline" / "bronze", after_ts=t0)
    b_sum      = read_json(b_sess / "summary.json")
    session_id = b_sum["session_id"]
    session_dir= b_sum["outputs"]["session_dir"]
    prog.progress(35)

    # Silver
    stat.markdown('<p class="fm-status">Detecting Reps...</p>', unsafe_allow_html=True)
    prog.progress(42)
    p = run_cmd([PY, "pipeline_silver_transform.py",
                 "--session_dir", session_dir], cwd=ROOT)
    if p.returncode != 0:
        st.error("Rep detection failed.")
        with st.expander("Error details"):
            st.code(p.stderr or p.stdout)
        st.stop()
    s_sum    = read_json(ROOT / "pipeline" / "silver" / session_id / "summary.json")
    num_reps = s_sum["num_reps_detected"]
    prog.progress(65)

    # Gold
    stat.markdown('<p class="fm-status">Scoring Form...</p>', unsafe_allow_html=True)
    prog.progress(72)
    p = run_cmd([PY, "pipeline_gold_score.py",
                 "--session_id", session_id,
                 "--exercise", exercise], cwd=ROOT)
    if p.returncode != 0:
        st.error("Scoring failed.")
        with st.expander("Error details"):
            st.code(p.stderr or p.stdout)
        st.stop()

    gold_dir = ROOT / "pipeline" / "gold" / session_id
    g_sum    = read_json(gold_dir / "summary.json")
    rep_df   = pd.read_parquet(gold_dir / "metrics_reps.parquet")
    prog.progress(100)
    stat.empty()
    prog.empty()

    # ── score banner ──
    scores  = g_sum["scores"]
    overall = safe_score(scores.get("overall", 0))
    reps    = g_sum.get("reps", num_reps) or num_reps

    st.markdown(
        '<div class="score-banner">'
        '<div class="score-box">'
        '<div class="score-big">' + str(int(overall)) + '</div>'
        '<div class="score-den">/ 100</div>'
        '</div>'
        '<div class="stat-grid">'
        '<div class="stat"><div class="stat-val">' + str(reps) + '</div><div class="stat-key">Reps</div></div>'
        '<div class="stat"><div class="stat-val ' + sc_cls(scores,"hinge_quality") + '">' + sc(scores,"hinge_quality") + '</div><div class="stat-key">Hinge</div></div>'
        '<div class="stat"><div class="stat-val ' + sc_cls(scores,"trunk_control") + '">' + sc(scores,"trunk_control") + '</div><div class="stat-key">Trunk</div></div>'
        '<div class="stat"><div class="stat-val ' + sc_cls(scores,"symmetry") + '">' + sc(scores,"symmetry") + '</div><div class="stat-key">Symmetry</div></div>'
        '<div class="stat"><div class="stat-val ' + sc_cls(scores,"tempo_consistency") + '">' + sc(scores,"tempo_consistency") + '</div><div class="stat-key">Tempo</div></div>'
        '<div class="stat"><div class="stat-val ' + sc_cls(scores,"setup_consistency") + '">' + sc(scores,"setup_consistency") + '</div><div class="stat-key">Setup</div></div>'
        '</div></div>',
        unsafe_allow_html=True
    )

    # ── two columns ──
    lc, rc = st.columns([1, 1], gap="medium")

    with lc:
        # Overlay
        st.markdown('<div class="sh"><div class="sh-bar"></div><span class="sh-txt">Pose Overlay</span></div>', unsafe_allow_html=True)
        ovp = b_sum["outputs"].get("overlay_mp4")
        overlay_path = Path(ovp) if ovp else None
        if overlay_path and overlay_path.exists():
            with open(str(overlay_path), "rb") as vf:
                st.video(vf.read())
        else:
            st.markdown('<div class="overlay-na">Overlay not available</div>', unsafe_allow_html=True)

        st.markdown('<hr class="div"/>', unsafe_allow_html=True)

        # Score bars
        st.markdown('<div class="sh"><div class="sh-bar"></div><span class="sh-txt">Score Breakdown</span></div>', unsafe_allow_html=True)
        score_items = [
            ("Hinge Quality",     "hinge_quality"),
            ("Trunk Control",     "trunk_control"),
            ("Symmetry",          "symmetry"),
            ("Tempo",             "tempo_consistency"),
            ("Setup Consistency", "setup_consistency"),
        ]
        for label, key in score_items:
            v  = scores.get(key, 0)
            bc = bar_cls(v)
            vi = int(v)
            st.markdown(
                '<div class="bar-wrap">'
                '<div class="bar-hd">'
                '<span class="bar-name">' + label + '</span>'
                '<span class="bar-val">' + str(vi) + '</span>'
                '</div>'
                '<div class="bar-track">'
                '<div class="bar-fill ' + bc + '" style="width:' + str(vi) + '%"></div>'
                '</div></div>',
                unsafe_allow_html=True
            )

    with rc:
        # Coach feedback
        pos, imp, nxt = coach_feedback(g_sum)
        st.markdown('<div class="sh"><div class="sh-bar"></div><span class="sh-txt">Coach Feedback</span></div>', unsafe_allow_html=True)
        for msg in pos:
            st.markdown('<div class="ci pos"><span class="ci-icon">&#10003;</span>' + msg + '</div>', unsafe_allow_html=True)
        for msg in imp:
            st.markdown('<div class="ci imp"><span class="ci-icon">&#8593;</span>' + msg + '</div>', unsafe_allow_html=True)
        for msg in nxt:
            st.markdown('<div class="ci foc"><span class="ci-icon">&#8594;</span>' + msg + '</div>', unsafe_allow_html=True)

        flags = g_sum.get("flags", [])
        if flags:
            st.markdown('<hr class="div"/>', unsafe_allow_html=True)
            st.markdown('<div class="sh"><div class="sh-bar"></div><span class="sh-txt">Form Flags</span></div>', unsafe_allow_html=True)
            for f in flags:
                st.markdown('<div class="flag"><div class="flag-dot"></div>' + f.get("message","") + '</div>', unsafe_allow_html=True)

        st.markdown('<hr class="div"/>', unsafe_allow_html=True)

        # AI Report
        st.markdown('<div class="sh"><div class="sh-bar"></div><span class="sh-txt">AI Coaching Report</span></div>', unsafe_allow_html=True)
        issues = g_sum.get("issues", [])
        issue_list = ", ".join([i["type"] for i in issues]) if issues else "none"
        prompt = (
            "You are a professional strength coach. Write a concise coaching report for a "
            + exercise + " session.\n\n"
            "Data:\n"
            "- Overall score: " + str(round(overall, 1)) + "/100\n"
            "- Reps: " + str(reps) + "\n"
            "- Hinge: " + sc(scores, "hinge_quality") + "/100\n"
            "- Trunk: " + sc(scores, "trunk_control") + "/100\n"
            "- Symmetry: " + sc(scores, "symmetry") + "/100\n"
            "- Tempo: " + sc(scores, "tempo_consistency") + "/100\n"
            "- Issues: " + issue_list + "\n\n"
            "Write 3 short paragraphs: what went well, what to fix, focus for next session. "
            "Be direct, specific, motivating. Plain text only."
        )
        with st.spinner("Generating coaching report..."):
            try:
                report = call_llm(prompt)
                (gold_dir / "llm_report.txt").write_text(report, encoding="utf-8")
            except Exception as e:
                st.error("LLM Error: " + str(e))
                report = (
                    "Score: " + str(int(overall)) + "/100 across " + str(reps) + " reps.\n\n"
                    "Focus on bracing before each rep and maintaining a consistent bar path. "
                    "Small improvements compound significantly over time.\n\n"
                    "Next session: nail your setup, brace hard, and aim for controlled reps."
                )
        st.markdown('<div class="report">' + report + '</div>', unsafe_allow_html=True)

        # Snapshots
        if issues:
            st.markdown('<hr class="div"/>', unsafe_allow_html=True)
            st.markdown('<div class="sh"><div class="sh-bar"></div><span class="sh-txt">Form Snapshots</span></div>', unsafe_allow_html=True)
            b_meta        = read_json(ROOT / "pipeline" / "bronze" / session_id / "meta.json")
            snapshots_dir = gold_dir / "snapshots"
            snapshots_dir.mkdir(exist_ok=True)
            cap = cv2.VideoCapture(b_meta["input_video"])
            for issue in issues:
                for fi in issue["frames"][:3]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(str(snapshots_dir / (issue["type"] + "_" + str(fi) + ".jpg")), frame)
            cap.release()
            for issue in issues:
                with st.expander(issue["type"].replace("_", " ").title()):
                    st.caption(issue["description"])
                    ic = st.columns(min(len(issue["frames"]), 3))
                    for i, fi in enumerate(issue["frames"][:3]):
                        ip = snapshots_dir / (issue["type"] + "_" + str(fi) + ".jpg")
                        if ip.exists():
                            ic[i].image(str(ip), use_container_width=True, caption="Frame " + str(fi))

        st.markdown('<hr class="div"/>', unsafe_allow_html=True)
        with st.expander("Per-Rep Breakdown"):
            st.dataframe(rep_df, use_container_width=True)

    st.markdown(
        '<p class="footer">ForMate &middot; ' + session_id[:16] + ' &middot; Powered by MediaPipe + AI</p>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)
