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

st.set_page_config(
    page_title="ForMate — AI Form Coach",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;800;900&family=Barlow:wght@300;400;500;600&display=swap');

:root {
  --lime:    #C8F400;
  --lime-dim:#8aaa00;
  --carbon:  #0C0C0C;
  --panel:   #121212;
  --border:  #1E1E1E;
  --border2: #2A2A2A;
  --text:    #F2F2F2;
  --muted:   #555;
  --muted2:  #333;
  --warn:    #FF6B35;
  --good:    #4ADE80;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
  font-family: 'Barlow', sans-serif;
  background: var(--carbon) !important;
  color: var(--text) !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden !important; display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--carbon); }
::-webkit-scrollbar-thumb { background: var(--lime); border-radius: 2px; }

/* ── NOISE OVERLAY ── */
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
  pointer-events: none; z-index: 0; opacity: 0.6;
}

/* ── OUTER WRAP ── */
.fm-wrap {
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 1.5rem 4rem;
}

/* ── NAV ── */
.fm-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.4rem 0 1rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2.5rem;
}
.fm-logo {
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 900;
  font-size: 1.6rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--text);
}
.fm-logo span { color: var(--lime); }
.fm-tag {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  border: 1px solid var(--border2);
  padding: 0.3rem 0.75rem;
  border-radius: 20px;
}

/* ── HERO ── */
.fm-hero {
  padding: 1rem 0 2.5rem;
  position: relative;
  overflow: hidden;
}
.fm-hero-line {
  position: absolute;
  top: 0; left: -5%;
  width: 110%;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--lime), transparent);
  opacity: 0.3;
}
.fm-hero-kicker {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--lime);
  margin-bottom: 0.75rem;
}
.fm-hero-headline {
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 900;
  font-size: clamp(3rem, 9vw, 7rem);
  line-height: 0.92;
  letter-spacing: -0.01em;
  text-transform: uppercase;
  color: var(--text);
}
.fm-hero-headline em {
  color: var(--lime);
  font-style: normal;
}
.fm-hero-sub {
  font-size: clamp(0.85rem, 1.5vw, 1rem);
  color: var(--muted);
  margin-top: 1.2rem;
  max-width: 540px;
  line-height: 1.6;
  font-weight: 300;
}

/* ── DIAGONAL ACCENT ── */
.fm-accent-bar {
  position: absolute;
  right: -40px; top: 0;
  width: 380px; height: 100%;
  background: linear-gradient(135deg, transparent 40%, rgba(200,244,0,0.04) 100%);
  pointer-events: none;
}
.fm-accent-num {
  position: absolute;
  right: 2rem; bottom: 1.5rem;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 8rem;
  font-weight: 900;
  color: rgba(200,244,0,0.04);
  letter-spacing: -0.05em;
  user-select: none;
  line-height: 1;
}

/* ── UPLOAD SECTION ── */
.fm-upload-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  margin-bottom: 1.5rem;
}
.fm-panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem;
  position: relative;
  overflow: hidden;
}
.fm-panel::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--lime), transparent);
  opacity: 0.5;
}
.fm-panel-label {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--lime);
  margin-bottom: 1rem;
}

/* ── CONFIG CONTROLS ── */
.stSelectbox > div > div {
  background: #1A1A1A !important;
  border: 1px solid var(--border2) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'Barlow', sans-serif !important;
}
.stSelectbox label { display: none !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
  background: #141414 !important;
  border: 1.5px dashed var(--border2) !important;
  border-radius: 12px !important;
  transition: border-color 0.25s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--lime) !important; }
[data-testid="stFileUploader"] label { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: var(--muted) !important; }

/* ── CTA BUTTON ── */
.stButton > button {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-weight: 800 !important;
  font-size: 1.1rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  background: var(--lime) !important;
  color: var(--carbon) !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 0.9rem 2rem !important;
  width: 100% !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 0 0 rgba(200,244,0,0) !important;
}
.stButton > button:hover {
  background: #d9ff1a !important;
  box-shadow: 0 0 30px rgba(200,244,0,0.25) !important;
  transform: translateY(-2px) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ── PROGRESS ── */
.stProgress { margin: 0.5rem 0; }
.stProgress > div { background: var(--border) !important; border-radius: 4px !important; height: 4px !important; }
.stProgress > div > div { background: var(--lime) !important; border-radius: 4px !important; }

/* ── RESULTS BANNER ── */
.fm-results-header {
  display: flex;
  align-items: stretch;
  gap: 1px;
  background: var(--border);
  border-radius: 16px;
  overflow: hidden;
  margin: 2rem 0 1.5rem;
}
.fm-score-hero {
  background: var(--lime);
  padding: 2rem 2.5rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 160px;
}
.fm-score-big {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 5rem;
  font-weight: 900;
  color: var(--carbon);
  line-height: 1;
  letter-spacing: -0.03em;
}
.fm-score-denom {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1rem;
  font-weight: 700;
  color: rgba(12,12,12,0.5);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 0.1rem;
}
.fm-stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
  flex: 1;
  background: var(--panel);
}
.fm-stat {
  padding: 1.25rem 1rem;
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.fm-stat:last-child { border-right: none; }
.fm-stat-val {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 2rem;
  font-weight: 700;
  color: var(--text);
  line-height: 1;
}
.fm-stat-val.good { color: var(--good); }
.fm-stat-val.warn { color: var(--warn); }
.fm-stat-key {
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 0.3rem;
}

/* ── RESULTS GRID ── */
.fm-results-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  margin-top: 1.5rem;
}

/* ── SECTION HEADERS ── */
.fm-sh {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 1rem;
}
.fm-sh-bar {
  width: 3px; height: 18px;
  background: var(--lime);
  border-radius: 2px;
  flex-shrink: 0;
}
.fm-sh-text {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
}

/* ── COACH ITEMS ── */
.fm-coach-item {
  display: flex;
  gap: 0.75rem;
  align-items: flex-start;
  padding: 0.9rem 1rem;
  border-radius: 10px;
  margin-bottom: 0.6rem;
  font-size: 0.9rem;
  line-height: 1.5;
  font-weight: 400;
}
.fm-coach-item.positive { background: rgba(74,222,128,0.07); border: 1px solid rgba(74,222,128,0.15); color: #a3f0b8; }
.fm-coach-item.improve  { background: rgba(255,107,53,0.07); border: 1px solid rgba(255,107,53,0.15); color: #ffc4ac; }
.fm-coach-item.focus    { background: rgba(200,244,0,0.06); border: 1px solid rgba(200,244,0,0.12); color: #e8ff80; }
.fm-coach-icon { font-size: 1rem; flex-shrink: 0; margin-top: 0.1rem; }

/* ── REPORT BOX ── */
.fm-report {
  background: #0E0E0E;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  font-size: 0.88rem;
  line-height: 1.8;
  color: #999;
  white-space: pre-wrap;
  position: relative;
}
.fm-report::before {
  content: 'AI COACHING REPORT';
  display: block;
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 0.2em;
  color: var(--lime);
  margin-bottom: 0.75rem;
}

/* ── FLAG CHIP ── */
.fm-flag {
  display: flex;
  align-items: flex-start;
  gap: 0.6rem;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  background: rgba(255,107,53,0.06);
  border: 1px solid rgba(255,107,53,0.15);
  font-size: 0.85rem;
  color: #ffc4ac;
  margin-bottom: 0.5rem;
  line-height: 1.5;
}
.fm-flag-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--warn);
  flex-shrink: 0;
  margin-top: 0.45rem;
}

/* ── SCORE BAR ── */
.fm-bar-wrap { margin-bottom: 0.9rem; }
.fm-bar-header { display: flex; justify-content: space-between; margin-bottom: 0.3rem; }
.fm-bar-name { font-size: 0.75rem; font-weight: 500; color: var(--muted); letter-spacing: 0.04em; }
.fm-bar-val { font-family: 'Barlow Condensed', sans-serif; font-size: 0.85rem; font-weight: 700; color: var(--text); }
.fm-bar-track { height: 4px; background: var(--border2); border-radius: 2px; overflow: hidden; }
.fm-bar-fill { height: 100%; border-radius: 2px; background: var(--lime); transition: width 0.6s ease; }
.fm-bar-fill.mid { background: #f0b429; }
.fm-bar-fill.low { background: var(--warn); }

/* ── EXPANDER ── */
.streamlit-expanderHeader {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  font-family: 'Barlow Condensed', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.06em !important;
  color: var(--text) !important;
  text-transform: uppercase !important;
  font-size: 0.85rem !important;
}
.streamlit-expanderContent {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 10px 10px !important;
}

/* ── EMPTY STATE ── */
.fm-empty {
  padding: 4rem 1rem;
  text-align: center;
}
.fm-empty-icon {
  font-size: 3.5rem;
  margin-bottom: 1rem;
  filter: grayscale(0.3);
}
.fm-empty-title {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.8rem;
  font-weight: 700;
  text-transform: uppercase;
  color: var(--muted);
  letter-spacing: 0.05em;
}
.fm-empty-sub {
  font-size: 0.85rem;
  color: var(--muted2);
  margin-top: 0.5rem;
}

/* ── STATUS TEXT ── */
.fm-status {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--lime);
}

/* ── ALERTS override ── */
[data-testid="stAlert"] {
  border-radius: 10px !important;
  border: 1px solid var(--border2) !important;
}

/* ── DIVIDER ── */
.fm-div { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* ── FOOTER ── */
.fm-footer {
  text-align: center;
  padding: 2rem 0 1rem;
  font-size: 0.7rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted2);
}

/* ── MOBILE ── */
@media (max-width: 768px) {
  .fm-wrap { padding: 0 1rem 3rem; }
  .fm-hero-headline { font-size: clamp(2.4rem, 11vw, 4rem); }
  .fm-upload-grid { grid-template-columns: 1fr; }
  .fm-results-header { flex-direction: column; }
  .fm-score-hero { padding: 1.5rem; flex-direction: row; gap: 1rem; min-width: unset; }
  .fm-score-big { font-size: 3.5rem; }
  .fm-stat-grid { grid-template-columns: repeat(3, 1fr); }
  .fm-stat { border-right: 1px solid var(--border); border-bottom: 1px solid var(--border); }
  .fm-results-grid { grid-template-columns: 1fr; }
  .fm-accent-num { display: none; }
}
@media (max-width: 480px) {
  .fm-stat-grid { grid-template-columns: repeat(2, 1fr); }
  .fm-score-big { font-size: 3rem; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def run_cmd(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def latest_session_dir(root: Path, after_ts: float) -> Path:
    if not root.exists():
        raise RuntimeError(f"Directory does not exist: {root}")
    candidates = [(p.stat().st_mtime, p) for p in root.iterdir()
                  if p.is_dir() and p.stat().st_mtime >= after_ts - 1.0]
    if candidates:
        return sorted(candidates)[-1][1]
    all_dirs = [(p.stat().st_mtime, p) for p in root.iterdir() if p.is_dir()]
    if not all_dirs:
        raise RuntimeError("No session directories found.")
    return sorted(all_dirs)[-1][1]

def get_secret(key):
    try:
        val = st.secrets[key]
        if val: return str(val).strip()
    except Exception:
        pass
    return os.getenv(key, "").strip()

def call_llm(prompt: str) -> str:
    groq_key     = get_secret("GROQ_API_KEY")
    openai_key   = get_secret("OPENAI_API_KEY")
    together_key = get_secret("TOGETHER_API_KEY")
    hf_key       = get_secret("HF_API_KEY")

    if groq_key:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": prompt}], "max_tokens": 600},
            timeout=30)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        raise Exception(f"Groq {r.status_code}: {r.text}")
    if openai_key:
        import openai
        c = openai.OpenAI(api_key=openai_key)
        return c.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600).choices[0].message.content
    if together_key:
        r = requests.post("https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {together_key}", "Content-Type": "application/json"},
            json={"model": "meta-llama/Llama-3-8b-chat-hf",
                  "messages": [{"role": "user", "content": prompt}], "max_tokens": 600},
            timeout=30)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        raise Exception(f"Together {r.status_code}: {r.text}")
    if hf_key:
        r = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            headers={"Authorization": f"Bearer {hf_key}"},
            json={"inputs": f"<s>[INST] {prompt} [/INST]",
                  "parameters": {"max_new_tokens": 500, "return_full_text": False}},
            timeout=60)
        if r.status_code == 200:
            return r.json()[0].get("generated_text", "").strip()
        raise Exception(f"HF {r.status_code}: {r.text}")
    raise Exception("No API key. Add GROQ_API_KEY to Streamlit secrets → console.groq.com")

def score_color(v):
    if v >= 75: return "good"
    if v >= 50: return "warn"
    return "warn"

def bar_class(v):
    if v >= 75: return ""
    if v >= 50: return "mid"
    return "low"

def coach_feedback(summary: dict):
    s      = summary.get("scores", {})
    flags  = summary.get("flags", [])
    pos, imp, nxt = [], [], []
    if s.get("hinge_quality",   0) >= 80: pos.append("Strong hip hinge — firing the right muscles.")
    if s.get("symmetry",        0) >= 80: pos.append("Excellent left-right balance throughout.")
    if s.get("trunk_control",   0) >= 80: pos.append("Solid torso control and bracing.")
    if s.get("setup_consistency",0)>= 80: pos.append("Consistent setup across all reps.")
    if s.get("tempo_consistency",100) < 75: imp.append("Pull speed varies between reps — aim for a steady cadence.")
    if s.get("setup_consistency",100) < 70: imp.append("Foot position shifts rep to rep — reset identically every time.")
    for f in flags[:2]: imp.append(f.get("message", "Form issue detected."))
    if s.get("tempo_consistency",100) < 75: nxt.append("Pause 1 sec at the bottom, then drive with consistent speed.")
    nxt.append("Keep the bar close to your body and brace your core before every pull.")
    if not pos: pos = ["Good effort — form was trackable throughout."]
    if not imp: imp = ["No major form issues detected. Keep pushing."]
    return pos[:2], imp[:2], nxt[:2]


# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────

st.markdown('<div class="fm-wrap">', unsafe_allow_html=True)

# NAV
st.markdown("""
<nav class="fm-nav">
  <div class="fm-logo">For<span>Mate</span></div>
  <div class="fm-tag">MVP · AI Form Coach</div>
</nav>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="fm-hero">
  <div class="fm-hero-line"></div>
  <div class="fm-accent-bar"></div>
  <p class="fm-hero-kicker">⚡ Computer Vision · Pose Analysis · AI Coaching</p>
  <h1 class="fm-hero-headline">Analyse<br><em>Your Form.</em><br>Train Smarter.</h1>
  <p class="fm-hero-sub">Upload your workout video and get instant AI-powered form analysis, rep detection, and personalised coaching feedback.</p>
  <div class="fm-accent-num">01</div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD + CONFIG ──
st.markdown('<div class="fm-upload-grid">', unsafe_allow_html=True)

col_cfg, col_up = st.columns([1, 1], gap="medium")

with col_cfg:
    st.markdown("""
    <div class="fm-panel">
      <p class="fm-panel-label">⚙ Session Setup</p>
    </div>
    """, unsafe_allow_html=True)
    exercise    = st.selectbox("Exercise", ["deadlift", "squat"])
    camera_view = st.selectbox("Camera Angle", ["front_oblique", "side"])
    st.markdown(f"""
    <div style="margin-top:0.75rem;padding:0.75rem 1rem;background:#1A1A1A;border-radius:8px;
                border:1px solid #222;font-size:0.8rem;color:#666;">
      <span style="color:var(--lime);font-weight:600;">{exercise.upper()}</span>
      &nbsp;·&nbsp; {camera_view.replace('_',' ').title()}
    </div>
    """, unsafe_allow_html=True)

with col_up:
    st.markdown("""
    <div class="fm-panel">
      <p class="fm-panel-label">📹 Upload Video</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["mp4", "mov", "m4v"], label_visibility="collapsed")
    if uploaded:
        fname = uploaded.name
        fsize = f"{uploaded.size/(1024*1024):.1f}"
        st.markdown(f"""
        <div style="margin-top:0.75rem;padding:0.75rem 1rem;background:rgba(200,244,0,0.06);
                    border-ra
