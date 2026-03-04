import json, os, time, tempfile, requests, subprocess, sys
from pathlib import Path
import streamlit as st
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parents[1]
PY   = sys.executable

st.set_page_config(page_title="FORMate", layout="wide", page_icon="F", initial_sidebar_state="collapsed")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Neue+Haas+Grotesk+Display+Pro:wght@400;500;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Archivo:wght@300;400;500;600&display=swap');

:root{
  --ink:#06060A; --surface:#0D0D12; --card:#13131A;
  --edge:#1D1D28; --edge2:#252533; --txt:#F0EEF8;
  --sub:#5A5870; --sub2:#2A2838;
  --acid:#BBFF00; --acid2:#D4FF4D;
  --red:#FF3F3F; --green:#2ECC71;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[class*="css"]{
  font-family:'Archivo',sans-serif;
  background:var(--ink)!important;
  color:var(--txt)!important;
}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden!important;display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stSidebar"]{display:none!important;}
::-webkit-scrollbar{width:3px;}
::-webkit-scrollbar-thumb{background:var(--acid);border-radius:2px;}

/* NAV */
.nav{
  display:flex;align-items:center;justify-content:space-between;
  padding:0 2.5rem;height:60px;
  background:var(--surface);
  border-bottom:1px solid var(--edge);
  position:sticky;top:0;z-index:100;
}
.logo{
  font-family:'Archivo Black',sans-serif;
  font-size:1.5rem;letter-spacing:-.01em;
}
.logo-form{color:var(--acid);}
.logo-ate{color:var(--txt);}
.nav-pills{display:flex;gap:.5rem;align-items:center;}
.pill{
  font-size:.62rem;font-weight:600;letter-spacing:.16em;text-transform:uppercase;
  padding:.3rem .75rem;border-radius:20px;
  background:var(--card);border:1px solid var(--edge2);color:var(--sub);
}
.pill.active{background:rgba(187,255,0,.1);border-color:rgba(187,255,0,.3);color:var(--acid);}

/* MAIN CONTENT AREA */
.main{max-width:1400px;margin:0 auto;padding:2.5rem 2.5rem 5rem;}

/* UPLOAD ZONE */
.upload-zone{
  border:1px solid var(--edge);
  border-radius:20px;
  background:var(--surface);
  padding:2.5rem;
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:2rem;
  margin-bottom:2rem;
}
.uz-left{display:flex;flex-direction:column;gap:1.25rem;}
.uz-headline{
  font-family:'Archivo Black',sans-serif;
  font-size:clamp(2rem,4vw,3.2rem);
  line-height:1.05;
  letter-spacing:-.02em;
  color:var(--txt);
}
.uz-headline span{color:var(--acid);}
.uz-desc{font-size:.9rem;color:var(--sub);line-height:1.7;font-weight:300;max-width:400px;}
.uz-controls{display:flex;flex-direction:column;gap:.75rem;}
.uz-right{display:flex;flex-direction:column;gap:.75rem;}

/* LABELS */
.lbl{
  font-size:.58rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;
  color:var(--sub);margin-bottom:.3rem;
  display:flex;align-items:center;gap:.5rem;
}
.lbl::after{content:'';flex:1;height:1px;background:var(--edge);}

/* CONTROLS */
.stSelectbox>div>div{background:var(--card)!important;border:1px solid var(--edge2)!important;border-radius:9px!important;color:var(--txt)!important;font-size:.87rem!important;}
.stSelectbox label{display:none!important;}
[data-testid="stFileUploader"]{background:var(--card)!important;border:1.5px dashed var(--edge2)!important;border-radius:12px!important;transition:border-color .2s!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--acid)!important;}
[data-testid="stFileUploader"] *{font-size:.85rem!important;}

/* RUN BUTTON */
.stButton>button{
  font-family:'Archivo Black',sans-serif!important;
  font-size:1rem!important;letter-spacing:.08em!important;
  background:var(--acid)!important;color:var(--ink)!important;
  border:none!important;border-radius:10px!important;
  padding:.85rem 2rem!important;width:100%!important;
  transition:all .18s!important;
}
.stButton>button:hover{background:var(--acid2)!important;transform:translateY(-2px)!important;box-shadow:0 8px 32px rgba(187,255,0,.2)!important;}

/* FILE OK */
.fok{
  padding:.65rem 1rem;border-radius:9px;font-size:.8rem;
  background:rgba(187,255,0,.06);border:1px solid rgba(187,255,0,.2);color:var(--acid);
}

/* PROGRESS */
.pstatus{
  font-family:'Archivo Black',sans-serif;
  font-size:.95rem;letter-spacing:.06em;color:var(--acid);
  margin-bottom:.35rem;
}
.stProgress>div{background:var(--edge2)!important;height:3px!important;border-radius:2px!important;}
.stProgress>div>div{background:var(--acid)!important;border-radius:2px!important;}

/* SCORE BANNER */
.sbanner{
  display:grid;
  grid-template-columns:180px 1fr;
  border-radius:18px;overflow:hidden;
  border:1px solid var(--edge);
  margin-bottom:2rem;
}
.sbanner-score{
  background:var(--acid);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:2rem 1.5rem;
}
.sbanner-num{
  font-family:'Archivo Black',sans-serif;
  font-size:5.5rem;line-height:1;color:var(--ink);letter-spacing:-.03em;
}
.sbanner-den{font-size:.65rem;font-weight:700;letter-spacing:.18em;color:rgba(6,6,10,.45);text-transform:uppercase;}
.sbanner-stats{
  background:var(--card);
  display:grid;grid-template-columns:repeat(6,1fr);
}
.ss{padding:1rem .75rem;border-left:1px solid var(--edge);display:flex;flex-direction:column;justify-content:center;}
.ss-v{font-family:'Archivo Black',sans-serif;font-size:1.9rem;line-height:1;color:var(--txt);}
.ss-v.g{color:var(--green);}.ss-v.w{color:var(--red);}
.ss-k{font-size:.57rem;font-weight:600;letter-spacing:.15em;text-transform:uppercase;color:var(--sub);margin-top:.25rem;}

/* CARD GRID */
.cgrid{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;}
.card{background:var(--card);border:1px solid var(--edge);border-radius:16px;padding:1.5rem;position:relative;overflow:hidden;}
.card::after{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(187,255,0,.02) 0%,transparent 60%);pointer-events:none;}
.card-full{grid-column:1/-1;}

/* SECTION TITLE */
.st2{
  font-size:.6rem;font-weight:700;letter-spacing:.22em;text-transform:uppercase;
  color:var(--acid);margin-bottom:1rem;
  display:flex;align-items:center;gap:.6rem;
}
.st2::after{content:'';flex:1;height:1px;background:var(--edge);}

/* COACH ITEMS */
.ci{
  display:flex;align-items:flex-start;gap:.7rem;
  padding:.75rem .9rem;border-radius:9px;
  margin-bottom:.45rem;font-size:.85rem;line-height:1.55;
}
.ci.pos{background:rgba(46,204,113,.07);border:1px solid rgba(46,204,113,.14);color:#7ee8a2;}
.ci.imp{background:rgba(255,63,63,.07);border:1px solid rgba(255,63,63,.14);color:#ffaaaa;}
.ci.foc{background:rgba(187,255,0,.05);border:1px solid rgba(187,255,0,.12);color:#cbff5e;}
.ci-icon{font-size:.8rem;margin-top:.18rem;flex-shrink:0;}

/* SCORE BARS */
.bw{margin-bottom:.8rem;}
.brow{display:flex;justify-content:space-between;margin-bottom:.2rem;}
.bname{font-size:.73rem;font-weight:500;color:var(--sub);}
.bval{font-family:'Archivo Black',sans-serif;font-size:.78rem;color:var(--txt);}
.btrack{height:3px;background:var(--edge2);border-radius:2px;overflow:hidden;}
.bfill{height:100%;border-radius:2px;background:var(--acid);}
.bfill.mid{background:#F59E0B;}.bfill.low{background:var(--red);}

/* AI REPORT */
.rbox{
  background:var(--surface);border-radius:10px;
  padding:1.25rem 1.4rem;
  font-size:.85rem;line-height:1.85;color:var(--sub);
  white-space:pre-wrap;border-top:2px solid var(--acid);
}

/* FLAG */
.flag{
  display:flex;gap:.6rem;align-items:flex-start;
  padding:.65rem .9rem;border-radius:8px;
  background:rgba(255,63,63,.05);border:1px solid rgba(255,63,63,.12);
  font-size:.82rem;color:#ffaaaa;margin-bottom:.4rem;line-height:1.5;
}
.fdot{width:5px;height:5px;border-radius:50%;background:var(--red);flex-shrink:0;margin-top:.5rem;}

/* DIVIDER */
.div{border:none;border-top:1px solid var(--edge);margin:1.1rem 0;}

/* EXPANDER */
.streamlit-expanderHeader{
  background:var(--surface)!important;border:1px solid var(--edge)!important;
  border-radius:9px!important;font-size:.8rem!important;color:var(--sub)!important;
}

/* EMPTY */
.empty-state{
  text-align:center;padding:5rem 1rem;
  display:flex;flex-direction:column;align-items:center;gap:1rem;
}
.empty-logo{
  font-family:'Archivo Black',sans-serif;
  font-size:clamp(4rem,12vw,10rem);
  line-height:1;letter-spacing:-.02em;
  color:rgba(187,255,0,.05);
}
.empty-logo b{color:rgba(187,255,0,.08);}
.empty-txt{font-size:.9rem;color:var(--sub2);max-width:300px;line-height:1.7;}

/* VIDEO in upload zone */
.vid-preview{border-radius:12px;overflow:hidden;border:1px solid var(--edge);}

/* FOOT */
.foot{text-align:center;font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:var(--sub2);margin-top:2.5rem;}

/* MOBILE */
@media(max-width:900px){
  .nav{padding:0 1rem;}
  .main{padding:1.5rem 1rem 4rem;}
  .upload-zone{grid-template-columns:1fr;}
  .sbanner{grid-template-columns:1fr;}
  .sbanner-score{flex-direction:row;gap:1.5rem;padding:1.25rem;}
  .sbanner-num{font-size:4rem;}
  .sbanner-stats{grid-template-columns:repeat(3,1fr);}
  .cgrid{grid-template-columns:1fr;}
  .card-full{grid-column:auto;}
}
@media(max-width:480px){
  .sbanner-stats{grid-template-columns:repeat(2,1fr);}
  .uz-headline{font-size:1.8rem;}
}
</style>""", unsafe_allow_html=True)


# ─── helpers ─────────────────────────────────────────────────────

def run_cmd(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def latest_session_dir(root, after_ts):
    root = Path(root)
    if not root.exists():
        raise RuntimeError("Missing: " + str(root))
    cands = [(p.stat().st_mtime, p) for p in root.iterdir()
             if p.is_dir() and p.stat().st_mtime >= after_ts - 1.0]
    if cands: return sorted(cands)[-1][1]
    all_d = [(p.stat().st_mtime, p) for p in root.iterdir() if p.is_dir()]
    if not all_d: raise RuntimeError("No session dirs.")
    return sorted(all_d)[-1][1]

def get_secret(key):
    try:
        v = st.secrets[key]
        if v: return str(v).strip()
    except Exception: pass
    return os.getenv(key, "").strip()

def call_llm(prompt):
    gk = get_secret("GROQ_API_KEY")
    ok = get_secret("OPENAI_API_KEY")
    tk = get_secret("TOGETHER_API_KEY")
    hk = get_secret("HF_API_KEY")
    if gk:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": "Bearer " + gk, "Content-Type": "application/json"},
            json={"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":prompt}],"max_tokens":600},
            timeout=30)
        if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
        raise Exception("Groq " + str(r.status_code))
    if ok:
        import openai
        c = openai.OpenAI(api_key=ok)
        return c.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],max_tokens=600).choices[0].message.content
    if tk:
        r = requests.post("https://api.together.xyz/v1/chat/completions",
            headers={"Authorization":"Bearer "+tk,"Content-Type":"application/json"},
            json={"model":"meta-llama/Llama-3-8b-chat-hf","messages":[{"role":"user","content":prompt}],"max_tokens":600},
            timeout=30)
        if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
        raise Exception("Together " + str(r.status_code))
    if hk:
        r = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            headers={"Authorization":"Bearer "+hk},
            json={"inputs":"<s>[INST] "+prompt+" [/INST]","parameters":{"max_new_tokens":500,"return_full_text":False}},
            timeout=60)
        if r.status_code == 200: return r.json()[0].get("generated_text","").strip()
        raise Exception("HF " + str(r.status_code))
    raise Exception("No API key. Add GROQ_API_KEY to Streamlit secrets.")

def safe(v):
    import math
    if v is None: return 0.0
    try:
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    except: return 0.0

def sv(sc, k): return str(int(safe(sc.get(k,0))))
def scls(sc, k): return "g" if safe(sc.get(k,0)) >= 75 else "w"
def bcls(v):
    v = safe(v)
    if v >= 75: return ""
    if v >= 50: return "mid"
    return "low"

def feedback(summary):
    s = summary.get("scores", {})
    flags = summary.get("flags", [])
    pos, imp, nxt = [], [], []
    if safe(s.get("hinge_quality",    0)) >= 80: pos.append("Strong hip hinge. Firing the right muscles.")
    if safe(s.get("symmetry",         0)) >= 80: pos.append("Excellent left-right balance throughout.")
    if safe(s.get("trunk_control",    0)) >= 80: pos.append("Solid torso control and bracing.")
    if safe(s.get("setup_consistency",0)) >= 80: pos.append("Consistent setup across all reps.")
    if safe(s.get("tempo_consistency",100)) < 75: imp.append("Pull speed varies. Aim for a steady cadence each rep.")
    if safe(s.get("setup_consistency",100)) < 70: imp.append("Foot position shifts. Reset identically every rep.")
    for f in flags[:2]: imp.append(f.get("message","Form issue detected."))
    if safe(s.get("tempo_consistency",100)) < 75: nxt.append("Pause 1 sec at the bottom then drive with consistent speed.")
    nxt.append("Keep bar close and brace your core before every single rep.")
    if not pos: pos = ["Good effort. Form was trackable throughout."]
    if not imp: imp = ["No major form issues detected. Keep pushing."]
    return pos[:2], imp[:2], nxt[:2]


# ─── NAV ─────────────────────────────────────────────────────────
st.markdown(
    '<nav class="nav">'
    '<div class="logo"><span class="logo-form">FORM</span><span class="logo-ate">ate</span></div>'
    '<div class="nav-pills">'
    '<span class="pill active">Analysis</span>'
    '<span class="pill">AI Coach</span>'
    '<span class="pill">History</span>'
    '</div>'
    '</nav>',
    unsafe_allow_html=True
)

st.markdown('<div class="main">', unsafe_allow_html=True)

# ─── UPLOAD ZONE ─────────────────────────────────────────────────
st.markdown('<div class="upload-zone">', unsafe_allow_html=True)

uz_left, uz_right = st.columns([1, 1], gap="large")

with uz_left:
    st.markdown(
        '<div class="uz-left">'
        '<h1 class="uz-headline">Analyse Your<br><span>Form.</span></h1>'
        '<p class="uz-desc">Upload a workout video and get instant AI-powered pose analysis, rep counting, and personalised coaching.</p>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('<p class="lbl">Exercise Type</p>', unsafe_allow_html=True)
    exercise = st.selectbox("Exercise", ["deadlift", "squat"])
    st.markdown('<p class="lbl">Camera Angle</p>', unsafe_allow_html=True)
    camera_view = st.selectbox("Camera", ["front_oblique", "side"])

with uz_right:
    st.markdown('<p class="lbl">Upload Video</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["mp4","mov","m4v"], label_visibility="collapsed")
    if uploaded:
        fname = uploaded.name
        fmb   = str(round(uploaded.size / (1024*1024), 1))
        st.markdown(
            '<div class="fok">&#10003; ' + fname + ' &middot; ' + fmb + ' MB</div>',
            unsafe_allow_html=True
        )
        tmp_dir   = Path(tempfile.mkdtemp())
        tmp_video = tmp_dir / uploaded.name
        tmp_video.write_bytes(uploaded.read())
        st.markdown('<p class="lbl" style="margin-top:.75rem;">Preview</p>', unsafe_allow_html=True)
        st.markdown('<div class="vid-preview">', unsafe_allow_html=True)
        st.video(str(tmp_video))
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# RUN BUTTON full width below upload zone
if uploaded:
    run_clicked = st.button("ANALYSE FORM", type="primary", use_container_width=True)
else:
    run_clicked = False

# ─── EMPTY STATE ─────────────────────────────────────────────────
if not uploaded:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-logo"><b>FORM</b>ate</div>'
        '<p class="empty-txt">Upload a video above to see your form breakdown, rep count, and AI coaching report.</p>'
        '</div>',
        unsafe_allow_html=True
    )

# ─── PIPELINE ────────────────────────────────────────────────────
elif run_clicked:
    prog = st.progress(0)
    stat = st.empty()

    stat.markdown('<p class="pstatus">Extracting pose data...</p>', unsafe_allow_html=True)
    prog.progress(8)
    t0 = time.time()
    p  = run_cmd([PY, "pipeline_bronze_extract.py",
                  "--video", str(tmp_video),
                  "--exercise", exercise,
                  "--camera_view", camera_view,
                  "--overlay"], cwd=ROOT)
    if p.returncode != 0:
        st.error("Pose extraction failed.")
        with st.expander("Error details"): st.code(p.stderr or p.stdout)
        st.stop()
    b_sess     = latest_session_dir(ROOT / "pipeline" / "bronze", after_ts=t0)
    b_sum      = read_json(b_sess / "summary.json")
    session_id = b_sum["session_id"]
    session_dir= b_sum["outputs"]["session_dir"]
    prog.progress(35)

    stat.markdown('<p class="pstatus">Detecting reps...</p>', unsafe_allow_html=True)
    prog.progress(42)
    p = run_cmd([PY, "pipeline_silver_transform.py", "--session_dir", session_dir], cwd=ROOT)
    if p.returncode != 0:
        st.error("Rep detection failed.")
        with st.expander("Error details"): st.code(p.stderr or p.stdout)
        st.stop()
    s_sum    = read_json(ROOT / "pipeline" / "silver" / session_id / "summary.json")
    num_reps = s_sum["num_reps_detected"]
    prog.progress(65)

    stat.markdown('<p class="pstatus">Scoring form...</p>', unsafe_allow_html=True)
    prog.progress(72)
    p = run_cmd([PY, "pipeline_gold_score.py", "--session_id", session_id, "--exercise", exercise], cwd=ROOT)
    if p.returncode != 0:
        st.error("Scoring failed.")
        with st.expander("Error details"): st.code(p.stderr or p.stdout)
        st.stop()
    gold_dir = ROOT / "pipeline" / "gold" / session_id
    g_sum    = read_json(gold_dir / "summary.json")
    rep_df   = pd.read_parquet(gold_dir / "metrics_reps.parquet")
    prog.progress(100)
    stat.empty()
    prog.empty()

    scores  = g_sum["scores"]
    overall = safe(scores.get("overall", 0))
    reps    = g_sum.get("reps", num_reps) or num_reps
    cam_lbl = camera_view.replace("_", " ").title()

    # ── SCORE BANNER ──
    st.markdown(
        '<div class="sbanner">'
        '<div class="sbanner-score">'
        '<div class="sbanner-num">' + str(int(overall)) + '</div>'
        '<div class="sbanner-den">/ 100</div>'
        '</div>'
        '<div class="sbanner-stats">'
        '<div class="ss"><div class="ss-v">' + str(reps) + '</div><div class="ss-k">Reps</div></div>'
        '<div class="ss"><div class="ss-v ' + scls(scores,"hinge_quality") + '">' + sv(scores,"hinge_quality") + '</div><div class="ss-k">Hinge</div></div>'
        '<div class="ss"><div class="ss-v ' + scls(scores,"trunk_control") + '">' + sv(scores,"trunk_control") + '</div><div class="ss-k">Trunk</div></div>'
        '<div class="ss"><div class="ss-v ' + scls(scores,"symmetry") + '">' + sv(scores,"symmetry") + '</div><div class="ss-k">Symmetry</div></div>'
        '<div class="ss"><div class="ss-v ' + scls(scores,"tempo_consistency") + '">' + sv(scores,"tempo_consistency") + '</div><div class="ss-k">Tempo</div></div>'
        '<div class="ss"><div class="ss-v ' + scls(scores,"setup_consistency") + '">' + sv(scores,"setup_consistency") + '</div><div class="ss-k">Setup</div></div>'
        '</div></div>',
        unsafe_allow_html=True
    )

    # ── CARD GRID ──
    st.markdown('<div class="cgrid">', unsafe_allow_html=True)
   