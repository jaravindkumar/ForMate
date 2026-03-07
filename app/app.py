import json, os, time, tempfile, requests, subprocess, sys, uuid, math
from pathlib import Path
import numpy as np
import streamlit as st
import pandas as pd
import cv2

# ROOT = directory containing the pipeline scripts
# On Streamlit Cloud, app.py may be in a subdirectory (e.g. /mount/src/formate/app/)
# while pipeline scripts sit in the repo root (/mount/src/formate/)
def _find_root():
    here = Path(__file__).resolve().parent
    if (here / "pipeline_bronze_extract.py").exists():
        return here
    if (here.parent / "pipeline_bronze_extract.py").exists():
        return here.parent
    return here  # fallback — will show clear error

ROOT = _find_root()
PY   = sys.executable

st.set_page_config(page_title="FORMate", layout="wide", page_icon="F", initial_sidebar_state="collapsed")

# ─── CSS ──────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root{
  --bg:#07070F;
  --surface:#0C0C18;
  --card:#10101E;
  --card2:#141428;
  --edge:#1A1A30;
  --edge2:#222238;
  --txt:#EAEAF8;
  --sub:#5C5C80;
  --sub2:#22223A;
  --muted:#30304A;
  --p1:#1D4ED8;
  --p2:#3B82F6;
  --p3:#93C5FD;
  --v1:#2563EB;
  --green:#38BDF8;
  --green-bg:rgba(56,189,248,.08);
  --green-bd:rgba(56,189,248,.2);
  --amber:#60A5FA;
  --amber-bg:rgba(96,165,250,.08);
  --amber-bd:rgba(96,165,250,.2);
  --red:#EF4444;
  --red-bg:rgba(239,68,68,.08);
  --red-bd:rgba(239,68,68,.2);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:var(--bg)!important;color:var(--txt)!important;-webkit-font-smoothing:antialiased;}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden!important;display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stSidebar"]{display:none!important;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-track{background:var(--bg);}::-webkit-scrollbar-thumb{background:var(--edge2);border-radius:4px;}

/* NAV */
.nav{display:flex;align-items:center;justify-content:space-between;padding:0 2rem;height:56px;background:rgba(7,7,15,.9);border-bottom:1px solid var(--edge);position:sticky;top:0;z-index:100;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);}
.logo{font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:700;letter-spacing:-.03em;display:flex;align-items:center;gap:.05rem;}
.logo-form{background:linear-gradient(135deg,var(--p1) 0%,var(--v1) 60%,var(--p2) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.logo-ate{color:rgba(255,255,255,.65);}
.logo-dot{width:5px;height:5px;border-radius:50%;background:var(--p1);margin-left:.2rem;margin-bottom:.8rem;box-shadow:0 0 8px rgba(29,78,216,.8);}
.nav-right{display:flex;gap:.5rem;align-items:center;}
.nbadge{font-size:.58rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;padding:.25rem .65rem;border-radius:20px;background:var(--card);border:1px solid var(--edge2);color:var(--sub);}
.nbadge.live{background:rgba(29,78,216,.15);border-color:rgba(59,130,246,.35);color:#60A5FA;}
.nbadge.live::before{content:'';display:inline-block;width:5px;height:5px;border-radius:50%;background:#3B82F6;margin-right:.4rem;animation:blink .9s infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.15;}}

/* MAIN */
.main{max-width:1320px;margin:0 auto;padding:2rem 2rem 6rem;}

/* TABS */
.stTabs [data-baseweb="tab-list"]{background:var(--card)!important;border:1px solid var(--edge)!important;border-radius:14px!important;padding:.3rem!important;gap:.2rem!important;margin-bottom:2rem!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;border-radius:10px!important;color:var(--sub)!important;font-family:'Space Grotesk',sans-serif!important;font-size:.75rem!important;font-weight:600!important;letter-spacing:.07em!important;text-transform:uppercase!important;padding:.5rem 1.25rem!important;border:none!important;transition:all .18s!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,var(--p1),var(--v1))!important;color:#fff!important;box-shadow:0 4px 18px rgba(29,78,216,.3)!important;}
.stTabs [data-baseweb="tab-panel"]{padding:0!important;}
.stTabs [data-baseweb="tab-border"]{display:none!important;}

/* UPLOAD / LIVE ZONES */
.zone,.live-zone{background:var(--card);border:1px solid var(--edge);border-radius:20px;padding:2.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}
.zone::before{content:'';position:absolute;top:-80px;right:-80px;width:300px;height:300px;border-radius:50%;background:radial-gradient(circle,rgba(29,78,216,.06) 0%,transparent 70%);pointer-events:none;}
.live-zone::before{content:'';position:absolute;top:-80px;left:-80px;width:300px;height:300px;border-radius:50%;background:radial-gradient(circle,rgba(37,99,235,.06) 0%,transparent 70%);pointer-events:none;}
.uz-headline,.live-headline{font-family:'Space Grotesk',sans-serif;font-size:clamp(1.75rem,4vw,2.75rem);line-height:1.1;font-weight:700;letter-spacing:-.03em;color:var(--txt);margin-bottom:.6rem;}
.uz-headline span{background:linear-gradient(135deg,var(--p1),var(--v1));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.live-headline span{color:#60A5FA;}
.uz-desc,.live-desc{font-size:.87rem;color:var(--sub);line-height:1.75;max-width:400px;margin-bottom:1.5rem;}

/* LABELS */
.lbl{font-size:.57rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--sub);margin-bottom:.3rem;display:flex;align-items:center;gap:.5rem;}
.lbl::after{content:'';flex:1;height:1px;background:var(--edge);}
.st2{font-size:.57rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--p2);margin-bottom:1rem;display:flex;align-items:center;gap:.6rem;}
.st2::after{content:'';flex:1;height:1px;background:var(--edge);}

/* CARDS */
.card{background:var(--card);border:1px solid var(--edge);border-radius:16px;padding:1.5rem;position:relative;overflow:hidden;}
.card::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(29,78,216,.02) 0%,transparent 50%);pointer-events:none;}

/* SCORE BANNER */
.sbanner{display:grid;grid-template-columns:160px 1fr;border-radius:20px;overflow:hidden;border:1px solid var(--edge);margin-bottom:2rem;}
.sbanner-score{background:linear-gradient(145deg,var(--p1) 0%,var(--v1) 100%);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:2rem 1.25rem;}
.sbanner-num{font-family:'Space Grotesk',sans-serif;font-size:5rem;line-height:1;color:#fff;font-weight:700;letter-spacing:-.04em;}
.sbanner-den{font-size:.58rem;font-weight:600;letter-spacing:.2em;color:rgba(255,255,255,.45);text-transform:uppercase;margin-top:.2rem;}
.sbanner-stats{display:grid;grid-template-columns:repeat(6,1fr);background:var(--card);}
.ss{padding:1.1rem .75rem;border-left:1px solid var(--edge);display:flex;flex-direction:column;justify-content:center;gap:.3rem;}
.ss-v{font-family:'Space Grotesk',sans-serif;font-size:1.7rem;line-height:1;color:var(--txt);font-weight:700;}
.ss-v.g{color:var(--green);}.ss-v.w{color:var(--red);}
.ss-k{font-size:.53rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--sub);}

/* SCORE BARS */
.bw{margin-bottom:.9rem;}
.brow{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.35rem;}
.bname{font-size:.78rem;font-weight:500;color:var(--sub);}
.bval{font-family:'Space Grotesk',sans-serif;font-size:.82rem;font-weight:600;color:var(--txt);}
.btrack{height:4px;background:var(--edge2);border-radius:4px;overflow:hidden;}
.bfill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--p1),var(--p2));}
.bfill.mid{background:linear-gradient(90deg,var(--amber),#93C5FD);}
.bfill.low{background:linear-gradient(90deg,var(--red),#F87171);}

/* COACH FEEDBACK */
.ci{display:flex;align-items:flex-start;gap:.75rem;padding:.8rem 1rem;border-radius:10px;margin-bottom:.4rem;font-size:.84rem;line-height:1.6;}
.ci.pos{background:var(--green-bg);border:1px solid var(--green-bd);color:#BAE6FD;}
.ci.imp{background:rgba(30,58,138,.15);border:1px solid rgba(59,130,246,.25);color:#93C5FD;}
.ci.foc{background:rgba(29,78,216,.07);border:1px solid rgba(29,78,216,.18);color:var(--p3);}
.ci-icon{font-size:.75rem;margin-top:.2rem;flex-shrink:0;opacity:.8;}

/* FLAGS */
.flag{display:flex;gap:.65rem;align-items:flex-start;padding:.7rem .95rem;border-radius:10px;background:rgba(30,58,138,.2);border:1px solid rgba(59,130,246,.3);font-size:.82rem;color:#93C5FD;margin-bottom:.4rem;line-height:1.55;}
.fdot{width:5px;height:5px;border-radius:50%;background:#3B82F6;flex-shrink:0;margin-top:.45rem;}

/* REPORT BOX */
.rbox{background:var(--surface);border-radius:12px;padding:1.25rem 1.4rem;font-size:.86rem;line-height:1.9;color:var(--sub);white-space:pre-wrap;border-left:3px solid var(--p1);}

/* BUTTONS */
.stButton>button{font-family:'Space Grotesk',sans-serif!important;font-size:.87rem!important;font-weight:700!important;letter-spacing:.06em!important;text-transform:uppercase!important;background:linear-gradient(135deg,var(--p1),var(--v1))!important;color:#fff!important;border:none!important;border-radius:12px!important;padding:.85rem 2rem!important;width:100%!important;transition:all .2s!important;box-shadow:0 4px 20px rgba(29,78,216,.22)!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 32px rgba(29,78,216,.4)!important;filter:brightness(1.08)!important;}
.stop-btn .stButton>button{background:var(--red-bg)!important;color:#F87171!important;border:1px solid var(--red-bd)!important;box-shadow:none!important;}
.stop-btn .stButton>button:hover{background:rgba(239,68,68,.14)!important;box-shadow:0 4px 18px rgba(239,68,68,.15)!important;}

/* FILE OK */
.fok{padding:.65rem 1rem;border-radius:10px;font-size:.8rem;background:rgba(29,78,216,.08);border:1px solid rgba(29,78,216,.2);color:var(--p2);display:flex;align-items:center;gap:.5rem;}

/* PROGRESS */
.pstatus{font-family:'Space Grotesk',sans-serif;font-size:.88rem;font-weight:600;color:var(--p2);margin-bottom:.4rem;}
.stProgress>div{background:var(--edge2)!important;height:3px!important;border-radius:3px!important;}
.stProgress>div>div{background:linear-gradient(90deg,var(--p1),var(--p2))!important;border-radius:3px!important;}

/* STATUS BADGE */
.status-badge{display:inline-flex;align-items:center;gap:.45rem;padding:.35rem .9rem;border-radius:20px;font-size:.63rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;margin-bottom:1rem;}
.status-badge.waiting{background:var(--amber-bg);border:1px solid var(--amber-bd);color:#93C5FD;}
.status-badge.detecting{background:rgba(29,78,216,.1);border:1px solid rgba(29,78,216,.25);color:var(--p2);}
.status-badge.recording{background:var(--red-bg);border:1px solid var(--red-bd);color:#F87171;}
.status-badge.done{background:var(--green-bg);border:1px solid var(--green-bd);color:#93C5FD;}
.status-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;}
.status-badge.waiting .status-dot{background:var(--amber);}
.status-badge.detecting .status-dot{background:var(--p1);animation:blink 1s infinite;}
.status-badge.recording .status-dot{background:var(--red);animation:blink .7s infinite;}
.status-badge.done .status-dot{background:var(--green);}

/* REP COUNTER */
.rep-counter{display:flex;flex-direction:column;align-items:center;justify-content:center;background:linear-gradient(145deg,var(--card),var(--card2));border:1px solid var(--edge);border-radius:20px;padding:2rem;text-align:center;}
.rep-num{font-family:'Space Grotesk',sans-serif;font-size:5.5rem;line-height:1;font-weight:700;letter-spacing:-.04em;background:linear-gradient(135deg,var(--p2),var(--v1));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.rep-label{font-size:.55rem;font-weight:600;letter-spacing:.2em;text-transform:uppercase;color:var(--sub);margin-top:.4rem;}
.rep-exercise{font-size:.7rem;color:var(--muted);margin-top:.2rem;text-transform:uppercase;letter-spacing:.1em;}

/* LIVE STEPS */
.live-steps{display:flex;flex-direction:column;gap:.5rem;margin-bottom:1.25rem;}
.live-step{display:flex;align-items:flex-start;gap:.8rem;padding:.7rem 1rem;border-radius:10px;background:var(--card2);border:1px solid var(--edge);font-size:.82rem;color:var(--sub);line-height:1.55;}
.step-num{font-family:'Space Grotesk',sans-serif;font-size:.9rem;font-weight:700;color:var(--p2);flex-shrink:0;line-height:1;margin-top:.05rem;}

/* FORM INPUTS */
.stSelectbox>div>div{background:var(--card)!important;border:1px solid var(--edge2)!important;border-radius:10px!important;color:var(--txt)!important;font-size:.87rem!important;}
.stSelectbox label{display:none!important;}
[data-testid="stFileUploader"]{background:var(--card)!important;border:1.5px dashed var(--edge2)!important;border-radius:14px!important;transition:border-color .2s!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--p1)!important;}
[data-testid="stFileUploader"] *{font-size:.85rem!important;}

/* MISC */
.div{border:none;border-top:1px solid var(--edge);margin:1.25rem 0;}
.foot{text-align:center;font-size:.6rem;letter-spacing:.14em;text-transform:uppercase;color:var(--sub2);margin-top:3rem;}
.vid-preview{border-radius:14px;overflow:hidden;border:1px solid var(--edge);}
.streamlit-expanderHeader{background:var(--card2)!important;border:1px solid var(--edge)!important;border-radius:10px!important;font-size:.8rem!important;color:var(--sub)!important;}

/* EMPTY STATE */
.empty-state{text-align:center;padding:5rem 1rem;display:flex;flex-direction:column;align-items:center;gap:1rem;}
.empty-logo{font-family:'Space Grotesk',sans-serif;font-size:clamp(4rem,12vw,9rem);line-height:1;font-weight:700;letter-spacing:-.04em;}
.empty-logo b{background:linear-gradient(135deg,rgba(29,78,216,.13),rgba(37,99,235,.2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.empty-logo span{color:rgba(255,255,255,.04);}
.empty-txt{font-size:.88rem;color:var(--sub2);max-width:280px;line-height:1.75;}

/* MOBILE */
@media(max-width:900px){
  .nav{padding:0 1rem;}.main{padding:1.25rem 1rem 5rem;}
  .sbanner{grid-template-columns:1fr;}
  .sbanner-score{flex-direction:row;gap:1.5rem;padding:1.25rem;}
  .sbanner-num{font-size:3.5rem;}
  .sbanner-stats{grid-template-columns:repeat(3,1fr);}
}
@media(max-width:480px){
  .sbanner-stats{grid-template-columns:repeat(2,1fr);}
  .uz-headline,.live-headline{font-size:1.75rem;}
  .zone,.live-zone{padding:1.5rem;}
}
</style>""", unsafe_allow_html=True)



# ─── HELPERS ──────────────────────────────────────────────────────

def run_cmd(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def latest_session_dir(root, after_ts):
    root = Path(root)
    if not root.exists(): raise RuntimeError("Missing: " + str(root))
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


# ─── LIVE TRAINER HELPERS ─────────────────────────────────────────

def decode_frame(img_file):
    """Decode st.camera_input image to OpenCV BGR frame."""
    arr = np.frombuffer(img_file.getvalue(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ── THRESHOLD DEFINITIONS ─────────────────────────────────────────
# Each check returns True if form is GOOD (within limits)
def check_thresholds(lms, exercise):
    """
    Evaluate per-joint form thresholds.
    Returns dict: joint_name -> ("ok"|"warn"|"bad", label_string)
    Uses MediaPipe normalised coords (0-1).
    """
    P = lms  # shorthand
    issues = {}

    def angle_2d(ax, ay, bx, by, cx, cy):
        """Angle at point B in triangle ABC."""
        v1x, v1y = ax - bx, ay - by
        v2x, v2y = cx - bx, cy - by
        dot  = v1x*v2x + v1y*v2y
        mag1 = math.sqrt(v1x**2 + v1y**2) + 1e-9
        mag2 = math.sqrt(v2x**2 + v2y**2) + 1e-9
        a    = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(a))

    if exercise == "deadlift":
        # 1. Back angle: shoulder-hip-knee angle — should be > 120 deg (upright)
        back_angle = angle_2d(
            P["l_shoulder_x"], P["l_shoulder_y"],
            P["l_hip_x"],      P["l_hip_y"],
            P["l_knee_x"],     P["l_knee_y"],
        )
        if back_angle >= 145:   issues["back"] = ("ok",   "Back OK")
        elif back_angle >= 120: issues["back"] = ("warn", "Back rounding")
        else:                   issues["back"] = ("bad",  "Severe back round!")

        # 2. Bar drift: shoulders should stay over hips horizontally
        shoulder_drift = abs(P["l_shoulder_x"] - P["l_hip_x"])
        if shoulder_drift < 0.08:   issues["drift"] = ("ok",   "Bar path OK")
        elif shoulder_drift < 0.15: issues["drift"] = ("warn", "Bar drifting")
        else:                       issues["drift"] = ("bad",  "Bar too far!")

        # 3. Hip hinge depth: hip should drop during pull (y increases)
        hip_knee_diff = P["l_knee_y"] - P["l_hip_y"]
        if hip_knee_diff > 0.05:    issues["hinge"] = ("ok",  "Good hinge")
        else:                       issues["hinge"] = ("warn","Hinge deeper")

    else:  # squat
        # 1. Knee cave: knee x should stay outside (or at) hip x
        l_knee_cave = P["l_knee_x"] - P["l_hip_x"]   # negative = caving inward
        r_knee_cave = P["r_hip_x"]  - P["r_knee_x"]  # negative = caving inward
        knee_cave   = min(l_knee_cave, r_knee_cave)
        if knee_cave >= -0.02:   issues["knee"] = ("ok",   "Knees tracking OK")
        elif knee_cave >= -0.06: issues["knee"] = ("warn", "Knee caving")
        else:                    issues["knee"] = ("bad",  "Knee cave!")

        # 2. Squat depth: hip y should be >= knee y (hip below knee at bottom)
        hip_depth = P["l_hip_y"] - P["l_knee_y"]
        if hip_depth >= 0.02:    issues["depth"] = ("ok",   "Good depth")
        elif hip_depth >= -0.05: issues["depth"] = ("warn", "Go deeper")
        else:                    issues["depth"] = ("bad",  "Shallow squat")

        # 3. Forward lean: shoulder should not be too far forward of hip
        lean = P["l_shoulder_x"] - P["l_hip_x"]
        if abs(lean) < 0.08:   issues["lean"] = ("ok",   "Upright OK")
        elif abs(lean) < 0.15: issues["lean"] = ("warn", "Leaning forward")
        else:                  issues["lean"] = ("bad",  "Too much lean!")

    return issues


def get_pose_result(frame):
    """Run MediaPipe Pose on a single frame. Returns (landmarks_dict, raw_result, mp_pose) or (None,None,None)."""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose    = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            def lx(lm_id): return lms[lm_id].x
            def ly(lm_id): return lms[lm_id].y
            L = mp_pose.PoseLandmark
            d = {
                "l_hip_y":      ly(L.LEFT_HIP),
                "r_hip_y":      ly(L.RIGHT_HIP),
                "l_knee_y":     ly(L.LEFT_KNEE),
                "r_knee_y":     ly(L.RIGHT_KNEE),
                "l_shoulder_y": ly(L.LEFT_SHOULDER),
                "r_shoulder_y": ly(L.RIGHT_SHOULDER),
                "l_hip_x":      lx(L.LEFT_HIP),
                "r_hip_x":      lx(L.RIGHT_HIP),
                "l_knee_x":     lx(L.LEFT_KNEE),
                "r_knee_x":     lx(L.RIGHT_KNEE),
                "l_shoulder_x": lx(L.LEFT_SHOULDER),
                "r_shoulder_x": lx(L.RIGHT_SHOULDER),
                "nose_y":       ly(L.NOSE),
                "l_hip_y":      ly(L.LEFT_HIP),
            }
            return d, res, mp_pose, pose
        pose.close()
    except Exception:
        pass
    return None, None, None, None


def draw_skeleton_threshold(frame, exercise):
    """
    Draw skeleton on frame with threshold-aware colouring.
    Green joints = good form. Orange = warning. Red = bad.
    Returns (annotated_bgr_frame, landmarks_dict_or_None, issues_dict)
    """
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp_pose.Pose(static_image_mode=True,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            res = pose.process(rgb)
            if not res.pose_landmarks:
                return frame, None, {}

            lms = res.pose_landmarks.landmark
            L   = mp_pose.PoseLandmark

            # Build landmarks dict
            def lx(i): return lms[i].x
            def ly(i): return lms[i].y
            lm_dict = {
                "l_hip_y": ly(L.LEFT_HIP),      "r_hip_y": ly(L.RIGHT_HIP),
                "l_knee_y": ly(L.LEFT_KNEE),     "r_knee_y": ly(L.RIGHT_KNEE),
                "l_shoulder_y": ly(L.LEFT_SHOULDER), "r_shoulder_y": ly(L.RIGHT_SHOULDER),
                "l_hip_x": lx(L.LEFT_HIP),       "r_hip_x": lx(L.RIGHT_HIP),
                "l_knee_x": lx(L.LEFT_KNEE),     "r_knee_x": lx(L.RIGHT_KNEE),
                "l_shoulder_x": lx(L.LEFT_SHOULDER), "r_shoulder_x": lx(L.RIGHT_SHOULDER),
                "nose_y": ly(L.NOSE),
            }

            # Get threshold issues
            issues = check_thresholds(lm_dict, exercise)

            # Determine worst global status for connection colour
            statuses = [v[0] for v in issues.values()]
            if "bad"  in statuses: conn_color = (255, 50,  50)   # red
            elif "warn" in statuses: conn_color = (255, 160, 20)  # orange
            else:                    conn_color = (187, 255,  0)  # acid green

            h, w = frame.shape[:2]

            # Draw connections first (underneath joints)
            for conn in mp_pose.POSE_CONNECTIONS:
                a_idx, b_idx = conn
                a = lms[a_idx]; b = lms[b_idx]
                if a.visibility < 0.4 or b.visibility < 0.4: continue
                ax, ay = int(a.x * w), int(a.y * h)
                bx, by = int(b.x * w), int(b.y * h)
                cv2.line(rgb, (ax, ay), (bx, by), conn_color, 2, cv2.LINE_AA)

            # Per-joint colouring — map landmark ids to issue keys
            KEY_JOINTS = {
                "back":  [L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP],
                "drift": [L.LEFT_SHOULDER, L.RIGHT_SHOULDER],
                "hinge": [L.LEFT_HIP, L.RIGHT_HIP],
                "knee":  [L.LEFT_KNEE, L.RIGHT_KNEE],
                "depth": [L.LEFT_HIP, L.RIGHT_HIP, L.LEFT_KNEE, L.RIGHT_KNEE],
                "lean":  [L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP],
            }
            joint_status = {}  # landmark_id -> worst status
            for key, joint_ids in KEY_JOINTS.items():
                if key in issues:
                    st_val = issues[key][0]
                    for jid in joint_ids:
                        prev = joint_status.get(jid, "ok")
                        if st_val == "bad" or (st_val == "warn" and prev == "ok"):
                            joint_status[jid] = st_val

            COLOR_MAP = {"ok": (100, 230, 100), "warn": (255, 160, 20), "bad": (255, 50, 50)}
            DEFAULT_COLOR = (187, 255, 0)

            for i, lm in enumerate(lms):
                if lm.visibility < 0.4: continue
                px, py = int(lm.x * w), int(lm.y * h)
                st_val = joint_status.get(i, "ok")
                color  = COLOR_MAP.get(st_val, DEFAULT_COLOR)
                cv2.circle(rgb, (px, py), 5, color, -1, cv2.LINE_AA)
                cv2.circle(rgb, (px, py), 5, (20, 20, 20), 1, cv2.LINE_AA)

            # Draw issue labels on frame
            y_offset = 24
            for key, (st_val, label) in issues.items():
                if st_val == "ok": continue
                txt_color = (255, 160, 20) if st_val == "warn" else (255, 80, 80)
                cv2.putText(rgb, label, (12, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(rgb, label, (12, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1, cv2.LINE_AA)
                y_offset += 22

            annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return annotated, lm_dict, issues

    except Exception as e:
        return frame, None, {}

def detect_rep_event(history, exercise):
    """
    Returns True when a new completed rep is detected.
    history: list of landmark dicts (most recent last), max ~30 frames.
    Deadlift: hip Y drops then rises back (hip goes down then up on screen = y increases then decreases).
    Squat:    hip Y rises (goes down screen = y increases) then falls back.
    MediaPipe Y: 0=top, 1=bottom of image.
    """
    if len(history) < 6:
        return False

    hip_y = [(h["l_hip_y"] + h["r_hip_y"]) / 2.0 for h in history]

    # smooth
    def smooth(arr, w=3):
        out = []
        for i in range(len(arr)):
            s = arr[max(0,i-w):i+w+1]
            out.append(sum(s)/len(s))
        return out

    hip_y = smooth(hip_y)
    n = len(hip_y)

    if exercise == "deadlift":
        # Hip goes DOWN on screen (y increases) during pull, then back up
        baseline = hip_y[0]
        peak     = max(hip_y)
        current  = hip_y[-1]
        threshold = 0.04
        # rep complete when: peak was significantly above baseline AND now back near baseline
        return (peak - baseline > threshold) and (peak - current > threshold * 0.7) and (hip_y[-1] < hip_y[n//2])
    else:
        # Squat: hip drops DOWN (y increases), then comes back up
        baseline = hip_y[0]
        peak     = max(hip_y)
        current  = hip_y[-1]
        threshold = 0.06
        return (peak - baseline > threshold) and (peak - current > threshold * 0.7) and (hip_y[-1] < hip_y[n//2])

def detect_activity_start(history, exercise):
    """
    Returns True when the user starts moving into their first rep.
    Detects initial hip drop (deadlift setup hinge / squat descent).
    """
    if len(history) < 4:
        return False
    hip_y = [(h["l_hip_y"] + h["r_hip_y"]) / 2.0 for h in history]
    delta = hip_y[-1] - hip_y[0]
    # Hip moving down (y increasing) = activity starting
    return delta > 0.025

def frames_to_video(frames, fps, out_path):
    """Save list of BGR frames to mp4 video."""
    if not frames: return
    h, w = frames[0].shape[:2]
    tmp = str(out_path) + "_raw.mp4"
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames: out.write(f)
    out.release()
    # Re-encode to H.264 for browser compatibility
    subprocess.run(["ffmpeg", "-y", "-i", tmp,
                    "-vcodec", "libx264", "-crf", "28", "-preset", "fast",
                    "-movflags", "+faststart", str(out_path)],
                   capture_output=True)
    try: os.remove(tmp)
    except: pass


# ─── RESULTS RENDERER (shared by both modes) ──────────────────────

def render_results(session_id, gold_dir, b_sum, g_sum, rep_df, num_reps, exercise, live_vid=None):
    scores  = g_sum["scores"]
    overall = safe(scores.get("overall", 0))
    reps    = g_sum.get("reps", num_reps) or num_reps

    # ── Detect poor body visibility ───────────────────────────────
    # If pipeline scored from bad framing, scores will be near 0 or 50
    # (50 = hinge fallback default, 0 = no data). Warn the user.
    sym   = safe(scores.get("symmetry", 0))
    setup = safe(scores.get("setup_consistency", 0))
    hinge = safe(scores.get("hinge_quality", 0))
    body_detected = not (sym == 0 and setup == 0 and reps == 0)
    poor_framing  = (sym == 0 and setup == 0) or (overall < 5 and reps > 0)

    if poor_framing:
        st.warning(
            "⚠️ **Body not fully visible in video.** Scores below are unreliable. "
            "For accurate analysis, record with your **full body in frame** — "
            "step back 2–3 metres from the camera.",
            icon=None
        )

    # Score banner
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

    row1_l, row1_r = st.columns([1, 1], gap="medium")

    with row1_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="st2">Pose Overlay</p>', unsafe_allow_html=True)
        # Live mode: show the annotated frames video
        if live_vid and Path(live_vid).exists():
            with open(live_vid, "rb") as vf: st.video(vf.read())
        else:
            ovp   = b_sum["outputs"].get("overlay_mp4") if b_sum else None
            opath = Path(ovp) if ovp else None
            if opath and opath.exists():
                with open(str(opath), "rb") as vf: st.video(vf.read())
            else:
                st.markdown('<div style="padding:2rem;text-align:center;color:#1D1D28;font-size:.85rem;">Overlay not available</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row1_r:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="st2">Score Breakdown</p>', unsafe_allow_html=True)
        for label, key in [("Hinge Quality","hinge_quality"),("Trunk Control","trunk_control"),
                            ("Symmetry","symmetry"),("Tempo Consistency","tempo_consistency"),
                            ("Setup Consistency","setup_consistency")]:
            v  = safe(scores.get(key, 0))
            vi = int(v)
            bc = bcls(v)
            st.markdown(
                '<div class="bw"><div class="brow"><span class="bname">' + label + '</span>'
                '<span class="bval">' + str(vi) + '</span></div>'
                '<div class="btrack"><div class="bfill ' + bc + '" style="width:' + str(vi) + '%"></div></div></div>',
                unsafe_allow_html=True
            )
        flags = g_sum.get("flags", [])
        if flags:
            st.markdown('<hr class="div"/>', unsafe_allow_html=True)
            st.markdown('<p class="st2">Flags</p>', unsafe_allow_html=True)
            for f in flags:
                st.markdown('<div class="flag"><div class="fdot"></div>' + f.get("message","") + '</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    row2_l, row2_r = st.columns([1, 1], gap="medium")

    with row2_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="st2">Coach Feedback</p>', unsafe_allow_html=True)
        pos, imp, nxt = feedback(g_sum)
        for msg in pos:
            st.markdown('<div class="ci pos"><span class="ci-icon">&#10003;</span>' + msg + '</div>', unsafe_allow_html=True)
        for msg in imp:
            st.markdown('<div class="ci imp"><span class="ci-icon">&#8593;</span>' + msg + '</div>', unsafe_allow_html=True)
        for msg in nxt:
            st.markdown('<div class="ci foc"><span class="ci-icon">&#8594;</span>' + msg + '</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row2_r:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="st2">AI Coaching Report</p>', unsafe_allow_html=True)
        issues     = g_sum.get("issues", [])
        issue_list = ", ".join([i["type"] for i in issues]) if issues else "none"
        prompt = (
            "You are an elite strength coach. Write a brief coaching report for a "
            + exercise + " session.\n"
            "Score: " + str(int(overall)) + "/100, Reps: " + str(reps) + "\n"
            "Hinge: " + sv(scores,"hinge_quality") + ", Trunk: " + sv(scores,"trunk_control")
            + ", Symmetry: " + sv(scores,"symmetry") + "\n"
            "Issues: " + issue_list + "\n\n"
            "Write 3 short sharp paragraphs: what went well, what to fix, next session focus. "
            "Be direct and motivating. Plain text only."
        )
        with st.spinner(""):
            try:
                report = call_llm(prompt)
                (gold_dir / "llm_report.txt").write_text(report, encoding="utf-8")
            except Exception as e:
                st.error("LLM: " + str(e))
                report = (
                    "Score: " + str(int(overall)) + "/100 over " + str(reps) + " reps.\n\n"
                    "Solid effort. Focus on consistent bar path and bracing.\n\n"
                    "Next session: perfect your setup, brace hard, drive with control."
                )
        st.markdown('<div class="rbox">' + report + '</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if issues and b_sum:
        try:
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
            snap_col, _ = st.columns([1, 1], gap="medium")
            with snap_col:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="st2">Form Snapshots</p>', unsafe_allow_html=True)
                for issue in issues:
                    with st.expander(issue["type"].replace("_"," ").title()):
                        st.caption(issue["description"])
                        ic = st.columns(min(len(issue["frames"]), 3))
                        for i, fi in enumerate(issue["frames"][:3]):
                            ip = snapshots_dir / (issue["type"] + "_" + str(fi) + ".jpg")
                            if ip.exists():
                                ic[i].image(str(ip), use_container_width=True, caption="Frame " + str(fi))
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception:
            pass

    rep_col, _ = st.columns([1, 1], gap="medium")
    with rep_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.expander("Per-Rep Breakdown"):
            st.dataframe(rep_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="foot">FORMate &middot; ' + session_id[:16] + ' &middot; Powered by MediaPipe + AI</p>',
        unsafe_allow_html=True
    )


def run_pipeline(tmp_video, exercise, camera_view):
    """Run bronze/silver/gold pipeline in-process."""
    import importlib.util, traceback as _tb

    def load_module(name):
        path = ROOT / f"{name}.py"
        if not path.exists():
            raise FileNotFoundError(f"Script not found: {path}\nROOT={ROOT}\nFiles in ROOT: {list(ROOT.glob('*.py'))}")
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    _orig_cwd = os.getcwd()
    os.chdir(str(ROOT))

    prog = st.progress(0)

    try:
        # ── BRONZE ────────────────────────────────────────────────
        with st.status("Step 1/3 — Extracting pose data…", expanded=True) as s1:
            st.write(f"Video: `{tmp_video}` ({Path(tmp_video).stat().st_size//1024} KB)")
            st.write(f"ROOT: `{ROOT}`")
            st.write(f"Exercise: `{exercise}`")
            bronze = load_module("pipeline_bronze_extract")
            session_id = bronze.extract_bronze(
                video_path   = str(tmp_video),
                out_root     = str(ROOT / "pipeline" / "bronze"),
                exercise     = exercise,
                camera_view  = camera_view,
                write_overlay= False,
            )
            b_sess = ROOT / "pipeline" / "bronze" / session_id
            b_sum  = read_json(b_sess / "summary.json")
            session_dir = b_sum["outputs"]["session_dir"]
            st.write(f"✅ Pose extracted — {b_sum['frames_processed']} frames, "
                     f"{b_sum['pose_detected_frames']} detected "
                     f"({b_sum['pose_detected_ratio']*100:.0f}%)")
            s1.update(label="Step 1/3 — Pose extraction complete ✅", state="complete")
        prog.progress(40)

        # ── SILVER ────────────────────────────────────────────────
        with st.status("Step 2/3 — Detecting reps…", expanded=True) as s2:
            silver   = load_module("pipeline_silver_transform")
            s_sum    = silver.run_silver(session_dir)
            num_reps = s_sum.get("num_reps_detected", 0)
            st.write(f"✅ {num_reps} reps detected")
            s2.update(label=f"Step 2/3 — Rep detection complete ✅  ({num_reps} reps)", state="complete")
        prog.progress(70)

        # ── GOLD ──────────────────────────────────────────────────
        with st.status("Step 3/3 — Scoring form…", expanded=True) as s3:
            gold     = load_module("pipeline_gold_score")
            gold.run_gold(session_id=session_id, exercise=exercise)
            gold_dir = ROOT / "pipeline" / "gold" / session_id
            g_sum    = read_json(gold_dir / "summary.json")
            rep_df   = pd.read_csv(gold_dir / "metrics_reps.csv")
            st.write(f"✅ Scoring complete")
            s3.update(label="Step 3/3 — Scoring complete ✅", state="complete")
        prog.progress(100)

    except Exception as e:
        st.error(f"❌ {type(e).__name__}: {e}")
        st.code(_tb.format_exc())
        os.chdir(_orig_cwd)
        prog.empty()
        return None

    os.chdir(_orig_cwd)
    prog.empty()
    return session_id, b_sum, g_sum, rep_df, num_reps, gold_dir


# ─── SESSION STATE INIT ───────────────────────────────────────────
for key, default in [
    ("live_active",        False),
    ("live_frames",        []),
    ("live_landmarks",     []),
    ("live_rep_count",     0),
    ("live_started",       False),
    ("live_last_move",     None),
    ("live_results",       None),
    ("live_annotated_vid", None),
    ("live_processing_done", False),
    ("upload_results",     None),
    ("upload_file_id",     None),
    ("upload_tmp_video",   None),
    ("upload_results",     None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── NAV ──────────────────────────────────────────────────────────
live_badge = ""
if st.session_state.live_active:
    live_badge = '<span class="nbadge live">Live</span>'

st.markdown(
    '<nav class="nav">'
    '<div class="logo"><span class="logo-form">FORM</span><span class="logo-ate">ate</span><div class="logo-dot"></div></div>'
    '<div class="nav-right">' + live_badge +
    '<span class="nbadge">AI Form Coach</span>'
    '</div></nav>',
    unsafe_allow_html=True
)

st.markdown('<div class="main">', unsafe_allow_html=True)

# ─── GLOBAL EXERCISE / CAMERA CONTROLS ───────────────────────────
ctrl_l, ctrl_r = st.columns([1, 1], gap="medium")
with ctrl_l:
    st.markdown('<p class="lbl">Exercise</p>', unsafe_allow_html=True)
    exercise = st.selectbox("Exercise", ["deadlift", "squat"])
with ctrl_r:
    st.markdown('<p class="lbl">Camera Angle</p>', unsafe_allow_html=True)
    camera_view = st.selectbox("Camera", ["front_oblique", "side"])

# ─── TABS ─────────────────────────────────────────────────────────
tab_upload, tab_live = st.tabs(["Upload Video", "Live Trainer"])


# ════════════════════════════════════════════
# TAB 1 — UPLOAD VIDEO
# ════════════════════════════════════════════
with tab_upload:
    import base64
    import streamlit.components.v1 as components

    st.markdown('<div class="zone">', unsafe_allow_html=True)
    uz_l, uz_r = st.columns([1, 1], gap="large")

    with uz_l:
        st.markdown(
            '<h2 class="uz-headline">Perfect Your<br><span>Form.</span></h2>'
            '<p class="uz-desc">Upload a workout video. The AI pipeline scores every rep, flags breakdowns in real-time, and generates a full coaching report.</p>',
            unsafe_allow_html=True
        )
        st.markdown('<p class="lbl">Upload Video</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["mp4","mov","m4v","webm"], label_visibility="collapsed")

        # ── Persist uploaded file to session state immediately ─────
        if uploaded is not None:
            # New file uploaded — save bytes and clear old results
            file_id = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.get("upload_file_id") != file_id:
                st.session_state.upload_file_id = file_id
                st.session_state.upload_results = None
                tmp_dir = Path(tempfile.mkdtemp())
                tmp_video = tmp_dir / uploaded.name
                uploaded.seek(0)
                tmp_video.write_bytes(uploaded.read())
                st.session_state.upload_tmp_video = str(tmp_video)

            fname = uploaded.name
            fmb   = str(round(uploaded.size / (1024*1024), 1))
            st.markdown('<div class="fok">&#10003; ' + fname + ' &middot; ' + fmb + ' MB</div>', unsafe_allow_html=True)
        else:
            st.session_state.pop("upload_file_id", None)
            st.session_state.pop("upload_tmp_video", None)

        # Retrieve persisted video path
        tmp_video_path = st.session_state.get("upload_tmp_video")

    with uz_r:
        if tmp_video_path and Path(tmp_video_path).exists():
            vid_b64  = base64.b64encode(Path(tmp_video_path).read_bytes()).decode()
            ex_label = exercise

            upload_movenet_html = """
<!DOCTYPE html><html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#0D0D12;font-family:Arial,sans-serif;color:#F0EEF8;}
.wrap{position:relative;background:#06060A;border-radius:14px;overflow:hidden;border:1px solid #1D1D28;}
video{width:100%;display:block;max-height:400px;object-fit:contain;background:#000;}
canvas{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;}
.bar{display:flex;align-items:center;justify-content:space-between;padding:.5rem .85rem;background:#13131A;border-top:1px solid #1D1D28;gap:.5rem;flex-wrap:wrap;}
.badge{font-size:.62rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:.25rem .65rem;border-radius:12px;}
.badge.green{background:rgba(37,99,235,.12);border:1px solid rgba(59,130,246,.3);color:#60A5FA;}
.badge.loading{background:rgba(90,88,112,.08);border:1px solid #1D1D28;color:#5A5870;}
.badge.err{background:rgba(30,58,138,.2);border:1px solid rgba(59,130,246,.3);color:#93C5FD;}
.flags{display:flex;gap:.35rem;flex-wrap:wrap;}
.flag{font-size:.65rem;font-weight:700;padding:.22rem .6rem;border-radius:10px;}
.flag.ok  {background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.2);color:#93C5FD;}
.flag.warn{background:rgba(255,170,0,.15);border:1px solid rgba(255,170,0,.4);color:#FFD060;}
.flag.bad {background:rgba(255,68,68,.18);border:1px solid rgba(255,68,68,.45);color:#FF9090;}
</style>
</head>
<body>
<div class="wrap">
  <video id="vid" controls playsinline></video>
  <canvas id="cvs"></canvas>
</div>
<div class="bar">
  <span class="badge loading" id="status-badge">Loading BlazePose...</span>
  <div class="flags" id="flags-wrap"></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.3/dist/pose-detection.min.js"></script>
<script>
const EXERCISE="EX_PLACEHOLDER";
const BP={L_SH:11,R_SH:12,L_EL:13,R_EL:14,L_WR:15,R_WR:16,L_HIP:23,R_HIP:24,L_KN:25,R_KN:26,L_AN:27,R_AN:28};
const CONNS=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]];
const KEY_JOINTS={back:[11,12,23,24],drift:[11,12],hinge:[23,24],knee:[25,26],depth:[23,24,25,26],lean:[11,12,23,24]};

function angleDeg(ax,ay,bx,by,cx,cy){
  const v1x=ax-bx,v1y=ay-by,v2x=cx-bx,v2y=cy-by;
  const dot=v1x*v2x+v1y*v2y;
  const mag=Math.sqrt((v1x**2+v1y**2)*(v2x**2+v2y**2))+1e-9;
  return Math.acos(Math.max(-1,Math.min(1,dot/mag)))*180/Math.PI;
}

// BlazePose returns normalised 0-1 x,y directly
function checkThresholds(kp){
  const xn=i=>kp[i]&&kp[i].score>.4?kp[i].x:null;
  const yn=i=>kp[i]&&kp[i].score>.4?kp[i].y:null;
  const ok=(...ids)=>ids.every(i=>xn(i)!==null);
  const f={};
  if(EXERCISE==="deadlift"){
    if(ok(11,23,25)){
      const a=angleDeg(xn(11),yn(11),xn(23),yn(23),xn(25),yn(25));
      f.back=a>=145?{st:"ok",lbl:"Back OK"}:a>=115?{st:"warn",lbl:"Back rounding"}:{st:"bad",lbl:"Back round!"};
    }
    if(ok(11,23)){const d=Math.abs(xn(11)-xn(23));f.drift=d<.07?{st:"ok",lbl:"Bar path OK"}:d<.14?{st:"warn",lbl:"Bar drifting"}:{st:"bad",lbl:"Bar too far!"};}
    if(ok(23,25)){const d=yn(25)-yn(23);f.hinge=d>.03?{st:"ok",lbl:"Good hinge"}:{st:"warn",lbl:"Hinge deeper"};}
  } else {
    if(ok(23,25)){const c=xn(25)-xn(23);f.knee=c>=-.02?{st:"ok",lbl:"Knees OK"}:c>=-.07?{st:"warn",lbl:"Knee caving"}:{st:"bad",lbl:"Knee cave!"};}
    if(ok(23,25)){const d=yn(23)-yn(25);f.depth=d<=.03?{st:"ok",lbl:"Good depth"}:d<=.12?{st:"warn",lbl:"Go deeper"}:{st:"bad",lbl:"Too shallow"};}
    if(ok(11,23)){const l=Math.abs(xn(11)-xn(23));f.lean=l<.07?{st:"ok",lbl:"Upright OK"}:l<.14?{st:"warn",lbl:"Leaning fwd"}:{st:"bad",lbl:"Too much lean!"};}
  }
  return f;
}

function jointColor(idx,flags){
  let w="ok";
  for(const[k,jts] of Object.entries(KEY_JOINTS)){
    if(!flags[k]||!jts.includes(idx))continue;
    if(flags[k].st==="bad")return"#FF4444";
    if(flags[k].st==="warn")w="warn";
  }
  return w==="warn"?"#FFAA00":"#3B82F6";
}

function drawSkeleton(ctx,kp,flags,W,H){
  const px=i=>kp[i].x*W, py=i=>kp[i].y*H, vs=i=>kp[i]?kp[i].score:0;
  const sts=Object.values(flags).map(f=>f.st);
  const cc=sts.includes("bad")?"#FF4444":sts.includes("warn")?"#FFAA00":"#3B82F6";
  ctx.lineWidth=3; ctx.lineCap="round";
  for(const[a,b] of CONNS){
    if(vs(a)<.3||vs(b)<.3)continue;
    ctx.globalAlpha=.9;ctx.strokeStyle=cc;
    ctx.beginPath();ctx.moveTo(px(a),py(a));ctx.lineTo(px(b),py(b));ctx.stroke();
  }
  ctx.globalAlpha=1;
  for(let i=0;i<kp.length;i++){
    if(vs(i)<.3)continue;
    ctx.beginPath();ctx.arc(px(i),py(i),7,0,2*Math.PI);
    ctx.fillStyle=jointColor(i,flags);ctx.fill();
    ctx.strokeStyle="rgba(0,0,0,.7)";ctx.lineWidth=2;ctx.stroke();
  }
  let yo=30;ctx.font="bold 14px Arial,sans-serif";ctx.globalAlpha=1;
  for(const[k,{st,lbl}] of Object.entries(flags)){
    if(st==="ok")continue;
    const col=st==="bad"?"#FF4444":"#FFAA00";
    const tw=ctx.measureText(lbl).width;
    ctx.fillStyle="rgba(0,0,0,.8)";ctx.fillRect(10,yo-16,tw+18,24);
    ctx.fillStyle=col;ctx.fillText(lbl,19,yo);yo+=30;
  }
}

function renderChips(flags){
  document.getElementById("flags-wrap").innerHTML=Object.values(flags).map(({st,lbl})=>
    '<span class="flag '+st+'">'+(st==="ok"?"✓":"▲")+" "+lbl+"</span>"
  ).join("");
}

const vid=document.getElementById("vid");
const cvs=document.getElementById("cvs");
const ctx=cvs.getContext("2d");
let detector=null,rafId=null;

const b64="VID_B64_PLACEHOLDER";
const blob=new Blob([Uint8Array.from(atob(b64),c=>c.charCodeAt(0))],{type:"video/mp4"});
vid.src=URL.createObjectURL(blob);

async function init(){
  try{
    detector=await poseDetection.createDetector(
      poseDetection.SupportedModels.BlazePose,
      {runtime:"tfjs",modelType:"lite",enableSmoothing:true}
    );
    document.getElementById("status-badge").textContent="BlazePose Ready ✓";
    document.getElementById("status-badge").className="badge green";
    vid.addEventListener("play",startLoop);
    vid.addEventListener("pause",()=>cancelAnimationFrame(rafId));
    vid.addEventListener("ended",()=>cancelAnimationFrame(rafId));
  }catch(e){
    document.getElementById("status-badge").textContent="Error: "+e.message;
    document.getElementById("status-badge").className="badge err";
  }
}

function startLoop(){
  async function loop(){
    if(vid.paused||vid.ended)return;
    const W=vid.videoWidth||640,H=vid.videoHeight||480;
    cvs.width=W;cvs.height=H;
    // Match canvas CSS size to actual rendered video box
    const r=vid.getBoundingClientRect();
    cvs.style.width=r.width+"px";cvs.style.height=r.height+"px";
    ctx.clearRect(0,0,W,H);
    try{
      const poses=await detector.estimatePoses(vid,{flipHorizontal:false});
      if(poses.length>0){
        const kp=poses[0].keypoints;
        const flags=checkThresholds(kp);
        drawSkeleton(ctx,kp,flags,W,H);
        renderChips(flags);
      }
    }catch(e){}
    rafId=requestAnimationFrame(loop);
  }
  loop();
}
init();
</script>
</body></html>""".replace("EX_PLACEHOLDER", ex_label).replace("VID_B64_PLACEHOLDER", vid_b64)


            st.markdown('<p class="lbl">MoveNet Preview</p>', unsafe_allow_html=True)
            components.html(upload_movenet_html, height=480, scrolling=False)

        else:
            st.markdown(
                '<div class="empty-state" style="padding:3rem 0;">'
                '<div class="empty-logo"><b>FORM</b><span>ate</span></div>'
                '<p class="empty-txt">Upload a video to begin analysis.</p>'
                '</div>',
                unsafe_allow_html=True
            )

    if tmp_video_path and Path(tmp_video_path).exists():
        if st.button("ANALYSE FORM", type="primary", use_container_width=True, key="run_upload"):
            result = run_pipeline(tmp_video_path, exercise, camera_view)
            if result:
                st.session_state.upload_results = result

    if st.session_state.upload_results:
        sid, b_sum, g_sum, rep_df, num_reps, gold_dir = st.session_state.upload_results
        render_results(sid, gold_dir, b_sum, g_sum, rep_df, num_reps, exercise)


# ════════════════════════════════════════════
# TAB 2 — LIVE TRAINER
# ════════════════════════════════════════════
with tab_live:

    # ── MoveNet component receives data back via query params trick
    # We use st.components to embed full TF.js MoveNet in browser
    import streamlit.components.v1 as components

    ex_js = exercise  # pass to JS

    movenet_html = """
<!DOCTYPE html><html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
html,body{width:100%;height:100%;overflow:hidden;
  font-family:'Inter',system-ui,sans-serif;-webkit-font-smoothing:antialiased;}
#cam-container{position:relative;width:100%;height:100vh;
  background:#000;overflow:hidden;}

/* Video fills container — object-fit:cover crops rather than letterboxes */
/* Front cam gets CSS mirror flip */
video{
  position:absolute;top:0;left:0;
  width:100%;height:100%;
  object-fit:cover;
  pointer-events:none;}
video.mirror{ transform:scaleX(-1); }
canvas{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;}

/* ── TOP HUD ── */
#hud-top{
  position:absolute;top:0;left:0;right:0;
  padding:1rem 1.1rem .8rem;
  background:linear-gradient(to bottom,rgba(0,0,0,.75) 0%,transparent 100%);
  display:flex;align-items:flex-start;justify-content:space-between;
  z-index:10;}

/* Rep counter block */
#rep-block{display:flex;flex-direction:column;gap:0;}
#rep-num{
  font-family:'Space Grotesk',sans-serif;
  font-size:6rem;font-weight:700;line-height:.95;
  color:#fff;
  text-shadow:0 2px 32px rgba(37,99,235,.5);
  transition:color .12s;}
#rep-sub{
  font-size:.5rem;font-weight:700;letter-spacing:.25em;
  text-transform:uppercase;color:rgba(255,255,255,.35);
  margin-top:.15rem;}
#rep-ex{
  font-size:.58rem;font-weight:600;letter-spacing:.18em;
  text-transform:uppercase;
  color:#3B82F6;margin-top:.1rem;}

/* Status pill */
#status-badge{
  font-size:.55rem;font-weight:700;letter-spacing:.12em;
  text-transform:uppercase;padding:.32rem .9rem;
  border-radius:20px;backdrop-filter:blur(12px);
  background:rgba(0,0,0,.5);
  border:1px solid rgba(255,255,255,.12);
  color:rgba(255,255,255,.4);}
#status-badge.waiting{
  border-color:rgba(59,130,246,.5);color:#93C5FD;
  background:rgba(37,99,235,.12);}
#status-badge.active{
  border-color:rgba(255,255,255,.3);color:#fff;
  background:rgba(37,99,235,.2);}
#status-badge.done{
  border-color:rgba(59,130,246,.4);color:#60A5FA;
  background:rgba(29,78,216,.15);}

/* FPS */
#fps-badge{
  position:absolute;top:.9rem;left:50%;transform:translateX(-50%);
  font-size:.5rem;font-weight:600;letter-spacing:.06em;
  background:rgba(0,0,0,.6);border:1px solid rgba(255,255,255,.07);
  border-radius:6px;padding:.18rem .5rem;
  color:rgba(255,255,255,.25);z-index:10;white-space:nowrap;}

/* ── FORM FLAGS ── */
#flags-wrap{
  position:absolute;bottom:6rem;left:.85rem;
  display:flex;flex-direction:column;gap:.35rem;z-index:10;}
.flag{
  font-size:.68rem;font-weight:600;letter-spacing:.04em;
  padding:.32rem .75rem;border-radius:8px;
  backdrop-filter:blur(10px);}
.flag.ok  {background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.15);color:rgba(255,255,255,.6);}
.flag.warn{background:rgba(255,170,0,.15);border:1px solid rgba(255,170,0,.45);color:#FFCC55;}
.flag.bad {background:rgba(220,38,38,.2);border:1px solid rgba(220,38,38,.5);color:#FCA5A5;}

/* ── BOTTOM CONTROLS ── */
#controls{
  position:absolute;bottom:0;left:0;right:0;
  padding:.85rem 1.1rem 2rem;
  background:linear-gradient(to top,rgba(0,0,0,.82) 0%,transparent 100%);
  display:flex;gap:.65rem;align-items:center;justify-content:center;
  z-index:10;}

/* Main CTA */
#btn-main{
  flex:1;max-width:240px;
  font-family:'Space Grotesk',sans-serif;
  font-weight:700;font-size:.85rem;
  letter-spacing:.1em;text-transform:uppercase;
  border:none;border-radius:14px;
  padding:.95rem 1.5rem;cursor:pointer;transition:all .18s;}
#btn-main.start{
  background:#1D4ED8;color:#fff;
  box-shadow:0 4px 28px rgba(29,78,216,.5);}
#btn-main.start:active{transform:scale(.96);box-shadow:none;}
#btn-main.stop{
  background:rgba(255,255,255,.08);color:#fff;
  border:1.5px solid rgba(255,255,255,.2);
  box-shadow:none;}
#btn-main.stop:active{background:rgba(255,255,255,.14);}

/* Save button */
#btn-save{
  font-family:'Space Grotesk',sans-serif;
  font-weight:700;font-size:.72rem;
  letter-spacing:.08em;text-transform:uppercase;
  border:1.5px solid #1D4ED8;border-radius:14px;
  padding:.85rem 1.1rem;
  background:rgba(29,78,216,.12);color:#60A5FA;
  cursor:pointer;backdrop-filter:blur(8px);
  display:none;white-space:nowrap;}
#btn-save:active{background:rgba(29,78,216,.25);}

/* Icon buttons (flip + voice) */
.icon-btn{
  width:48px;height:48px;border-radius:14px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  flex-shrink:0;transition:all .18s;
  backdrop-filter:blur(10px);}
#btn-flip{
  border:1.5px solid rgba(255,255,255,.15);
  background:rgba(255,255,255,.07);
  display:none;}
#btn-flip:active{background:rgba(255,255,255,.15);}
#btn-voice{
  border:1.5px solid rgba(29,78,216,.5);
  background:rgba(29,78,216,.18);}
#btn-voice:active{background:rgba(29,78,216,.35);}
#btn-voice.muted{
  border-color:rgba(255,255,255,.12);
  background:rgba(255,255,255,.05);}

/* ── CAM OFF SCREEN ── */
#cam-off{
  position:absolute;inset:0;
  display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:1.25rem;
  background:linear-gradient(160deg,#000 0%,#0A0F1E 60%,#0D1533 100%);
  z-index:5;}
.cam-icon-wrap{
  width:80px;height:80px;border-radius:22px;
  background:rgba(29,78,216,.15);
  border:1.5px solid rgba(29,78,216,.35);
  display:flex;align-items:center;justify-content:center;
  animation:glow 2.5s ease-in-out infinite;}
@keyframes glow{
  0%,100%{box-shadow:0 0 0 0 rgba(29,78,216,.0);}
  50%{box-shadow:0 0 24px 6px rgba(29,78,216,.25);}}
.cam-icon-wrap svg{width:38px;height:38px;}
.cam-tagline{
  font-size:.72rem;color:rgba(255,255,255,.28);
  max-width:190px;text-align:center;line-height:1.65;
  letter-spacing:.02em;}

/* ── GESTURE OVERLAY ── */
#gesture-overlay{
  position:absolute;inset:0;
  display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:.75rem;
  z-index:8;pointer-events:none;
  background:linear-gradient(to top,rgba(0,0,0,.55) 0%,transparent 60%);}
#gesture-ring-wrap{
  width:88px;height:88px;
  filter:drop-shadow(0 0 16px rgba(59,130,246,.5));}
#gesture-countdown{
  font-family:'Space Grotesk',sans-serif;
  font-size:5rem;font-weight:700;color:#fff;
  line-height:1;min-height:5rem;
  text-shadow:0 0 32px rgba(255,255,255,.5);}
#gesture-hint{
  font-size:.85rem;font-weight:600;letter-spacing:.06em;
  color:#fff;text-align:center;
  text-shadow:0 2px 8px rgba(0,0,0,.8);}
#gesture-sub{
  font-size:.65rem;font-weight:500;letter-spacing:.08em;
  color:rgba(255,255,255,.45);text-align:center;}

/* ── ORIENTATION PICKER ── */
#orient-picker{
  display:flex;gap:.75rem;margin-top:.25rem;}
.orient-btn{
  display:flex;flex-direction:column;align-items:center;gap:.5rem;
  padding:.85rem 1.1rem;border-radius:16px;cursor:pointer;
  border:1.5px solid rgba(255,255,255,.12);
  background:rgba(255,255,255,.05);
  transition:all .18s;min-width:90px;}
.orient-btn:active{transform:scale(.96);}
.orient-btn.selected{
  border-color:#3B82F6;
  background:rgba(29,78,216,.2);
  box-shadow:0 0 20px rgba(59,130,246,.2);}
.orient-btn svg{opacity:.7;transition:opacity .18s;}
.orient-btn.selected svg{opacity:1;}
.orient-lbl{
  font-size:.6rem;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:rgba(255,255,255,.5);
  transition:color .18s;}
.orient-btn.selected .orient-lbl{color:#93C5FD;}
.orient-hint{
  font-size:.62rem;color:rgba(255,255,255,.3);
  text-align:center;max-width:200px;line-height:1.5;}

/* ── ANIMATIONS ── */
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.15;}}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.25;}}
.pulsing{animation:pulse .8s infinite;}
</style>
</head>
<body>
<div id="cam-container">
  <div id="cam-off">
    <div class="cam-icon-wrap">
      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M23 19C23 19.5304 22.7893 20.0391 22.4142 20.4142C22.0391 20.7893 21.5304 21 21 21H3C2.46957 21 1.96086 20.7893 1.58579 20.4142C1.21071 20.0391 1 19.5304 1 19V8C1 7.46957 1.21071 6.96086 1.58579 6.58579C1.96086 6.21071 2.46957 6 3 6H7L9 3H15L17 6H21C21.5304 6 22.0391 6.21071 22.4142 6.58579C22.7893 6.96086 23 7.46957 23 8V19Z" stroke="#3B82F6" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="12" cy="13" r="4" stroke="#60A5FA" stroke-width="1.5"/>
        <circle cx="12" cy="13" r="1.5" fill="#93C5FD"/>
      </svg>
    </div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;letter-spacing:-.02em;color:#fff;">
      FORM<span style="color:#3B82F6;">ate</span>
    </div>

    <!-- Orientation picker -->
    <div id="orient-picker">
      <!-- Portrait -->
      <div class="orient-btn selected" id="btn-portrait" onclick="setOrientation('portrait')">
        <svg width="28" height="38" viewBox="0 0 28 38" fill="none">
          <rect x="1" y="1" width="26" height="36" rx="4" stroke="white" stroke-width="2"/>
          <rect x="5" y="5" width="18" height="26" rx="2" fill="rgba(59,130,246,.35)"/>
          <line x1="9" y1="33" x2="19" y2="33" stroke="white" stroke-width="2" stroke-linecap="round"/>
        </svg>
        <span class="orient-lbl">Portrait</span>
      </div>
      <!-- Landscape -->
      <div class="orient-btn" id="btn-landscape" onclick="setOrientation('landscape')">
        <svg width="38" height="28" viewBox="0 0 38 28" fill="none">
          <rect x="1" y="1" width="36" height="26" rx="4" stroke="white" stroke-width="2"/>
          <rect x="5" y="5" width="26" height="18" rx="2" fill="rgba(59,130,246,.35)"/>
          <line x1="33" y1="9" x2="33" y2="19" stroke="white" stroke-width="2" stroke-linecap="round"/>
        </svg>
        <span class="orient-lbl">Landscape</span>
      </div>
    </div>
    <div class="orient-hint" id="orient-hint">Phone propped upright — see head to toe</div>
  </div>
  <video id="video" autoplay playsinline muted></video>
  <canvas id="overlay"></canvas>
  <div id="hud-top">
    <div id="rep-block">
      <div id="rep-num">0</div>
      <div id="rep-sub">REPS</div>
      <div id="rep-ex">EXERCISE_PLACEHOLDER</div>
    </div>
    <div id="status-badge">OFF</div>
  </div>
  <div id="fps-badge" style="display:none">-- FPS</div>
  <div id="flags-wrap"></div>

  <!-- Gesture-to-start overlay — hidden until body detected -->
  <div id="gesture-overlay" style="display:none">
    <div id="gesture-ring-wrap">
      <svg width="88" height="88" viewBox="0 0 88 88">
        <!-- track -->
        <circle cx="44" cy="44" r="36" fill="none"
          stroke="rgba(255,255,255,.1)" stroke-width="4"/>
        <!-- progress ring -->
        <circle id="gesture-ring" cx="44" cy="44" r="36" fill="none"
          stroke="#3B82F6" stroke-width="4"
          stroke-linecap="round"
          stroke-dasharray="226"
          stroke-dashoffset="226"
          transform="rotate(-90 44 44)"
          style="transition:stroke-dashoffset .08s linear,stroke .2s;"/>
        <!-- hands icon -->
        <text x="44" y="50" text-anchor="middle"
          font-size="26" fill="white" style="font-family:system-ui">🙌</text>
      </svg>
    </div>
    <div id="gesture-countdown"></div>
    <div id="gesture-hint">Raise both hands to begin</div>
    <div id="gesture-sub">Hold for 1.5 seconds</div>
  </div>

  <div id="controls">
    <button id="btn-flip" class="icon-btn" onclick="flipCamera()">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,.8)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M1 4v6h6"/><path d="M23 20v-6h-6"/>
        <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4-4.64 4.36A9 9 0 0 1 3.51 15"/>
      </svg>
    </button>
    <button id="btn-main" class="start" onclick="toggleCamera()">START CAMERA</button>
    <button id="btn-voice" class="icon-btn" onclick="toggleVoice()" title="Voice feedback">
      <svg id="voice-icon" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
        <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
        <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
      </svg>
    </button>
    <button id="btn-save" onclick="saveSession()">SAVE &amp; ANALYSE</button>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.3/dist/pose-detection.min.js"></script>
<script>
const EXERCISE="EXERCISE_PLACEHOLDER";

// MoveNet keypoint indices (17 keypoints, pixel coords)
const MV={
  NOSE:0,L_EYE:1,R_EYE:2,L_EAR:3,R_EAR:4,
  L_SH:5,R_SH:6,L_EL:7,R_EL:8,L_WR:9,R_WR:10,
  L_HIP:11,R_HIP:12,L_KN:13,R_KN:14,L_AN:15,R_AN:16
};
const CONNS=[
  [5,6],[5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],
  [11,13],[13,15],[12,14],[14,16]
];
const KEY_JOINTS={
  back:[5,6,11,12],drift:[5,6],hinge:[11,12],
  knee:[13,14],depth:[11,12,13,14],lean:[5,6,11,12]
};

// ── angle (works on any consistent unit — norm or pixel) ──────────
function angleDeg(ax,ay,bx,by,cx,cy){
  const v1x=ax-bx,v1y=ay-by,v2x=cx-bx,v2y=cy-by;
  const dot=v1x*v2x+v1y*v2y;
  const mag=Math.sqrt((v1x**2+v1y**2)*(v2x**2+v2y**2))+1e-9;
  return Math.acos(Math.max(-1,Math.min(1,dot/mag)))*180/Math.PI;
}

// ── thresholds on NORMALISED 0-1 coords ──────────────────────────
// MoveNet returns pixel coords → we normalise by W,H before checking
function checkThresholds(kp,W,H){
  const xn=i=>(kp[i]&&kp[i].score>.3)?kp[i].x/W:null;
  const yn=i=>(kp[i]&&kp[i].score>.3)?kp[i].y/H:null;
  const ok=(...ids)=>ids.every(i=>xn(i)!==null);
  const f={};
  if(EXERCISE==="deadlift"){
    // Back angle: shoulder(5)-hip(11)-knee(13) — higher angle = straighter
    if(ok(5,11,13)){
      const a=angleDeg(xn(5),yn(5),xn(11),yn(11),xn(13),yn(13));
      f.back=a>=145?{st:"ok",lbl:"Back OK"}:a>=115?{st:"warn",lbl:"Back rounding"}:{st:"bad",lbl:"Back round!"};
    }
    // Bar drift: horizontal distance shoulder vs hip (side view)
    if(ok(5,11)){
      const d=Math.abs(xn(5)-xn(11));
      f.drift=d<.08?{st:"ok",lbl:"Bar OK"}:d<.16?{st:"warn",lbl:"Bar drifting"}:{st:"bad",lbl:"Bar too far!"};
    }
    // Hinge: knee Y should be clearly below hip Y (knee in front/lower)
    if(ok(11,13)){
      const d=yn(13)-yn(11); // positive = knee lower than hip (good)
      f.hinge=d>.04?{st:"ok",lbl:"Hinge OK"}:{st:"warn",lbl:"Hinge deeper"};
    }
  }else{
    // SQUAT
    // Knee cave: left knee X should NOT be inside (greater than) left hip X
    // Only fire during descent — check hip is below standing threshold
    if(ok(11,12,13,14)){
      // Use both sides — check if knees are collapsing inward
      const hipW=Math.abs(xn(12)-xn(11));  // hip width
      const kneeW=Math.abs(xn(14)-xn(13)); // knee width
      // knees caving = knee width < hip width significantly
      const ratio = hipW>0.01 ? kneeW/hipW : 1;
      f.knee=ratio>=0.75?{st:"ok",lbl:"Knees OK"}:ratio>=0.55?{st:"warn",lbl:"Knee caving"}:{st:"bad",lbl:"Knee cave!"};
    }
    // Squat depth: hip Y vs knee Y — in squat, hip drops toward knee level
    // hip Y approaches knee Y from above (both increase downward)
    // gap = knee Y - hip Y → smaller gap = deeper squat
    if(ok(11,13)){
      const gap=yn(13)-yn(11); // knee Y minus hip Y (both normalised 0-1)
      // gap shrinks as you squat — parallel = gap ~0, standing = gap ~0.2+
      f.depth=gap<=.08?{st:"ok",lbl:"Good depth"}:gap<=.16?{st:"warn",lbl:"Go deeper"}:{st:"bad",lbl:"Too shallow"};
    }
    // Forward lean: trunk angle from vertical
    // use shoulder Y vs hip Y relative to their X offset
    if(ok(5,11)){
      const dx=Math.abs(xn(5)-xn(11));
      const dy=Math.abs(yn(5)-yn(11));
      // lean angle = atan(dx/dy) — 0 = vertical, higher = leaning
      const lean=dy>0.01?dx/dy:0;
      f.lean=lean<.25?{st:"ok",lbl:"Upright OK"}:lean<.45?{st:"warn",lbl:"Leaning fwd"}:{st:"bad",lbl:"Too much lean!"};
    }
  }
  return f;
}

function jointColor(idx,flags){
  let w="ok";
  for(const[k,jts]of Object.entries(KEY_JOINTS)){
    if(!flags[k]||!jts.includes(idx))continue;
    if(flags[k].st==="bad")return"#FF4444";   // red = bad form
    if(flags[k].st==="warn")w="warn";
  }
  return w==="warn"?"#FFAA00":"#3B82F6";       // orange = warn, blue = ok
}

// ── draw skeleton — MoveNet returns PIXEL coords ─────────────────
// Video is CSS-mirrored so we flip x: draw at (W - kp.x)
function drawSkeleton(ctx,kp,flags,vW,vH,rW,rH,offX,offY){
  // MoveNet returns pixel coords in video space (vW x vH)
  // Scale to rendered area, then offset for letterbox
  // Front cam: mirror within the rendered area
  const mirror=facingMode==="user";
  const px=i=>{
    if(!kp[i])return 0;
    const sx=kp[i].x/vW*rW;
    return offX+(mirror?rW-sx:sx);
  };
  const py=i=>kp[i]?offY+kp[i].y/vH*rH:0;
  const vs=i=>kp[i]?kp[i].score:0;
  const sts=Object.values(flags).map(f=>f.st);
  const cc=sts.includes("bad")?"#FF4444":sts.includes("warn")?"#FFAA00":"#3B82F6";

  // connections
  ctx.lineWidth=4;ctx.lineCap="round";
  for(const[a,b]of CONNS){
    if(vs(a)<.25||vs(b)<.25)continue;
    ctx.globalAlpha=.9;ctx.strokeStyle=cc;
    ctx.beginPath();ctx.moveTo(px(a),py(a));ctx.lineTo(px(b),py(b));ctx.stroke();
  }
  // joints
  ctx.globalAlpha=1;
  for(let i=0;i<kp.length;i++){
    if(vs(i)<.25)continue;
    ctx.beginPath();ctx.arc(px(i),py(i),8,0,2*Math.PI);
    ctx.fillStyle=jointColor(i,flags);ctx.fill();
    ctx.strokeStyle="rgba(0,0,0,.65)";ctx.lineWidth=2;ctx.stroke();
  }
  // flag labels
  let yo=offY+32;ctx.font="bold 15px Arial,sans-serif";ctx.globalAlpha=1;
  for(const[k,{st,lbl}]of Object.entries(flags)){
    if(st==="ok")continue;
    const col=st==="bad"?"#FF4444":"#FFAA00";
    const tw=ctx.measureText(lbl).width;
    ctx.fillStyle="rgba(0,0,0,.78)";ctx.fillRect(10,yo-17,tw+18,24);
    ctx.fillStyle=col;ctx.fillText(lbl,19,yo);yo+=30;
  }
}

function updateFlags(flags){
  document.getElementById("flags-wrap").innerHTML=
    Object.entries(flags)
      .filter(([,{st}])=>st!=="ok")
      .map(([,{st,lbl}])=>'<div class="flag '+st+'">'+(st==="bad"?"▲ ":"△ ")+lbl+"</div>")
      .join("");
  // Voice form cues
  speakFlags(flags);
}

// ── Voice Feedback Engine ─────────────────────────────────────────
let voiceEnabled=true;
let lastFlagState={};    // key → last severity spoken
let voiceQueue=null;     // pending speak timeout
let lastRepSpoken=0;     // last rep count spoken
const FLAG_HOLD=8;       // frames flag must persist before speaking
let flagHoldCount={};    // key → consecutive frames at this severity

function speak(text, priority=false){
  if(!voiceEnabled) return;
  if(!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  clearTimeout(voiceQueue);
  const delay = priority ? 0 : 80;
  voiceQueue = setTimeout(()=>{
    const u=new SpeechSynthesisUtterance(text);
    u.rate=1.0; u.pitch=1.0; u.volume=1.0;
    const voices=window.speechSynthesis.getVoices();
    const preferred=voices.find(v=>
      v.name.includes("Google US")||v.name.includes("Samantha")||
      v.name.includes("Daniel")||v.name.includes("Karen")
    );
    if(preferred) u.voice=preferred;
    window.speechSynthesis.speak(u);
  }, delay);
}

const VOICE_CUES={
  "Knee cave!":   "Knees out",
  "Knee caving":  "Watch your knees",
  "Too shallow":  "Go deeper",
  "Go deeper":    "Deeper",
  "Too much lean!":"Stay upright",
  "Leaning fwd":  "Chest up",
  "Bar too far!": "Bar close to body",
  "Bar drifting": "Keep bar close",
  "Back round!":  "Neutral spine",
  "Back rounding":"Brace your back",
  "Hinge deeper": "Push hips back",
};

function speakFlags(flags){
  const activeKeys=new Set(Object.keys(flags).filter(k=>flags[k].st!=="ok"));

  for(const key of Object.keys(flagHoldCount)){
    if(!activeKeys.has(key)){
      flagHoldCount[key]=0;
      // State cleared — reset so it can speak again next time it appears
      if(lastFlagState[key] && lastFlagState[key]!=="ok"){
        lastFlagState[key]="ok";
      }
    }
  }

  for(const key of activeKeys){
    const {st, lbl}=flags[key];
    const prev=lastFlagState[key]||"ok";

    // Only speak if severity changed (ok→warn, ok→bad, warn→bad)
    const changed = (prev==="ok") ||
                    (prev==="warn" && st==="bad");

    if(changed){
      // Increment hold counter — must persist FLAG_HOLD frames
      flagHoldCount[key]=(flagHoldCount[key]||0)+1;
      if(flagHoldCount[key]>=FLAG_HOLD){
        const cue=VOICE_CUES[lbl];
        if(cue){
          speak(cue);
          lastFlagState[key]=st;
          flagHoldCount[key]=0; // reset so it doesn't re-trigger immediately
        }
      }
    } else {
      // Same severity — don't speak, just hold
      flagHoldCount[key]=0;
      lastFlagState[key]=st;
    }
  }
}

function speakRepCount(n){
  if(n===lastRepSpoken) return;
  lastRepSpoken=n;
  speak(String(n), true);
  if(n===5)  setTimeout(()=>speak("5 reps, keep going"),500);
  if(n===10) setTimeout(()=>speak("10 reps, excellent"),500);
}


// ── Rep counting — exercise-specific, adaptive ────────────────────
//
// SQUAT:  track hip Y (goes DOWN = higher Y value)
//         Rep = any meaningful descent + return, no fixed depth required
//         Uses person's own range — 30% of THEIR measured range = 1 rep
//
// DEADLIFT: track hip Y (goes UP then DOWN)
//           Rep = hip rises from hinge position back to lockout
//           Bar-off-floor = hip at low point, lockout = hip at high point
//
// Both use the same adaptive state machine but with opposite polarity:
//   SQUAT:    standing=hipY low, bottom=hipY high  → descent triggers rep
//   DEADLIFT: setup=hipY high,  lockout=hipY low   → ascent triggers rep
//
// Key insight: rep fires when person returns to START position,
// regardless of HOW FAR they went. 30% of personal range = minimum.

let repCount=0,hipHist=[],repState="IDLE";
let repStart=0,peakY=0,standY=null,botY=null,calibN=0;
let rangeHistory=[];
let wakeLock=null;

// ── Gesture-to-start system ───────────────────────────────────────
// User must raise both wrists above shoulders for GESTURE_HOLD frames
// before rep counting begins. This prevents accidental early starts.
let repActive=false;          // true only after gesture confirmed
let gestureHoldCount=0;       // consecutive frames wrists above shoulders
let countdownActive=false;    // 3-2-1 countdown in progress
let countdownVal=3;           // current countdown number
const GESTURE_HOLD=22;        // ~1.5s at 15fps

function checkGesture(kp, vW, vH){
  // Wrists above shoulders = both wrist Y < both shoulder Y (lower Y = higher on screen)
  const lw=kp[MV.L_WR], rw=kp[MV.R_WR];
  const ls=kp[MV.L_SH], rs=kp[MV.R_SH];
  if(!lw||!rw||!ls||!rs) return false;
  if(lw.score<.3||rw.score<.3||ls.score<.3||rs.score<.3) return false;
  return (lw.y < ls.y) && (rw.y < rs.y);
}

function startCountdown(){
  if(countdownActive) return;
  countdownActive=true;
  countdownVal=5;
  // Clear flags during countdown
  document.getElementById("flags-wrap").innerHTML="";
  // Show countdown in overlay
  const el=document.getElementById("gesture-overlay");
  if(el){
    el.style.display="flex";
    document.getElementById("gesture-ring-wrap").style.display="none";
    document.getElementById("gesture-sub").style.display="none";
  }
  function tick(){
    if(countdownVal>0){
      document.getElementById("gesture-hint").textContent="Starting in";
      document.getElementById("gesture-countdown").textContent=countdownVal;
      speak(String(countdownVal));
      countdownVal--;
      setTimeout(tick,1000);
    } else {
      // GO
      countdownActive=false;
      repActive=true;
      gestureHoldCount=0;
      repCount=0;hipHist=[];repState="IDLE";
      calibN=0;standY=null;botY=null;rangeHistory=[];peakY=0;
      document.getElementById("rep-num").textContent="0";
      document.getElementById("rep-num").style.color="#60A5FA";
      setStatus("active");
      speak("Go!");
      hideGestureOverlay();
      // Restore ring wrap for next time
      const rw=document.getElementById("gesture-ring-wrap");
      const gs=document.getElementById("gesture-sub");
      if(rw) rw.style.display="";
      if(gs) gs.style.display="";
    }
  }
  tick();
}

function updateGestureOverlay(holdCount, isReady){
  // Only show overlay once user is detected and in gesture phase
  // Don't show if repActive or countdownActive
  if(repActive || countdownActive) return;
  const el=document.getElementById("gesture-overlay");
  if(!el) return;
  el.style.display="flex";
  const pct=Math.min(holdCount/GESTURE_HOLD,1);
  const ring=document.getElementById("gesture-ring");
  if(ring){
    const c=2*Math.PI*36;
    ring.style.strokeDashoffset= c-(c*pct);
    ring.style.stroke = pct>=1?"#fff":"#3B82F6";
  }
  document.getElementById("gesture-countdown").textContent="";
  document.getElementById("gesture-hint").textContent=
    isReady ? "Hold..." : "Raise both hands to begin";
}

function hideGestureOverlay(){
  const el=document.getElementById("gesture-overlay");
  if(el) el.style.display="none";
}


const CALIB=40; // frames before we start counting (~2-3s at 15fps)
const MIN_RANGE=0.04; // min normalised hip movement to count anything
const REP_THRESHOLD=0.30; // must move 30% of personal range (very lenient)

function updateRep(kp, vW, vH){
  // ── Pick signal based on exercise ──────────────────────────────
  // SQUAT: use hip Y (higher Y = lower body = squat)
  // DEADLIFT: use hip Y inverted (higher Y = hips back/down = hinge start)
  const lh=kp[MV.L_HIP], rh=kp[MV.R_HIP];
  const lk=kp[MV.L_KN],  rk=kp[MV.R_KN];
  const ls=kp[MV.L_SH],  rs=kp[MV.R_SH];

  if(!lh||!rh||lh.score<.25||rh.score<.25) return;

  const hipYn = ((lh.y+rh.y)/2) / vH;  // 0=top of frame, 1=bottom

  // For deadlift, also check if we have good hip signal
  // Signal is always hipY for both — just the movement direction differs
  hipHist.push(hipYn);
  if(hipHist.length>120) hipHist.shift(); // 8s rolling window at 15fps

  const idx = sessionFrames.length;

  // ── Calibration phase: learn standing position ─────────────────
  if(calibN < CALIB){
    calibN++;
    // Standing/start position: for squat = hip high (low Y)
    //                           for deadlift = hip in hinge (varies)
    // Just collect data — don't set thresholds yet
    if(standY===null) standY=hipYn;
    if(EXERCISE==="squat"){
      // Standing = lowest Y (highest on screen = standing tall)
      if(hipYn < standY) standY = hipYn;
    }else{
      // Deadlift standing = also low Y (upright lockout)
      if(hipYn < standY) standY = hipYn;
    }
    return;
  }

  // ── Adaptive range estimation ──────────────────────────────────
  // Recalculate range every 15 frames from rolling window
  if(idx % 15 === 0 || botY===null){
    const sorted=[...hipHist].sort((a,b)=>a-b);
    const newStand = sorted[Math.floor(sorted.length*.1)];  // 10th pct = standing
    const newBot   = sorted[Math.floor(sorted.length*.9)];  // 90th pct = lowest
    // Smooth updates — don't jump suddenly
    standY = standY===null ? newStand : standY*.7 + newStand*.3;
    botY   = botY===null   ? newBot   : botY*.7   + newBot*.3;
  }

  const range = botY - standY;
  if(range < MIN_RANGE) return; // not enough movement detected yet

  // ── Adaptive thresholds based on personal range ────────────────
  // Trigger descent at 30% of range (very lenient — catches partial reps)
  // Complete rep when back within 20% of start
  const descTrigger = standY + range * REP_THRESHOLD;      // started going down
  const returnTrig  = standY + range * 0.20;               // back near top = rep done
  const minDepth    = standY + range * REP_THRESHOLD;      // minimum depth required

  // ── State machine ──────────────────────────────────────────────
  if(repState==="IDLE"){
    if(hipYn > descTrigger){
      repState="DESCENDING";
      repStart=idx;
      peakY=hipYn;
      setStatus("active");
    }
  }
  else if(repState==="DESCENDING"){
    if(hipYn > peakY) peakY=hipYn; // track deepest point
    // If they reverse direction and came back up past the trigger
    if(hipYn < descTrigger && peakY > minDepth){
      // Moved enough — now ascending
      repState="ASCENDING";
    }
    // Abandoned movement (barely moved, came back immediately)
    else if(hipYn < returnTrig && peakY <= minDepth && (idx-repStart) < 8){
      repState="IDLE";
    }
  }
  else if(repState==="ASCENDING"){
    if(hipYn > peakY) peakY=hipYn; // still going deeper
    // Completed rep: returned close to standing position
    if(hipYn <= returnTrig && (idx-repStart) >= 6){
      repCount++;
      const el=document.getElementById("rep-num");
      el.textContent=repCount;
      el.style.color="#fff";
      setTimeout(()=>el.style.color="#60A5FA",300);
      speakRepCount(repCount);
      repState="IDLE";
      peakY=0;
    }
  }
}


function setStatus(s){
  const el=document.getElementById("status-badge");
  const m={
    off:     ["OFF",""],
    loading: ["LOADING","pulsing"],
    waiting: ["DETECTING","waiting pulsing"],
    gesture: ["RAISE HANDS","waiting pulsing"],
    active:  ["REC ●","active pulsing"],
    done:    ["DONE ✓","done"]
  };
  const[t,c]=m[s]||m.off;el.textContent=t;el.className=c;
}

// ── State ─────────────────────────────────────────────────────────
let detector=null,stream=null,rafId=null,running=false;
let facingMode="environment"; // start with back camera
let sessionFrames=[],fpsT=performance.now(),fpsN=0;
let mediaRecorder=null,recordedChunks=[];

async function detect(){
  if(!running)return;
  const video =document.getElementById("video");
  const canvas=document.getElementById("overlay");
  const ctx   =canvas.getContext("2d");
  const vW=video.videoWidth||720, vH=video.videoHeight||1280;

  // object-fit:cover — video fills container, edges cropped (not letterboxed)
  // Canvas must match container size; keypoint coords scale with cover math
  const cW=canvas.offsetWidth, cH=canvas.offsetHeight;
  // Cover scale = fill both dimensions, crop the overflow
  const scale=Math.max(cW/vW, cH/vH);
  const rW=vW*scale, rH=vH*scale;
  const offX=(cW-rW)/2, offY=(cH-rH)/2;  // negative = cropped amount

  canvas.width=cW; canvas.height=cH;
  try{
    const mirror=(facingMode==="user");
    const poses=await detector.estimatePoses(video,{flipHorizontal:mirror});
    ctx.clearRect(0,0,cW,cH);

    if(poses.length>0){
      const kp=poses[0].keypoints;

      if(!repActive && !countdownActive){
        // ── GESTURE PHASE: detect raise-hands, no skeleton yet ────
        const ready=checkGesture(kp,vW,vH);
        if(ready){
          gestureHoldCount++;
          if(gestureHoldCount>=GESTURE_HOLD){
            startCountdown();
          }
        } else {
          gestureHoldCount=0;
        }
        updateGestureOverlay(gestureHoldCount, ready);
        // No skeleton, no flags during gesture phase

      } else if(countdownActive){
        // ── COUNTDOWN PHASE: show skeleton, no flags, overlay shows countdown
        drawSkeleton(ctx,kp,{},vW,vH,rW,rH,offX,offY);

      } else if(repActive){
        // ── ACTIVE PHASE: full skeleton + flags + rep counting ────
        hideGestureOverlay();
        const flags=checkThresholds(kp,vW,vH);
        drawSkeleton(ctx,kp,flags,vW,vH,rW,rH,offX,offY);
        updateFlags(flags);
        const lh=kp[MV.L_HIP],rh=kp[MV.R_HIP];
        if(lh&&rh&&lh.score>.3&&rh.score>.3)
          updateRep(kp, vW, vH);
      }

      // Recording composite: video frame behind skeleton
      if(mediaRecorder && mediaRecorder.state==="recording"){
        ctx.save();
        ctx.globalCompositeOperation="destination-over";
        if(mirror){
          ctx.translate(cW,0);ctx.scale(-1,1);
          ctx.drawImage(video,cW-offX-rW,offY,rW,rH);
        } else {
          ctx.drawImage(video,offX,offY,rW,rH);
        }
        ctx.restore();
      }

      sessionFrames.push(canvas.toDataURL("image/jpeg",.7));

    }else{
      // No pose detected
      if(!repActive && !countdownActive){
        // Hide gesture overlay when body not in frame
        hideGestureOverlay();
        // Simple hint text
        ctx.font="bold 13px Inter,system-ui";
        ctx.fillStyle="rgba(255,255,255,.4)";
        ctx.textAlign="center";
        ctx.fillText("Point camera at your full body",cW/2,cH/2);
        ctx.textAlign="left";
      }
    }
  }catch(e){console.warn("detect:",e);}
  fpsN++;const now=performance.now();
  if(now-fpsT>1000){
    document.getElementById("fps-badge").textContent=
      fpsN+" FPS · "+sessionFrames.length+" frames";
    fpsN=0;fpsT=now;
  }
  rafId=requestAnimationFrame(detect);
}

function toggleVoice(){
  voiceEnabled=!voiceEnabled;
  const btn=document.getElementById("btn-voice");
  const icon=document.getElementById("voice-icon");
  btn.className=voiceEnabled?"icon-btn":"icon-btn muted";
  if(icon){
    icon.setAttribute("stroke", voiceEnabled?"#60A5FA":"rgba(255,255,255,.2)");
  }
  if(voiceEnabled) speak("Voice on");
  else window.speechSynthesis.cancel();
}


// ── Orientation + Camera ─────────────────────────────────────────
let camOrientation = "portrait";

function setOrientation(o){
  camOrientation = o;
  document.getElementById("btn-portrait").className =
    "orient-btn" + (o==="portrait" ? " selected" : "");
  document.getElementById("btn-landscape").className =
    "orient-btn" + (o==="landscape" ? " selected" : "");
  document.getElementById("orient-hint").textContent =
    o==="portrait"
      ? "Phone propped upright — see head to toe"
      : "Phone on its side — wider field of view";
}

async function getCameraStream(facing){
  // Front camera always portrait regardless of orientation selection
  const portrait = (camOrientation === "portrait") || (facing === "user");
  try {
    // Try ideal resolution first
    return await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: facing },
        width:  { ideal: portrait ? 720  : 1280 },
        height: { ideal: portrait ? 1280 : 720  },
      }
    });
  } catch(e) {
    // Fallback — just request the camera with no size constraints
    return await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: facing } }
    });
  }
}

async function toggleCamera(){
  const btn=document.getElementById("btn-main");
  if(!running){
    btn.textContent="Loading...";btn.disabled=true;setStatus("loading");
    try{
      // ── Wake Lock — keep screen on during session ───────────────
      if('wakeLock' in navigator){
        try{
          wakeLock=await navigator.wakeLock.request('screen');
          wakeLock.addEventListener('release',()=>{wakeLock=null;});
        }catch(e){console.warn('Wake lock failed:',e);}
      }
      // MoveNet Lightning — most reliable on mobile browsers
      detector=await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {modelType:poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
         enableSmoothing:true}
      );
      stream=await getCameraStream(facingMode);
      const video=document.getElementById("video");
      video.srcObject=stream;
      document.getElementById("video").className = facingMode==="user"?"mirror":"";
      await new Promise(r=>video.onloadedmetadata=r);
      video.play();
      document.getElementById("cam-off").style.display="none";
      document.getElementById("fps-badge").style.display="block";
      document.getElementById("btn-flip").style.display="flex";
      document.getElementById("rep-ex").textContent=EXERCISE.toUpperCase();
      running=true;repCount=0;hipHist=[];repState="IDLE";
      calibN=0;standY=null;botY=null;sessionFrames=[];recordedChunks=[];
      rangeHistory=[];peakY=0;repStart=0;
      // Reset gesture state
      repActive=false;gestureHoldCount=0;countdownActive=false;countdownVal=5;
      document.getElementById("rep-num").textContent="0";
      document.getElementById("flags-wrap").innerHTML="";
      btn.textContent="STOP & ANALYSE";btn.className="stop";btn.disabled=false;
      setStatus("gesture");
      // Preload voices and greet
      window.speechSynthesis.getVoices();
      setTimeout(()=>speak("Raise both hands when you're ready to start."), 600);
      // Start recording the canvas overlay stream
      const canvas=document.getElementById("overlay");
      const mimeType=MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ?"video/webm;codecs=vp9":"video/webm";
      try{
        mediaRecorder=new MediaRecorder(canvas.captureStream(15),
          {mimeType,videoBitsPerSecond:2000000});
        mediaRecorder.ondataavailable=e=>{if(e.data.size>0)recordedChunks.push(e.data);};
        mediaRecorder.start(500);
      }catch(err){console.warn("MediaRecorder:",err);mediaRecorder=null;}
      detect();
    }catch(e){
      btn.textContent="START CAMERA";btn.className="start";btn.disabled=false;
      setStatus("off");alert("Camera error: "+e.message);
    }
    // Re-acquire wake lock if OS releases it (e.g. battery saver)
    document.addEventListener('visibilitychange',async()=>{
      if(running && document.visibilityState==='visible' && !wakeLock && 'wakeLock' in navigator){
        try{wakeLock=await navigator.wakeLock.request('screen');}catch(e){}
      }
    },{once:false});
  }else{
    running=false;cancelAnimationFrame(rafId);
    if(stream)stream.getTracks().forEach(t=>t.stop());
    // Release wake lock
    if(wakeLock){try{await wakeLock.release();}catch(e){}wakeLock=null;}
    // Reset gesture
    repActive=false;gestureHoldCount=0;countdownActive=false;
    hideGestureOverlay();
    document.getElementById("cam-off").style.display="flex";
    document.getElementById("fps-badge").style.display="none";
    document.getElementById("btn-flip").style.display="none";
    btn.textContent="START CAMERA";btn.className="start";
    setStatus("done");
    window.speechSynthesis.cancel();
    clearTimeout(voiceQueue);
    lastFlagState={};flagHoldCount={};lastRepSpoken=0;
    setTimeout(()=>speak(repCount+" reps. Session done."),300);
    // Stop recording and auto-download the annotated video
    if(mediaRecorder&&mediaRecorder.state!=="inactive"){
      mediaRecorder.onstop=()=>{
        const blob=new Blob(recordedChunks,{type:"video/webm"});
        const url=URL.createObjectURL(blob);
        const a=document.createElement("a");
        a.href=url;a.download="formate_live_session_"+repCount+"reps.webm";
        document.body.appendChild(a);a.click();
        document.body.removeChild(a);
        setTimeout(()=>URL.revokeObjectURL(url),1000);
        // Show save button as fallback
        document.getElementById("btn-save").style.display="inline-block";
      };
      mediaRecorder.stop();
    }else{
      if(sessionFrames.length>5)
        document.getElementById("btn-save").style.display="inline-block";
    }
  }
}

async function flipCamera(){
  if(!running)return;
  facingMode=facingMode==="environment"?"user":"environment";
  if(stream)stream.getTracks().forEach(t=>t.stop());
  const video=document.getElementById("video");
  stream=await getCameraStream(facingMode);
  video.srcObject=stream;
  document.getElementById("video").className = facingMode==="user"?"mirror":"";
  await new Promise(r=>video.onloadedmetadata=r);
  video.play();
  // Orientation picker only relevant for back camera
  const op=document.getElementById("orient-picker");
  if(op) op.style.opacity = facingMode==="user"?"0.3":"1";
}

function saveSession(){
  // Fallback: if MediaRecorder didn't work, re-trigger download reminder
  const btn=document.getElementById("btn-save");
  btn.textContent="Check your Downloads folder \u2713";
  btn.disabled=true;
}
</script>
</body></html>
""".replace("EXERCISE_PLACEHOLDER", ex_js)




    # ── Kill Streamlit padding around the iframe so it's edge-to-edge ──
    st.markdown("""
    <style>
    /* Remove all padding from the live tab so iframe fills width */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] > div,
    .stTabs [data-baseweb="tab-panel"] > div > div {
        padding-left: 0 !important;
        padding-right: 0 !important;
        padding-top: 0 !important;
    }
    iframe { display: block; border: none; }
    </style>
    """, unsafe_allow_html=True)

    result_data = components.html(movenet_html, height=780, scrolling=False)

    # ── Receive frames from iframe and auto-process ───────────────
    # JS sends postMessage with base64 JPEG frames on Stop
    # We poll st.query_params for the trigger, but since postMessage
    # can't directly update Python state, we use a hidden file uploader
    # that JS programmatically fills via a Blob URL download trick.
    # Simpler: we embed a second tiny component that writes to a shared temp file.

    st.markdown('<p class="lbl" style="margin-top:1rem">Live Session Analysis</p>', unsafe_allow_html=True)

    # Guard: don't re-process if we already have results for this upload
    if not st.session_state.live_results:
        live_upload = st.file_uploader(
            "Session video will appear here automatically after stopping — or upload manually",
            type=["mp4","mov","webm"],
            key="live_video_upload",
            label_visibility="visible"
        )

        if live_upload and not st.session_state.get("live_processing_done"):
            import re as _re
            m = _re.search(r'_(\d+)reps', live_upload.name)
            js_rep_count = int(m.group(1)) if m else None

            with st.spinner("Running pipeline on live session..."):
                tmp_dir   = Path(tempfile.mkdtemp())
                tmp_video = tmp_dir / "live_session.mp4"
                live_upload.seek(0)
                tmp_video.write_bytes(live_upload.read())
                try:
                    result = run_pipeline(tmp_video, exercise, camera_view)
                    # Inject JS rep count — ground truth from live counter
                    if js_rep_count and js_rep_count > 0:
                        sid, b_sum, g_sum, rep_df, num_reps, gold_dir = result
                        g_sum["reps"] = js_rep_count
                        result = (sid, b_sum, g_sum, rep_df, js_rep_count, gold_dir)
                        st.session_state.live_rep_count = js_rep_count
                    st.session_state.live_results = result
                    st.session_state.live_processing_done = True
                except Exception as e:
                    st.error(f"Pipeline error: {e}")

    # Live results
    if st.session_state.live_results:
        sid, b_sum, g_sum, rep_df, num_reps, gold_dir = st.session_state.live_results
        live_vid = st.session_state.get("live_annotated_vid", None)
        st.markdown('<hr class="div"/>', unsafe_allow_html=True)
        st.markdown('<p class="st2">Live Session Results</p>', unsafe_allow_html=True)
        render_results(sid, gold_dir, b_sum, g_sum, rep_df, num_reps, exercise, live_vid=live_vid)

        if st.button("New Session", key="new_live"):
            st.session_state.live_results           = None
            st.session_state.live_rep_count         = 0
            st.session_state.live_frames            = []
            st.session_state.live_landmarks         = []
            st.session_state.live_started           = False
            st.session_state.live_annotated_vid     = None
            st.session_state.live_processing_done   = False
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
