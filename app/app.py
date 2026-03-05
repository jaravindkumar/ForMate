import json, os, time, tempfile, requests, subprocess, sys, uuid, math
from pathlib import Path
import numpy as np
import streamlit as st
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parents[1]
PY   = sys.executable

st.set_page_config(page_title="FORMate", layout="wide", page_icon="F", initial_sidebar_state="collapsed")

# ─── CSS ──────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Archivo:wght@300;400;500;600&display=swap');

:root{
  --ink:#06060A; --surface:#0D0D12; --card:#13131A;
  --edge:#1D1D28; --edge2:#252533; --txt:#F0EEF8;
  --sub:#5A5870;  --sub2:#2A2838;
  --acid:#BBFF00; --acid2:#D4FF4D;
  --red:#FF3F3F;  --green:#2ECC71; --orange:#F59E0B;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[class*="css"]{font-family:'Archivo',sans-serif;background:var(--ink)!important;color:var(--txt)!important;}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden!important;display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stSidebar"]{display:none!important;}
::-webkit-scrollbar{width:3px;}::-webkit-scrollbar-thumb{background:var(--acid);border-radius:2px;}

/* NAV */
.nav{display:flex;align-items:center;justify-content:space-between;padding:0 2.5rem;height:60px;background:var(--surface);border-bottom:1px solid var(--edge);position:sticky;top:0;z-index:100;}
.logo{font-family:'Archivo Black',sans-serif;font-size:1.5rem;letter-spacing:-.02em;}
.logo-form{background:linear-gradient(135deg,#3B82F6 0%,#60A5FA 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.logo-ate{color:var(--txt);}
.nav-right{display:flex;gap:.5rem;align-items:center;}
.nbadge{font-size:.6rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;padding:.28rem .7rem;border-radius:20px;background:var(--card);border:1px solid var(--edge2);color:var(--sub);}
.nbadge.live{background:rgba(255,63,63,.12);border-color:rgba(255,63,63,.3);color:#ff6b6b;}
.nbadge.live::before{content:'';display:inline-block;width:6px;height:6px;border-radius:50%;background:#ff3f3f;margin-right:.4rem;animation:pulse 1.2s infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.3;}}

/* MAIN */
.main{max-width:1400px;margin:0 auto;padding:2.5rem 2.5rem 5rem;}

/* TABS */
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border:1px solid var(--edge)!important;border-radius:12px!important;padding:.3rem!important;gap:.25rem!important;margin-bottom:1.5rem!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;border-radius:9px!important;color:var(--sub)!important;font-family:'Archivo Black',sans-serif!important;font-size:.78rem!important;letter-spacing:.08em!important;padding:.55rem 1.25rem!important;border:none!important;}
.stTabs [aria-selected="true"]{background:var(--acid)!important;color:var(--ink)!important;}
.stTabs [data-baseweb="tab-panel"]{padding:0!important;}
.stTabs [data-baseweb="tab-border"]{display:none!important;}

/* UPLOAD ZONE */
.zone{border:1px solid var(--edge);border-radius:20px;background:var(--surface);padding:2.5rem;margin-bottom:1.5rem;}
.uz-headline{font-family:'Archivo Black',sans-serif;font-size:clamp(1.8rem,4vw,3rem);line-height:1.05;letter-spacing:-.02em;color:var(--txt);margin-bottom:.5rem;}
.uz-headline span{color:var(--acid);}
.uz-desc{font-size:.88rem;color:var(--sub);line-height:1.7;font-weight:300;max-width:420px;margin-bottom:1.25rem;}

/* LIVE ZONE */
.live-zone{border:1px solid rgba(255,63,63,.25);border-radius:20px;background:var(--surface);padding:2.5rem;margin-bottom:1.5rem;position:relative;}
.live-zone::before{content:'';position:absolute;inset:0;border-radius:20px;background:radial-gradient(ellipse at top left,rgba(255,63,63,.04) 0%,transparent 60%);pointer-events:none;}
.live-headline{font-family:'Archivo Black',sans-serif;font-size:clamp(1.8rem,4vw,3rem);line-height:1.05;letter-spacing:-.02em;color:var(--txt);margin-bottom:.5rem;}
.live-headline span{color:#ff6b6b;}
.live-desc{font-size:.88rem;color:var(--sub);line-height:1.7;font-weight:300;max-width:420px;margin-bottom:1.25rem;}

/* REP COUNTER */
.rep-counter{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  background:var(--card);border:1px solid var(--edge);border-radius:16px;
  padding:1.5rem;text-align:center;
}
.rep-num{font-family:'Archivo Black',sans-serif;font-size:5rem;line-height:1;color:var(--acid);letter-spacing:-.03em;}
.rep-label{font-size:.6rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--sub);margin-top:.3rem;}
.rep-exercise{font-size:.75rem;color:var(--sub2);margin-top:.25rem;text-transform:uppercase;letter-spacing:.1em;}

/* STATUS BADGE */
.status-badge{display:inline-flex;align-items:center;gap:.5rem;padding:.4rem 1rem;border-radius:20px;font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:1rem;}
.status-badge.waiting{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.25);color:#fbbf24;}
.status-badge.detecting{background:rgba(187,255,0,.08);border:1px solid rgba(187,255,0,.2);color:var(--acid);}
.status-badge.recording{background:rgba(255,63,63,.1);border:1px solid rgba(255,63,63,.25);color:#ff6b6b;}
.status-badge.done{background:rgba(46,204,113,.1);border:1px solid rgba(46,204,113,.25);color:var(--green);}
.status-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.status-badge.waiting .status-dot{background:#fbbf24;}
.status-badge.detecting .status-dot{background:var(--acid);animation:pulse 1s infinite;}
.status-badge.recording .status-dot{background:#ff3f3f;animation:pulse .8s infinite;}
.status-badge.done .status-dot{background:var(--green);}

/* LIVE INSTRUCTIONS */
.live-steps{display:flex;flex-direction:column;gap:.6rem;margin-bottom:1.25rem;}
.live-step{display:flex;align-items:flex-start;gap:.75rem;padding:.75rem 1rem;border-radius:9px;background:var(--card);border:1px solid var(--edge);font-size:.83rem;color:var(--sub);line-height:1.5;}
.step-num{font-family:'Archivo Black',sans-serif;font-size:1rem;color:var(--acid);flex-shrink:0;line-height:1;margin-top:.05rem;}

/* LABELS */
.lbl{font-size:.58rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--sub);margin-bottom:.3rem;display:flex;align-items:center;gap:.5rem;}
.lbl::after{content:'';flex:1;height:1px;background:var(--edge);}

/* CONTROLS */
.stSelectbox>div>div{background:var(--card)!important;border:1px solid var(--edge2)!important;border-radius:9px!important;color:var(--txt)!important;font-size:.87rem!important;}
.stSelectbox label{display:none!important;}
[data-testid="stFileUploader"]{background:var(--card)!important;border:1.5px dashed var(--edge2)!important;border-radius:12px!important;transition:border-color .2s!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--acid)!important;}
[data-testid="stFileUploader"] *{font-size:.85rem!important;}
[data-testid="stCameraInput"]{border-radius:12px!important;overflow:hidden!important;}
[data-testid="stCameraInput"] *{border-radius:12px!important;}

/* BUTTONS */
.stButton>button{font-family:'Archivo Black',sans-serif!important;font-size:1rem!important;letter-spacing:.08em!important;background:var(--acid)!important;color:var(--ink)!important;border:none!important;border-radius:10px!important;padding:.85rem 2rem!important;width:100%!important;transition:all .18s!important;}
.stButton>button:hover{background:var(--acid2)!important;transform:translateY(-2px)!important;box-shadow:0 8px 32px rgba(187,255,0,.2)!important;}
.stop-btn .stButton>button{background:rgba(255,63,63,.15)!important;color:#ff6b6b!important;border:1px solid rgba(255,63,63,.3)!important;}
.stop-btn .stButton>button:hover{background:rgba(255,63,63,.25)!important;box-shadow:0 8px 24px rgba(255,63,63,.15)!important;}

/* FILE OK */
.fok{padding:.65rem 1rem;border-radius:9px;font-size:.8rem;background:rgba(187,255,0,.06);border:1px solid rgba(187,255,0,.2);color:var(--acid);}

/* PROGRESS */
.pstatus{font-family:'Archivo Black',sans-serif;font-size:.95rem;letter-spacing:.06em;color:var(--acid);margin-bottom:.35rem;}
.stProgress>div{background:var(--edge2)!important;height:3px!important;border-radius:2px!important;}
.stProgress>div>div{background:var(--acid)!important;border-radius:2px!important;}

/* SCORE BANNER */
.sbanner{display:grid;grid-template-columns:180px 1fr;border-radius:18px;overflow:hidden;border:1px solid var(--edge);margin-bottom:2rem;}
.sbanner-score{background:var(--acid);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:2rem 1.5rem;}
.sbanner-num{font-family:'Archivo Black',sans-serif;font-size:5.5rem;line-height:1;color:var(--ink);letter-spacing:-.03em;}
.sbanner-den{font-size:.65rem;font-weight:700;letter-spacing:.18em;color:rgba(6,6,10,.45);text-transform:uppercase;}
.sbanner-stats{background:var(--card);display:grid;grid-template-columns:repeat(6,1fr);}
.ss{padding:1rem .75rem;border-left:1px solid var(--edge);display:flex;flex-direction:column;justify-content:center;}
.ss-v{font-family:'Archivo Black',sans-serif;font-size:1.9rem;line-height:1;color:var(--txt);}
.ss-v.g{color:var(--green);}.ss-v.w{color:var(--red);}
.ss-k{font-size:.57rem;font-weight:600;letter-spacing:.15em;text-transform:uppercase;color:var(--sub);margin-top:.25rem;}

/* CARDS */
.card{background:var(--card);border:1px solid var(--edge);border-radius:16px;padding:1.5rem;position:relative;overflow:hidden;}
.card::after{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(187,255,0,.02) 0%,transparent 60%);pointer-events:none;}
.st2{font-size:.6rem;font-weight:700;letter-spacing:.22em;text-transform:uppercase;color:var(--acid);margin-bottom:1rem;display:flex;align-items:center;gap:.6rem;}
.st2::after{content:'';flex:1;height:1px;background:var(--edge);}
.ci{display:flex;align-items:flex-start;gap:.7rem;padding:.75rem .9rem;border-radius:9px;margin-bottom:.45rem;font-size:.85rem;line-height:1.55;}
.ci.pos{background:rgba(46,204,113,.07);border:1px solid rgba(46,204,113,.14);color:#7ee8a2;}
.ci.imp{background:rgba(255,63,63,.07);border:1px solid rgba(255,63,63,.14);color:#ffaaaa;}
.ci.foc{background:rgba(187,255,0,.05);border:1px solid rgba(187,255,0,.12);color:#cbff5e;}
.ci-icon{font-size:.8rem;margin-top:.18rem;flex-shrink:0;}
.bw{margin-bottom:.8rem;}
.brow{display:flex;justify-content:space-between;margin-bottom:.2rem;}
.bname{font-size:.73rem;font-weight:500;color:var(--sub);}
.bval{font-family:'Archivo Black',sans-serif;font-size:.78rem;color:var(--txt);}
.btrack{height:3px;background:var(--edge2);border-radius:2px;overflow:hidden;}
.bfill{height:100%;border-radius:2px;background:var(--acid);}
.bfill.mid{background:var(--orange);}.bfill.low{background:var(--red);}
.rbox{background:var(--surface);border-radius:10px;padding:1.25rem 1.4rem;font-size:.85rem;line-height:1.85;color:var(--sub);white-space:pre-wrap;border-top:2px solid var(--acid);}
.flag{display:flex;gap:.6rem;align-items:flex-start;padding:.65rem .9rem;border-radius:8px;background:rgba(255,63,63,.05);border:1px solid rgba(255,63,63,.12);font-size:.82rem;color:#ffaaaa;margin-bottom:.4rem;line-height:1.5;}
.fdot{width:5px;height:5px;border-radius:50%;background:var(--red);flex-shrink:0;margin-top:.5rem;}
.div{border:none;border-top:1px solid var(--edge);margin:1.1rem 0;}
.streamlit-expanderHeader{background:var(--surface)!important;border:1px solid var(--edge)!important;border-radius:9px!important;font-size:.8rem!important;color:var(--sub)!important;}

/* EMPTY */
.empty-state{text-align:center;padding:5rem 1rem;display:flex;flex-direction:column;align-items:center;gap:1rem;}
.empty-logo{font-family:'Archivo Black',sans-serif;font-size:clamp(4rem,12vw,10rem);line-height:1;letter-spacing:-.02em;color:rgba(255,255,255,.04);}
.empty-logo b{background:linear-gradient(135deg,rgba(59,130,246,.15) 0%,rgba(96,165,250,.2) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.empty-txt{font-size:.9rem;color:var(--sub2);max-width:300px;line-height:1.7;}
.vid-preview{border-radius:12px;overflow:hidden;border:1px solid var(--edge);}
.foot{text-align:center;font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:var(--sub2);margin-top:2.5rem;}

/* MOBILE */
@media(max-width:900px){
  .nav{padding:0 1rem;}.main{padding:1.5rem 1rem 4rem;}
  .sbanner{grid-template-columns:1fr;}
  .sbanner-score{flex-direction:row;gap:1.5rem;padding:1.25rem;}
  .sbanner-num{font-size:4rem;}
  .sbanner-stats{grid-template-columns:repeat(3,1fr);}
}
@media(max-width:480px){
  .sbanner-stats{grid-template-columns:repeat(2,1fr);}
  .uz-headline,.live-headline{font-size:1.8rem;}
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
    """Run bronze/silver/gold pipeline on a video file. Returns (session_id, b_sum, g_sum, rep_df, num_reps)."""
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
        return None

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
        return None
    s_sum    = read_json(ROOT / "pipeline" / "silver" / session_id / "summary.json")
    num_reps = s_sum["num_reps_detected"]
    prog.progress(65)

    stat.markdown('<p class="pstatus">Scoring form...</p>', unsafe_allow_html=True)
    prog.progress(72)
    p = run_cmd([PY, "pipeline_gold_score.py", "--session_id", session_id, "--exercise", exercise], cwd=ROOT)
    if p.returncode != 0:
        st.error("Scoring failed.")
        with st.expander("Error details"): st.code(p.stderr or p.stdout)
        return None

    gold_dir = ROOT / "pipeline" / "gold" / session_id
    g_sum    = read_json(gold_dir / "summary.json")
    rep_df   = pd.read_parquet(gold_dir / "metrics_reps.parquet")
    prog.progress(100)
    stat.empty()
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
    '<div class="logo"><span class="logo-form">FORM</span><span class="logo-ate">ate</span></div>'
    '<div class="nav-right">' + live_badge +
    '<span class="nbadge">AI Form Coach &middot; MVP</span>'
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
            '<h2 class="uz-headline">Analyse Your<br><span>Form.</span></h2>'
            '<p class="uz-desc">Upload a workout video — MoveNet previews your skeleton live as it plays, then the full AI pipeline scores every rep.</p>',
            unsafe_allow_html=True
        )
        st.markdown('<p class="lbl">Upload Video</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["mp4","mov","m4v"], label_visibility="collapsed")
        if uploaded:
            fname = uploaded.name
            fmb   = str(round(uploaded.size / (1024*1024), 1))
            st.markdown('<div class="fok">&#10003; ' + fname + ' &middot; ' + fmb + ' MB</div>', unsafe_allow_html=True)

    with uz_r:
        if uploaded:
            tmp_dir   = Path(tempfile.mkdtemp())
            tmp_video = tmp_dir / uploaded.name
            uploaded.seek(0)
            tmp_video.write_bytes(uploaded.read())

            # Encode video as base64 so TF.js component can play it inline
            vid_b64  = base64.b64encode(tmp_video.read_bytes()).decode()
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
.badge.green{background:rgba(187,255,0,.1);border:1px solid rgba(187,255,0,.25);color:#BBFF00;}
.badge.loading{background:rgba(90,88,112,.08);border:1px solid #1D1D28;color:#5A5870;}
.badge.err{background:rgba(255,63,63,.1);border:1px solid rgba(255,63,63,.25);color:#ff8080;}
.flags{display:flex;gap:.35rem;flex-wrap:wrap;}
.flag{font-size:.65rem;font-weight:700;padding:.22rem .6rem;border-radius:10px;}
.flag.ok  {background:rgba(46,204,113,.1);border:1px solid rgba(46,204,113,.2);color:#7ee8a2;}
.flag.warn{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.25);color:#fbbf24;}
.flag.bad {background:rgba(255,63,63,.1);border:1px solid rgba(255,63,63,.25);color:#ff7070;}
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
    if(flags[k].st==="bad")return"#FF3F3F";
    if(flags[k].st==="warn")w="warn";
  }
  return w==="warn"?"#F59E0B":"#BBFF00";
}

function drawSkeleton(ctx,kp,flags,W,H){
  const px=i=>kp[i].x*W, py=i=>kp[i].y*H, vs=i=>kp[i]?kp[i].score:0;
  const sts=Object.values(flags).map(f=>f.st);
  const cc=sts.includes("bad")?"#FF3F3F":sts.includes("warn")?"#F59E0B":"#BBFF00";
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
    const col=st==="bad"?"#FF4444":"#F59E0B";
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
                '<div class="empty-logo"><b>FORM</b>ate</div>'
                '<p class="empty-txt">Upload a video to begin analysis.</p>'
                '</div>',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        if st.button("ANALYSE FORM", type="primary", use_container_width=True, key="run_upload"):
            result = run_pipeline(tmp_video, exercise, camera_view)
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
*{box-sizing:border-box;margin:0;padding:0;}
html,body{width:100%;height:100%;background:#000;overflow:hidden;}
#cam-container{position:relative;width:100%;height:100vh;background:#000;overflow:hidden;}
video{position:absolute;top:0;left:0;width:1px;height:1px;opacity:0;pointer-events:none;}
canvas{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;}

/* TOP HUD */
#hud-top{position:absolute;top:0;left:0;right:0;padding:.75rem 1rem .5rem;
  background:linear-gradient(to bottom,rgba(0,0,0,.65) 0%,transparent 100%);
  display:flex;align-items:flex-start;justify-content:space-between;z-index:10;}
#rep-num{font-size:5.5rem;font-weight:900;line-height:1;color:#BBFF00;
  font-family:Arial Black,sans-serif;text-shadow:0 2px 16px rgba(0,0,0,.9);transition:color .15s;}
#rep-sub{font-size:.55rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;
  color:rgba(255,255,255,.55);margin-top:.1rem;}
#rep-ex{font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;color:#60A5FA;opacity:.85;}
#status-badge{font-size:.6rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  padding:.3rem .8rem;border-radius:20px;backdrop-filter:blur(4px);
  background:rgba(0,0,0,.45);border:1px solid rgba(255,255,255,.15);color:rgba(255,255,255,.6);}
#status-badge.waiting{border-color:rgba(245,158,11,.6);color:#fbbf24;}
#status-badge.active{border-color:rgba(255,63,63,.6);color:#ff6b6b;}
#status-badge.done{border-color:rgba(187,255,0,.5);color:#BBFF00;}

/* DEBUG / FPS */
#fps-badge{position:absolute;top:.75rem;left:50%;transform:translateX(-50%);
  font-size:.6rem;font-weight:700;background:rgba(0,0,0,.55);
  border:1px solid rgba(255,255,255,.1);border-radius:6px;padding:.2rem .55rem;
  color:rgba(255,255,255,.45);z-index:10;white-space:nowrap;}

/* FORM FLAGS */
#flags-wrap{position:absolute;bottom:5.5rem;left:.75rem;
  display:flex;flex-direction:column;gap:.3rem;z-index:10;}
.flag{font-size:.72rem;font-weight:700;padding:.28rem .65rem;border-radius:8px;
  backdrop-filter:blur(6px);}
.flag.ok  {background:rgba(46,204,113,.2);border:1px solid rgba(46,204,113,.35);color:#a0f0c0;}
.flag.warn{background:rgba(245,158,11,.22);border:1px solid rgba(245,158,11,.4);color:#ffd060;}
.flag.bad {background:rgba(255,63,63,.25);border:1px solid rgba(255,63,63,.45);color:#ff8888;}

/* BOTTOM CONTROLS */
#controls{position:absolute;bottom:0;left:0;right:0;padding:.75rem 1rem 1.25rem;
  background:linear-gradient(to top,rgba(0,0,0,.7) 0%,transparent 100%);
  display:flex;gap:.6rem;align-items:center;justify-content:center;z-index:10;}
#btn-main{flex:1;max-width:260px;font-weight:900;font-size:.95rem;letter-spacing:.07em;
  border:none;border-radius:50px;padding:.85rem 1.5rem;cursor:pointer;transition:all .18s;}
#btn-main.start{background:#BBFF00;color:#06060A;}
#btn-main.stop{background:rgba(220,50,50,.9);color:#fff;}
#btn-save{font-weight:700;font-size:.78rem;letter-spacing:.05em;
  border:1px solid rgba(187,255,0,.4);border-radius:50px;padding:.75rem 1.25rem;
  background:rgba(0,0,0,.55);color:#BBFF00;cursor:pointer;backdrop-filter:blur(4px);display:none;}
#btn-flip{width:44px;height:44px;border-radius:50%;border:1px solid rgba(255,255,255,.2);
  background:rgba(0,0,0,.5);color:#fff;font-size:1.2rem;cursor:pointer;
  backdrop-filter:blur(4px);display:flex;align-items:center;justify-content:center;
  flex-shrink:0;display:none;}

/* OFF STATE */
#cam-off{position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:.85rem;background:#06060A;color:#2A2838;z-index:5;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.25;}}
.pulsing{animation:pulse .8s infinite;}
</style>
</head>
<body>
<div id="cam-container">
  <div id="cam-off">
    <div style="font-size:3rem">&#128247;</div>
    <div style="font-size:.95rem;font-weight:900;font-family:Arial Black,sans-serif;background:linear-gradient(135deg,#3B82F6,#60A5FA);-webkit-background-clip:text;-webkit-text-fill-color:transparent">FORM</div><div style="font-size:.95rem;font-weight:900;font-family:Arial Black,sans-serif;color:#fff;display:inline">ate</div>
    <div style="font-size:.72rem;color:#1D1D28;max-width:200px;text-align:center;line-height:1.6">
      Point camera at your full body. MoveNet tracks your skeleton in real-time.</div>
  </div>
  <video id="video" autoplay playsinline muted style="display:none"></video>
  <canvas id="overlay"></canvas>
  <div id="hud-top">
    <div>
      <div id="rep-num">0</div>
      <div id="rep-sub">REPS</div>
      <div id="rep-ex">EXERCISE_PLACEHOLDER</div>
    </div>
    <div id="status-badge">OFF</div>
  </div>
  <div id="fps-badge" style="display:none">-- FPS</div>
  <div id="flags-wrap"></div>
  <div id="controls">
    <button id="btn-flip" onclick="flipCamera()">&#x1F504;</button>
    <button id="btn-main" class="start" onclick="toggleCamera()">START CAMERA</button>
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
    if(ok(5,11,13)){
      const a=angleDeg(xn(5),yn(5),xn(11),yn(11),xn(13),yn(13));
      f.back=a>=145?{st:"ok",lbl:"Back OK"}:a>=115?{st:"warn",lbl:"Back rounding"}:{st:"bad",lbl:"Back round!"};
    }
    if(ok(5,11)){const d=Math.abs(xn(5)-xn(11));
      f.drift=d<.07?{st:"ok",lbl:"Bar OK"}:d<.14?{st:"warn",lbl:"Bar drifting"}:{st:"bad",lbl:"Bar too far!"};}
    if(ok(11,13)){const d=yn(13)-yn(11);
      f.hinge=d>.05?{st:"ok",lbl:"Hinge OK"}:{st:"warn",lbl:"Hinge deeper"};}
  }else{
    if(ok(11,13)){const c=xn(13)-xn(11);
      f.knee=c>=-.02?{st:"ok",lbl:"Knees OK"}:c>=-.07?{st:"warn",lbl:"Knee caving"}:{st:"bad",lbl:"Knee cave!"};}
    if(ok(11,13)){const d=yn(11)-yn(13);
      f.depth=d<=.04?{st:"ok",lbl:"Good depth"}:d<=.14?{st:"warn",lbl:"Go deeper"}:{st:"bad",lbl:"Too shallow"};}
    if(ok(5,11)){const l=Math.abs(xn(5)-xn(11));
      f.lean=l<.07?{st:"ok",lbl:"Upright OK"}:l<.14?{st:"warn",lbl:"Leaning fwd"}:{st:"bad",lbl:"Too much lean!"};}
  }
  return f;
}

function jointColor(idx,flags){
  let w="ok";
  for(const[k,jts]of Object.entries(KEY_JOINTS)){
    if(!flags[k]||!jts.includes(idx))continue;
    if(flags[k].st==="bad")return"#FF3F3F";
    if(flags[k].st==="warn")w="warn";
  }
  return w==="warn"?"#F59E0B":"#BBFF00";
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
  const cc=sts.includes("bad")?"#FF3F3F":sts.includes("warn")?"#F59E0B":"#BBFF00";

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
    const col=st==="bad"?"#FF4444":"#F59E0B";
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
}

// ── Rep state machine (normalised hip Y) ─────────────────────────
let repCount=0,hipHist=[],repState="IDLE";
let repStart=0,peakY=0,standY=null,botY=null,calibN=0;
const CALIB=50;

function updateRep(hipYnorm,idx){
  hipHist.push(hipYnorm);if(hipHist.length>90)hipHist.shift();
  if(calibN<CALIB){calibN++;if(standY===null||hipYnorm<standY)standY=hipYnorm;return;}
  if(botY===null){
    const s=[...hipHist].sort((a,b)=>a-b);
    standY=s[Math.floor(s.length*.15)];
    botY  =s[Math.floor(s.length*.85)];
  }
  const r=botY-standY;if(r<.04)return;
  const dT=standY+r*.35,bT=standY+r*.65,rT=standY+r*.30;
  if(repState==="IDLE"){
    if(hipYnorm>dT){repState="DESCENDING";repStart=idx;peakY=hipYnorm;setStatus("active");}
  }else if(repState==="DESCENDING"){
    if(hipYnorm>peakY)peakY=hipYnorm;
    if(hipYnorm>=bT)repState="AT_BOTTOM";
    else if(hipYnorm<rT&&(idx-repStart)<10)repState="IDLE";
  }else if(repState==="AT_BOTTOM"){
    if(hipYnorm>peakY)peakY=hipYnorm;
    if(hipYnorm<dT)repState="ASCENDING";
  }else if(repState==="ASCENDING"){
    if(hipYnorm>peakY){peakY=hipYnorm;repState="AT_BOTTOM";}
    else if(hipYnorm<=rT&&(idx-repStart)>=8){
      repCount++;
      const el=document.getElementById("rep-num");
      el.textContent=repCount;
      el.style.color="#fff";setTimeout(()=>el.style.color="#BBFF00",250);
      repState="IDLE";
    }
  }
}

function setStatus(s){
  const el=document.getElementById("status-badge");
  const m={off:["OFF",""],loading:["LOADING","pulsing"],
           waiting:["DETECTING","waiting pulsing"],
           active:["REC ●","active pulsing"],done:["DONE ✓","done"]};
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
  const vW=video.videoWidth||640, vH=video.videoHeight||480;

  // With object-fit:contain, video is letterboxed inside the container.
  // Calculate the actual rendered rect so canvas coords match exactly.
  const cW=canvas.offsetWidth, cH=canvas.offsetHeight;
  const scale=Math.min(cW/vW, cH/vH);
  const rW=vW*scale, rH=vH*scale;
  const offX=(cW-rW)/2, offY=(cH-rH)/2;

  // Canvas internal resolution = container size (so CSS 100%/100% aligns)
  canvas.width=cW; canvas.height=cH;
  try{
    const poses=await detector.estimatePoses(video,{flipHorizontal:false});
    ctx.clearRect(0,0,cW,cH);
    // Draw video frame onto canvas first — this makes captureStream() record the composite
    ctx.save();
    if(facingMode==="user"){
      // mirror the video draw to match CSS scaleX(-1)
      ctx.translate(cW,0);ctx.scale(-1,1);
      ctx.drawImage(video,cW-offX-rW,offY,rW,rH);
    }else{
      ctx.drawImage(video,offX,offY,rW,rH);
    }
    ctx.restore();
    if(poses.length>0){
      const kp=poses[0].keypoints;
      const flags=checkThresholds(kp,vW,vH);
      drawSkeleton(ctx,kp,flags,vW,vH,rW,rH,offX,offY);
      updateFlags(flags);
      const lh=kp[MV.L_HIP],rh=kp[MV.R_HIP];
      if(lh&&rh&&lh.score>.3&&rh.score>.3)
        updateRep(((lh.y+rh.y)/2)/vH, sessionFrames.length);
      sessionFrames.push(canvas.toDataURL("image/jpeg",.7));
    }else{
      ctx.font="bold 14px Arial";ctx.fillStyle="rgba(255,255,255,.35)";
      ctx.textAlign="center";
      ctx.fillText("Point camera at full body",cW/2,cH-20);
      ctx.textAlign="left";
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

async function toggleCamera(){
  const btn=document.getElementById("btn-main");
  if(!running){
    btn.textContent="Loading...";btn.disabled=true;setStatus("loading");
    try{
      // MoveNet Lightning — most reliable on mobile browsers
      detector=await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {modelType:poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
         enableSmoothing:true}
      );
      stream=await navigator.mediaDevices.getUserMedia(
        {video:{facingMode:facingMode,width:{ideal:1280},height:{ideal:720}},audio:false}
      );
      const video=document.getElementById("video");
      video.srcObject=stream;
      await new Promise(r=>video.onloadedmetadata=r);
      video.play();
      document.getElementById("cam-off").style.display="none";
      document.getElementById("fps-badge").style.display="block";
      document.getElementById("btn-flip").style.display="flex";
      document.getElementById("rep-ex").textContent=EXERCISE.toUpperCase();
      running=true;repCount=0;hipHist=[];repState="IDLE";
      calibN=0;standY=null;botY=null;sessionFrames=[];recordedChunks=[];
      document.getElementById("rep-num").textContent="0";
      btn.textContent="STOP & ANALYSE";btn.className="stop";btn.disabled=false;
      setStatus("waiting");
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
  }else{
    running=false;cancelAnimationFrame(rafId);
    if(stream)stream.getTracks().forEach(t=>t.stop());
    document.getElementById("cam-off").style.display="flex";
    document.getElementById("fps-badge").style.display="none";
    document.getElementById("btn-flip").style.display="none";
    btn.textContent="START CAMERA";btn.className="start";
    setStatus("done");
    // Stop recording and auto-download the annotated video
    if(mediaRecorder&&mediaRecorder.state!=="inactive"){
      mediaRecorder.onstop=()=>{
        const blob=new Blob(recordedChunks,{type:"video/webm"});
        const url=URL.createObjectURL(blob);
        const a=document.createElement("a");
        a.href=url;a.download="formate_live_session.webm";
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
  stream=await navigator.mediaDevices.getUserMedia(
    {video:{facingMode:facingMode,width:{ideal:1280},height:{ideal:720}},audio:false}
  );
  video.srcObject=stream;
  await new Promise(r=>video.onloadedmetadata=r);
  video.play();
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
    live_upload = st.file_uploader(
        "Session video will appear here automatically after stopping — or upload manually",
        type=["mp4","mov","webm"],
        key="live_video_upload",
        label_visibility="visible"
    )

    if live_upload:
        with st.spinner("Running pipeline on live session..."):
            tmp_dir   = Path(tempfile.mkdtemp())
            tmp_video = tmp_dir / "live_session.mp4"
            live_upload.seek(0)
            tmp_video.write_bytes(live_upload.read())
            try:
                result = run_pipeline(tmp_video, exercise, camera_view)
                st.session_state.live_results = result
                st.rerun()
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
            st.session_state.live_results       = None
            st.session_state.live_rep_count     = 0
            st.session_state.live_frames        = []
            st.session_state.live_landmarks     = []
            st.session_state.live_started       = False
            st.session_state.live_annotated_vid = None
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
