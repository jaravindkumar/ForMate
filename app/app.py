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
.logo{font-family:'Archivo Black',sans-serif;font-size:1.5rem;letter-spacing:-.01em;}
.logo-form{color:var(--acid);}.logo-ate{color:var(--txt);}
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
.empty-logo{font-family:'Archivo Black',sans-serif;font-size:clamp(4rem,12vw,10rem);line-height:1;letter-spacing:-.02em;color:rgba(187,255,0,.05);}
.empty-logo b{color:rgba(187,255,0,.08);}
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
    st.markdown('<div class="zone">', unsafe_allow_html=True)
    uz_l, uz_r = st.columns([1, 1], gap="large")

    with uz_l:
        st.markdown(
            '<h2 class="uz-headline">Analyse Your<br><span>Form.</span></h2>'
            '<p class="uz-desc">Upload a workout video and get instant AI-powered pose analysis, rep counting, and personalised coaching.</p>',
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
            tmp_video.write_bytes(uploaded.read())
            st.markdown('<p class="lbl">Preview</p>', unsafe_allow_html=True)
            st.markdown('<div class="vid-preview">', unsafe_allow_html=True)
            st.video(str(tmp_video))
            st.markdown('</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="live-zone">', unsafe_allow_html=True)

    live_l, live_r = st.columns([1, 1], gap="large")

    with live_l:
        st.markdown(
            '<h2 class="live-headline">Live<br><span>Trainer.</span></h2>'
            '<p class="live-desc">Turn on your camera, walk into position, and start your set. FORMate auto-detects your first rep and counts as you go.</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="live-steps">'
            '<div class="live-step"><span class="step-num">01</span>Select your exercise above and hit Start Camera.</div>'
            '<div class="live-step"><span class="step-num">02</span>Walk into position. App auto-detects when you start your first rep.</div>'
            '<div class="live-step"><span class="step-num">03</span>Complete your set. Rep counter updates live after each rep.</div>'
            '<div class="live-step"><span class="step-num">04</span>Hit Stop or walk away — full AI analysis runs automatically.</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # Status badge
        if not st.session_state.live_active and st.session_state.live_results is None:
            st.markdown('<div class="status-badge waiting"><div class="status-dot"></div>Camera Off</div>', unsafe_allow_html=True)
        elif st.session_state.live_active and not st.session_state.live_started:
            st.markdown('<div class="status-badge detecting"><div class="status-dot"></div>Waiting for movement...</div>', unsafe_allow_html=True)
        elif st.session_state.live_active and st.session_state.live_started:
            st.markdown('<div class="status-badge recording"><div class="status-dot"></div>Recording &middot; Rep ' + str(st.session_state.live_rep_count) + '</div>', unsafe_allow_html=True)
        elif st.session_state.live_results:
            st.markdown('<div class="status-badge done"><div class="status-dot"></div>Session Complete</div>', unsafe_allow_html=True)

        # Rep counter
        st.markdown(
            '<div class="rep-counter">'
            '<div class="rep-num">' + str(st.session_state.live_rep_count) + '</div>'
            '<div class="rep-label">Reps Completed</div>'
            '<div class="rep-exercise">' + exercise.upper() + '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # Buttons
        if not st.session_state.live_active:
            if st.button("START CAMERA", type="primary", use_container_width=True, key="start_live"):
                st.session_state.live_active    = True
                st.session_state.live_frames    = []
                st.session_state.live_landmarks = []
                st.session_state.live_rep_count = 0
                st.session_state.live_started   = False
                st.session_state.live_last_move = time.time()
                st.session_state.live_results   = None
                st.rerun()
        else:
            st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
            if st.button("STOP & ANALYSE", use_container_width=True, key="stop_live"):
                st.session_state.live_active = False
                if len(st.session_state.live_frames) > 10:
                    with st.spinner("Processing your session..."):
                        tmp_dir   = Path(tempfile.mkdtemp())
                        vid_path  = tmp_dir / "live_session.mp4"
                        # live_frames already contain annotated BGR frames
                        frames_to_video(st.session_state.live_frames, fps=10, out_path=vid_path)
                        if vid_path.exists():
                            result = run_pipeline(vid_path, exercise, camera_view)
                            if result:
                                # Store annotated video path separately for replay
                                st.session_state.live_annotated_vid = str(vid_path)
                                st.session_state.live_results = result
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    with live_r:
        if st.session_state.live_active:
            st.markdown('<p class="lbl">Live Camera Feed + Skeleton</p>', unsafe_allow_html=True)

            # Hidden camera input — we render our own annotated image
            cam_frame = st.camera_input("", label_visibility="collapsed", key="live_cam")
            frame_display = st.empty()

            if cam_frame is not None:
                raw_frame = decode_frame(cam_frame)

                if raw_frame is not None:
                    # Draw threshold-aware skeleton on frame
                    annotated, lm, issues = draw_skeleton_threshold(raw_frame, exercise)

                    # Show annotated frame (replaces camera feed visually)
                    frame_display.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                        caption=None
                    )

                    if lm:
                        st.session_state.live_landmarks.append(lm)
                        st.session_state.live_frames.append(annotated)  # save annotated frame
                        st.session_state.live_last_move = time.time()

                        history = st.session_state.live_landmarks[-30:]

                        # Auto-detect activity start
                        if not st.session_state.live_started:
                            if detect_activity_start(history, exercise):
                                st.session_state.live_started = True

                        # Count reps once started
                        if st.session_state.live_started and len(history) >= 8:
                            if detect_rep_event(history, exercise):
                                st.session_state.live_rep_count += 1
                                st.session_state.live_landmarks = st.session_state.live_landmarks[-4:]

                        # Show live form status chips
                        if issues:
                            chips = ""
                            for key, (st_val, label) in issues.items():
                                if st_val == "ok":
                                    chips += '<span style="display:inline-flex;align-items:center;gap:.35rem;padding:.25rem .65rem;border-radius:20px;font-size:.68rem;font-weight:600;background:rgba(46,204,113,.1);border:1px solid rgba(46,204,113,.2);color:#7ee8a2;margin:.2rem .2rem 0 0;">&#10003; ' + label + '</span>'
                                elif st_val == "warn":
                                    chips += '<span style="display:inline-flex;align-items:center;gap:.35rem;padding:.25rem .65rem;border-radius:20px;font-size:.68rem;font-weight:600;background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.25);color:#fbbf24;margin:.2rem .2rem 0 0;">&#9651; ' + label + '</span>'
                                else:
                                    chips += '<span style="display:inline-flex;align-items:center;gap:.35rem;padding:.25rem .65rem;border-radius:20px;font-size:.68rem;font-weight:600;background:rgba(255,63,63,.1);border:1px solid rgba(255,63,63,.25);color:#ff8080;margin:.2rem .2rem 0 0;">&#9650; ' + label + '</span>'
                            st.markdown('<div style="margin-top:.5rem;">' + chips + '</div>', unsafe_allow_html=True)

                    # Auto-stop after 10s of no movement
                    last_move = st.session_state.live_last_move
                    if last_move and (time.time() - last_move) > 10 and st.session_state.live_started:
                        st.session_state.live_active = False
                        if len(st.session_state.live_frames) > 10:
                            tmp_dir  = Path(tempfile.mkdtemp())
                            vid_path = tmp_dir / "live_session.mp4"
                            frames_to_video(st.session_state.live_frames, fps=10, out_path=vid_path)
                            if vid_path.exists():
                                result = run_pipeline(vid_path, exercise, camera_view)
                                if result:
                                    st.session_state.live_results = result
                        st.rerun()

        elif not st.session_state.live_results:
            st.markdown(
                '<div class="empty-state" style="padding:3rem 0;">'
                '<div style="font-size:4rem;margin-bottom:.5rem;">📷</div>'
                '<p class="empty-txt">Hit Start Camera to activate your front camera and begin live tracking.</p>'
                '</div>',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

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
