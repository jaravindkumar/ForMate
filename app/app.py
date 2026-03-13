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


/* ── WORKOUT LIBRARY ── */
.prog-header{margin-bottom:1.5rem;}
.prog-title{font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:700;letter-spacing:-.02em;color:var(--txt);margin-bottom:.25rem;}
.prog-sub{font-size:.8rem;color:var(--sub);line-height:1.6;}
.day-strip{display:flex;gap:.5rem;margin-bottom:1.5rem;overflow-x:auto;padding-bottom:.25rem;}
.day-chip{flex-shrink:0;padding:.45rem 1rem;border-radius:20px;font-size:.65rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;cursor:pointer;border:1.5px solid var(--edge2);background:var(--card);color:var(--sub);transition:all .15s;}
.day-chip:hover{border-color:var(--p2);color:var(--p3);}
.day-chip.active{background:linear-gradient(135deg,var(--p1),var(--v1));border-color:transparent;color:#fff;box-shadow:0 4px 14px rgba(29,78,216,.35);}
.day-chip.rest{background:var(--card2);color:var(--muted);cursor:default;}
.cat-label{font-size:.55rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--p2);margin:1.25rem 0 .75rem;}
.ex-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:.85rem;margin-bottom:1.5rem;}
.ex-card{background:var(--card);border:1px solid var(--edge);border-radius:16px;overflow:hidden;cursor:pointer;transition:all .18s;position:relative;}
.ex-card:hover{border-color:var(--p2);transform:translateY(-2px);box-shadow:0 8px 24px rgba(29,78,216,.15);}
.ex-card.supported{border-color:rgba(59,130,246,.3);}
.ex-card.supported::after{content:'AI';position:absolute;top:.5rem;right:.5rem;font-size:.45rem;font-weight:800;letter-spacing:.1em;background:var(--p1);color:#fff;padding:.15rem .4rem;border-radius:6px;}
.ex-pic{width:100%;height:110px;display:flex;align-items:center;justify-content:center;background:var(--card2);}
.ex-pic svg{width:72px;height:72px;}
.ex-info{padding:.6rem .75rem .75rem;}
.ex-name{font-family:'Space Grotesk',sans-serif;font-size:.8rem;font-weight:700;color:var(--txt);margin-bottom:.2rem;}
.ex-meta{font-size:.62rem;color:var(--sub);display:flex;gap:.4rem;flex-wrap:wrap;}
.ex-tag{padding:.1rem .4rem;border-radius:6px;background:var(--card2);border:1px solid var(--edge2);}
.day-focus{display:flex;align-items:center;gap:.75rem;padding:.85rem 1.1rem;border-radius:14px;background:var(--card2);border:1px solid var(--edge);margin-bottom:1.25rem;}
.day-focus-icon{font-size:1.4rem;}
.day-focus-text{font-size:.8rem;font-weight:600;color:var(--txt);}
.day-focus-sub{font-size:.68rem;color:var(--sub);}

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
.ss-v{font-family:'Space Grotesk',sans-serif;font-size:1.7rem;line-height:1;color:#fff;font-weight:700;}
.ss-v.g{color:var(--green);}.ss-v.w{color:var(--red);}
.ss-k{font-size:.60rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#fff;}

/* SCORE BARS */
.bw{margin-bottom:0;}
.brow-wrap{display:grid;grid-template-columns:180px 1fr 52px;align-items:center;gap:1rem;padding:.85rem 1.2rem;border-bottom:1px solid var(--edge2);}
.brow-wrap:last-child{border-bottom:none;}
.bname{font-size:.95rem;font-weight:500;color:#fff;white-space:nowrap;}
.btrack{height:6px;background:var(--edge2);border-radius:6px;overflow:hidden;}
.bfill{height:100%;border-radius:6px;background:linear-gradient(90deg,#1D4ED8,#3B82F6);}
.bfill.mid{background:linear-gradient(90deg,#D97706,#F59E0B);}
.bfill.low{background:linear-gradient(90deg,#DC2626,#EF4444);}
.bval{font-family:'Space Grotesk',sans-serif;font-size:1.05rem;font-weight:700;color:#fff;text-align:right;}
.bval.g{color:#22C55E;}.bval.m{color:#F59E0B;}.bval.r{color:#EF4444;}

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
            opath = Path(ovp) if (ovp and ovp != "None" and ovp != "null") else None
            if opath and opath.exists() and opath.stat().st_size > 1000:
                st.video(str(opath))
            else:
                # Try fallback path
                fb = ROOT / "pipeline" / "bronze" / session_id / "overlay_h264.mp4"
                if not fb.exists():
                    fb = ROOT / "pipeline" / "bronze" / session_id / "overlay.mp4"
                if fb.exists() and fb.stat().st_size > 1000:
                    st.video(str(fb))
                else:
                    st.markdown('<div style="padding:1.5rem;text-align:center;color:var(--sub);font-size:.85rem;">Overlay generating… refresh after analysis.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row1_r:
        st.empty()  # keep column layout balanced

    # ── Score Breakdown full-width ────────────────────────────
    st.markdown('<div class="card" style="padding:0;overflow:hidden;">', unsafe_allow_html=True)
    st.markdown('<p class="st2" style="padding:1rem 1.2rem .5rem;">Score Breakdown</p>', unsafe_allow_html=True)
    for label, key in [("Hinge Quality","hinge_quality"),("Trunk Control","trunk_control"),
                        ("Symmetry","symmetry"),("Tempo Consistency","tempo_consistency"),
                        ("Setup Consistency","setup_consistency")]:
        v  = safe(scores.get(key, 0))
        vi = int(v)
        bc = bcls(v)
        vc = "g" if v >= 75 else ("m" if v >= 50 else "r")
        st.markdown(
            '<div class="brow-wrap">'
            '<span class="bname">' + label + '</span>'
            '<div class="btrack"><div class="bfill ' + bc + '" style="width:' + str(vi) + '%;height:100%;"></div></div>'
            '<span class="bval ' + vc + '">' + str(vi) + '</span>'
            '</div>',
            unsafe_allow_html=True
        )
    flags = g_sum.get("flags", [])
    if flags:
        st.markdown('<div style="padding:.75rem 1.2rem 0;"><p class="st2">Flags</p></div>', unsafe_allow_html=True)
        for f in flags:
            sev = f.get("severity","warn")
            cls = "bad" if sev=="bad" else ("ok" if sev=="ok" else "warn")
            st.markdown('<div style="padding:0 1.2rem .5rem;"><div class="flag ' + cls + '"><div class="fdot"></div>' + f.get("message","") + '</div></div>', unsafe_allow_html=True)
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
            import math as _math
            b_meta        = read_json(ROOT / "pipeline" / "bronze" / session_id / "meta.json")
            snapshots_dir = gold_dir / "snapshots"
            snapshots_dir.mkdir(exist_ok=True)
            vid_for_snap  = b_meta.get("input_video", "")
            if not vid_for_snap or not Path(vid_for_snap).exists():
                vid_for_snap = str(ROOT / "pipeline" / "bronze" / session_id / "input.mp4")

            # Load keypoints for overlay drawing
            kp_path  = ROOT / "pipeline" / "bronze" / session_id / "keypoints.jsonl"
            kp_index = {}  # frame_idx -> landmarks dict
            if kp_path.exists():
                with kp_path.open() as kpf:
                    for line in kpf:
                        line = line.strip()
                        if not line: continue
                        rec = json.loads(line)
                        if rec.get("pose_detected"):
                            kp_index[rec["frame_idx"]] = {lm["id"]: lm for lm in rec["landmarks"]}

            MP_CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),
                       (11,23),(12,24),(23,24),
                       (23,25),(25,27),(24,26),(26,28)]
            BODY_LMS  = list(range(11,17)) + list(range(23,29))
            COL_OK   = (56,189,96);  COL_WARN = (60,165,248);  COL_BAD = (68,68,239)
            COL_BONE = (180,180,180)

            def draw_on(frame, lms_d, exercise_name):
                if not lms_d: return frame
                frame = frame.copy()
                fH, fW = frame.shape[:2]
                def px(i):
                    lm = lms_d.get(i)
                    if lm is None or lm.get("vis",0)<0.15: return None
                    return int(lm["x"]*fW), int(lm["y"]*fH)
                for a,b in MP_CONN:
                    pa,pb = px(a),px(b)
                    if pa and pb: cv2.line(frame,pa,pb,COL_BONE,2,cv2.LINE_AA)
                # Colour joints by form check (inlined)
                def _check_form_inline(lms, ex, W, H):
                    import math as _m
                    if not lms: return {}
                    def pt(i):
                        lm = lms.get(i)
                        return (float(lm["x"]), float(lm["y"]), float(lm.get("vis",0))) if lm else None
                    def ang(ax,ay,bx,by,cx,cy):
                        v1x,v1y=ax-bx,ay-by; v2x,v2y=cx-bx,cy-by
                        dot=v1x*v2x+v1y*v2y
                        mag=(_m.sqrt(v1x**2+v1y**2)+1e-9)*(_m.sqrt(v2x**2+v2y**2)+1e-9)
                        return _m.degrees(_m.acos(max(-1,min(1,dot/mag))))
                    flags = {}
                    def flag(idxs, sev):
                        for i in idxs: flags[i] = sev
                    ls=pt(11);rs=pt(12);lh=pt(23);rh=pt(24)
                    lk=pt(25);rk=pt(26);la=pt(27);ra=pt(28)
                    lw=pt(15);rw=pt(16);le=pt(13);re=pt(14)
                    if ex in ("deadlift","romanian_deadlift","dumbbell_deadlift","bent_over_row","single_arm_row"):
                        if ls and lh and lk and ls[2]>.3 and lh[2]>.3 and lk[2]>.3:
                            a=ang(ls[0],ls[1],lh[0],lh[1],lk[0],lk[1])
                            flag([11,12,23,24],"ok" if a>=120 else ("warn" if a>=100 else "bad"))
                        if lk and rk and la and ra:
                            kw=abs(lk[0]-rk[0]); fw=abs(la[0]-ra[0])
                            if kw < fw*0.7: flag([25,26,27,28],"warn")
                    elif "squat" in ex or ex=="squat":
                        if lk and la and lk[2]>.3 and la[2]>.3:
                            ll=abs(lk[1]-la[1])+1e-9; r=(lk[0]-la[0])/ll
                            flag([25,26,27,28],"ok" if r<0.25 else ("warn" if r<0.45 else "bad"))
                        if lh and lk and lh[2]>.3 and lk[2]>.3:
                            flag([23,24],"ok" if lh[1]>=lk[1] else "warn")
                    elif "press" in ex or "shoulder" in ex:
                        if lw and le and lw[2]>.3 and le[2]>.3:
                            flag([13,14,15,16],"ok" if lw[1]<le[1] else "warn")
                    elif "row" in ex or "curl" in ex:
                        if le and ls and le[2]>.3 and ls[2]>.3:
                            d=abs(le[0]-ls[0])
                            flag([13,14,15,16],"ok" if d<0.15 else ("warn" if d<0.25 else "bad"))
                    return flags
                try: flags = _check_form_inline(lms_d, exercise_name, fW, fH)
                except: flags = {}
                for i in BODY_LMS:
                    p = px(i)
                    if not p: continue
                    sev = flags.get(i,"ok")
                    col = COL_OK if sev=="ok" else (COL_WARN if sev=="warn" else COL_BAD)
                    cv2.circle(frame,p,6,col,-1,cv2.LINE_AA)
                    cv2.circle(frame,p,7,(0,0,0),1,cv2.LINE_AA)
                return frame

            cap = cv2.VideoCapture(str(vid_for_snap), cv2.CAP_FFMPEG)
            total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
            guard_s = int(total_frames * 0.05)
            guard_e = int(total_frames * 0.95)

            for issue in issues:
                saved = 0
                for fi in issue["frames"][:9]:
                    is_sample = issue["type"] == "form_sample"
                    if not is_sample and not (guard_s <= fi <= guard_e):
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(fi)-2))
                        cap.read(); ret, frame = cap.read()
                    if ret and frame is not None and float(frame.mean()) > 6:
                        lms_d = kp_index.get(fi, {})
                        annotated = draw_on(frame, lms_d, exercise)
                        label = issue["type"].upper().replace("_"," ")
                        cv2.rectangle(annotated,(0,0),(annotated.shape[1],28),(0,0,0),-1)
                        cv2.putText(annotated, f"{label}  frame {fi}",
                                    (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                        out_p = snapshots_dir / f"{issue['type']}_{fi}.jpg"
                        cv2.imwrite(str(out_p), annotated)
                        saved += 1
                    if saved >= 3: break
            cap.release()

            snap_col, _ = st.columns([1, 1], gap="medium")
            with snap_col:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="st2">Form Snapshots</p>', unsafe_allow_html=True)
                for issue in issues:
                    label = issue["type"].replace("_"," ").title()
                    saved_snaps = sorted(snapshots_dir.glob(f"{issue['type']}_*.jpg"))[:3]
                    with st.expander(label, expanded=(issue["type"] != "form_sample")):
                        st.caption(issue["description"])
                        if saved_snaps:
                            ic = st.columns(len(saved_snaps))
                            for i, ip in enumerate(saved_snaps):
                                ic[i].image(str(ip), width="stretch")
                        else:
                            st.caption("No frames captured for this issue.")
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as _snap_e:
            import traceback as _tb
            st.warning(f"Snapshots: {_snap_e}")
            st.code(_tb.format_exc())

    rep_col, _ = st.columns([1, 1], gap="medium")
    with rep_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.expander("Per-Rep Breakdown"):
            st.dataframe(rep_df, width="stretch")
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
                write_overlay= True,
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
            gold.run_gold(session_id=session_id, exercise=exercise, root=str(ROOT))
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
    ("u_fid",              None),
    ("u_bytes",            None),
    ("u_name",             None),
    ("u_result",           None),
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
    exercise = st.selectbox("Exercise", [
        "deadlift", "squat",
        "romanian_deadlift", "goblet_squat", "sumo_squat", "bulgarian_split_squat",
        "shoulder_press", "floor_press", "lateral_raise",
        "bent_over_row", "bicep_curl", "single_arm_row",
        "dumbbell_swing", "russian_twist", "renegade_row",
    ], format_func=lambda x: x.replace("_"," ").title(), key="sel_exercise")
with ctrl_r:
    st.markdown('<p class="lbl">Camera Angle</p>', unsafe_allow_html=True)
    camera_view = st.selectbox("Camera", ["front_oblique", "side"], key="sel_camera")

# ─── TABS ─────────────────────────────────────────────────────────
tab_upload, tab_live, tab_body, tab_library = st.tabs(["Upload Video", "Live Trainer", "Body Assessment", "Workout Library"])


# ════════════════════════════════════════════
# TAB 1 — UPLOAD VIDEO
# ════════════════════════════════════════════
with tab_upload:

    st.markdown('<div class="zone">', unsafe_allow_html=True)
    st.markdown(
        '<h2 class="uz-headline">Perfect Your <span>Form.</span></h2>'
        '<p class="uz-desc">Upload a workout video — AI scores every rep and generates a coaching report.</p>',
        unsafe_allow_html=True
    )

    # ── File uploader ─────────────────────────────────────────────
    # type=None — mobile browsers send video/quicktime or application/octet-stream
    # explicit type lists block these silently on iOS/Android
    st.caption("📹 Tap to upload — MP4, MOV, WebM, MKV, AVI supported")
    uploaded = st.file_uploader(
        "Choose a video file",
        type=None,
        key="u_file",
        label_visibility="collapsed"
    )

    # ── Cache bytes immediately while uploader object is live ─────
    if uploaded is not None:
        fid = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("u_fid") != fid:
            uploaded.seek(0)
            raw_bytes = uploaded.read()
            if len(raw_bytes) < 1000:
                st.error(f"❌ File too small ({len(raw_bytes)} bytes) — try again.")
            else:
                st.session_state["u_fid"]    = fid
                st.session_state["u_bytes"]  = raw_bytes
                st.session_state["u_name"]   = uploaded.name
                st.session_state["u_result"] = None

    # ── Read from session_state (survives reruns) ─────────────────
    vid_bytes = st.session_state.get("u_bytes")
    vid_name  = st.session_state.get("u_name", "video.mp4")

    if vid_bytes:
        mb = round(len(vid_bytes) / 1024 / 1024, 1)
        st.success(f"✓ {vid_name}  ({mb} MB) — ready to analyse")

        # Preview — use bytes copy so original in session_state stays intact
        try:
            st.video(bytes(vid_bytes))
        except Exception:
            pass  # preview failure is cosmetic — don't block analysis

        col_btn, col_clr = st.columns([3, 1])
        with col_btn:
            analyse = st.button("🔬 ANALYSE FORM", type="primary",
                                width="stretch", key="u_analyse")
        with col_clr:
            if st.button("✕ Clear", key="u_clear"):
                for k in ["u_fid", "u_bytes", "u_name", "u_result"]:
                    st.session_state.pop(k, None)
                st.rerun()

        if analyse:
            # ── Detect extension from magic bytes (most reliable) ─
            VIDEO_EXTS = {".mp4",".mov",".m4v",".webm",".mkv",".avi",
                          ".3gp",".ts",".flv",".wmv",".mpeg",".mpg",".mts",".m2ts"}
            raw_suffix = Path(vid_name).suffix.lower()
            header = vid_bytes[:16]
            if header[4:8] in (b"ftyp", b"moov", b"mdat"):
                ext = ".mp4"
            elif header[:4] == b"\x1a\x45\xdf\xa3":
                ext = ".webm"
            elif header[:3] == b"FLV":
                ext = ".flv"
            elif header[:4] == b"RIFF":
                ext = ".avi"
            elif raw_suffix in VIDEO_EXTS:
                ext = raw_suffix
            else:
                ext = ".mp4"  # ffmpeg handles anything

            # ── Write to unique temp path (avoids concurrent-user collisions) ─
            uid  = uuid.uuid4().hex[:8]
            tp   = Path(tempfile.gettempdir()) / f"formate_{uid}{ext}"
            try:
                tp.write_bytes(vid_bytes)
            except Exception as we:
                st.error(f"❌ Could not write temp file: {we}")
                st.stop()

            if tp.stat().st_size < 1000:
                st.error("❌ Temp file write failed — zero bytes on disk.")
                st.stop()

            # ── Run pipeline ──────────────────────────────────────
            res = run_pipeline(str(tp), exercise, camera_view)

            # Clean up temp file
            try: tp.unlink()
            except Exception: pass

            if res:
                st.session_state["u_result"] = res
                st.rerun()
            else:
                st.error("Pipeline returned no result — see errors above ↑")

    elif uploaded is None and not st.session_state.get("u_bytes"):
        # Nothing uploaded yet — show empty state
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-logo"><b>🎥</b></div>'
            '<p style="color:var(--sub);font-size:.85rem;">Upload a video to get started</p>'
            '</div>', unsafe_allow_html=True
        )

    # ── Render results (persists across reruns via session_state) ─
    if st.session_state.get("u_result"):
        sid, b_sum, g_sum, rep_df, num_reps, gold_dir = st.session_state["u_result"]
        render_results(sid, gold_dir, b_sum, g_sum, rep_df, num_reps, exercise)

    st.markdown('</div>', unsafe_allow_html=True)

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

/* Video uses contain — show full frame, no cropping */
/* Front cam gets CSS mirror flip */
video{
  position:absolute;top:0;left:0;
  width:100%;height:100%;
  object-fit:cover;
  pointer-events:none;}
/* portrait-fix — dimensions set by JS, transform set here */
video.portrait-fix{
  transform:translate(-50%,-50%) rotate(90deg);
  object-fit:cover;}
video.portrait-fix.mirror{
  transform:translate(-50%,-50%) rotate(90deg) scaleX(-1);}
video.normal{top:0;left:0;width:100%;height:100%;object-fit:cover;}
video.normal.mirror{transform:scaleX(-1);}
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
    <div id="shake-hint" style="display:none;font-size:.62rem;color:rgba(255,255,255,.35);
         text-align:center;padding:.2rem 0;letter-spacing:.05em">
      👈 SHAKE HEAD LEFT-RIGHT TO STOP 👉
    </div>
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

  // ── DEADLIFT ─────────────────────────────────────────────────
  if(EXERCISE==="deadlift"||EXERCISE==="romanian_deadlift"||EXERCISE==="dumbbell_deadlift"){
    // Back angle: shoulder(5)-hip(11)-knee(13)
    if(ok(5,11,13)){
      const a=angleDeg(xn(5),yn(5),xn(11),yn(11),xn(13),yn(13));
      f.back=a>=145?{st:"ok",lbl:"Back OK"}:a>=115?{st:"warn",lbl:"Back rounding"}:{st:"bad",lbl:"Back round!"};
    }
    // Bar drift: shoulder vs hip horizontal distance (side view)
    if(ok(5,11)){
      const d=Math.abs(xn(5)-xn(11));
      f.drift=d<.08?{st:"ok",lbl:"Bar OK"}:d<.16?{st:"warn",lbl:"Bar drifting"}:{st:"bad",lbl:"Bar too far!"};
    }
    // Hinge depth
    if(ok(11,13)){
      const d=yn(13)-yn(11);
      f.hinge=d>.04?{st:"ok",lbl:"Hinge OK"}:{st:"warn",lbl:"Hinge deeper"};
    }

  // ── SQUAT / GOBLET SQUAT / SUMO SQUAT ────────────────────────
  }else if(EXERCISE==="squat"||EXERCISE==="goblet_squat"||EXERCISE==="sumo_squat"){
    // Knee cave
    if(ok(11,12,13,14)){
      const hipW=Math.abs(xn(12)-xn(11));
      const kneeW=Math.abs(xn(14)-xn(13));
      const ratio=hipW>0.01?kneeW/hipW:1;
      // Sumo squat naturally has wider stance — more lenient threshold
      const warnT=EXERCISE==="sumo_squat"?0.6:0.75;
      const badT =EXERCISE==="sumo_squat"?0.4:0.55;
      f.knee=ratio>=warnT?{st:"ok",lbl:"Knees OK"}:ratio>=badT?{st:"warn",lbl:"Knee caving"}:{st:"bad",lbl:"Knee cave!"};
    }
    // Squat depth
    if(ok(11,13)){
      const gap=yn(13)-yn(11);
      f.depth=gap<=.08?{st:"ok",lbl:"Good depth"}:gap<=.16?{st:"warn",lbl:"Go deeper"}:{st:"bad",lbl:"Too shallow"};
    }
    // Forward lean — more lenient for goblet (counterweight helps stay upright)
    if(ok(5,11)){
      const dx=Math.abs(xn(5)-xn(11));
      const dy=Math.abs(yn(5)-yn(11));
      const lean=dy>0.01?dx/dy:0;
      const warnT=EXERCISE==="goblet_squat"?0.35:0.25;
      f.lean=lean<warnT?{st:"ok",lbl:"Chest up OK"}:lean<.5?{st:"warn",lbl:"Leaning fwd"}:{st:"bad",lbl:"Too much lean!"};
    }

  // ── BULGARIAN SPLIT SQUAT ────────────────────────────────────
  }else if(EXERCISE==="bulgarian_split_squat"){
    // Front shin angle (knee tracking over toes)
    if(ok(13,15)){
      const dx=Math.abs(xn(13)-xn(15));
      const dy=Math.abs(yn(13)-yn(15));
      const shin=dy>0.01?dx/dy:0;
      f.shin=shin<.3?{st:"ok",lbl:"Shin OK"}:shin<.55?{st:"warn",lbl:"Knee too far fwd"}:{st:"bad",lbl:"Knee over toes!"};
    }
    // Torso upright
    if(ok(5,11)){
      const dx=Math.abs(xn(5)-xn(11));
      const dy=Math.abs(yn(5)-yn(11));
      const lean=dy>0.01?dx/dy:0;
      f.torso=lean<.3?{st:"ok",lbl:"Upright OK"}:lean<.5?{st:"warn",lbl:"Leaning fwd"}:{st:"bad",lbl:"Stay upright!"};
    }
    // Hip drop (hip level)
    if(ok(11,12)){
      const hipDiff=Math.abs(yn(11)-yn(12));
      f.hip=hipDiff<.06?{st:"ok",lbl:"Hips level"}:hipDiff<.12?{st:"warn",lbl:"Hip dropping"}:{st:"bad",lbl:"Level hips!"};
    }

  // ── SHOULDER PRESS ────────────────────────────────────────────
  }else if(EXERCISE==="shoulder_press"){
    // Elbow flare: wrists (15,16) should be above elbows (13,14) at lockout
    if(ok(13,14,15,16)){
      const lElbowAbove=yn(13)>yn(15); // elbow higher than wrist = bad
      const rElbowAbove=yn(14)>yn(16);
      f.lockout=(!lElbowAbove&&!rElbowAbove)?{st:"ok",lbl:"Full lockout"}:
                (lElbowAbove||rElbowAbove)?{st:"warn",lbl:"Extend fully"}:{st:"bad",lbl:"No lockout!"};
    }
    // Rib flare / back arch: excessive lean back
    if(ok(5,6,11,12)){
      const shoulderMidX=(xn(5)+xn(6))/2;
      const hipMidX=(xn(11)+xn(12))/2;
      const lean=Math.abs(shoulderMidX-hipMidX);
      f.arch=lean<.08?{st:"ok",lbl:"Core tight"}:lean<.15?{st:"warn",lbl:"Watch rib flare"}:{st:"bad",lbl:"Arching back!"};
    }
    // Symmetry: both elbows at similar height
    if(ok(13,14)){
      const diff=Math.abs(yn(13)-yn(14));
      f.sym=diff<.06?{st:"ok",lbl:"Even press"}:{st:"warn",lbl:"Uneven press"};
    }

  // ── FLOOR PRESS ───────────────────────────────────────────────
  }else if(EXERCISE==="floor_press"){
    // Elbow angle at bottom (elbows should be ~45° from torso)
    if(ok(5,7,9)){
      const a=angleDeg(xn(5),yn(5),xn(7),yn(7),xn(9),yn(9));
      f.elbow=a>=130?{st:"ok",lbl:"Elbow OK"}:a>=100?{st:"warn",lbl:"Open elbows more"}:{st:"bad",lbl:"Elbows too tight!"};
    }
    // Wrist over elbow at lockout
    if(ok(7,9)){
      const diff=Math.abs(xn(7)-xn(9));
      f.lockout=diff<.06?{st:"ok",lbl:"Lockout OK"}:{st:"warn",lbl:"Full extension"};
    }

  // ── LATERAL RAISE ─────────────────────────────────────────────
  }else if(EXERCISE==="lateral_raise"){
    // Arms at shoulder height (wrist Y ~ shoulder Y at top)
    if(ok(5,6,9,10)){
      const shoulderY=(yn(5)+yn(6))/2;
      const wristY=(yn(9)+yn(10))/2;
      const diff=wristY-shoulderY; // negative = wrists above shoulders
      f.height=diff<=.05?{st:"ok",lbl:"Good height"}:diff<=.15?{st:"warn",lbl:"Raise higher"}:{st:"bad",lbl:"Too low!"};
    }
    // Symmetry
    if(ok(9,10)){
      const diff=Math.abs(yn(9)-yn(10));
      f.sym=diff<.06?{st:"ok",lbl:"Even raise"}:{st:"warn",lbl:"Uneven arms"};
    }
    // Elbow slight bend (elbow should not be fully straight = locked)
    if(ok(5,7,9)){
      const a=angleDeg(xn(5),yn(5),xn(7),yn(7),xn(9),yn(9));
      f.bend=a<170?{st:"ok",lbl:"Soft elbow OK"}:{st:"warn",lbl:"Soften elbows"};
    }

  // ── BENT-OVER ROW ─────────────────────────────────────────────
  }else if(EXERCISE==="bent_over_row"||EXERCISE==="single_arm_row"){
    // Back flatness: shoulder-hip-knee angle
    if(ok(5,11,13)){
      const a=angleDeg(xn(5),yn(5),xn(11),yn(11),xn(13),yn(13));
      f.back=a>=140?{st:"ok",lbl:"Flat back"}:a>=115?{st:"warn",lbl:"Back rounding"}:{st:"bad",lbl:"Back round!"};
    }
    // Row height: elbow should travel above torso at top
    if(ok(5,7)){
      const elbowAboveShoulder=yn(7)<yn(5);
      f.row=elbowAboveShoulder?{st:"ok",lbl:"Full row"}:{st:"warn",lbl:"Pull higher"};
    }
    // Hip hinge depth (torso should be ~parallel to floor)
    if(ok(5,11)){
      const dx=Math.abs(xn(5)-xn(11));
      const dy=Math.abs(yn(5)-yn(11));
      const angle=dy>0.01?dx/dy:0;
      f.hinge=angle>.8?{st:"ok",lbl:"Good hinge"}:angle>.5?{st:"warn",lbl:"Hinge more"}:{st:"bad",lbl:"Hinge deeper!"};
    }

  // ── BICEP CURL ────────────────────────────────────────────────
  }else if(EXERCISE==="bicep_curl"){
    // Elbow flare: upper arm should stay close to torso
    // Elbow X should be close to hip X (not flaring out)
    if(ok(5,7,11)){
      const elbowFlare=Math.abs(xn(7)-xn(11));
      f.flare=elbowFlare<.12?{st:"ok",lbl:"Elbows pinned"}:elbowFlare<.2?{st:"warn",lbl:"Tuck elbows"}:{st:"bad",lbl:"Elbow flare!"};
    }
    // Full extension at bottom: arm angle should be near straight
    if(ok(5,7,9)){
      const a=angleDeg(xn(5),yn(5),xn(7),yn(7),xn(9),yn(9));
      f.extend=a>=160?{st:"ok",lbl:"Full extension"}:a>=140?{st:"warn",lbl:"Lower fully"}:{st:"bad",lbl:"Extend fully!"};
    }

  // ── DUMBBELL SWING ────────────────────────────────────────────
  }else if(EXERCISE==="dumbbell_swing"){
    // Hip hinge dominance: hip angle
    if(ok(5,11,13)){
      const a=angleDeg(xn(5),yn(5),xn(11),yn(11),xn(13),yn(13));
      f.hinge=a<=140?{st:"ok",lbl:"Hip hinge OK"}:a<=160?{st:"warn",lbl:"Hinge more"}:{st:"bad",lbl:"Not hinging!"};
    }
    // Spine neutral at bottom
    if(ok(5,11)){
      const dx=Math.abs(xn(5)-xn(11));
      const dy=Math.abs(yn(5)-yn(11));
      const lean=dy>0.01?dx/dy:0;
      f.spine=lean<.5?{st:"ok",lbl:"Spine OK"}:{st:"warn",lbl:"Neutral spine"};
    }

  // ── RUSSIAN TWIST ─────────────────────────────────────────────
  }else if(EXERCISE==="russian_twist"){
    // Lean back angle (should be ~45°)
    if(ok(5,11)){
      const dx=Math.abs(xn(5)-xn(11));
      const dy=Math.abs(yn(5)-yn(11));
      const lean=dy>0.01?dx/dy:0;
      f.lean=lean>.2&&lean<.7?{st:"ok",lbl:"Good lean"}:lean<=.2?{st:"warn",lbl:"Lean back more"}:{st:"bad",lbl:"Too far back!"};
    }
    // Rotation: shoulders should rotate (shoulder width projection changes)
    if(ok(5,6)){
      const shoulderW=Math.abs(xn(5)-xn(6));
      f.rotation=shoulderW<.15?{st:"ok",lbl:"Rotating OK"}:{st:"warn",lbl:"Rotate more"};
    }

  // ── RENEGADE ROW ──────────────────────────────────────────────
  }else if(EXERCISE==="renegade_row"){
    // Hip rotation (must stay flat — hips shouldn't rotate during row)
    if(ok(11,12)){
      const hipDiff=Math.abs(yn(11)-yn(12));
      f.hip=hipDiff<.05?{st:"ok",lbl:"Hips flat"}:hipDiff<.1?{st:"warn",lbl:"Control hips"}:{st:"bad",lbl:"Hips rotating!"};
    }
    // Plank position: shoulder-hip-ankle alignment
    if(ok(5,11)){
      const dx=Math.abs(xn(5)-xn(11));
      const dy=Math.abs(yn(5)-yn(11));
      const sag=dy>0.01?dx/dy:0;
      f.plank=sag<.15?{st:"ok",lbl:"Plank solid"}:sag<.3?{st:"warn",lbl:"Hips sagging"}:{st:"bad",lbl:"Hips too low!"};
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
  // Deadlift / RDL
  "Knee cave!":      "Knees out",
  "Knee caving":     "Watch your knees",
  "Too shallow":     "Go deeper",
  "Go deeper":       "Deeper",
  "Too much lean!":  "Stay upright",
  "Leaning fwd":     "Chest up",
  "Bar too far!":    "Bar close to body",
  "Bar drifting":    "Keep bar close",
  "Back round!":     "Neutral spine",
  "Back rounding":   "Brace your back",
  "Hinge deeper":    "Push hips back",
  "Hinge deeper!":   "Drive hips back",
  // Shoulder press / floor press
  "No lockout!":     "Lock it out",
  "Extend fully":    "Full extension",
  "Arching back!":   "Tighten your core",
  "Watch rib flare": "Ribs down",
  "Uneven press":    "Press evenly",
  "Elbows too tight!":"Open your elbows",
  // Lateral raise
  "Too low!":        "Raise to shoulder height",
  "Raise higher":    "Higher",
  "Uneven arms":     "Even it out",
  "Soften elbows":   "Soft bend in elbows",
  // Row
  "Pull higher":     "Elbow past your torso",
  "Hinge more":      "Hinge at the hips",
  "Hinge deeper!":   "Parallel to the floor",
  // Bicep curl
  "Elbow flare!":    "Pin your elbows",
  "Tuck elbows":     "Keep elbows in",
  "Extend fully!":   "Lower all the way down",
  // Split squat
  "Knee over toes!": "Shin vertical",
  "Hip dropping":    "Level your hips",
  "Level hips!":     "Square your hips",
  // Swing
  "Not hinging!":    "Hip hinge, not a squat",
  // Renegade row
  "Hips rotating!":  "Keep hips still",
  "Hips sagging":    "Squeeze your core",
  "Hips too low!":   "Raise your hips",
  // Russian twist
  "Lean back more":  "Lean back forty five degrees",
  "Too far back!":   "Sit up slightly",
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

// ── Head-shake stop detection ─────────────────────────────────────
// Simple approach: track running nose X, detect crossings of a centre
// line in alternating directions. 3 crossings = 1.5 shakes = done.
//
// "Crossing" = nose X moves from one side of the face-centre to other
// by at least SHAKE_MIN normalised width. At 15fps this is very stable.

let shakeNoseHist=[];   // {x, t} ring buffer, normalised 0-1
let shakeSide=null;     // last confirmed side: "L" or "R"
let shakeCrossings=0;   // alternating L→R or R→L crossings
let shakeWindowStart=0; // timestamp of first crossing in current sequence
const SHAKE_MIN=0.08;   // nose must move 8% of frame width per direction
const SHAKE_WINDOW=2500;// all crossings must happen within 2.5s
const SHAKE_CROSSES=3;  // 3 crossings = done (L→R→L or R→L→R)

// Show visual countdown when shake is in progress
let shakeProgress=0;

function checkHeadShake(kp, vW){
  const nose=kp[MV.NOSE];
  if(!nose || nose.score<0.35) return false;

  const nx = nose.x / vW;         // normalise to 0–1
  const now = Date.now();

  shakeNoseHist.push({x:nx, t:now});
  if(shakeNoseHist.length > 45) shakeNoseHist.shift(); // 3s window

  // Need at least 10 frames to compute a stable mean
  if(shakeNoseHist.length < 10) return false;

  // Face centre = mean of recent nose positions
  const centre = shakeNoseHist.reduce((s,p)=>s+p.x,0) / shakeNoseHist.length;

  // Current side of centre
  const side = nx < centre - SHAKE_MIN ? "L" :
               nx > centre + SHAKE_MIN ? "R" : null;

  if(side && side !== shakeSide){
    // Crossed to the other side
    if(shakeSide === null){
      // First detection — start sequence
      shakeSide = side;
      shakeCrossings = 1;
      shakeWindowStart = now;
    } else if(now - shakeWindowStart < SHAKE_WINDOW){
      shakeSide = side;
      shakeCrossings++;
      shakeProgress = shakeCrossings / SHAKE_CROSSES;
      if(shakeCrossings >= SHAKE_CROSSES){
        // CONFIRMED — reset and fire
        shakeNoseHist=[]; shakeSide=null; shakeCrossings=0; shakeProgress=0;
        return true;
      }
    } else {
      // Timed out — restart
      shakeSide = side;
      shakeCrossings = 1;
      shakeWindowStart = now;
      shakeProgress = 1/SHAKE_CROSSES;
    }
  }

  // Draw progress arc on canvas as visual feedback
  _drawShakeProgress(shakeProgress);
  return false;
}

function _drawShakeProgress(pct){
  if(pct <= 0) return;
  const canvas=document.getElementById("overlay");
  if(!canvas) return;
  const ctx=canvas.getContext("2d");
  const cx=canvas.width/2, cy=40, r=18;
  ctx.save();
  ctx.strokeStyle="rgba(255,255,255,0.25)";
  ctx.lineWidth=3;
  ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.stroke();
  ctx.strokeStyle="#60A5FA";
  ctx.lineWidth=3;
  ctx.beginPath();
  ctx.arc(cx,cy,r,-Math.PI/2,-Math.PI/2+Math.PI*2*pct);
  ctx.stroke();
  ctx.fillStyle="rgba(255,255,255,0.7)";
  ctx.font="bold 9px Inter,system-ui";
  ctx.textAlign="center";
  ctx.fillText("STOP",cx,cy+3);
  ctx.textAlign="left";
  ctx.restore();
}

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
      shakeNoseHist=[];shakeSide=null;shakeCrossings=0;shakeProgress=0;shakeWindowStart=0;
      document.getElementById("rep-num").textContent="0";
      document.getElementById("rep-num").style.color="#60A5FA";
      setStatus("active");
      speak("Go! Shake your head to stop.");
      hideGestureOverlay();
      // Start recording CLEAN video (no overlay) from the camera stream
      recordedChunks=[];
      try{
        const vid=document.getElementById("video");
        const rawStream=vid.srcObject;
        // Prefer VP8 — VP9 causes "Not all references available" errors on mobile webm
        // VP8 produces clean, seekable webm that ffmpeg can process reliably
        const mimePrefs=[
          "video/webm;codecs=vp8",
          "video/webm;codecs=vp8,opus",
          "video/mp4;codecs=avc1",
          "video/webm"
        ];
        const mimeType=mimePrefs.find(m=>MediaRecorder.isTypeSupported(m))||"video/webm";
        console.log("[FORMate] Recording codec:", mimeType);
        mediaRecorder=new MediaRecorder(rawStream,
          {mimeType, videoBitsPerSecond:2500000});
        mediaRecorder.ondataavailable=e=>{if(e.data.size>0)recordedChunks.push(e.data);};
        mediaRecorder.start(1000); // 1s chunks — larger chunks = more complete frames
        console.log("[FORMate] Recording started (clean video)");
      }catch(err){
        console.warn("[FORMate] MediaRecorder failed:",err);
        mediaRecorder=null;
      }
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
  const sh=document.getElementById("shake-hint");
  if(sh) sh.style.display=(s==="active")?"block":"none";
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
  // Account for rotation: if landscape stream rotated to portrait, swap dims
  const _vW=video.videoWidth||720, _vH=video.videoHeight||1280;
  const _rotated = video.className.includes("portrait-fix");
  const vW = _rotated ? _vH : _vW;
  const vH = _rotated ? _vW : _vH;

  const cW=canvas.offsetWidth, cH=canvas.offsetHeight;
  const scale=Math.max(cW/vW, cH/vH);
  const rW=vW*scale, rH=vH*scale;
  const offX=(cW-rW)/2, offY=(cH-rH)/2;

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

        // ── Head shake = stop session ─────────────────────────────
        if(checkHeadShake(kp, vW)){
          stopSession();
          return;
        }
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
  // Just request the camera — don't fight the browser with size hints
  try {
    return await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: facing } }
    });
  } catch(e) {
    return await navigator.mediaDevices.getUserMedia({ audio:false, video:true });
  }
}


function applyVideoOrientation(video){
  const vW = video.videoWidth;
  const vH = video.videoHeight;
  const needsRotate = vW > vH;
  const isMirror = facingMode === "user";
  const container = document.getElementById("cam-container");
  const cW = container.offsetWidth;
  const cH = container.offsetHeight;

  if(needsRotate){
    // Stream is landscape (e.g. 1280x720), phone is portrait.
    // Rotate 90deg. After rotation the video's rendered w/h are swapped.
    // To fill the portrait container: set video width=cH, height=cW
    // so after rotate(90deg) it fills width=cW, height=cH correctly.
    video.style.position = "absolute";
    video.style.width    = cH + "px";
    video.style.height   = cW + "px";
    video.style.top      = "50%";
    video.style.left     = "50%";
    video.style.objectFit = "cover";
    video.className = isMirror ? "portrait-fix mirror" : "portrait-fix";
  } else {
    video.style.position = "absolute";
    video.style.width    = "100%";
    video.style.height   = "100%";
    video.style.top      = "0";
    video.style.left     = "0";
    video.style.objectFit = "cover";
    video.className = isMirror ? "normal mirror" : "normal";
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
      await new Promise(r=>video.onloadedmetadata=r);
      video.play();
      applyVideoOrientation(video);
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
      // MediaRecorder started in startCountdown() when workout actually begins
      // (not here — avoids recording the gesture/countdown phase)
      mediaRecorder=null; recordedChunks=[];
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
    stopSession();
  }
}

async function stopSession(){
  if(!running) return;
  running=false;
  cancelAnimationFrame(rafId);
  if(stream) stream.getTracks().forEach(t=>t.stop());
  if(wakeLock){try{await wakeLock.release();}catch(e){}wakeLock=null;}

  repActive=false;gestureHoldCount=0;countdownActive=false;
  shakeNoseHist=[];shakeSide=null;shakeCrossings=0;shakeProgress=0;
  hideGestureOverlay();

  const btn=document.getElementById("btn-main");
  document.getElementById("cam-off").style.display="flex";
  document.getElementById("fps-badge").style.display="none";
  document.getElementById("btn-flip").style.display="none";
  btn.textContent="START CAMERA"; btn.className="start";
  setStatus("done");

  // ── "Workout ended" announcement ─────────────────────────────
  window.speechSynthesis.cancel();
  clearTimeout(voiceQueue);
  lastFlagState={};flagHoldCount={};lastRepSpoken=0;
  const msg = repCount > 0
    ? "Workout ended. " + repCount + " reps completed. Great work!"
    : "Workout ended.";
  setTimeout(()=>speak(msg), 300);

  // ── Show "WORKOUT ENDED" banner on overlay ────────────────────
  const canvas=document.getElementById("overlay");
  const ctx=canvas.getContext("2d");
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle="rgba(0,0,0,0.55)";
  ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle="#60A5FA";
  ctx.font="bold 28px Inter,system-ui";
  ctx.textAlign="center";
  ctx.fillText("WORKOUT ENDED", canvas.width/2, canvas.height/2 - 20);
  ctx.fillStyle="#F0EEF8";
  ctx.font="18px Inter,system-ui";
  ctx.fillText(repCount + " reps completed", canvas.width/2, canvas.height/2 + 16);
  ctx.textAlign="left";

  // ── Stop recording and trigger download ──────────────────────
  if(mediaRecorder && mediaRecorder.state!=="inactive"){
    mediaRecorder.onstop=()=>{
      if(recordedChunks.length===0) return;
      const blob=new Blob(recordedChunks,{type:"video/webm"});
      const url=URL.createObjectURL(blob);
      const a=document.createElement("a");
      a.href=url;
      a.download="formate_live_session_"+repCount+"reps.webm";
      document.body.appendChild(a);a.click();
      document.body.removeChild(a);
      setTimeout(()=>URL.revokeObjectURL(url),2000);
      document.getElementById("btn-save").style.display="inline-block";
    };
    mediaRecorder.stop();
  }
}

async function flipCamera(){
  if(!running)return;
  facingMode=facingMode==="environment"?"user":"environment";
  if(stream)stream.getTracks().forEach(t=>t.stop());
  const video=document.getElementById("video");
  stream=await getCameraStream(facingMode);
  video.srcObject=stream;
  await new Promise(r=>video.onloadedmetadata=r);
  video.play();
  applyVideoOrientation(video);
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
            type=None,  # Accept all — ffmpeg handles conversion
            key="live_video_upload",
            label_visibility="visible"
        )

        if live_upload and not st.session_state.get("live_processing_done"):
            import re as _re
            m = _re.search(r'_(\d+)reps', live_upload.name)
            js_rep_count = int(m.group(1)) if m else None

            # Preserve original extension — webm must stay webm for opencv
            ext = Path(live_upload.name).suffix or ".webm"

            with st.spinner("Running pipeline on live session..."):
                # Write to stable /tmp path — not mkdtemp (new dir each rerun)
                tmp_video = Path(tempfile.gettempdir()) / f"formate_live{ext}"
                live_upload.seek(0)
                tmp_video.write_bytes(live_upload.read())
                try:
                    result = run_pipeline(tmp_video, exercise, camera_view)
                    if result is None:
                        st.error("Pipeline failed — check the error above. You can still re-upload the video.")
                    else:
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
                    import traceback
                    st.code(traceback.format_exc())

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


# ════════════════════════════════════════════
# ════════════════════════════════════════════
# TAB 3 — BODY ASSESSMENT
# ════════════════════════════════════════════
with tab_body:

    # ── CSS for this tab ─────────────────────────────────────────
    st.markdown("""<style>
.ba-zone{background:var(--card);border:1px solid var(--edge);border-radius:20px;padding:2rem 2.25rem;margin-bottom:1.5rem;}
.ba-headline{font-family:'Space Grotesk',sans-serif;font-size:clamp(1.6rem,3.5vw,2.4rem);font-weight:700;letter-spacing:-.03em;line-height:1.15;margin-bottom:.4rem;}
.ba-headline span{background:linear-gradient(135deg,var(--p2),var(--green));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.ba-desc{color:var(--sub);font-size:.88rem;line-height:1.7;margin-bottom:1.5rem;}
.ba-section{font-family:'Space Grotesk',sans-serif;font-size:.65rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--sub);margin:1.5rem 0 .75rem;}
.ba-metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:.75rem;margin-bottom:1.25rem;}
.ba-metric{background:var(--card2);border:1px solid var(--edge);border-radius:14px;padding:1rem 1.1rem;}
.ba-metric-label{font-size:.65rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--sub);margin-bottom:.3rem;}
.ba-metric-value{font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;line-height:1;color:var(--txt);}
.ba-metric-unit{font-size:.72rem;color:var(--sub);margin-top:.2rem;}
.ba-metric-sub{font-size:.68rem;color:var(--sub2);margin-top:.15rem;}
.ba-metric.good .ba-metric-value{color:#34D399;}
.ba-metric.warn .ba-metric-value{color:#FBBF24;}
.ba-metric.bad  .ba-metric-value{color:var(--red);}
.ba-metric.blue .ba-metric-value{color:var(--p2);}
.ba-bar-row{display:flex;align-items:center;gap:.75rem;margin-bottom:.6rem;}
.ba-bar-label{font-size:.75rem;color:var(--sub);width:130px;flex-shrink:0;}
.ba-bar-track{flex:1;background:var(--edge);border-radius:4px;height:7px;overflow:hidden;}
.ba-bar-fill{height:100%;border-radius:4px;transition:width .6s ease;}
.ba-bar-val{font-size:.75rem;font-weight:600;color:var(--txt);width:38px;text-align:right;flex-shrink:0;}
.ba-flag{display:flex;align-items:flex-start;gap:.75rem;background:var(--card2);border:1px solid var(--edge);border-radius:12px;padding:.85rem 1rem;margin-bottom:.6rem;}
.ba-flag-icon{font-size:1.1rem;flex-shrink:0;margin-top:.05rem;}
.ba-flag-title{font-size:.8rem;font-weight:600;color:var(--txt);margin-bottom:.15rem;}
.ba-flag-body{font-size:.73rem;color:var(--sub);line-height:1.55;}
.ba-flag.ok{border-color:rgba(52,211,153,.2);}
.ba-flag.ok .ba-flag-title{color:#34D399;}
.ba-flag.warn{border-color:rgba(251,191,36,.2);}
.ba-flag.warn .ba-flag-title{color:#FBBF24;}
.ba-flag.bad{border-color:rgba(239,68,68,.2);}
.ba-flag.bad .ba-flag-title{color:var(--red);}
.ba-report-card{background:linear-gradient(135deg,var(--card2),var(--card));border:1px solid var(--edge2);border-radius:16px;padding:1.5rem;margin-top:1rem;}
.ba-report-title{font-family:'Space Grotesk',sans-serif;font-size:.95rem;font-weight:700;color:var(--p2);margin-bottom:.75rem;display:flex;align-items:center;gap:.5rem;}
.ba-report-body{font-size:.82rem;color:var(--sub);line-height:1.75;white-space:pre-wrap;}
.ba-upload-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.25rem;}
.ba-photo-label{font-size:.75rem;font-weight:600;color:var(--sub);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;}
.ba-disclaimer{font-size:.67rem;color:var(--sub2);background:var(--card2);border:1px solid var(--edge);border-radius:10px;padding:.6rem .8rem;line-height:1.6;margin-top:1rem;}
.ba-somatotype{display:inline-block;background:linear-gradient(135deg,var(--p1),var(--p2));color:white;font-family:'Space Grotesk',sans-serif;font-size:.75rem;font-weight:700;letter-spacing:.06em;padding:.3rem .75rem;border-radius:20px;margin-top:.3rem;}
@media(max-width:600px){.ba-upload-grid{grid-template-columns:1fr;}.ba-metric-grid{grid-template-columns:repeat(2,1fr);}}
</style>""", unsafe_allow_html=True)

    # ── Imports needed ────────────────────────────────────────────
    import io
    from PIL import Image as PILImage

    # ── Body Assessment helper functions ─────────────────────────

    def ba_get_landmarks(img_array):
        """Run MediaPipe Pose on a single image, return normalised landmarks or None."""
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.4
            ) as pose:
                rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    return {i: {"x": lm.x, "y": lm.y, "z": lm.z, "v": lm.visibility}
                            for i, lm in enumerate(lms)}
        except Exception as e:
            st.warning(f"Landmark detection issue: {e}")
        return None

    def ba_pixel_to_cm(img_h_px, height_cm):
        """Return cm-per-pixel scale factor using known height."""
        # Assume person occupies ~85% of image height
        body_px = img_h_px * 0.85
        return height_cm / body_px

    def ba_angle_3pts(a, b, c):
        """Angle at point b formed by a-b-c (degrees)."""
        v1 = np.array([a[0]-b[0], a[1]-b[1]])
        v2 = np.array([c[0]-b[0], c[1]-b[1]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6: return 0.0
        cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        return float(np.degrees(np.arccos(cos)))

    def ba_landmark_px(lms, idx, W, H):
        """Convert normalised landmark to pixel coords."""
        if idx not in lms: return None
        lm = lms[idx]
        if lm["v"] < 0.3: return None
        return (lm["x"] * W, lm["y"] * H)

    def ba_compute_body_fat(waist_cm, neck_cm, height_cm, sex, hip_cm=None):
        """U.S. Navy body fat formula."""
        try:
            if sex == "Male":
                bf = 495 / (1.0324 - 0.19077 * math.log10(waist_cm - neck_cm) + 0.15456 * math.log10(height_cm)) - 450
            else:
                if hip_cm is None: hip_cm = waist_cm * 1.05
                bf = 495 / (1.29579 - 0.35004 * math.log10(waist_cm + hip_cm - neck_cm) + 0.22100 * math.log10(height_cm)) - 450
            return max(3.0, min(50.0, round(bf, 1)))
        except:
            return None

    def ba_compute_ffmi(lean_kg, height_m):
        """Fat-Free Mass Index."""
        return round(lean_kg / (height_m ** 2), 1)

    def ba_segment_lengths(lms, W, H, scale_cm_px):
        """Estimate key segment lengths in cm."""
        def dist(a, b):
            if a is None or b is None: return None
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) * scale_cm_px

        # Landmark indices (MediaPipe)
        # 0=nose,11=l_shoulder,12=r_shoulder,13=l_elbow,14=r_elbow
        # 15=l_wrist,16=r_wrist,23=l_hip,24=r_hip
        # 25=l_knee,26=r_knee,27=l_ankle,28=r_ankle

        lm = lambda i: ba_landmark_px(lms, i, W, H)

        torso    = dist(lm(11) and lm(12) and ((lm(11)[0]+lm(12)[0])/2, (lm(11)[1]+lm(12)[1])/2),
                        lm(23) and lm(24) and ((lm(23)[0]+lm(24)[0])/2, (lm(23)[1]+lm(24)[1])/2))
        l_femur  = dist(lm(23), lm(25))
        r_femur  = dist(lm(24), lm(26))
        l_tibia  = dist(lm(25), lm(27))
        r_tibia  = dist(lm(26), lm(28))
        l_arm    = dist(lm(11), lm(13))
        r_arm    = dist(lm(12), lm(14))
        l_fore   = dist(lm(13), lm(15))
        r_fore   = dist(lm(14), lm(16))
        sh_width = dist(lm(11), lm(12))
        hip_width= dist(lm(23), lm(24))

        femur  = (l_femur or 0 + r_femur or 0) / 2 if (l_femur and r_femur) else (l_femur or r_femur)
        tibia  = (l_tibia or 0 + r_tibia or 0) / 2 if (l_tibia and r_tibia) else (l_tibia or r_tibia)
        arm    = (l_arm or 0 + r_arm or 0) / 2 if (l_arm and r_arm) else (l_arm or r_arm)
        fore   = (l_fore or 0 + r_fore or 0) / 2 if (l_fore and r_fore) else (l_fore or r_fore)

        return {
            "torso_cm": round(torso, 1) if torso else None,
            "femur_cm": round(femur, 1) if femur else None,
            "tibia_cm": round(tibia, 1) if tibia else None,
            "upper_arm_cm": round(arm, 1) if arm else None,
            "forearm_cm": round(fore, 1) if fore else None,
            "shoulder_width_cm": round(sh_width, 1) if sh_width else None,
            "hip_width_cm": round(hip_width, 1) if hip_width else None,
        }

    def ba_postural_analysis(lms_front, lms_side, W_f, H_f, W_s, H_s):
        """Compute postural deviations. Returns dict of findings."""
        findings = {}

        # ── From FRONT photo ──────────────────────────────────────
        if lms_front:
            lm = lambda i: ba_landmark_px(lms_front, i, W_f, H_f)

            # Shoulder level (y coords — lower y = higher on screen)
            ls, rs = lm(11), lm(12)
            if ls and rs:
                diff_px = abs(ls[1] - rs[1])
                diff_cm = diff_px / (H_f * 0.85) * 170  # approx
                findings["shoulder_imbalance_cm"] = round(diff_cm, 1)
                findings["shoulder_high_side"] = "Left" if ls[1] < rs[1] else "Right"

            # Hip level
            lh, rh = lm(23), lm(24)
            if lh and rh:
                diff_px = abs(lh[1] - rh[1])
                diff_cm = diff_px / (H_f * 0.85) * 170
                findings["hip_imbalance_cm"] = round(diff_cm, 1)
                findings["hip_high_side"] = "Left" if lh[1] < rh[1] else "Right"

            # Knee valgus proxy — knee x vs ankle x spread
            lk, la = lm(25), lm(27)
            rk, ra = lm(26), lm(28)
            if lk and la and rk and ra:
                # In neutral: knees should be above ankles
                l_valgus = (lk[0] - la[0]) / W_f  # +ve = knee inside ankle
                r_valgus = (ra[0] - rk[0]) / W_f
                findings["knee_valgus_l"] = round(l_valgus * 100, 1)
                findings["knee_valgus_r"] = round(r_valgus * 100, 1)

            # Lateral trunk lean
            nose = lm(0)
            if nose and lh and rh:
                mid_hip_x = (lh[0] + rh[0]) / 2
                lean_px = nose[0] - mid_hip_x
                findings["lateral_lean_pct"] = round(lean_px / W_f * 100, 1)

        # ── From SIDE photo ───────────────────────────────────────
        if lms_side:
            lm = lambda i: ba_landmark_px(lms_side, i, W_s, H_s)

            # Forward head posture — ear (0/nose proxy) vs shoulder
            nose, shoulder = lm(0), lm(11) or lm(12)
            if nose and shoulder:
                fhp_px = nose[0] - shoulder[0]  # +ve = head forward
                findings["forward_head_cm"] = round(abs(fhp_px) / (H_s * 0.85) * 170 * 0.15, 1)
                findings["forward_head_dir"] = "forward" if fhp_px > 0 else "back"

            # Anterior pelvic tilt — angle at hip
            hip = lm(23) or lm(24)
            knee = lm(25) or lm(26)
            shoulder2 = lm(11) or lm(12)
            if hip and knee and shoulder2:
                # Trunk angle from vertical
                trunk_angle = math.degrees(math.atan2(
                    abs(shoulder2[0] - hip[0]),
                    abs(shoulder2[1] - hip[1])
                ))
                findings["trunk_forward_lean_deg"] = round(trunk_angle, 1)

            # Knee hyperextension
            hip2, knee2, ankle = lm(23) or lm(24), lm(25) or lm(26), lm(27) or lm(28)
            if hip2 and knee2 and ankle:
                knee_angle = ba_angle_3pts(hip2, knee2, ankle)
                findings["knee_angle_standing"] = round(knee_angle, 1)
                findings["knee_hyperextension"] = knee_angle > 185

        return findings

    def ba_estimate_circumferences(lms_front, W, H, scale):
        """Estimate waist/hip/neck width from front photo as proxy for circumference."""
        if not lms_front: return {}
        lm = lambda i: ba_landmark_px(lms_front, i, W, H)

        # Shoulder width → neck proxy
        ls, rs = lm(11), lm(12)
        sh_w = math.sqrt((ls[0]-rs[0])**2 + (ls[1]-rs[1])**2) * scale if ls and rs else None

        # Hip width from landmarks 23,24
        lh, rh = lm(23), lm(24)
        hip_w = math.sqrt((lh[0]-rh[0])**2 + (lh[1]-rh[1])**2) * scale if lh and rh else None

        # Waist: midpoint between shoulder and hip
        waist_w = None
        if ls and rs and lh and rh:
            # Estimate waist as 75% of hip width (approximation)
            waist_w = hip_w * 0.75 if hip_w else None

        # Convert widths → circumferences (approximate: circ ≈ width × π × 0.75)
        # This is a rough anthropometric approximation
        circ = lambda w: round(w * math.pi * 0.75, 1) if w else None

        neck_circ  = round(sh_w * 0.28, 1) if sh_w else None  # neck ~ 28% of shoulder width
        waist_circ = circ(waist_w)
        hip_circ   = circ(hip_w * 1.1) if hip_w else None

        return {
            "neck_cm_est": neck_circ,
            "waist_cm_est": waist_circ,
            "hip_cm_est": hip_circ,
        }

    def ba_somatotype(sh_w, hip_w, waist_w):
        """Simple somatotype classification from proportions."""
        if not all([sh_w, hip_w, waist_w]):
            return "Unknown", "Insufficient landmark data for classification."
        ratio_sh_hip = sh_w / hip_w
        if ratio_sh_hip > 1.25:
            return "Mesomorph", "Broad shoulders, narrow hips — naturally muscular build. Responds well to strength training."
        elif ratio_sh_hip < 0.90:
            return "Endomorph", "Wider hips relative to shoulders — tends to store fat more easily. Responds well to metabolic conditioning."
        else:
            return "Ectomorph / Balanced", "Balanced shoulder-to-hip ratio — lean frame. Responds well to volume-based hypertrophy training."

    def ba_bmi_category(bmi):
        if bmi < 18.5: return "Underweight", "bad"
        elif bmi < 25: return "Healthy", "good"
        elif bmi < 30: return "Overweight", "warn"
        else: return "Obese", "bad"

    def ba_bf_category(bf, sex):
        if sex == "Male":
            if bf < 6: return "Essential fat", "warn"
            elif bf < 14: return "Athletic", "good"
            elif bf < 18: return "Fitness", "good"
            elif bf < 25: return "Average", "warn"
            else: return "Above average", "bad"
        else:
            if bf < 14: return "Essential fat", "warn"
            elif bf < 21: return "Athletic", "good"
            elif bf < 25: return "Fitness", "good"
            elif bf < 32: return "Average", "warn"
            else: return "Above average", "bad"

    def ba_call_llm(data: dict) -> str:
        """Generate personalised coaching report via LLM."""
        prompt = f"""You are a sports scientist and personal trainer writing a personalised body assessment report.

ASSESSMENT DATA:
- Sex: {data.get('sex')}
- Age: {data.get('age')} years
- Height: {data.get('height_cm')} cm
- Weight: {data.get('weight_kg')} kg
- BMI: {data.get('bmi')} ({data.get('bmi_cat')})
- Est. Body Fat: {data.get('bf_pct')}%  ({data.get('bf_cat')})
- Lean Mass: {data.get('lean_kg')} kg
- Fat Mass: {data.get('fat_kg')} kg
- FFMI: {data.get('ffmi')}
- Shoulder width: {data.get('shoulder_width_cm')} cm
- Hip width: {data.get('hip_width_cm')} cm
- Femur length: {data.get('femur_cm')} cm
- Torso length: {data.get('torso_cm')} cm
- Somatotype: {data.get('somatotype')}

POSTURAL FINDINGS:
- Forward head posture: {data.get('forward_head_cm')} cm forward
- Trunk forward lean: {data.get('trunk_lean')} degrees
- Shoulder imbalance: {data.get('shoulder_imbal')} cm ({data.get('shoulder_high')} higher)
- Hip imbalance: {data.get('hip_imbal')} cm ({data.get('hip_high')} higher)
- Standing knee angle: {data.get('knee_angle')} degrees

Write a concise, plain-English assessment (250-300 words) covering:
1. What their body composition means for them RIGHT NOW (honest, not alarming)
2. What their proportions mean for their training (e.g. long femurs, torso:leg ratio)
3. Their 2-3 most important postural findings and what causes them in everyday life
4. 3 specific, actionable recommendations tailored to their exact numbers

Be direct, warm, and specific. Avoid generic advice. Reference their actual numbers.
No bullet points — write in flowing paragraphs. Keep it under 300 words."""

        try:
            # Increase token limit for body assessment
            gk = get_secret("GROQ_API_KEY")
            if gk:
                r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": "Bearer " + gk, "Content-Type": "application/json"},
                    json={"model": "llama-3.3-70b-versatile",
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 900},
                    timeout=40)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
            return call_llm(prompt)
        except Exception as e:
            return f"AI report unavailable: {e}"

    # ── UI ────────────────────────────────────────────────────────
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="ba-zone">', unsafe_allow_html=True)
    st.markdown(
        '<h2 class="ba-headline">Know Your <span>Body.</span></h2>'
        '<p class="ba-desc">Upload a front and side photo. Add your height and weight. '
        'FORMate maps your posture, estimates body composition, and gives you a personalised training blueprint — '
        'in 60 seconds, from your phone camera.</p>',
        unsafe_allow_html=True)

    # ── Step 1: Inputs ────────────────────────────────────────────
    st.markdown('<p class="ba-section">Step 1 — Your Details</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ba_sex = st.selectbox("Sex", ["Male", "Female"], key="ba_sex",
                              label_visibility="visible")
    with c2:
        ba_age = st.number_input("Age", min_value=16, max_value=80, value=28, key="ba_age",
                                 label_visibility="visible")
    with c3:
        ba_height = st.number_input("Height (cm)", min_value=140, max_value=220, value=175,
                                    key="ba_height", label_visibility="visible")
    with c4:
        ba_weight = st.number_input("Weight (kg)", min_value=40.0, max_value=200.0, value=75.0,
                                    step=0.5, key="ba_weight", label_visibility="visible")

    # ── Step 2: Capture mode ──────────────────────────────────────
    st.markdown('<p class="ba-section">Step 2 — Capture Photos</p>', unsafe_allow_html=True)

    ba_mode = st.radio("Capture method", ["📷  Live Camera", "📁  Upload Photos"],
                       horizontal=True, key="ba_mode", label_visibility="collapsed")

    # initialise session state for captured images
    if "ba_img_front" not in st.session_state: st.session_state.ba_img_front = None
    if "ba_img_side"  not in st.session_state: st.session_state.ba_img_side  = None
    if "ba_cam_done"  not in st.session_state: st.session_state.ba_cam_done  = False

    ba_front = None
    ba_side  = None

    # ── LIVE CAMERA MODE ─────────────────────────────────────────
    if "Live Camera" in ba_mode:
        import streamlit.components.v1 as components

        # Reset button
        col_rst, _ = st.columns([1,4])
        with col_rst:
            if st.button("🔄 Reset Camera", key="ba_reset"):
                st.session_state.ba_img_front = None
                st.session_state.ba_img_side  = None
                st.session_state.ba_cam_done  = False
                st.rerun()

        if not st.session_state.ba_cam_done:
            st.markdown("""
            <div style="font-size:.75rem;color:var(--sub);background:var(--card2);
            border:1px solid var(--edge);border-radius:10px;padding:.65rem .85rem;
            margin-bottom:.75rem;line-height:1.65;">
            ✋ <strong>How to capture:</strong>
            Raise <strong>one hand above your head</strong> and hold for 5 seconds.
            First shot = front view &nbsp;·&nbsp; Second shot = side view (turn 90°).
            Stand 2–3 m away, full body visible, fitted clothing.
            </div>""", unsafe_allow_html=True)

        gesture_html = """<!DOCTYPE html><html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
html,body{width:100%;height:100%;background:#07070F;overflow:hidden;font-family:'Inter',sans-serif;}
#cam-container{position:relative;width:100%;height:100vh;background:#000;overflow:hidden;}
video{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;pointer-events:none;}
video.normal{top:0;left:0;width:100%;height:100%;object-fit:cover;}
video.normal.mirror{transform:scaleX(-1);}
video.portrait-fix{transform:translate(-50%,-50%) rotate(90deg);object-fit:cover;}
video.portrait-fix.mirror{transform:translate(-50%,-50%) rotate(90deg) scaleX(-1);}
canvas{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;}

/* ── CAM-OFF SCREEN ── */
#cam-off{
  position:absolute;top:0;left:0;right:0;bottom:0;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:1.2rem;z-index:30;background:#07070F;}
#cam-off-title{
  font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;color:#fff;}
#cam-off-sub{
  font-size:.8rem;color:rgba(255,255,255,.45);text-align:center;
  max-width:240px;line-height:1.6;}
#btn-start{
  background:linear-gradient(135deg,#1D4ED8,#3B82F6);
  color:#fff;border:none;border-radius:14px;
  font-family:'Space Grotesk',sans-serif;font-size:.95rem;font-weight:700;
  padding:.85rem 2.2rem;cursor:pointer;
  box-shadow:0 4px 24px rgba(59,130,246,.35);
  transition:opacity .15s;}
#btn-start:hover{opacity:.88;}
#btn-start:disabled{opacity:.5;cursor:default;}

/* ── TOP STATUS BAR ── */
#status-bar{
  position:absolute;top:0;left:0;right:0;display:none;
  background:linear-gradient(to bottom,rgba(0,0,0,.82),transparent);
  padding:1.1rem 1.2rem .9rem;z-index:20;
  align-items:center;justify-content:space-between;}
#status-bar.visible{display:flex;}
#phase-label{
  font-family:'Space Grotesk',sans-serif;font-size:.65rem;font-weight:700;
  letter-spacing:.14em;text-transform:uppercase;color:rgba(255,255,255,.4);}
#phase-name{
  font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:700;
  color:#fff;line-height:1;margin-top:.1rem;}
#shot-dots{display:flex;gap:.5rem;}
.dot{width:10px;height:10px;border-radius:50%;background:rgba(255,255,255,.18);transition:background .3s;}
.dot.done{background:#34D399;}
.dot.active{background:#3B82F6;box-shadow:0 0 8px rgba(59,130,246,.7);}

/* ── BOTTOM GESTURE BOX ── */
#gesture-box{
  position:absolute;bottom:0;left:0;right:0;display:none;
  background:linear-gradient(to top,rgba(0,0,0,.9) 0%,rgba(0,0,0,.6) 60%,transparent 100%);
  padding:1.5rem 1.2rem 2rem;z-index:20;
  flex-direction:column;align-items:center;gap:.75rem;}
#gesture-box.visible{display:flex;}

/* Ring */
#ring-wrap{position:relative;width:84px;height:84px;}
#ring-wrap svg{position:absolute;top:0;left:0;width:100%;height:100%;}
#ring-num{
  position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  font-family:'Space Grotesk',sans-serif;font-size:1.9rem;font-weight:700;
  color:#fff;line-height:1;transition:color .2s;}
#ring-wrap.counting #ring-num{color:#3B82F6;}
#ring-wrap.done     #ring-num{color:#34D399;}

#gesture-text{
  font-size:.84rem;font-weight:600;color:rgba(255,255,255,.9);
  text-align:center;line-height:1.55;max-width:260px;}
#gesture-hint{
  font-size:.68rem;color:rgba(255,255,255,.38);text-align:center;}

/* ── FLASH ── */
#flash{position:absolute;inset:0;background:#fff;opacity:0;pointer-events:none;z-index:50;transition:opacity .07s;}

/* ── DONE OVERLAY ── */
#done-overlay{
  position:absolute;inset:0;background:rgba(7,7,15,.9);z-index:60;
  display:none;flex-direction:column;align-items:center;justify-content:center;
  gap:1.25rem;padding:2rem;}
#done-overlay.visible{display:flex;}
#done-title{
  font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;
  color:#34D399;text-align:center;}
#done-sub{font-size:.8rem;color:rgba(255,255,255,.55);text-align:center;line-height:1.6;}
#thumbs-wrap{display:flex;gap:1rem;margin-top:.25rem;}
.thumb-col{display:flex;flex-direction:column;align-items:center;gap:.4rem;}
.thumb-box{border-radius:10px;overflow:hidden;width:110px;height:150px;
  border:1.5px solid rgba(255,255,255,.15);}
.thumb-box img{width:100%;height:100%;object-fit:cover;}
.thumb-lbl{font-size:.58rem;color:rgba(255,255,255,.35);
  text-transform:uppercase;letter-spacing:.1em;}

/* ── TRANSITION MSG ── */
#turn-msg{
  position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  background:rgba(0,0,0,.88);border-radius:18px;padding:1.5rem 2.5rem;
  z-index:40;text-align:center;display:none;}
#turn-msg.visible{display:block;}
#turn-msg-icon{font-size:2.2rem;margin-bottom:.5rem;}
#turn-msg-text{
  font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;
  color:#fff;margin-bottom:.3rem;}
#turn-msg-sub{font-size:.72rem;color:rgba(255,255,255,.5);}
</style>
</head>
<body>
<div id="cam-container">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="canvas"></canvas>
  <div id="flash"></div>

  <!-- Off screen -->
  <div id="cam-off">
    <div style="font-size:2.8rem">📷</div>
    <div id="cam-off-title">Body Assessment Camera</div>
    <div id="cam-off-sub">Stand 2–3 m away · Full body visible · Fitted clothing</div>
    <button id="btn-start" onclick="startCapture()">Start Camera</button>
    <div id="err-msg" style="font-size:.72rem;color:#EF4444;display:none;text-align:center;max-width:240px;"></div>
  </div>

  <!-- Top bar (hidden until camera on) -->
  <div id="status-bar">
    <div>
      <div id="phase-label">CAPTURING</div>
      <div id="phase-name">Front View</div>
    </div>
    <div id="shot-dots">
      <div class="dot active" id="dot0"></div>
      <div class="dot"        id="dot1"></div>
    </div>
  </div>

  <!-- Turn msg -->
  <div id="turn-msg">
    <div id="turn-msg-icon">↩️</div>
    <div id="turn-msg-text">Turn sideways</div>
    <div id="turn-msg-sub">Face left or right, then raise one hand</div>
  </div>

  <!-- Bottom gesture box -->
  <div id="gesture-box">
    <div id="ring-wrap">
      <svg viewBox="0 0 84 84">
        <circle cx="42" cy="42" r="36" fill="none" stroke="rgba(255,255,255,.1)" stroke-width="6"/>
        <circle id="ring-arc" cx="42" cy="42" r="36" fill="none"
          stroke="#3B82F6" stroke-width="6"
          stroke-dasharray="226" stroke-dashoffset="226"
          stroke-linecap="round" transform="rotate(-90 42 42)"/>
      </svg>
      <div id="ring-num">✋</div>
    </div>
    <div id="gesture-text">Raise one hand above your head<br>and hold for <strong>5 seconds</strong></div>
    <div id="gesture-hint">Front photo · Shot 1 of 2</div>
  </div>

  <!-- Done overlay -->
  <div id="done-overlay">
    <div id="done-title">✅ Both shots captured!</div>
    <div id="done-sub">Scroll down and click<br><strong>Analyse My Body</strong></div>
    <div id="thumbs-wrap">
      <div class="thumb-col">
        <div class="thumb-box"><img id="thumb-front" src="data:,"/></div>
        <div class="thumb-lbl">Front</div>
      </div>
      <div class="thumb-col">
        <div class="thumb-box"><img id="thumb-side" src="data:,"/></div>
        <div class="thumb-lbl">Side</div>
      </div>
    </div>
  </div>
</div>

<!-- TF.js + MoveNet — exact same versions as Live Trainer -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.3/dist/pose-detection.min.js"></script>
<script>
const HOLD_SECS=5, CIRC=226;
let phase='front',detector=null,stream=null,rafId=null;
let holdStart=null,holdActive=false,imgFront=null,imgSide=null;
const MV={NOSE:0,L_SH:5,R_SH:6,L_EL:7,R_EL:8,L_WR:9,R_WR:10,
          L_HIP:11,R_HIP:12,L_KN:13,R_KN:14,L_AN:15,R_AN:16};
const CONNS=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],
             [11,13],[13,15],[12,14],[14,16]];

async function startCapture(){
  const btn=document.getElementById('btn-start');
  const err=document.getElementById('err-msg');
  btn.textContent='Loading…';btn.disabled=true;err.style.display='none';
  try{
    stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'user'}}})
      .catch(()=>navigator.mediaDevices.getUserMedia({audio:false,video:true}));
    const video=document.getElementById('video');
    video.srcObject=stream;
    await new Promise(r=>video.onloadedmetadata=r);
    video.play();
    applyVideoOrientation(video);
    document.getElementById('cam-off').style.display='none';
    document.getElementById('status-bar').classList.add('visible');
    document.getElementById('gesture-box').classList.add('visible');
    detector=await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      {modelType:poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,enableSmoothing:true}
    );
    detect();
  }catch(e){
    btn.textContent='Start Camera';btn.disabled=false;
    err.textContent='Camera error: '+(e.message||e);err.style.display='block';
  }
}

function applyVideoOrientation(video){
  const vW=video.videoWidth,vH=video.videoHeight,needsRotate=vW>vH;
  const c=document.getElementById('cam-container');
  const cW=c.offsetWidth,cH=c.offsetHeight;
  if(needsRotate){
    video.style.cssText='position:absolute;top:50%;left:50%;object-fit:cover;';
    video.style.width=cH+'px';video.style.height=cW+'px';
    video.className='portrait-fix mirror';
  }else{
    video.style.cssText='position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;';
    video.className='normal mirror';
  }
}

function drawSkel(ctx,kp,vW,vH,rW,rH,offX,offY){
  const px=i=>kp[i]?offX+(rW-kp[i].x/vW*rW):0;
  const py=i=>kp[i]?offY+kp[i].y/vH*rH:0;
  const vs=i=>kp[i]?kp[i].score:0;
  ctx.lineWidth=4;ctx.lineCap='round';
  for(const[a,b]of CONNS){
    if(vs(a)<0.25||vs(b)<0.25)continue;
    ctx.globalAlpha=0.9;ctx.strokeStyle='#3B82F6';
    ctx.shadowColor='#3B82F6';ctx.shadowBlur=8;
    ctx.beginPath();ctx.moveTo(px(a),py(a));ctx.lineTo(px(b),py(b));ctx.stroke();
  }
  ctx.globalAlpha=1;ctx.shadowBlur=0;
  for(let i=5;i<=16;i++){
    if(vs(i)<0.25)continue;
    ctx.beginPath();ctx.arc(px(i),py(i),5,0,Math.PI*2);
    ctx.fillStyle='#60A5FA';ctx.fill();
  }
}

function isHandRaised(kp){
  // Mirror of Live Trainer checkGesture — raw pixel y, wrist above shoulder
  // ONE hand raised = either wrist clearly above its shoulder
  const lw=kp[MV.L_WR],rw=kp[MV.R_WR];
  const ls=kp[MV.L_SH],rs=kp[MV.R_SH];
  if(!ls||!rs) return false;
  if(ls.score<0.2||rs.score<0.2) return false;
  const lRaised = lw && lw.score>0.2 && lw.y < ls.y - 10;
  const rRaised = rw && rw.score>0.2 && rw.y < rs.y - 10;
  return lRaised || rRaised;
}

function setRing(pct){
  const arc=document.getElementById('ring-arc');
  const num=document.getElementById('ring-num');
  const wrap=document.getElementById('ring-wrap');
  arc.style.strokeDashoffset=CIRC*(1-pct);
  if(pct<=0){arc.style.stroke='rgba(255,255,255,.15)';wrap.className='';num.textContent='✋';}
  else if(pct>=1){arc.style.stroke='#34D399';wrap.className='done';num.textContent='✓';}
  else{arc.style.stroke='#3B82F6';wrap.className='counting';num.textContent=Math.ceil(HOLD_SECS*(1-pct));}
}

function captureFrame(){
  const video=document.getElementById('video');
  const fl=document.getElementById('flash');
  fl.style.opacity='1';setTimeout(()=>fl.style.opacity='0',100);
  const cap=document.createElement('canvas');
  cap.width=video.videoWidth;cap.height=video.videoHeight;
  const c2=cap.getContext('2d');
  c2.save();c2.scale(-1,1);c2.translate(-cap.width,0);
  c2.drawImage(video,0,0);c2.restore();
  return cap.toDataURL('image/jpeg',0.92);
}

function goSide(){
  phase='side';holdStart=null;holdActive=false;setRing(0);
  document.getElementById('phase-name').textContent='Side View';
  document.getElementById('gesture-hint').textContent='Side photo · Shot 2 of 2';
  document.getElementById('dot0').className='dot done';
  document.getElementById('dot1').className='dot active';
  const msg=document.getElementById('turn-msg');
  msg.classList.add('visible');setTimeout(()=>msg.classList.remove('visible'),2500);
}

function goDone(){
  phase='done';cancelAnimationFrame(rafId);
  document.getElementById('gesture-box').classList.remove('visible');
  document.getElementById('status-bar').classList.remove('visible');
  document.getElementById('thumb-front').src=imgFront;
  document.getElementById('thumb-side').src=imgSide;
  document.getElementById('done-overlay').classList.add('visible');
  setTimeout(()=>window.parent.postMessage({type:'ba_photos',front:imgFront,side:imgSide},'*'),500);
}

async function detect(){
  if(phase==='done')return;
  const video=document.getElementById('video');
  const canvas=document.getElementById('canvas');
  const ctx=canvas.getContext('2d');
  const _vW=video.videoWidth||640,_vH=video.videoHeight||480;
  const rotated=video.className.includes('portrait-fix');
  const vW=rotated?_vH:_vW,vH=rotated?_vW:_vH;
  const cW=canvas.offsetWidth||canvas.width,cH=canvas.offsetHeight||canvas.height;
  canvas.width=cW;canvas.height=cH;
  const scale=Math.max(cW/vW,cH/vH);
  const rW=vW*scale,rH=vH*scale,offX=(cW-rW)/2,offY=(cH-rH)/2;
  ctx.clearRect(0,0,cW,cH);
  try{
    if(detector&&video.readyState>=2){
      const poses=await detector.estimatePoses(video,{flipHorizontal:true});
      if(poses.length>0){
        const kp=poses[0].keypoints;
        drawSkel(ctx,kp,vW,vH,rW,rH,offX,offY);
        const raised=isHandRaised(kp);
        if(raised){
          if(!holdActive){holdActive=true;holdStart=performance.now();}
          const pct=Math.min((performance.now()-holdStart)/1000/HOLD_SECS,1);
          setRing(pct);
          if(pct>=1){
            const url=captureFrame();
            if(phase==='front'){imgFront=url;goSide();}
            else{imgSide=url;goDone();return;}
          }
        }else{holdActive=false;holdStart=null;setRing(0);}
      }else{
        ctx.font='bold 14px Inter,sans-serif';ctx.fillStyle='rgba(255,255,255,.35)';
        ctx.textAlign='center';
        ctx.fillText('Point camera at your full body',cW/2,cH/2);
      }
    }
  }catch(e){console.warn('detect:',e);}
  rafId=requestAnimationFrame(detect);
}

window.addEventListener('resize',()=>{
  const v=document.getElementById('video');
  if(v.srcObject)applyVideoOrientation(v);
});
</script>
</body></html>"""

        # Embed camera iframe
        cam_component = components.html(gesture_html, height=640, scrolling=False)

        # Receive postMessage via hidden bridge
        # We use a second tiny component to relay postMessage → session_state
        bridge_html = """<script>
window.addEventListener('message', function(e){
  if(e.data && e.data.type === 'ba_photos'){
    // Relay to Streamlit via query param trick — write to sessionStorage
    sessionStorage.setItem('ba_front', e.data.front);
    sessionStorage.setItem('ba_side',  e.data.side);
    // Signal Streamlit to rerun via URL fragment change
    window.parent.location.hash = 'ba_captured_' + Date.now();
  }
});
// On load, check if photos already in sessionStorage and relay up
const f = sessionStorage.getItem('ba_front');
const s = sessionStorage.getItem('ba_side');
if(f && s){
  window.parent.postMessage({type:'ba_ready',front:f,side:s},'*');
}
</script>"""

        # Show captured previews if we have them
        if st.session_state.ba_img_front:
            st.success("✅ Both photos captured — scroll down to analyse.")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown('<p class="ba-photo-label">Front</p>', unsafe_allow_html=True)
                # Decode base64 and show
                import base64
                front_b64 = st.session_state.ba_img_front.split(',')[1] if ',' in st.session_state.ba_img_front else st.session_state.ba_img_front
                front_bytes = base64.b64decode(front_b64)
                st.image(front_bytes, use_column_width=True)
            with col_p2:
                st.markdown('<p class="ba-photo-label">Side</p>', unsafe_allow_html=True)
                if st.session_state.ba_img_side:
                    side_b64 = st.session_state.ba_img_side.split(',')[1] if ',' in st.session_state.ba_img_side else st.session_state.ba_img_side
                    side_bytes = base64.b64decode(side_b64)
                    st.image(side_bytes, use_column_width=True)

        # JS→Python bridge: file uploader that JS programmatically triggers
        st.markdown('<p style="font-size:.7rem;color:var(--sub);margin-top:.5rem">'
                    'After both shots are captured above, click <strong>Send to Analysis</strong> '
                    'in the camera view, or use the uploader below as fallback:</p>',
                    unsafe_allow_html=True)
        ba_cam_upload = st.file_uploader(
            "Captured photos (auto-filled by camera)",
            type=["jpg","jpeg","png"],
            accept_multiple_files=True,
            key="ba_cam_upload",
            label_visibility="collapsed"
        )
        if ba_cam_upload and len(ba_cam_upload) >= 1:
            st.session_state.ba_img_front = ba_cam_upload[0]
            if len(ba_cam_upload) >= 2:
                st.session_state.ba_img_side = ba_cam_upload[1]

        # Set ba_front / ba_side from session state for analysis
        ba_front = st.session_state.ba_img_front
        ba_side  = st.session_state.ba_img_side

    # ── UPLOAD MODE ──────────────────────────────────────────────
    else:
        st.markdown("""
        <div style="font-size:.75rem;color:var(--sub);background:var(--card2);border:1px solid var(--edge);
        border-radius:10px;padding:.65rem .85rem;margin-bottom:1rem;line-height:1.65;">
        📸 <strong>Tips for best results:</strong>
        Stand 2–3m from camera &nbsp;·&nbsp; Full body visible head to toe &nbsp;·&nbsp;
        Wear fitted clothing &nbsp;·&nbsp; Stand on a flat surface &nbsp;·&nbsp;
        Arms slightly away from body
        </div>""", unsafe_allow_html=True)

        col_f, col_s = st.columns(2)
        with col_f:
            st.markdown('<p class="ba-photo-label">📷 Front View</p>', unsafe_allow_html=True)
            ba_front = st.file_uploader("Front photo", type=["jpg","jpeg","png","webp"],
                                        key="ba_front", label_visibility="collapsed")
            if ba_front:
                st.image(ba_front, use_column_width=True)
        with col_s:
            st.markdown('<p class="ba-photo-label">📷 Side View (optional but recommended)</p>',
                        unsafe_allow_html=True)
            ba_side = st.file_uploader("Side photo", type=["jpg","jpeg","png","webp"],
                                       key="ba_side", label_visibility="collapsed")
            if ba_side:
                st.image(ba_side, use_column_width=True)

    st.markdown('<div class="ba-disclaimer">⚠️ <strong>Estimates only.</strong> '
                'Body fat and muscle mass are calculated from anthropometric formulas — not medical-grade measurements. '
                'Accuracy is typically ±4–6%. For clinical accuracy use DEXA or InBody scanning. '
                'This tool is for fitness guidance only, not medical advice.</div>',
                unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close ba-zone

    # ── Analyse button — works for both modes ─────────────────────
    # Normalise: convert base64 string → bytes-like for load_img
    def ba_to_bytes(src):
        """Accept file-uploader object OR base64 string, return bytes."""
        if src is None: return None
        if hasattr(src, 'read'):
            src.seek(0)
            return src.read()
        import base64
        b64 = src.split(',')[1] if ',' in src else src
        return base64.b64decode(b64)

    if ba_front is not None:
        if st.button("🔬 Analyse My Body", type="primary", key="ba_run",
                     use_container_width=True):

            with st.spinner("Detecting landmarks and computing metrics…"):

                # Load images — works for file-uploader objects AND base64 strings
                def load_img(src):
                    raw = ba_to_bytes(src)
                    if raw is None: return None
                    arr = np.frombuffer(raw, np.uint8)
                    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

                img_front = load_img(ba_front)
                img_side  = load_img(ba_side) if ba_side else None

                H_f, W_f = img_front.shape[:2]
                H_s, W_s = (img_side.shape[:2] if img_side is not None else (0, 0))

                # Get landmarks
                lms_front = ba_get_landmarks(img_front)
                lms_side  = ba_get_landmarks(img_side) if img_side is not None else None

                # Scale factor
                scale = ba_pixel_to_cm(H_f, ba_height)

                # Segment lengths
                segs = ba_segment_lengths(lms_front, W_f, H_f, scale) if lms_front else {}

                # Circumference estimates
                circs = ba_estimate_circumferences(lms_front, W_f, H_f, scale) if lms_front else {}

                # Body composition
                bmi   = round(ba_weight / ((ba_height / 100) ** 2), 1)
                bmi_cat, bmi_cls = ba_bmi_category(bmi)

                # Body fat
                neck_c  = circs.get("neck_cm_est") or (ba_height * 0.21)
                waist_c = circs.get("waist_cm_est") or (ba_height * 0.47 if ba_sex == "Male" else ba_height * 0.44)
                hip_c   = circs.get("hip_cm_est")

                bf_pct = ba_compute_body_fat(waist_c, neck_c, ba_height, ba_sex, hip_c)
                if bf_pct is None:
                    bf_pct = 22.0 if ba_sex == "Male" else 28.0  # fallback average
                bf_cat, bf_cls = ba_bf_category(bf_pct, ba_sex)

                lean_kg  = round(ba_weight * (1 - bf_pct / 100), 1)
                fat_kg   = round(ba_weight - lean_kg, 1)
                ffmi     = ba_compute_ffmi(lean_kg, ba_height / 100)

                # Waist:height ratio
                whr = round(waist_c / ba_height, 3) if waist_c else None

                # Somatotype
                sh_w  = segs.get("shoulder_width_cm")
                hip_w = segs.get("hip_width_cm")
                soma_type, soma_desc = ba_somatotype(sh_w, hip_w, waist_c)

                # Postural analysis
                posture = ba_postural_analysis(
                    lms_front, lms_side,
                    W_f, H_f, W_s, H_s
                )

                # Femur:torso ratio
                femur_cm = segs.get("femur_cm")
                torso_cm = segs.get("torso_cm")
                ft_ratio = round(femur_cm / torso_cm, 2) if femur_cm and torso_cm else None

            # ── RESULTS ──────────────────────────────────────────
            st.markdown('<div class="ba-zone">', unsafe_allow_html=True)
            st.markdown('<p class="ba-section">Body Composition</p>', unsafe_allow_html=True)

            # Metric grid
            def metric_card(label, value, unit="", sub="", cls=""):
                return f"""<div class="ba-metric {cls}">
                <div class="ba-metric-label">{label}</div>
                <div class="ba-metric-value">{value}</div>
                <div class="ba-metric-unit">{unit}</div>
                {'<div class="ba-metric-sub">'+sub+'</div>' if sub else ''}
                </div>"""

            cards_html = ""
            cards_html += metric_card("BMI", bmi, "kg/m²", bmi_cat, bmi_cls)
            cards_html += metric_card("Body Fat", f"{bf_pct}%", "estimated", bf_cat, bf_cls)
            cards_html += metric_card("Lean Mass", f"{lean_kg}", "kg", "muscle + bone + organ", "blue")
            cards_html += metric_card("Fat Mass", f"{fat_kg}", "kg", "", "")
            cards_html += metric_card("FFMI", ffmi, "", "Fat-free mass index", "blue" if ffmi >= 18 else "")
            if whr:
                whr_cls = "good" if whr < 0.5 else "warn" if whr < 0.6 else "bad"
                cards_html += metric_card("Waist:Height", whr, "ratio", "< 0.5 is optimal", whr_cls)

            st.markdown(f'<div class="ba-metric-grid">{cards_html}</div>', unsafe_allow_html=True)

            # ── Body proportions ─────────────────────────────────
            st.markdown('<p class="ba-section">Body Proportions</p>', unsafe_allow_html=True)
            prop_cards = ""
            if sh_w:  prop_cards += metric_card("Shoulder Width", f"{sh_w}", "cm", "", "blue")
            if hip_w: prop_cards += metric_card("Hip Width", f"{hip_w}", "cm", "", "")
            if sh_w and hip_w:
                shr = round(sh_w / hip_w, 2)
                shr_cls = "good" if shr > 1.15 else ""
                prop_cards += metric_card("Shoulder:Hip", shr, "ratio", "V-taper indicator", shr_cls)
            if femur_cm: prop_cards += metric_card("Femur Length", f"{femur_cm}", "cm", "affects squat stance", "")
            if torso_cm: prop_cards += metric_card("Torso Length", f"{torso_cm}", "cm", "", "")
            if ft_ratio:
                ft_cls = "warn" if ft_ratio > 1.1 else ""
                ft_sub = "Long femurs — widen squat stance" if ft_ratio > 1.0 else "Balanced proportions"
                prop_cards += metric_card("Femur:Torso", ft_ratio, "ratio", ft_sub, ft_cls)
            if segs.get("upper_arm_cm"):
                prop_cards += metric_card("Upper Arm", f"{segs['upper_arm_cm']}", "cm", "", "")

            if prop_cards:
                st.markdown(f'<div class="ba-metric-grid">{prop_cards}</div>', unsafe_allow_html=True)

            # Somatotype badge
            st.markdown(
                f'<div style="margin-bottom:.5rem"><span class="ba-somatotype">{soma_type}</span>'
                f'<span style="font-size:.75rem;color:var(--sub);margin-left:.75rem">{soma_desc}</span></div>',
                unsafe_allow_html=True)

            # ── Postural findings ─────────────────────────────────
            st.markdown('<p class="ba-section">Postural Analysis</p>', unsafe_allow_html=True)

            flags = []

            # Forward head
            fhp = posture.get("forward_head_cm", 0)
            if fhp > 0:
                cls = "ok" if fhp < 2 else "warn" if fhp < 4 else "bad"
                icon = "✅" if fhp < 2 else "⚠️" if fhp < 4 else "🔴"
                flags.append((cls, icon, "Forward Head Posture",
                    f"{fhp} cm forward. " +
                    ("Within normal range." if fhp < 2 else
                     "Mild forward head — common with desk work. Chin tucks + thoracic extension will help." if fhp < 4 else
                     "Significant forward head. Prioritise deep neck flexor strengthening and reduce screen time.")))

            # Trunk lean
            trunk = posture.get("trunk_forward_lean_deg", 0)
            if trunk > 0:
                cls = "ok" if trunk < 5 else "warn" if trunk < 12 else "bad"
                icon = "✅" if trunk < 5 else "⚠️" if trunk < 12 else "🔴"
                flags.append((cls, icon, "Trunk Posture",
                    f"{trunk}° forward inclination. " +
                    ("Neutral spine — excellent." if trunk < 5 else
                     "Mild anterior lean — check hip flexor tightness and core activation." if trunk < 12 else
                     "Significant forward lean. Prioritise hip flexor stretching, glute activation, and posterior chain work.")))

            # Shoulder imbalance
            sh_im = posture.get("shoulder_imbalance_cm", 0)
            if sh_im > 0:
                cls = "ok" if sh_im < 1.5 else "warn" if sh_im < 3 else "bad"
                icon = "✅" if sh_im < 1.5 else "⚠️"
                high = posture.get("shoulder_high_side", "")
                flags.append((cls, icon, "Shoulder Level",
                    f"{sh_im} cm difference — {high} side higher. " +
                    ("Symmetrical — within normal variation." if sh_im < 1.5 else
                     f"Mild imbalance — check unilateral pressing volume. Consider single-arm carries.")))

            # Hip imbalance
            hip_im = posture.get("hip_imbalance_cm", 0)
            if hip_im > 0:
                cls = "ok" if hip_im < 1 else "warn" if hip_im < 2.5 else "bad"
                icon = "✅" if hip_im < 1 else "⚠️"
                high_h = posture.get("hip_high_side", "")
                flags.append((cls, icon, "Pelvic Level",
                    f"{hip_im} cm difference — {high_h} hip higher. " +
                    ("Symmetrical." if hip_im < 1 else
                     f"Mild pelvic tilt — check {high_h.lower()} hip flexor tightness and opposite glute weakness.")))

            # Knee alignment
            kl = posture.get("knee_valgus_l", 0)
            kr = posture.get("knee_valgus_r", 0)
            if abs(kl) > 2 or abs(kr) > 2:
                avg_v = (abs(kl) + abs(kr)) / 2
                cls = "ok" if avg_v < 3 else "warn" if avg_v < 6 else "bad"
                icon = "✅" if avg_v < 3 else "⚠️" if avg_v < 6 else "🔴"
                flags.append((cls, icon, "Knee Alignment",
                    f"L: {kl:+.1f}  R: {kr:+.1f} (% frame width). " +
                    ("Knees track well over ankles." if avg_v < 3 else
                     "Mild knee valgus — strengthen glutes and hip abductors. Focus on knee tracking during squats.")))

            if not flags:
                flags.append(("ok", "✅", "Posture Scan Complete",
                    "Not enough landmark data for detailed postural analysis. "
                    "Try a full-length photo with better lighting and fitted clothing."))

            for cls, icon, title, body in flags:
                st.markdown(
                    f'<div class="ba-flag {cls}">'
                    f'<div class="ba-flag-icon">{icon}</div>'
                    f'<div><div class="ba-flag-title">{title}</div>'
                    f'<div class="ba-flag-body">{body}</div></div></div>',
                    unsafe_allow_html=True)

            # ── Score bars ────────────────────────────────────────
            st.markdown('<p class="ba-section">Fitness Readiness Profile</p>', unsafe_allow_html=True)

            # Compute simple scores (0–100)
            def clamp_score(v): return max(0, min(100, int(v)))

            bmi_score  = clamp_score(100 - abs(bmi - 22) * 6)
            bf_score   = clamp_score(100 - max(0, bf_pct - (10 if ba_sex=="Male" else 18)) * 3)
            lean_score = clamp_score((ffmi / 25) * 100) if ba_sex=="Male" else clamp_score((ffmi / 22) * 100)
            posture_score = clamp_score(100 - (fhp * 8) - (trunk * 2) - (sh_im * 6))
            proportion_score = clamp_score(70 + (20 if ft_ratio and 0.8 < ft_ratio < 1.1 else 0) +
                                            (10 if sh_w and hip_w and sh_w/hip_w > 1.1 else 0))

            bars = [
                ("Body Composition", bmi_score, "#3B82F6"),
                ("Body Fat Level",   bf_score,  "#34D399"),
                ("Lean Mass Index",  lean_score, "#60A5FA"),
                ("Posture Quality",  posture_score, "#FBBF24"),
                ("Proportions",      proportion_score, "#A78BFA"),
            ]

            bars_html = ""
            for label, val, colour in bars:
                bars_html += f"""<div class="ba-bar-row">
                <div class="ba-bar-label">{label}</div>
                <div class="ba-bar-track">
                  <div class="ba-bar-fill" style="width:{val}%;background:{colour};"></div>
                </div>
                <div class="ba-bar-val">{val}</div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)  # close results zone

            # ── AI Report ─────────────────────────────────────────
            st.markdown('<div class="ba-zone">', unsafe_allow_html=True)
            st.markdown('<p class="ba-section">Your AI Coach Report</p>', unsafe_allow_html=True)

            with st.spinner("Generating personalised report with Groq LLM…"):
                report_data = {
                    "sex": ba_sex, "age": ba_age,
                    "height_cm": ba_height, "weight_kg": ba_weight,
                    "bmi": bmi, "bmi_cat": bmi_cat,
                    "bf_pct": bf_pct, "bf_cat": bf_cat,
                    "lean_kg": lean_kg, "fat_kg": fat_kg, "ffmi": ffmi,
                    "shoulder_width_cm": sh_w, "hip_width_cm": hip_w,
                    "femur_cm": femur_cm, "torso_cm": torso_cm,
                    "somatotype": soma_type,
                    "forward_head_cm": fhp,
                    "trunk_lean": trunk,
                    "shoulder_imbal": sh_im,
                    "shoulder_high": posture.get("shoulder_high_side", "N/A"),
                    "hip_imbal": hip_im,
                    "hip_high": posture.get("hip_high_side", "N/A"),
                    "knee_angle": posture.get("knee_angle_standing", "N/A"),
                }
                ai_report = ba_call_llm(report_data)

            st.markdown(
                f'<div class="ba-report-card">'
                f'<div class="ba-report-title">🤖 Personalised Report — Powered by Groq LLM</div>'
                f'<div class="ba-report-body">{ai_report}</div>'
                f'</div>',
                unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div style="text-align:center;padding:4rem 1rem;color:var(--sub);">
          <div style="font-size:3.5rem;margin-bottom:1rem;">🧍</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:600;
               color:var(--txt);margin-bottom:.5rem;">Upload a front photo to begin</div>
          <div style="font-size:.82rem;max-width:320px;margin:0 auto;line-height:1.7;">
          A full-length photo (head to toe) gives the best results.
          Side view adds postural depth analysis.
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close main


# TAB 4 — WORKOUT LIBRARY
# ════════════════════════════════════════════
with tab_library:

    # ── Workout data ─────────────────────────────────────────────
    EXERCISES = {
        # ── LOWER BODY ───────────────────────────────────────────
        "Goblet Squat": {
            "cat": "Lower Body", "muscles": ["Quads","Glutes","Core"],
            "sets": "3×12", "tip": "Elbows inside knees. Chest tall.",
            "supported": True,
            "svg": """<svg viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
              <!-- body deep squat -->
              <circle cx="36" cy="10" r="6" fill="#3B82F6"/>
              <line x1="36" y1="16" x2="36" y2="36" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <!-- arms holding dumbbell -->
              <line x1="36" y1="22" x2="26" y2="28" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <line x1="36" y1="22" x2="46" y2="28" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="22" y="26" width="28" height="5" rx="2.5" fill="#1D4ED8"/>
              <rect x="20" y="25" width="5" height="7" rx="2" fill="#3B82F6"/>
              <rect x="47" y="25" width="5" height="7" rx="2" fill="#3B82F6"/>
              <!-- legs in squat -->
              <line x1="36" y1="36" x2="22" y2="50" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="36" y1="36" x2="50" y2="50" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="22" y1="50" x2="18" y2="64" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="50" y1="50" x2="54" y2="64" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
            </svg>"""
        },
        "Romanian Deadlift": {
            "cat": "Lower Body", "muscles": ["Hamstrings","Glutes","Lower Back"],
            "sets": "3×10", "tip": "Hinge at hips. Soft knee. Bar close.",
            "supported": True,
            "svg": """<svg viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="36" cy="10" r="6" fill="#3B82F6"/>
              <!-- hinge torso -->
              <line x1="36" y1="16" x2="20" y2="38" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <!-- arms down holding bar -->
              <line x1="28" y1="27" x2="22" y2="46" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <line x1="28" y1="27" x2="34" y2="46" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="16" y="44" width="28" height="4" rx="2" fill="#1D4ED8"/>
              <circle cx="14" cy="46" r="5" fill="none" stroke="#3B82F6" stroke-width="2"/>
              <circle cx="46" cy="46" r="5" fill="none" stroke="#3B82F6" stroke-width="2"/>
              <!-- legs standing -->
              <line x1="20" y1="38" x2="18" y2="62" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="20" y1="38" x2="30" y2="62" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
            </svg>"""
        },
        "Bulgarian Split Squat": {
            "cat": "Lower Body", "muscles": ["Quads","Glutes","Balance"],
            "sets": "3×10 each", "tip": "Rear foot elevated. Front shin vertical.",
            "supported": False,
            "svg": """<svg viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="28" cy="10" r="6" fill="#3B82F6"/>
              <line x1="28" y1="16" x2="28" y2="36" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <!-- arms holding dumbbells -->
              <line x1="28" y1="24" x2="18" y2="30" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <line x1="28" y1="24" x2="38" y2="30" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="13" y="28" width="10" height="4" rx="2" fill="#1D4ED8"/>
              <rect x="33" y="28" width="10" height="4" rx="2" fill="#1D4ED8"/>
              <!-- front leg in lunge -->
              <line x1="28" y1="36" x2="18" y2="52" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="18" y1="52" x2="16" y2="64" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <!-- rear leg elevated -->
              <line x1="28" y1="36" x2="42" y2="44" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="42" y1="44" x2="52" y2="38" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <!-- bench -->
              <rect x="48" y="38" width="18" height="4" rx="2" fill="#1A1A30"/>
              <rect x="50" y="42" width="3" height="8" rx="1" fill="#1A1A30"/>
              <rect x="61" y="42" width="3" height="8" rx="1" fill="#1A1A30"/>
            </svg>"""
        },
        "Sumo Squat": {
            "cat": "Lower Body", "muscles": ["Inner Thighs","Glutes","Quads"],
            "sets": "3×15", "tip": "Wide stance. Toes out 45°. Knees track toes.",
            "supported": False,
            "svg": """<svg viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="36" cy="9" r="6" fill="#3B82F6"/>
              <line x1="36" y1="15" x2="36" y2="34" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <!-- dumbbell between legs -->
              <line x1="36" y1="28" x2="36" y2="44" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="30" y="42" width="12" height="5" rx="2.5" fill="#1D4ED8"/>
              <rect x="27" y="41" width="5" height="7" rx="2" fill="#3B82F6"/>
              <rect x="40" y="41" width="5" height="7" rx="2" fill="#3B82F6"/>
              <!-- wide squat legs -->
              <line x1="36" y1="34" x2="16" y2="52" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="36" y1="34" x2="56" y2="52" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="16" y1="52" x2="12" y2="64" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="56" y1="52" x2="60" y2="64" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
            </svg>"""
        },
        # ── PUSH ─────────────────────────────────────────────────
        "Shoulder Press": {
            "cat": "Push", "muscles": ["Shoulders","Triceps","Upper Chest"],
            "sets": "3×12", "tip": "Core tight. Don't flare ribs. Full lockout.",
            "supported": False,
            "svg": """<svg viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="36" cy="10" r="6" fill="#3B82F6"/>
              <line x1="36" y1="16" x2="36" y2="44" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <!-- arms pressed up -->
              <line x1="36" y1="22" x2="16" y2="14" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <line x1="36" y1="22" x2="56" y2="14" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="8" y="10" width="10" height="4" rx="2" fill="#1D4ED8"/>
              <rect x="5" y="9" width="5" height="6" rx="2" fill="#3B82F6"/>
              <rect x="17" y="9" width="5" height="6" rx="2" fill="#3B82F6"/>
              <rect x="54" y="10" width="10" height="4" rx="2" fill="#1D4ED8"/>
              <rect x="51" y="9" width="5" height="6" rx="2" fill="#3B82F6"/>
              <rect x="63" y="9" width="5" height="6" rx="2" fill="#3B82F6"/>
              <!-- legs -->
              <line x1="36" y1="44" x2="28" y2="62" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="36" y1="44" x2="44" y2="62" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
            </svg>"""
        },
        "Floor Press": {
            "cat": "Push", "muscles": ["Chest","Triceps","Shoulders"],
            "sets": "3×12", "tip": "Elbows 45° from torso. Full extension.",
            "supported": False,
            "svg": """<svg viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
              <!-- floor line -->
              <line x1="4" y1="58" x2="68" y2="58" stroke="#1A1A30" stroke-width="2"/>
              <!-- lying body -->
              <circle cx="14" cy="44" r="6" fill="#3B82F6"/>
              <line x1="20" y1="44" x2="58" y2="44" stroke="#60A5FA" stroke-width="3" stroke-linecap="round"/>
              <!-- arms pressing up -->
              <line x1="30" y1="44" x2="24" y2="28" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <line x1="44" y1="44" x2="50" y2="28" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="20" y="25" width="32" height="5" rx="2.5" fill="#1D4ED8"/>
              <rect x="15" y="24" width="7" height="7" rx="3" fill="#3B82F6"/>
              <rect x="50" y="24" width="7" height="7" rx="3" fill="#3B82F6"/>
              <!-- legs flat -->
              <line x1="58" y1="44" x2="64" y2="56" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="58" y1="44" x2="58" y2="56" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
            </svg>"""
        },
        "Lateral Raise": {
            "cat": "Push", "muscles": ["Side Delts","Traps"],
            "sets": "3×15", "tip": "Slight bend in elbow. Lead with elbows, not wrists.",
            "supported": False,
            "svg": """<svg viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="36" cy="10" r="6" fill="#3B82F6"/>
              <line x1="36" y1="16" x2="36" y2="44" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <!-- arms out to sides -->
              <line x1="36" y1="22" x2="10" y2="30" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <line x1="36" y1="22" x2="62" y2="30" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="4" y="27" width="8" height="4" rx="2" fill="#1D4ED8"/>
              <rect x="1" y="26" width="4" height="6" rx="2" fill="#3B82F6"/>
              <rect x="11" y="26" width="4" height="6" rx="2" fill="#3B82F6"/>
              <rect x="60" y="27" width="8" height="4" rx="2" fill="#1D4ED8"/>
              <rect x="57" y="26" width="4" height="6" rx="2" fill="#3B82F6"/>
              <rect x="67" y="26" width="4" height="6" rx="2" fill="#3B82F6"/>
              <line x1="36" y1="44" x2="28" y2="62" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="36" y1="44" x2="44" y2="62" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
            </svg>"""
        },
        # ── PULL ─────────────────────────────────────────────────
        "Bent-Over Row": {
            "cat": "Pull", "muscles": ["Lats","Rhomboids","Biceps"],
            "sets": "3×10", "tip": "Flat back. Pull elbows past torso. Squeeze.",
            "supported": True,
            "svg": """<svg viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="48" cy="10" r="6" fill="#3B82F6"/>
              <!-- hinged torso -->
              <line x1="48" y1="16" x2="20" y2="36" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <!-- arms rowing up -->
              <line x1="34" y1="26" x2="28" y2="18" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <line x1="34" y1="26" x2="40" y2="18" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="24" y="15" width="20" height="5" rx="2.5" fill="#1D4ED8"/>
              <rect x="20" y="14" width="6" height="7" rx="3" fill="#3B82F6"/>
              <rect x="42" y="14" width="6" height="7" rx="3" fill="#3B82F6"/>
              <!-- legs standing -->
              <line x1="20" y1="36" x2="16" y2="60" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="20" y1="36" x2="28" y2="60" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
            </svg>"""
        },
        "Bicep Curl": {
            "cat": "Pull", "muscles": ["Biceps","Forearms"],
            "sets": "3×12", "tip": "Elbows pinned. Full extension at bottom.",
            "supported": False,
            "svg": """<svg viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="36" cy="10" r="6" fill="#3B82F6"/>
              <line x1="36" y1="16" x2="36" y2="42" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <!-- one arm curled up -->
              <line x1="36" y1="26" x2="20" y2="30" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <line x1="20" y1="30" x2="16" y2="18" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="10" y="14" width="10" height="4" rx="2" fill="#1D4ED8"/>
              <rect x="7" y="13" width="5" height="6" rx="2" fill="#3B82F6"/>
              <rect x="19" y="13" width="5" height="6" rx="2" fill="#3B82F6"/>
              <!-- other arm down -->
              <line x1="36" y1="26" x2="52" y2="30" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <line x1="52" y1="30" x2="54" y2="46" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="50" y="44" width="10" height="4" rx="2" fill="#1D4ED8"/>
              <rect x="47" y="43" width="5" height="6" rx="2" fill="#3B82F6"/>
              <rect x="59" y="43" width="5" height="6" rx="2" fill="#3B82F6"/>
              <line x1="36" y1="42" x2="28" y2="62" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="36" y1="42" x2="44" y2="62" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
            </svg>"""
        },
        "Single-Arm Row": {
            "cat": "Pull", "muscles": ["Lats","Rear Delt","Biceps"],
            "sets": "3×12 each", "tip": "Brace on knee. Elbow close to body.",
            "supported": False,
            "svg": """<svg viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
              <!-- bench -->
              <rect x="38" y="36" width="28" height="6" rx="3" fill="#0C0C18" stroke="#1A1A30" stroke-width="1.5"/>
              <rect x="40" y="42" width="4" height="10" rx="2" fill="#0C0C18" stroke="#1A1A30" stroke-width="1"/>
              <rect x="60" y="42" width="4" height="10" rx="2" fill="#0C0C18" stroke="#1A1A30" stroke-width="1"/>
              <!-- body -->
              <circle cx="22" cy="22" r="6" fill="#3B82F6"/>
              <line x1="22" y1="28" x2="22" y2="46" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <!-- support arm on bench -->
              <line x1="22" y1="34" x2="40" y2="38" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <!-- rowing arm -->
              <line x1="22" y1="34" x2="14" y2="40" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <line x1="14" y1="40" x2="12" y2="28" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="6" y="24" width="10" height="4" rx="2" fill="#1D4ED8"/>
              <rect x="3" y="23" width="5" height="6" rx="2" fill="#3B82F6"/>
              <rect x="15" y="23" width="5" height="6" rx="2" fill="#3B82F6"/>
              <!-- legs -->
              <line x1="22" y1="46" x2="14" y2="62" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="22" y1="46" x2="32" y2="62" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
            </svg>"""
        },
        # ── CORE ─────────────────────────────────────────────────
        "Russian Twist": {
            "cat": "Core", "muscles": ["Obliques","Abs","Hip Flexors"],
            "sets": "3×20", "tip": "Lean back 45°. Rotate shoulder-to-shoulder.",
            "supported": False,
            "svg": """<svg viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
              <!-- floor -->
              <line x1="4" y1="62" x2="68" y2="62" stroke="#1A1A30" stroke-width="2"/>
              <!-- seated V position -->
              <circle cx="36" cy="26" r="6" fill="#3B82F6"/>
              <line x1="36" y1="32" x2="36" y2="50" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="36" y1="50" x2="20" y2="60" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="36" y1="50" x2="52" y2="60" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <!-- arms holding dumbbell, rotated to side -->
              <line x1="36" y1="38" x2="52" y2="32" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="52" y="29" width="12" height="4" rx="2" fill="#1D4ED8"/>
              <rect x="49" y="28" width="5" height="6" rx="2" fill="#3B82F6"/>
              <rect x="63" y="28" width="5" height="6" rx="2" fill="#3B82F6"/>
            </svg>"""
        },
        "Renegade Row": {
            "cat": "Core", "muscles": ["Core","Lats","Shoulders"],
            "sets": "3×8 each", "tip": "Plank position. No hip rotation.",
            "supported": False,
            "svg": """<svg viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
              <!-- floor -->
              <line x1="4" y1="62" x2="68" y2="62" stroke="#1A1A30" stroke-width="2"/>
              <!-- plank body -->
              <circle cx="58" cy="28" r="6" fill="#3B82F6"/>
              <line x1="52" y1="30" x2="18" y2="46" stroke="#60A5FA" stroke-width="3" stroke-linecap="round"/>
              <!-- support arm -->
              <line x1="40" y1="38" x2="34" y2="56" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="28" y="54" width="10" height="4" rx="2" fill="#1D4ED8"/>
              <!-- rowing arm up -->
              <line x1="26" y1="42" x2="22" y2="30" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="16" y="26" width="10" height="4" rx="2" fill="#1D4ED8"/>
              <rect x="13" y="25" width="5" height="6" rx="2" fill="#3B82F6"/>
              <!-- feet -->
              <line x1="18" y1="46" x2="10" y2="58" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="18" y1="46" x2="22" y2="58" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
            </svg>"""
        },
        # ── FULL BODY ────────────────────────────────────────────
        "Dumbbell Deadlift": {
            "cat": "Full Body", "muscles": ["Hamstrings","Glutes","Back","Traps"],
            "sets": "4×8", "tip": "Neutral spine throughout. Drive through heels.",
            "supported": True,
            "svg": """<svg viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="36" cy="10" r="6" fill="#3B82F6"/>
              <line x1="36" y1="16" x2="36" y2="40" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <!-- arms hanging with dumbbells -->
              <line x1="36" y1="24" x2="20" y2="32" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <line x1="20" y1="32" x2="18" y2="48" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="12" y="46" width="10" height="4" rx="2" fill="#1D4ED8"/>
              <rect x="9" y="45" width="5" height="6" rx="2" fill="#3B82F6"/>
              <rect x="21" y="45" width="5" height="6" rx="2" fill="#3B82F6"/>
              <line x1="36" y1="24" x2="52" y2="32" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <line x1="52" y1="32" x2="54" y2="48" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="50" y="46" width="10" height="4" rx="2" fill="#1D4ED8"/>
              <rect x="47" y="45" width="5" height="6" rx="2" fill="#3B82F6"/>
              <rect x="59" y="45" width="5" height="6" rx="2" fill="#3B82F6"/>
              <line x1="36" y1="40" x2="26" y2="62" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="36" y1="40" x2="46" y2="62" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
            </svg>"""
        },
        "Dumbbell Swing": {
            "cat": "Full Body", "muscles": ["Glutes","Hamstrings","Core","Shoulders"],
            "sets": "3×15", "tip": "Hip hinge drive, not squat. Hike then explode.",
            "supported": False,
            "svg": """<svg viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="36" cy="10" r="6" fill="#3B82F6"/>
              <!-- torso hinged -->
              <line x1="36" y1="16" x2="20" y2="34" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <!-- arms swinging forward with dumbbell -->
              <line x1="28" y1="25" x2="36" y2="8" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <line x1="36" y1="8" x2="44" y2="25" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
              <rect x="30" y="4" width="12" height="5" rx="2.5" fill="#1D4ED8"/>
              <rect x="27" y="3" width="5" height="7" rx="2" fill="#3B82F6"/>
              <rect x="40" y="3" width="5" height="7" rx="2" fill="#3B82F6"/>
              <line x1="20" y1="34" x2="14" y2="60" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="20" y1="34" x2="30" y2="60" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>
            </svg>"""
        },
    }

    # ── Weekly program ────────────────────────────────────────────
    PROGRAM = [
        {"day": "Mon", "label": "Day 1", "focus": "Lower Body Power", "icon": "🦵",
         "desc": "Quad & glute dominant. 3–4 sets, moderate weight.",
         "exercises": ["Goblet Squat","Romanian Deadlift","Sumo Squat","Bulgarian Split Squat"]},
        {"day": "Tue", "label": "Day 2", "focus": "Push", "icon": "💪",
         "desc": "Chest, shoulders, triceps. Control the descent.",
         "exercises": ["Shoulder Press","Floor Press","Lateral Raise"]},
        {"day": "Wed", "label": "Rest", "focus": "Rest & Recovery", "icon": "😴",
         "desc": "Active recovery — walk, stretch, foam roll.",
         "exercises": [], "rest": True},
        {"day": "Thu", "label": "Day 3", "focus": "Pull", "icon": "🏋️",
         "desc": "Back, biceps, rear delts. Focus on the squeeze.",
         "exercises": ["Bent-Over Row","Single-Arm Row","Bicep Curl"]},
        {"day": "Fri", "label": "Day 4", "focus": "Full Body", "icon": "🔥",
         "desc": "Compound movements. High effort, full range.",
         "exercises": ["Dumbbell Deadlift","Dumbbell Swing","Renegade Row","Russian Twist"]},
        {"day": "Sat", "label": "Day 5", "focus": "Active Rest", "icon": "🧘",
         "desc": "Yoga, light walk, mobility work.",
         "exercises": [], "rest": True},
        {"day": "Sun", "label": "Rest", "focus": "Full Rest", "icon": "💤",
         "desc": "Sleep, eat, recover.",
         "exercises": [], "rest": True},
    ]

    # ── UI ────────────────────────────────────────────────────────
    from datetime import datetime
    today_idx = datetime.now().weekday()  # Mon=0 … Sun=6

    st.markdown("""
    <div class="prog-header">
      <div class="prog-title">🏠 Home Dumbbell Program</div>
      <div class="prog-sub">7-day structured plan · All you need is a pair of dumbbells · <span style="color:#3B82F6">AI badge</span> = FORMate can analyse this exercise</div>
    </div>
    """, unsafe_allow_html=True)

    # Day strip
    day_chips = ""
    for i, d in enumerate(PROGRAM):
        active = "active" if i == today_idx else ""
        rest_cls = " rest" if d.get("rest") else ""
        day_chips += f'<div class="day-chip {active}{rest_cls}">{d["day"]}<br><span style="font-size:.5rem;opacity:.6">{d["label"]}</span></div>'
    st.markdown(f'<div class="day-strip">{day_chips}</div>', unsafe_allow_html=True)

    # Today's session
    today = PROGRAM[today_idx]
    rest_color = "#30304A" if today.get("rest") else "#1D4ED8"
    st.markdown(f"""
    <div class="day-focus">
      <div class="day-focus-icon">{today["icon"]}</div>
      <div>
        <div class="day-focus-text">Today — {today["focus"]}</div>
        <div class="day-focus-sub">{today["desc"]}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if today.get("rest"):
        st.markdown('<p style="color:var(--sub);font-size:.85rem;padding:1rem 0">Rest day — no workout scheduled. Focus on recovery.</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="cat-label">Today\'s Exercises — {today["focus"]}</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(today["exercises"]), 4))
        for i, ex_name in enumerate(today["exercises"]):
            ex = EXERCISES[ex_name]
            sup = " supported" if ex["supported"] else ""
            tags = "".join(f'<span class="ex-tag">{m}</span>' for m in ex["muscles"][:2])
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div class="ex-card{sup}">
                  <div class="ex-pic">{ex["svg"]}</div>
                  <div class="ex-info">
                    <div class="ex-name">{ex_name}</div>
                    <div class="ex-meta">{tags}</div>
                    <div style="font-size:.6rem;color:var(--p3);margin-top:.3rem">{ex["sets"]}</div>
                    <div style="font-size:.58rem;color:var(--sub);margin-top:.2rem;line-height:1.4">{ex["tip"]}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # Full library by category
    st.markdown('<div class="cat-label" style="margin-top:2rem">Full Exercise Library</div>', unsafe_allow_html=True)
    cats = {}
    for name, ex in EXERCISES.items():
        cats.setdefault(ex["cat"], []).append((name, ex))

    for cat, exercises in cats.items():
        st.markdown(f'<div class="cat-label">{cat}</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(exercises), 4))
        for i, (ex_name, ex) in enumerate(exercises):
            sup = " supported" if ex["supported"] else ""
            tags = "".join(f'<span class="ex-tag">{m}</span>' for m in ex["muscles"][:2])
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div class="ex-card{sup}">
                  <div class="ex-pic">{ex["svg"]}</div>
                  <div class="ex-info">
                    <div class="ex-name">{ex_name}</div>
                    <div class="ex-meta">{tags}</div>
                    <div style="font-size:.6rem;color:var(--p3);margin-top:.3rem">{ex["sets"]}</div>
                    <div style="font-size:.58rem;color:var(--sub);margin-top:.2rem;line-height:1.4">{ex["tip"]}</div>
                  </div>
                </div>""", unsafe_allow_html=True)
