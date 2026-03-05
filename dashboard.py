# dashboard.py — Violence Detection Dashboard v5
# Fixed: pred.txt drives status/analytics/report, clear-all button,
#        better visible UI, login centered properly, no confusing label map
# Fixed: StreamlitDuplicateElementKey errors (all widget keys are now unique)

import os, io, re, time, json, shutil, hashlib, zipfile, subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
@dataclass
class CFG:
    APP_TITLE: str           = "Violence Detection Dashboard"
    OUTPUT_DIR: str          = "outputs_dashboard"
    DEFAULT_FPS: int         = 25
    MAX_FRAMES: int          = 180
    THRESH_VIOLENCE: float   = 0.70
    THRESH_SUSPICIOUS: float = 0.45

os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
UPLOAD_ROOT = Path(CFG.OUTPUT_DIR) / "uploads"
LOGS_ROOT   = Path(CFG.OUTPUT_DIR) / "logs"
USERS_FILE  = Path(CFG.OUTPUT_DIR) / "users.json"
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "hockeyfight": ["Fight", "NonFight"],
    "rwf":         ["Fight", "NonFight"],
}


# ──────────────────────────────────────────
# Auth
# ──────────────────────────────────────────
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def load_users():
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE) as f: return json.load(f)
        except: pass
    return {}

def save_users(u):
    with open(USERS_FILE,"w") as f: json.dump(u,f,indent=2)

def try_login(u,p):
    return load_users().get(u) == hash_pw(p)

def try_register(u,p):
    if not u or not p: return False,"Username and password required."
    if len(p)<4: return False,"Password must be at least 4 characters."
    users = load_users()
    if u in users: return False,"Username already exists."
    users[u] = hash_pw(p); save_users(users)
    return True,"Account created! You can now log in."


# ──────────────────────────────────────────
# Core utils
# ──────────────────────────────────────────
def confidence_to_status(conf_str: str) -> str:
    try:
        c = float(conf_str)
        if c >= CFG.THRESH_VIOLENCE:   return "ALERT"
        if c >= CFG.THRESH_SUSPICIOUS: return "SUSPICIOUS"
        return "NORMAL"
    except: return "UNKNOWN"

def pred_label_to_status(pred_label: str) -> str:
    if "fight" in str(pred_label).lower() and "non" not in str(pred_label).lower():
        return "ALERT"
    return "NORMAL"

def color_from_status(s):
    return {"ALERT":"🔴","SUSPICIOUS":"🟡","NORMAL":"🟢","UNKNOWN":"⚪"}.get(s,"⚪")

def fmt_time(sec):
    if sec is None: return "N/A"
    try: return f"{int(float(sec)//60):02d}:{int(float(sec)%60):02d}"
    except: return str(sec)

def to_rgb(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def resize_keep(frame, w=640):
    h,ww = frame.shape[:2]
    if ww==w: return frame
    return cv2.resize(frame,(w,int(h*(w/ww))),interpolation=cv2.INTER_AREA)

def read_video_frames(path, max_frames=CFG.MAX_FRAMES):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise FileNotFoundError(f"Cannot open: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or CFG.DEFAULT_FPS
    frames,i = [],0
    while True:
        ret,f = cap.read()
        if not ret or i>=max_frames: break
        frames.append(f); i+=1
    cap.release()
    return frames, float(fps)

def fake_model_scores(frames, fps):
    """Placeholder motion heuristic — replace with real R3D-18."""
    ps,prev = [],None
    for f in frames:
        g = cv2.GaussianBlur(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(5,5),0)
        if prev is None: ps.append(0.10)
        else: ps.append(float(np.clip(float(np.mean(cv2.absdiff(g,prev)))/255*2.8,0,1)))
        prev=g
    return np.array(ps,dtype=np.float32)

def scores_from_pred(pred: dict, n_frames: int, fps: float):
    conf = 0.5
    try: conf = float(pred.get("confidence", 0.5))
    except: pass

    onset_frame = 0
    try: onset_frame = int(pred.get("onset_frame", 0))
    except: pass

    pred_label = pred.get("pred_label","").lower()
    is_fight   = "fight" in pred_label and "non" not in pred_label

    scores = np.zeros(n_frames, dtype=np.float32)
    if is_fight:
        for i in range(n_frames):
            if i < onset_frame:
                scores[i] = max(0.05, conf * 0.1)
            else:
                ramp = min(1.0, (i - onset_frame) / max(1, fps))
                scores[i] = float(np.clip(conf * (0.7 + 0.3*ramp), 0, 1))
    else:
        scores = np.clip(np.random.normal(0.15, 0.05, n_frames), 0, 0.4).astype(np.float32)
        scores[0] = 0.10

    return scores

def detect_fight_start(scores, fps, thr=None):
    thr = thr or CFG.THRESH_VIOLENCE
    N = max(3,int(0.2*fps)); run=0
    for i,a in enumerate(scores>=thr):
        run = run+1 if a else 0
        if run>=N: sf=i-(N-1); return sf, sf/fps
    return None, None

def ffmpeg_ok():
    try:
        subprocess.run(["ffmpeg","-version"],stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,timeout=5)
        return True
    except: return False

def make_web_preview(src,dst):
    dst = Path(dst); src = Path(src)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() and dst.stat().st_mtime>=src.stat().st_mtime: return True
    if ffmpeg_ok():
        try:
            subprocess.run(["ffmpeg","-y","-i",str(src),"-c:v","libx264",
                            "-pix_fmt","yuv420p","-preset","veryfast","-crf","23",
                            "-c:a","aac","-b:a","128k",str(dst)],
                           check=True,stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,timeout=120)
            return True
        except: pass
    try:
        cap=cv2.VideoCapture(str(src))
        if not cap.isOpened(): return False
        fps=cap.get(cv2.CAP_PROP_FPS) or CFG.DEFAULT_FPS
        w,h=int(cap.get(3)),int(cap.get(4))
        out=cv2.VideoWriter(str(dst),cv2.VideoWriter_fourcc(*"mp4v"),float(fps),(w,h))
        while True:
            ret,f=cap.read()
            if not ret: break
            out.write(f)
        cap.release(); out.release()
        return dst.exists()
    except: return False

def describe_onset(pred: dict) -> str:
    onset_t = pred.get("onset_time","?")
    onset_f = pred.get("onset_frame","?")
    spike   = pred.get("spike_delta","?")
    thr     = pred.get("onset_threshold","?")
    return (f"Fight onset detected at frame {onset_f} ({onset_t}). "
            f"Onset threshold: {thr}, spike delta: {spike}.")


# ──────────────────────────────────────────
# Plots
# ──────────────────────────────────────────
def make_timeline_plot(scores, fps, pred=None):
    t = np.arange(len(scores))/fps
    fig = plt.figure(figsize=(6,2.5))
    ax  = fig.add_subplot(111)

    is_fight = False
    if pred:
        lbl = pred.get("pred_label","").lower()
        is_fight = "fight" in lbl and "non" not in lbl
    color = "#e05252" if is_fight else "#52e08a"

    ax.plot(t, scores, color=color, linewidth=1.8)
    ax.axhline(CFG.THRESH_SUSPICIOUS, linestyle="--", color="orange", linewidth=1,
               label=f"Suspicious ({CFG.THRESH_SUSPICIOUS})")
    ax.axhline(CFG.THRESH_VIOLENCE,   linestyle="--", color="red",    linewidth=1,
               label=f"Violence ({CFG.THRESH_VIOLENCE})")

    if pred:
        try:
            onset_f = int(pred.get("onset_frame",0))
            onset_t_val = onset_f / fps
            if 0 < onset_t_val < t[-1]:
                ax.axvline(onset_t_val, color="yellow", linewidth=1.5,
                           linestyle=":", label=f"Onset ({pred.get('onset_time','?')})")
        except: pass

    ax.set_xlabel("Time (s)",fontsize=9); ax.set_ylabel("Prob",fontsize=9)
    ax.set_title("Violence Probability Timeline",fontsize=10)
    ax.legend(fontsize=7); plt.tight_layout(); return fig

def make_hist_plot(scores):
    fig=plt.figure(figsize=(4,2.5))
    plt.hist(scores,bins=20,color="#5271e0",edgecolor="white",linewidth=0.3)
    plt.xlabel("Probability",fontsize=9); plt.ylabel("Count",fontsize=9)
    plt.title("Confidence Distribution",fontsize=10); plt.tight_layout(); return fig

def make_confusion_matrix(records):
    labels=["Fight","NonFight"]
    cm=np.zeros((2,2),dtype=int)
    lmap={"fight":0,"nonfight":1,"Fight":0,"NonFight":1}
    for r in records:
        t=lmap.get(r.get("true_label",""),-1)
        p=lmap.get(r.get("pred_label",""),-1)
        if t>=0 and p>=0: cm[t][p]+=1
    fig,ax=plt.subplots(figsize=(4,3))
    im=ax.imshow(cm,interpolation="nearest",cmap=plt.cm.Blues)
    plt.colorbar(im,ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i][j]),ha="center",va="center",
                    color="white" if cm[i][j]>cm.max()/2 else "black",
                    fontsize=14,fontweight="bold")
    plt.tight_layout(); return fig,cm


# ──────────────────────────────────────────
# pred.txt
# ──────────────────────────────────────────
def parse_pred_txt(path) -> dict:
    out={}
    try:
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k,v=line.strip().split(":",1)
                    out[k.strip()]=v.strip()
    except: pass
    return out

def render_pred_card(pred: dict):
    if not pred: st.info("No pred.txt found."); return
    true_lbl = pred.get("true_label","?")
    pred_lbl = pred.get("pred_label","?")
    correct  = pred.get("correct","?")
    conf     = pred.get("confidence","?")
    dataset  = pred.get("dataset","?")
    o_time   = pred.get("onset_time","?")
    o_frame  = pred.get("onset_frame","?")
    tot_fr   = pred.get("total_frames","?")
    ok_emoji   = "✅" if str(correct).lower()=="true" else "❌"
    pred_color = "🔴" if ("fight" in str(pred_lbl).lower() and "non" not in str(pred_lbl).lower()) else "🟢"
    c1,c2,c3,c4=st.columns(4)
    c1.metric("True Label",true_lbl)
    c2.metric("Predicted",f"{pred_color} {pred_lbl}")
    c3.metric("Correct?",f"{ok_emoji} {correct}")
    try:    c4.metric("Confidence",f"{float(conf):.1%}")
    except: c4.metric("Confidence",conf)
    c5,c6,c7,c8=st.columns(4)
    c5.metric("Dataset",dataset); c6.metric("Onset Frame",o_frame)
    c7.metric("Onset Time",o_time); c8.metric("Total Frames",tot_fr)
    with st.expander("📋 Full pred.txt details",expanded=False):
        ca,cb=st.columns(2)
        with ca:
            for k in ["model_path","model_val_acc","probs","pred_class"]:
                st.markdown(f"**{k}:** `{pred.get(k,'?')}`")
        with cb:
            for k in ["window_size","window_stride","target_layer","img_size","cam_target_class","spike_delta","onset_threshold"]:
                st.markdown(f"**{k}:** `{pred.get(k,'?')}`")


# ──────────────────────────────────────────
# Folder helpers
# ──────────────────────────────────────────
def class_root(ds,cls): return UPLOAD_ROOT/ds/cls

def list_video_folders(ds,cls):
    root=class_root(ds,cls)
    if not root.exists(): return []
    f=[x for x in root.iterdir() if x.is_dir() and not x.name.startswith("_")]
    f.sort(key=lambda x:x.name.lower()); return f

def find_file(folder,pattern):
    m=list(folder.glob(pattern)); return m[0] if m else None

def get_files(folder) -> dict:
    files={}
    orig=find_file(folder,"*original*.mp4")
    if not orig:
        cands=[f for f in folder.glob("*.mp4") if "gradcam" not in f.name.lower()]
        if len(cands)==1: orig=cands[0]
    if orig: files["original"]=orig
    for f in sorted(folder.glob("*.mp4")):
        n=f.name.lower()
        if "gradcampp" in n or "gradcam++" in n: files["gradcampp"]=f
        elif "gradcam" in n: files["gradcam"]=f
    for key,pat in [("raw_grid","raw_grid.png"),("gradcam_grid","gradcam_grid.png"),
                    ("gradcampp_grid","gradcampp_grid.png"),("pred","pred.txt")]:
        f=find_file(folder,pat)
        if f: files[key]=f
    return files

def get_all_pred_records():
    records=[]
    for ds in DATASETS:
        for cls in DATASETS[ds]:
            for folder in list_video_folders(ds,cls):
                files=get_files(folder)
                if "pred" in files:
                    p=parse_pred_txt(files["pred"])
                    p["_dataset"]=ds; p["_class"]=cls; p["_folder"]=folder.name
                    records.append(p)
    return records

def clear_all_uploads():
    if UPLOAD_ROOT.exists():
        shutil.rmtree(UPLOAD_ROOT)
    for ds in DATASETS:
        for cls in DATASETS[ds]:
            class_root(ds,cls).mkdir(parents=True,exist_ok=True)


# ──────────────────────────────────────────
# PDF Export
# ──────────────────────────────────────────
def generate_pdf_report(pred, scores, fps, folder_name) -> bytes:
    fig=plt.figure(figsize=(11,8.5))
    fig.patch.set_facecolor("#0e1117")
    gs=gridspec.GridSpec(3,3,figure=fig,hspace=0.55,wspace=0.4)
    title_ax=fig.add_subplot(gs[0,:]); title_ax.axis("off"); title_ax.set_facecolor("#0e1117")
    title_ax.text(0.5,0.75,"Violence Detection — Incident Report",ha="center",va="center",
                  fontsize=18,fontweight="bold",color="white")
    title_ax.text(0.5,0.3,f"Video: {folder_name}   |   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                  ha="center",va="center",fontsize=10,color="#aaaaaa")
    ax_tl=fig.add_subplot(gs[1,0:2]); ax_tl.set_facecolor("#1a1a2e")
    t=np.arange(len(scores))/fps
    is_fight = "fight" in pred.get("pred_label","").lower() and "non" not in pred.get("pred_label","").lower()
    ax_tl.plot(t,scores,color="#e05252" if is_fight else "#52e08a",linewidth=1.5)
    ax_tl.axhline(CFG.THRESH_VIOLENCE,linestyle="--",color="red",linewidth=1)
    ax_tl.set_xlabel("Time (s)",color="white",fontsize=8)
    ax_tl.set_ylabel("Probability",color="white",fontsize=8)
    ax_tl.set_title("Violence Timeline",color="white",fontsize=10)
    ax_tl.tick_params(colors="white"); ax_tl.spines[:].set_color("#333355")
    ax_h=fig.add_subplot(gs[1,2]); ax_h.set_facecolor("#1a1a2e")
    ax_h.hist(scores,bins=15,color="#5271e0")
    ax_h.set_title("Confidence Dist.",color="white",fontsize=10)
    ax_h.tick_params(colors="white"); ax_h.spines[:].set_color("#333355")
    ax_info=fig.add_subplot(gs[2,:]); ax_info.axis("off"); ax_info.set_facecolor("#0e1117")
    lines=[
        f"STATUS: {'🔴 FIGHT DETECTED' if is_fight else '🟢 NO FIGHT'}   |   Confidence: {pred.get('confidence','?')}",
        f"Dataset: {pred.get('dataset','?')}   |   True: {pred.get('true_label','?')}   |   Predicted: {pred.get('pred_label','?')}   |   Correct: {pred.get('correct','?')}",
        f"Onset Frame: {pred.get('onset_frame','?')}   |   Onset Time: {pred.get('onset_time','?')}   |   Total Frames: {pred.get('total_frames','?')}",
        f"Model: {pred.get('model_path','?')}   |   Val Acc: {pred.get('model_val_acc','?')}",
    ]
    for i,line in enumerate(lines):
        ax_info.text(0.02,0.88-i*0.22,line,transform=ax_info.transAxes,fontsize=9,
                     color="#ff6666" if i==0 and is_fight else ("white" if i>0 else "#66ff66"),
                     fontweight="bold" if i==0 else "normal")
    buf=io.BytesIO()
    plt.savefig(buf,format="pdf",facecolor=fig.get_facecolor(),bbox_inches="tight")
    plt.close(fig); buf.seek(0); return buf.read()


# ──────────────────────────────────────────
# Fight Alert
# ──────────────────────────────────────────
def show_fight_alert(folder_name, confidence):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#ff0000 0%,#8b0000 100%);
                border:2px solid #ff4444;border-radius:12px;padding:20px 28px;
                margin:12px 0;box-shadow:0 0 28px rgba(255,0,0,0.6);">
        <div style="font-size:26px;font-weight:900;color:white;letter-spacing:2px;">
            🚨 FIGHT DETECTED — ALERT
        </div>
        <div style="font-size:15px;color:#ffcccc;margin-top:8px;">
            Video: <b>{folder_name}</b> &nbsp;|&nbsp; Confidence: <b>{confidence}</b>
        </div>
        <div style="font-size:13px;color:#ffaaaa;margin-top:4px;">
            ⚠️ Immediate review recommended
        </div>
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# ZIP upload
# ──────────────────────────────────────────
def extract_zip_to_uploads(zip_bytes, dataset, cls):
    n_folders,n_files=0,0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names=zf.namelist()
        folder_map={}
        for name in names:
            parts=Path(name).parts
            if len(parts)>=2:
                fk=parts[-2]
                folder_map.setdefault(fk,[]).append(name)
            elif len(parts)==1 and not name.endswith("/"):
                folder_map.setdefault("misc",[]).append(name)
        for fn,flist in folder_map.items():
            dest=class_root(dataset,cls)/fn
            dest.mkdir(parents=True,exist_ok=True); n_folders+=1
            for zp in flist:
                if zp.endswith("/"): continue
                try:
                    with open(dest/Path(zp).name,"wb") as f: f.write(zf.read(zp))
                    n_files+=1
                except: pass
    return n_folders,n_files


# ──────────────────────────────────────────
# Session state
# ──────────────────────────────────────────
def init_state():
    defaults={
        "logged_in":False,"username":"",
        "active_pred":{}, "active_scores":None, "active_fps":None,
        "active_frames":None, "active_folder_name":"",
        "active_video_path":None, "active_dataset":"", "active_class":"",
        "run_id":datetime.now().strftime("run_%Y%m%d_%H%M%S"),
        "_confirm_clear":False,
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

init_state()
for ds in DATASETS:
    for cls in DATASETS[ds]:
        class_root(ds,cls).mkdir(parents=True,exist_ok=True)


# ──────────────────────────────────────────
# Page config & CSS
# ──────────────────────────────────────────
st.set_page_config(page_title=CFG.APP_TITLE, layout="wide", page_icon="🎯")

st.markdown("""
<style>
body, .stApp { background-color: #0e1117; }
[data-testid="stMetricValue"] { font-size:18px !important; font-weight:700 !important; }
[data-testid="stMetricLabel"] { font-size:12px !important; color:#aaa !important; }
.block-container { padding-top:1rem !important; max-width:1400px !important; }
h1  { font-size:1.6rem !important; font-weight:900 !important; color:#ffffff !important; }
h2  { font-size:1.2rem !important; font-weight:700 !important; color:#dddddd !important; }
h3  { font-size:1.0rem !important; font-weight:600 !important; color:#cccccc !important; }
.stTabs [data-baseweb="tab-list"] { gap:2px; background:#1a1a2e; border-radius:8px; padding:4px; }
.stTabs [data-baseweb="tab"] {
    font-size:12px; padding:6px 10px; border-radius:6px;
    color:#aaaaaa !important; background:transparent !important;
}
.stTabs [aria-selected="true"] {
    background:#e05252 !important; color:white !important; font-weight:700 !important;
}
div[data-testid="stImage"] img { border-radius:8px; border:1px solid #2d2d4e; }
.stButton>button[kind="primary"] {
    background:#e05252 !important; border:none !important;
    color:white !important; font-weight:700 !important; border-radius:8px !important;
}
.stButton>button { border-radius:8px !important; }
div[data-testid="stAlert"] { border-radius:8px !important; }
.login-container {
    max-width:420px; margin:0 auto; padding:32px;
    background:#1a1a2e; border-radius:16px;
    border:1px solid #2d2d4e;
}
hr { margin:0.6rem 0 !important; border-color:#2d2d4e !important; }
[data-testid="metric-container"] {
    background:#1a1a2e; border-radius:10px;
    padding:12px 16px !important; border:1px solid #2d2d4e;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# LOGIN WALL
# ══════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""
    <div style="text-align:center;padding:60px 0 20px 0;">
        <div style="font-size:52px;">🎯</div>
        <div style="font-size:2.2rem;font-weight:900;color:white;margin-top:8px;">
            Violence Detection Dashboard
        </div>
        <div style="color:#888;margin-top:8px;font-size:1rem;">
            Sign in or create an account to continue
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        auth_tabs = st.tabs(["🔐  Login", "📝  Create Account"])

        with auth_tabs[0]:
            st.markdown("#### Welcome back")
            lu = st.text_input("Username", key="login_u", placeholder="Enter username")
            lp = st.text_input("Password", type="password", key="login_p", placeholder="Enter password")
            if st.button("Login →", type="primary", use_container_width=True, key="login_btn"):
                if try_login(lu.strip(), lp):
                    st.session_state.logged_in = True
                    st.session_state.username  = lu.strip()
                    st.rerun()
                else:
                    st.error("❌ Incorrect username or password.")

        with auth_tabs[1]:
            st.markdown("#### Create your account")
            ru  = st.text_input("Username",         key="reg_u",  placeholder="Choose a username")
            rp  = st.text_input("Password",         type="password", key="reg_p",  placeholder="Choose a password")
            rp2 = st.text_input("Confirm Password", type="password", key="reg_p2", placeholder="Repeat password")
            if st.button("Create Account →", type="primary", use_container_width=True, key="reg_btn"):
                if rp != rp2:
                    st.error("❌ Passwords do not match.")
                else:
                    ok,msg = try_register(ru.strip(), rp)
                    st.success(f"✅ {msg}") if ok else st.error(f"❌ {msg}")

        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════
# HEADER (logged in)
# ══════════════════════════════════════════
hdr_l, hdr_r = st.columns([5, 2])
with hdr_l:
    st.title(f"🎯 {CFG.APP_TITLE}")
with hdr_r:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    u_col, btn_col = st.columns([1.2, 1])
    with u_col:
        st.markdown(f"""
        <div style='background:#1a1a2e;border:1px solid #2d2d4e;border-radius:8px;
                    padding:8px 14px;text-align:center;margin-top:2px;'>
            👤 <b style='color:white;font-size:13px;'>{st.session_state.username}</b>
        </div>""", unsafe_allow_html=True)
    with btn_col:
        if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

st.warning("⚠️ **Cloud mode:** Uploads reset on restart. Run locally for permanent storage.", icon="⚠️")

# ── Active folder status bar ──
if st.session_state.active_folder_name:
    pred = st.session_state.active_pred
    is_fight = ("fight" in pred.get("pred_label","").lower()
                and "non" not in pred.get("pred_label","").lower())
    status_color = "#ff4444" if is_fight else "#44ff88"
    status_text  = "🔴 FIGHT" if is_fight else "🟢 NORMAL"
    conf = pred.get("confidence","?")
    st.markdown(f"""
    <div style="background:#1a1a2e;border:1px solid #2d2d4e;border-left:4px solid {status_color};
                border-radius:8px;padding:10px 18px;margin:6px 0;display:flex;
                align-items:center;gap:20px;">
        <span style="color:#888;font-size:12px;">ACTIVE FOLDER</span>
        <span style="color:white;font-weight:700;">{st.session_state.active_folder_name}</span>
        <span style="color:#888;font-size:12px;">|</span>
        <span style="color:#888;font-size:12px;">DATASET</span>
        <span style="color:white;font-weight:600;">{st.session_state.active_dataset} / {st.session_state.active_class}</span>
        <span style="color:#888;font-size:12px;">|</span>
        <span style="color:#888;font-size:12px;">RESULT</span>
        <span style="color:{status_color};font-weight:700;">{status_text}</span>
        <span style="color:#888;font-size:12px;">|</span>
        <span style="color:#888;font-size:12px;">CONFIDENCE</span>
        <span style="color:white;font-weight:600;">{conf}</span>
    </div>
    """, unsafe_allow_html=True)

tabs = st.tabs([
    "📁 Video Explorer",
    "🔍 Search & Filter",
    "⚖️ Video Comparison",
    "🎥 Control Room",
    "🟦 Multi-Camera",
    "📤 Upload Manager",
    "📈 Analytics",
    "📊 Dataset Stats",
    "❌ FP/FN Browser",
    "🎛️ Threshold Tuner",
    "🧩 Confusion Matrix",
    "🧾 Incident Report",
])


# ══════════════════════════════════════════
# TAB 1 — VIDEO EXPLORER
# ══════════════════════════════════════════
with tabs[0]:
    st.subheader("📁 Video Explorer")

    s1,s2,s3 = st.columns([1,1,2])
    with s1: sel_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="ex_ds")
    with s2: sel_cls = st.selectbox("Class", DATASETS[sel_ds], key="ex_cls")
    with s3:
        vfolders = list_video_folders(sel_ds, sel_cls)
        if not vfolders:
            st.info(f"No folders yet for **{sel_ds}/{sel_cls}**. Use Upload Manager ➡️")
            sel_folder = None
        else:
            sel_folder = st.selectbox(
                f"Video folder  ({len(vfolders)} available)",
                [f.name for f in vfolders], key="ex_folder"
            )

    if st.button("🔍 Analyze this folder", type="primary",
                 disabled=(sel_folder is None), key="analyze_btn"):
        folder_path = class_root(sel_ds, sel_cls) / sel_folder
        files = get_files(folder_path)
        pred  = parse_pred_txt(files["pred"]) if "pred" in files else {}

        frames, fps = [], float(CFG.DEFAULT_FPS)
        if "original" in files:
            try:
                with st.spinner("Loading video..."):
                    frames, fps = read_video_frames(files["original"], max_frames=140)
                    frames = [resize_keep(f,720) for f in frames]
            except: pass

        n = len(frames) if frames else 100
        scores = scores_from_pred(pred, n, fps)

        st.session_state.active_pred        = pred
        st.session_state.active_scores      = scores
        st.session_state.active_fps         = fps
        st.session_state.active_frames      = frames
        st.session_state.active_folder_name = sel_folder
        st.session_state.active_video_path  = str(files.get("original",""))
        st.session_state.active_dataset     = sel_ds
        st.session_state.active_class       = sel_cls
        st.session_state["_active_files"]   = {k:str(v) for k,v in files.items()}
        st.rerun()

    # ── Results area ──
    if st.session_state.active_folder_name:
        pred        = st.session_state.active_pred
        scores      = st.session_state.active_scores
        fps         = st.session_state.active_fps
        folder_name = st.session_state.active_folder_name
        files       = {k:Path(v) for k,v in st.session_state.get("_active_files",{}).items()}

        is_fight = ("fight" in pred.get("pred_label","").lower()
                    and "non" not in pred.get("pred_label","").lower())

        if is_fight:
            show_fight_alert(folder_name, pred.get("confidence","?"))

        st.markdown("---")
        st.markdown("#### 📋 Prediction Results")
        render_pred_card(pred)

        st.markdown("---")
        st.markdown("#### 🎬 Videos")
        vid_keys   = [k for k in ["original","gradcam","gradcampp"] if k in files]
        vid_labels = {"original":"📹 Original","gradcam":"🔥 Grad-CAM","gradcampp":"🔥 Grad-CAM++"}
        if vid_keys:
            vcols = st.columns(len(vid_keys))
            for i,vk in enumerate(vid_keys):
                with vcols[i]:
                    st.markdown(f"**{vid_labels[vk]}**")
                    vpath = files[vk]
                    fp    = vpath.parent / f"_preview_{vk}.mp4"
                    if not fp.exists():
                        with st.spinner("Converting..."): make_web_preview(vpath,fp)
                    st.video(str(fp)) if fp.exists() else st.warning("Preview unavailable")
        else:
            st.info("No .mp4 files found.")

        st.markdown("---")
        st.markdown("#### 🖼️ Frame Grids")
        img_keys   = [k for k in ["raw_grid","gradcam_grid","gradcampp_grid"] if k in files]
        img_labels = {"raw_grid":"📷 Raw Frames","gradcam_grid":"🌡️ Grad-CAM","gradcampp_grid":"🌡️ Grad-CAM++"}
        if img_keys:
            icols = st.columns(len(img_keys))
            for i,ik in enumerate(img_keys):
                with icols[i]:
                    st.markdown(f"**{img_labels[ik]}**")
                    st.image(str(files[ik]),use_container_width=True)
        else:
            st.info("No grid images found.")

        if "gradcam_grid" in files and "gradcampp_grid" in files:
            st.markdown("---")
            st.markdown("#### 🔬 Grad-CAM vs Grad-CAM++ Comparison")
            gc1,gc2 = st.columns(2)
            with gc1:
                st.markdown("**Grad-CAM**")
                st.image(str(files["gradcam_grid"]),use_container_width=True)
            with gc2:
                st.markdown("**Grad-CAM++**")
                st.image(str(files["gradcampp_grid"]),use_container_width=True)
            st.caption("Grad-CAM++ generally produces sharper, more precise activation maps.")

        if scores is not None and fps:
            st.markdown("---")
            st.markdown("#### 📊 Quick Analysis")
            _, start_sec = detect_fight_start(scores, fps)
            peak = float(np.max(scores))
            stat = pred_label_to_status(pred.get("pred_label",""))
            conf_str = pred.get("confidence","?")

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Status",      f"{color_from_status(stat)} {stat}")
            m2.metric("Confidence",  conf_str)
            m3.metric("Onset Time",  pred.get("onset_time","N/A"))
            m4.metric("Total Frames",pred.get("total_frames","?"))

            c1,c2 = st.columns(2)
            with c1: st.pyplot(make_timeline_plot(scores,fps,pred), clear_figure=True, use_container_width=False)
            with c2: st.pyplot(make_hist_plot(scores),               clear_figure=True, use_container_width=False)

        st.markdown("---")
        st.markdown("#### 📄 Export PDF Report")
        if st.button("📄 Generate PDF Report", key="ex_pdf_btn"):
            with st.spinner("Generating PDF..."):
                pdf_bytes = generate_pdf_report(pred, scores if scores is not None else np.zeros(10), fps or 25.0, folder_name)
            st.download_button("⬇️ Download PDF",data=pdf_bytes,
                               file_name=f"report_{folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                               mime="application/pdf")
    else:
        st.info("👆 Select a dataset, class, and folder above — then click **Analyze this folder**.")


# ══════════════════════════════════════════
# TAB 2 — SEARCH & FILTER
# ══════════════════════════════════════════
with tabs[1]:
    st.subheader("🔍 Search & Filter Videos")
    records = get_all_pred_records()
    if not records:
        st.info("No pred.txt files found. Upload some folders first.")
    else:
        df_all = pd.DataFrame(records)
        fc1,fc2,fc3,fc4 = st.columns(4)
        with fc1: sq   = st.text_input("Search folder name", placeholder="e.g. fi1", key="sf_search")
        with fc2: dsf  = st.selectbox("Dataset",["All"]+list(DATASETS.keys()),key="sf_ds")
        with fc3: clsf = st.selectbox("Class",["All","Fight","NonFight"],key="sf_cls")
        with fc4: corf = st.selectbox("Correct?",["All","True","False"],key="sf_cor")
        df=df_all.copy()
        if sq:   df=df[df["_folder"].str.contains(sq,case=False,na=False)]
        if dsf!="All":  df=df[df["_dataset"]==dsf]
        if clsf!="All": df=df[df["_class"]==clsf]
        if corf!="All": df=df[df["correct"].str.lower()==corf.lower()]
        st.markdown(f"**{len(df)} results**")
        show=["_folder","_dataset","_class","true_label","pred_label","correct","confidence","onset_time"]
        show=[c for c in show if c in df.columns]
        st.dataframe(df[show].rename(columns={"_folder":"Folder","_dataset":"Dataset","_class":"Class"}),
                     use_container_width=True,height=350)
        st.download_button("⬇️ Download CSV",data=df[show].to_csv(index=False).encode(),
                           file_name="filtered.csv",mime="text/csv",key="sf_csv")


# ══════════════════════════════════════════
# TAB 3 — VIDEO COMPARISON
# ══════════════════════════════════════════
with tabs[2]:
    st.subheader("⚖️ Side-by-Side Video Comparison")

    # Use unique prefixes: "cmp_l" and "cmp_r" to avoid any collision
    def folder_sel(prefix, col):
        with col:
            ds  = st.selectbox("Dataset",  list(DATASETS.keys()), key=f"{prefix}_ds")
            cls = st.selectbox("Class",    DATASETS[ds],           key=f"{prefix}_cls")
            vf  = list_video_folders(ds,cls)
            if not vf: st.info("No folders."); return None,None,None
            fn = st.selectbox(f"Folder ({len(vf)})",[f.name for f in vf],key=f"{prefix}_fn")
            return ds,cls,fn

    lc,rc = st.columns(2,gap="large")
    l_ds,l_cls,l_fn = folder_sel("cmp_l",lc)
    r_ds,r_cls,r_fn = folder_sel("cmp_r",rc)

    if st.button("⚖️ Compare",type="primary",disabled=(not l_fn or not r_fn),key="cmp_btn"):
        for col,ds,cls,fn,side in [(lc,l_ds,l_cls,l_fn,"LEFT"),(rc,r_ds,r_cls,r_fn,"RIGHT")]:
            fp   = class_root(ds,cls)/fn
            fls  = get_files(fp)
            pred = parse_pred_txt(fls["pred"]) if "pred" in fls else {}
            with col:
                st.markdown(f"### {side}: `{fn}`")
                is_f = "fight" in pred.get("pred_label","").lower() and "non" not in pred.get("pred_label","").lower()
                if is_f: st.error(f"🔴 FIGHT — {pred.get('confidence','?')} confidence")
                else:    st.success(f"🟢 NON-FIGHT — {pred.get('confidence','?')} confidence")
                ok = "✅" if str(pred.get("correct","")).lower()=="true" else "❌"
                st.markdown(f"**True:** {pred.get('true_label','?')} | **Pred:** {pred.get('pred_label','?')} | **Correct:** {ok}")
                if "original" in fls:
                    pp=fp/"_preview_original.mp4"
                    if not pp.exists(): make_web_preview(fls["original"],pp)
                    if pp.exists(): st.video(str(pp))
                if "gradcam_grid" in fls:
                    st.image(str(fls["gradcam_grid"]),use_container_width=True,caption="Grad-CAM")
                if "gradcampp_grid" in fls:
                    st.image(str(fls["gradcampp_grid"]),use_container_width=True,caption="Grad-CAM++")


# ══════════════════════════════════════════
# TAB 4 — CONTROL ROOM
# ══════════════════════════════════════════
with tabs[3]:
    st.subheader("🎥 Single Camera Control Room")
    cA,cB = st.columns([1.6,1.0],gap="large")
    with cA:
        # FIX: Use unique keys "ctrl_room_*" — avoids collision with any other tab
        src=st.radio("Source",["Pick from uploaded folders","Use local path"],
                     horizontal=True, key="ctrl_room_src")
        vpath=None
        if src=="Pick from uploaded folders":
            ctrl_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="ctrl_room_ds")
            ctrl_cls = st.selectbox("Class",   DATASETS[ctrl_ds],     key="ctrl_room_cls")
            vf=list_video_folders(ctrl_ds, ctrl_cls)
            if not vf: st.warning("No folders yet.")
            else:
                ctrl_fn=st.selectbox("Folder",[f.name for f in vf],key="ctrl_room_folder")
                ff=get_files(class_root(ctrl_ds, ctrl_cls)/ctrl_fn)
                if "original" in ff: vpath=str(ff["original"])
        else:
            vpath=st.text_input("Local path",value=st.session_state.active_video_path or "",
                                key="ctrl_room_path")
        if st.button("▶️ Run Preview",type="primary",disabled=not bool(vpath),key="ctrl_room_run"):
            try:
                frames,fps=read_video_frames(vpath,max_frames=140)
                frames=[resize_keep(f,720) for f in frames]
                scores=st.session_state.active_scores
                if scores is None or len(scores)!=len(frames):
                    scores=fake_model_scores(frames,fps)
                ph,il=st.empty(),st.empty()
                for i in range(min(len(frames),90)):
                    p=float(scores[i]); stat=pred_label_to_status(st.session_state.active_pred.get("pred_label",""))
                    e=color_from_status(stat)
                    ph.image(to_rgb(frames[i]),caption=f"Frame {i} | {e} | p={p:.2f}",use_container_width=True)
                    il.markdown(f"**Time:** {fmt_time(i/fps)} | **{e}** | p=`{p:.2f}`")
                    time.sleep(0.02)
                st.success("Done ✅")
            except Exception as e: st.error(str(e))
    with cB:
        st.write("### Status")
        pred=st.session_state.active_pred
        if not pred: st.info("Analyze a folder in Video Explorer first.")
        else:
            is_f="fight" in pred.get("pred_label","").lower() and "non" not in pred.get("pred_label","").lower()
            if is_f: st.error(f"🔴 FIGHT DETECTED\nConfidence: {pred.get('confidence','?')}")
            else:    st.success(f"🟢 NO FIGHT\nConfidence: {pred.get('confidence','?')}")
            st.metric("Onset Time",    pred.get("onset_time","N/A"))
            st.metric("Total Frames",  pred.get("total_frames","?"))
            scores=st.session_state.active_scores; fps=st.session_state.active_fps
            if scores is not None and fps:
                st.pyplot(make_timeline_plot(scores,fps,pred),clear_figure=True,use_container_width=False)


# ══════════════════════════════════════════
# TAB 5 — MULTI-CAMERA
# ══════════════════════════════════════════
with tabs[4]:
    st.subheader("🟦 Multi-Camera Wall")
    use_same=st.checkbox("Use same video for all cameras",value=True,key="wall_same")
    cams=[]
    if use_same:
        wall_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="wall_ds")
        wall_cls = st.selectbox("Class",   DATASETS[wall_ds],     key="wall_cls")
        wf=list_video_folders(wall_ds, wall_cls)
        if not wf: st.info("No folders yet.")
        else:
            wfn=st.selectbox("Folder",[f.name for f in wf],key="wall_fn")
            wfiles=get_files(class_root(wall_ds, wall_cls)/wfn)
            if "original" in wfiles: cams=[str(wfiles["original"])]*4
    else:
        cams=[st.text_input(f"Camera {i+1} path","",key=f"wall_cam_{i}") for i in range(4)]
    if st.button("🧱 Render Wall",disabled=(not cams or not all(bool(x) for x in cams)),key="wall_render"):
        cols=st.columns(2,gap="large")
        for idx,vp in enumerate(cams):
            try:
                frames,fps=read_video_frames(vp,max_frames=45)
                frames=[resize_keep(f,480) for f in frames]
                scores=fake_model_scores(frames,fps)
                peak=float(np.max(scores)); s="ALERT" if peak>=CFG.THRESH_VIOLENCE else "NORMAL"
                with cols[idx%2]:
                    st.image(to_rgb(frames[min(10,len(frames)-1)]),use_container_width=True,
                             caption=f"Cam {idx+1} | {color_from_status(s)} | p={peak:.2f}")
                    st.progress(min(1.0,peak))
            except Exception as e:
                with cols[idx%2]: st.error(f"Cam {idx+1}: {e}")


# ══════════════════════════════════════════
# TAB 6 — UPLOAD MANAGER
# ══════════════════════════════════════════
with tabs[5]:
    st.subheader("📤 Upload Manager")

    with st.expander("🗑️ Start Fresh — Clear all uploads", expanded=False):
        st.warning("⚠️ This will permanently delete ALL uploaded folders and files.")
        if not st.session_state._confirm_clear:
            if st.button("🗑️ Clear all uploads", key="clear_btn"):
                st.session_state._confirm_clear = True
                st.rerun()
        else:
            st.error("Are you sure? This cannot be undone.")
            col_y, col_n = st.columns(2)
            with col_y:
                if st.button("✅ Yes, delete everything", type="primary", key="confirm_yes"):
                    clear_all_uploads()
                    for k in ["active_pred","active_scores","active_fps","active_frames",
                              "active_folder_name","active_video_path","_active_files"]:
                        st.session_state[k] = {} if "pred" in k or "files" in k else None
                    st.session_state.active_folder_name = ""
                    st.session_state._confirm_clear = False
                    st.success("✅ All uploads cleared. Starting fresh!")
                    st.rerun()
            with col_n:
                if st.button("❌ Cancel", key="confirm_no"):
                    st.session_state._confirm_clear = False
                    st.rerun()

    st.divider()

    upload_mode = st.radio("Upload mode",
                           ["📁 Single folder (select files)","🗜️ ZIP file (whole dataset)"],
                           horizontal=True, key="up_mode")

    if upload_mode=="📁 Single folder (select files)":
        uc1,uc2,uc3=st.columns(3)
        with uc1: up_ds  = st.selectbox("Dataset",list(DATASETS.keys()),key="up_ds")
        with uc2: up_cls = st.selectbox("Class",DATASETS[up_ds],key="up_cls")
        with uc3: fn_inp = st.text_input("Folder name",placeholder="e.g. fi1_xvid",key="up_fn_inp")
        up_files=st.file_uploader("Select all files (mp4 + png + pred.txt)",
                                   type=["mp4","avi","mov","mkv","png","jpg","txt"],
                                   accept_multiple_files=True,key="up_files")
        if st.button("💾 Save",type="primary",disabled=(not up_files or not fn_inp.strip()),key="up_save"):
            dest=class_root(up_ds,up_cls)/fn_inp.strip()
            dest.mkdir(parents=True,exist_ok=True)
            for uf in up_files:
                with open(dest/uf.name,"wb") as f: f.write(uf.getbuffer())
            st.success(f"✅ Saved {len(up_files)} file(s) → `{up_ds}/{up_cls}/{fn_inp.strip()}`")
            st.info("Go to **Video Explorer** → select dataset/class → pick this folder → Analyze.")
    else:
        uc1,uc2=st.columns(2)
        with uc1: zip_ds  = st.selectbox("Dataset",list(DATASETS.keys()),key="zip_ds")
        with uc2: zip_cls = st.selectbox("Class",DATASETS[zip_ds],key="zip_cls")
        st.markdown("""
**ZIP structure expected:**
Each folder inside the zip = one video folder.
```
yourzip.zip
├── fi1_xvid/
│   ├── fi1_xvid_original.mp4
│   ├── gradcam_grid.png
│   └── pred.txt
└── fi3_xvid/
    └── ...
```""")
        zf=st.file_uploader("Upload ZIP",type=["zip"],key="zip_up")
        if st.button("📦 Extract ZIP",type="primary",disabled=(not zf),key="zip_extract"):
            with st.spinner("Extracting..."):
                n_f,n_files=extract_zip_to_uploads(zf.read(),zip_ds,zip_cls)
            st.success(f"✅ Extracted {n_f} folder(s), {n_files} file(s) into `{zip_ds}/{zip_cls}/`")

    st.divider()
    st.markdown("### 📂 Currently uploaded")
    found=False
    for ds in DATASETS:
        for cls in DATASETS[ds]:
            flist=list_video_folders(ds,cls)
            if flist:
                found=True
                with st.expander(f"**{ds}/{cls}** — {len(flist)} folder(s)",expanded=False):
                    for fl in flist:
                        items=list(fl.iterdir())
                        st.markdown(f"📁 **{fl.name}** — {len(items)} file(s)")
    if not found: st.info("Nothing uploaded yet.")


# ══════════════════════════════════════════
# TAB 7 — ANALYTICS
# ══════════════════════════════════════════
with tabs[6]:
    st.subheader("📈 Analytics")
    scores = st.session_state.active_scores
    fps    = st.session_state.active_fps
    pred   = st.session_state.active_pred
    fname  = st.session_state.active_folder_name

    if scores is None or fps is None:
        st.info("Analyze a folder in **Video Explorer** first.")
    else:
        st.markdown(f"Showing analytics for: **{fname}**")
        _, start_sec = detect_fight_start(scores, fps)
        step = max(1,int(0.5*fps))
        rows=[{"time":fmt_time(i/fps),"frame":i,"prob":round(float(scores[i]),3)}
              for i in range(0,len(scores),step)]
        df=pd.DataFrame(rows)
        cA,cB=st.columns([1.3,1.0],gap="large")
        with cA:
            st.write("#### Frame-by-Frame Log")
            st.dataframe(df,use_container_width=True,height=260)
        with cB:
            st.write("#### Summary from pred.txt")
            st.metric("Prediction",  pred.get("pred_label","?"))
            st.metric("Confidence",  pred.get("confidence","?"))
            st.metric("Onset Frame", pred.get("onset_frame","?"))
            st.metric("Onset Time",  pred.get("onset_time","?"))
            st.metric("Total Frames",pred.get("total_frames","?"))
            st.metric("Model Val Acc",pred.get("model_val_acc","?"))
        c1,c2=st.columns(2)
        with c1: st.pyplot(make_timeline_plot(scores,fps,pred),clear_figure=True,use_container_width=False)
        with c2: st.pyplot(make_hist_plot(scores),              clear_figure=True,use_container_width=False)


# ══════════════════════════════════════════
# TAB 8 — DATASET STATS
# ══════════════════════════════════════════
with tabs[7]:
    st.subheader("📊 Dataset Statistics")
    records=get_all_pred_records()
    if not records:
        st.info("No pred.txt files found. Upload folders first.")
    else:
        df=pd.DataFrame(records)
        total=len(df)
        n_correct=df["correct"].str.lower().eq("true").sum() if "correct" in df.columns else 0
        acc=n_correct/total if total>0 else 0
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Total Videos",str(total))
        m2.metric("Correct",str(n_correct))
        m3.metric("Overall Accuracy",f"{acc:.1%}")
        m4.metric("Datasets",str(df["_dataset"].nunique()) if "_dataset" in df.columns else "?")
        st.markdown("---")
        ca,cb=st.columns(2)
        with ca:
            st.markdown("#### Per Dataset")
            if "_dataset" in df.columns:
                g=df.groupby("_dataset").apply(lambda x: pd.Series({
                    "Total":len(x),
                    "Correct":x["correct"].str.lower().eq("true").sum() if "correct" in x else 0,
                    "Accuracy":f"{x['correct'].str.lower().eq('true').mean():.1%}" if "correct" in x else "?"})).reset_index()
                st.dataframe(g,use_container_width=True,hide_index=True)
        with cb:
            st.markdown("#### Per Class")
            if "_class" in df.columns:
                g=df.groupby("_class").apply(lambda x: pd.Series({
                    "Total":len(x),
                    "Correct":x["correct"].str.lower().eq("true").sum() if "correct" in x else 0,
                    "Accuracy":f"{x['correct'].str.lower().eq('true').mean():.1%}" if "correct" in x else "?"})).reset_index()
                st.dataframe(g,use_container_width=True,hide_index=True)
        st.markdown("---")
        st.markdown("#### All Records")
        show=["_folder","_dataset","_class","true_label","pred_label","correct","confidence","onset_time"]
        show=[c for c in show if c in df.columns]
        st.dataframe(df[show].rename(columns={"_folder":"Folder","_dataset":"Dataset","_class":"Class"}),
                     use_container_width=True,height=260)


# ══════════════════════════════════════════
# TAB 9 — FP/FN BROWSER
# ══════════════════════════════════════════
with tabs[8]:
    st.subheader("❌ False Positive / False Negative Browser")
    records=get_all_pred_records()
    if not records:
        st.info("No pred.txt files found.")
    else:
        df=pd.DataFrame(records)
        if "correct" not in df.columns:
            st.info("pred.txt files need a 'correct' field.")
        else:
            wrong=df[df["correct"].str.lower()!="true"]
            if wrong.empty:
                st.success("🎉 No errors found — all predictions correct!")
            else:
                tl="true_label"; pl="pred_label"
                fp=wrong[(wrong.get(tl,"").str.lower()=="nonfight")&(wrong.get(pl,"").str.lower()=="fight")] if tl in wrong.columns and pl in wrong.columns else pd.DataFrame()
                fn=wrong[(wrong.get(tl,"").str.lower()=="fight")&(wrong.get(pl,"").str.lower()=="nonfight")] if tl in wrong.columns and pl in wrong.columns else pd.DataFrame()
                t1,t2,t3=st.tabs([f"All Wrong ({len(wrong)})",f"False Positives ({len(fp)})",f"False Negatives ({len(fn)})"])
                for tab,data,label in [(t1,wrong,"wrong"),(t2,fp,"fp"),(t3,fn,"fn")]:
                    with tab:
                        if data.empty: st.info("None found.")
                        else:
                            show=["_folder","_dataset","_class","true_label","pred_label","confidence","onset_time"]
                            show=[c for c in show if c in data.columns]
                            st.dataframe(data[show].rename(columns={"_folder":"Folder","_dataset":"Dataset","_class":"Class"}),
                                         use_container_width=True,height=280)
                            st.download_button(f"⬇️ CSV",data=data[show].to_csv(index=False).encode(),
                                               file_name=f"{label}.csv",mime="text/csv",key=f"fpfn_{label}_csv")


# ══════════════════════════════════════════
# TAB 10 — THRESHOLD TUNER
# ══════════════════════════════════════════
with tabs[9]:
    st.subheader("🎛️ Confidence Threshold Tuner")
    records=get_all_pred_records()
    if not records:
        st.info("No pred.txt files found.")
    else:
        t1c,t2c=st.columns(2)
        with t1c: new_thr_v=st.slider("Violence threshold",   0.10,0.99,CFG.THRESH_VIOLENCE,   0.01,key="thr_violence")
        with t2c: new_thr_s=st.slider("Suspicious threshold", 0.10,0.99,CFG.THRESH_SUSPICIOUS, 0.01,key="thr_suspicious")
        df=pd.DataFrame(records)
        if "confidence" in df.columns and "true_label" in df.columns:
            try:
                df["conf_f"]=df["confidence"].astype(float)
                df["pred_fight_new"]=df["conf_f"]>=new_thr_v
                df["true_fight"]=df["true_label"].str.lower()=="fight"
                df["correct_new"]=df["pred_fight_new"]==df["true_fight"]
                new_acc=df["correct_new"].mean()
                m1,m2,m3=st.columns(3)
                m1.metric("Accuracy at this threshold",f"{new_acc:.1%}")
                m2.metric("Violence threshold",f"{new_thr_v:.2f}")
                m3.metric("Suspicious threshold",f"{new_thr_s:.2f}")
                thresholds=np.arange(0.1,1.0,0.05)
                accs=[(((df["conf_f"]>=t)==df["true_fight"]).mean()) for t in thresholds]
                fig=plt.figure(figsize=(7,3))
                plt.plot(thresholds,[a*100 for a in accs],color="#5271e0",linewidth=2,marker="o",markersize=4)
                plt.axvline(new_thr_v,color="red",linestyle="--",linewidth=1.5,label=f"Current={new_thr_v:.2f}")
                plt.xlabel("Threshold",fontsize=10); plt.ylabel("Accuracy (%)",fontsize=10)
                plt.title("Accuracy vs Threshold",fontsize=11); plt.legend(fontsize=9)
                plt.grid(alpha=0.3); plt.tight_layout()
                st.pyplot(fig,clear_figure=True,use_container_width=False)
            except Exception as e: st.error(f"Error: {e}")
        else:
            st.info("Need 'confidence' and 'true_label' in pred.txt files.")


# ══════════════════════════════════════════
# TAB 11 — CONFUSION MATRIX
# ══════════════════════════════════════════
with tabs[10]:
    st.subheader("🧩 Confusion Matrix & Per-Class Metrics")
    records=get_all_pred_records()
    if not records:
        st.info("No pred.txt files found.")
    else:
        df=pd.DataFrame(records)
        if "true_label" not in df.columns or "pred_label" not in df.columns:
            st.info("Need 'true_label' and 'pred_label' in pred.txt.")
        else:
            cm_col,met_col=st.columns([1,1.2],gap="large")
            with cm_col:
                fig,cm=make_confusion_matrix(records)
                st.pyplot(fig,clear_figure=True,use_container_width=False)
            with met_col:
                st.markdown("#### Per-Class Metrics")
                TP,FN=int(cm[0][0]),int(cm[0][1])
                FP,TN=int(cm[1][0]),int(cm[1][1])
                pf=TP/(TP+FP) if (TP+FP)>0 else 0
                rf=TP/(TP+FN) if (TP+FN)>0 else 0
                f1f=2*pf*rf/(pf+rf) if (pf+rf)>0 else 0
                pn=TN/(TN+FN) if (TN+FN)>0 else 0
                rn=TN/(TN+FP) if (TN+FP)>0 else 0
                f1n=2*pn*rn/(pn+rn) if (pn+rn)>0 else 0
                oa=(TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0
                mdf=pd.DataFrame([
                    {"Class":"Fight",   "Precision":f"{pf:.1%}","Recall":f"{rf:.1%}","F1":f"{f1f:.1%}","Support":TP+FN},
                    {"Class":"NonFight","Precision":f"{pn:.1%}","Recall":f"{rn:.1%}","F1":f"{f1n:.1%}","Support":FP+TN},
                ])
                st.dataframe(mdf,use_container_width=True,hide_index=True)
                st.metric("Overall Accuracy",f"{oa:.1%}")
                st.markdown(f"**TP:** `{TP}` | **FN:** `{FN}` | **FP:** `{FP}` | **TN:** `{TN}`")


# ══════════════════════════════════════════
# TAB 12 — INCIDENT REPORT
# ══════════════════════════════════════════
with tabs[11]:
    st.subheader("🧾 Incident Report Generator")
    pred   = st.session_state.active_pred
    scores = st.session_state.active_scores
    fps    = st.session_state.active_fps
    vpath  = st.session_state.active_video_path
    fname  = st.session_state.active_folder_name

    if not pred or scores is None:
        st.info("Analyze a folder in **Video Explorer** first — the report will be based on that folder.")
    else:
        st.success(f"Report will be generated for: **{fname}**")
        col1,col2=st.columns([1.1,1.2],gap="large")
        with col1:
            st.write("### Inputs")
            cam_nm = st.text_input("Camera name",  value="Entrance Camera", key="inc_cam")
            loc    = st.text_input("Location",      value="Main Gate / Hallway", key="inc_loc")
            notes  = st.text_area("Notes",          value="", key="inc_notes")
            gen    = st.button("🧾 Generate Report",type="primary",key="inc_gen")
        with col2:
            st.write("### Preview")
            is_f = "fight" in pred.get("pred_label","").lower() and "non" not in pred.get("pred_label","").lower()
            if is_f: st.error(f"🔴 FIGHT DETECTED — {pred.get('confidence','?')} confidence")
            else:    st.success(f"🟢 NO FIGHT — {pred.get('confidence','?')} confidence")
            st.markdown(f"- **Video folder:** `{fname}`")
            st.markdown(f"- **Dataset:** {pred.get('dataset','?')}")
            st.markdown(f"- **True Label:** {pred.get('true_label','?')}")
            st.markdown(f"- **Predicted:** {pred.get('pred_label','?')}")
            st.markdown(f"- **Correct:** {pred.get('correct','?')}")
            st.markdown(f"- **Onset Time:** {pred.get('onset_time','?')}")
            st.markdown(f"- **Model:** {pred.get('model_path','?')}")
            c1,c2=st.columns(2)
            with c1: st.pyplot(make_timeline_plot(scores,fps,pred),clear_figure=True,use_container_width=False)
            with c2: st.pyplot(make_hist_plot(scores),              clear_figure=True,use_container_width=False)

        if gen:
            sf,start_sec=detect_fight_start(scores,fps)
            report={
                "incident_id":  f"INC-{int(time.time())}",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generated_by": st.session_state.username,
                "camera":cam_nm,"location":loc,
                "video_folder": fname,
                "model":        pred.get("model_path","?"),
                "dataset":      pred.get("dataset","?"),
                "thresholds":   {"violence":CFG.THRESH_VIOLENCE,"suspicious":CFG.THRESH_SUSPICIOUS},
                "prediction": {
                    "true_label":    pred.get("true_label","?"),
                    "pred_label":    pred.get("pred_label","?"),
                    "confidence":    pred.get("confidence","?"),
                    "correct":       pred.get("correct","?"),
                    "onset_frame":   pred.get("onset_frame","?"),
                    "onset_time":    pred.get("onset_time","?"),
                    "total_frames":  pred.get("total_frames","?"),
                    "how_started":   describe_onset(pred),
                },
                "notes": notes,
            }
            ts=datetime.now().strftime("%Y%m%d_%H%M%S")
            path=Path(CFG.OUTPUT_DIR)/f"incident_{fname}_{ts}.json"
            with open(path,"w") as f: json.dump(report,f,indent=2)
            st.success(f"✅ Report saved!")
            st.download_button("⬇️ Download JSON",
                               data=io.BytesIO(json.dumps(report,indent=2).encode()),
                               file_name=path.name,mime="application/json",key="inc_json_dl")
            pdf=generate_pdf_report(pred,scores,fps,fname)
            st.download_button("⬇️ Download PDF",data=pdf,
                               file_name=f"report_{fname}_{ts}.pdf",mime="application/pdf",
                               key="inc_pdf_dl")

st.divider()
st.caption("Replace fake_model_scores() / scores_from_pred() with real R3D-18 inference when ready.")
