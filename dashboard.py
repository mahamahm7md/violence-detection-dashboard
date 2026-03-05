# dashboard.py — Violence Detection Dashboard v4
# Features: Login/Signup, Fight Alert, PDF Export, Confusion Matrix,
#           Grad-CAM Comparison, Dataset Stats, FP/FN Browser,
#           Threshold Tuner, Search/Filter, Video Comparison, ZIP Upload

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
    APP_TITLE: str         = "Violence Detection Dashboard"
    OUTPUT_DIR: str        = "outputs_dashboard"
    DEFAULT_FPS: int       = 25
    MAX_FRAMES: int        = 180
    THRESH_VIOLENCE: float = 0.70
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
# Auth helpers
# ──────────────────────────────────────────
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def load_users() -> dict:
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def try_login(username: str, password: str) -> bool:
    users = load_users()
    return users.get(username) == hash_pw(password)

def try_register(username: str, password: str) -> tuple:
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = hash_pw(password)
    save_users(users)
    return True, "Account created!"


# ──────────────────────────────────────────
# Core utilities
# ──────────────────────────────────────────
def status_from_score(p: float, label_map: int = 1, thr_v=None, thr_s=None) -> str:
    thr_v = thr_v or CFG.THRESH_VIOLENCE
    thr_s = thr_s or CFG.THRESH_SUSPICIOUS
    if label_map == 0: p = 1.0 - p
    if p >= thr_v:  return "ALERT"
    if p >= thr_s:  return "SUSPICIOUS"
    return "NORMAL"

def color_from_status(s: str) -> str:
    return {"ALERT":"🔴","SUSPICIOUS":"🟡","NORMAL":"🟢"}.get(s,"⚪")

def fmt_time(sec):
    if sec is None: return "N/A"
    return f"{int(sec//60):02d}:{int(sec%60):02d}"

def to_rgb(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def resize_keep(frame, w=640):
    h, ww = frame.shape[:2]
    if ww == w: return frame
    return cv2.resize(frame, (w, int(h*(w/ww))), interpolation=cv2.INTER_AREA)

def read_video_frames(path, max_frames=CFG.MAX_FRAMES):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise FileNotFoundError(f"Cannot open: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or CFG.DEFAULT_FPS
    frames, i = [], 0
    while True:
        ret, f = cap.read()
        if not ret or i >= max_frames: break
        frames.append(f); i += 1
    cap.release()
    return frames, float(fps)

def fake_model_scores(frames, fps):
    """Placeholder — replace with real R3D-18 inference."""
    ps, prev = [], None
    for f in frames:
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),(5,5),0)
        if prev is None: ps.append(0.10)
        else: ps.append(float(np.clip(float(np.mean(cv2.absdiff(g,prev)))/255*2.8,0,1)))
        prev = g
    return np.array(ps, dtype=np.float32)

def detect_fight_start(scores, fps, thr=None):
    thr = thr or CFG.THRESH_VIOLENCE
    N = max(3, int(0.2*fps)); run = 0
    for i, a in enumerate(scores >= thr):
        run = run+1 if a else 0
        if run >= N:
            sf = i-(N-1); return sf, sf/fps
    return None, None

def ffmpeg_ok() -> bool:
    try:
        subprocess.run(["ffmpeg","-version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=5)
        return True
    except: return False

def make_web_preview(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime: return True
    if ffmpeg_ok():
        try:
            subprocess.run(["ffmpeg","-y","-i",str(src),"-c:v","libx264",
                            "-pix_fmt","yuv420p","-preset","veryfast","-crf","23",
                            "-c:a","aac","-b:a","128k",str(dst)],
                           check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=120)
            return True
        except: pass
    try:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened(): return False
        fps = cap.get(cv2.CAP_PROP_FPS) or CFG.DEFAULT_FPS
        w,h = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), float(fps),(w,h))
        while True:
            ret, f = cap.read()
            if not ret: break
            out.write(f)
        cap.release(); out.release()
        return dst.exists()
    except: return False

def describe_onset(frames, scores, fps, sf):
    if sf is None: return "No clear onset detected."
    win = int(max(5,fps)); a,b = max(0,sf-win), min(len(frames)-1,sf+1)
    prev, mv = None, []
    for i in range(a,b):
        g = cv2.GaussianBlur(cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY),(5,5),0)
        if prev is not None: mv.append(float(np.mean(cv2.absdiff(g,prev)))/255.0)
        prev = g
    if not mv: return "Onset detected but motion evidence insufficient."
    return ("Sudden spike in movement — rapid interaction/contact."
            if max(mv)>np.mean(mv)*2.0
            else "Gradual onset — movement increases steadily.")


# ──────────────────────────────────────────
# Plots
# ──────────────────────────────────────────
def make_timeline_plot(scores, fps, thr_v=None, thr_s=None):
    thr_v = thr_v or CFG.THRESH_VIOLENCE
    thr_s = thr_s or CFG.THRESH_SUSPICIOUS
    t = np.arange(len(scores))/fps
    fig = plt.figure(figsize=(6,2.5))
    plt.plot(t, scores, color="#e05252", linewidth=1.5)
    plt.axhline(thr_s, linestyle="--", color="orange", linewidth=1, label=f"Suspicious ({thr_s:.2f})")
    plt.axhline(thr_v, linestyle="--", color="red",    linewidth=1, label=f"Violence ({thr_v:.2f})")
    plt.xlabel("Time (s)",fontsize=9); plt.ylabel("Prob",fontsize=9)
    plt.title("Violence Timeline",fontsize=10); plt.legend(fontsize=8)
    plt.tight_layout(); return fig

def make_hist_plot(scores):
    fig = plt.figure(figsize=(4,2.5))
    plt.hist(scores, bins=20, color="#5271e0")
    plt.xlabel("Probability",fontsize=9); plt.ylabel("Count",fontsize=9)
    plt.title("Confidence Dist.",fontsize=10); plt.tight_layout(); return fig

def make_confusion_matrix(records: list):
    """records = list of dicts with true_label, pred_label"""
    labels = ["Fight","NonFight"]
    cm = np.zeros((2,2), dtype=int)
    lmap = {"fight":0,"nonfight":1,"Fight":0,"NonFight":1}
    for r in records:
        t = lmap.get(r.get("true_label","?"), -1)
        p = lmap.get(r.get("pred_label","?"), -1)
        if t >= 0 and p >= 0: cm[t][p] += 1
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i][j]),ha="center",va="center",
                    color="white" if cm[i][j]>cm.max()/2 else "black",fontsize=14,fontweight="bold")
    plt.tight_layout(); return fig, cm


# ──────────────────────────────────────────
# pred.txt
# ──────────────────────────────────────────
def parse_pred_txt(path: Path) -> dict:
    out = {}
    try:
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k,v = line.strip().split(":",1)
                    out[k.strip()] = v.strip()
    except: pass
    return out

def render_pred_card(pred: dict, label_map: int=1):
    if not pred: st.info("No pred.txt found."); return
    true_lbl  = pred.get("true_label","?")
    pred_lbl  = pred.get("pred_label","?")
    correct   = pred.get("correct","?")
    conf      = pred.get("confidence","?")
    dataset   = pred.get("dataset","?")
    o_time    = pred.get("onset_time","?")
    o_frame   = pred.get("onset_frame","?")
    tot_fr    = pred.get("total_frames","?")
    if label_map==0:
        flip = {"Fight":"NonFight","NonFight":"Fight","fight":"nonfight","nonfight":"fight"}
        pred_lbl = flip.get(pred_lbl, pred_lbl)
    ok_emoji   = "✅" if str(correct).lower()=="true" else "❌"
    pred_color = "🔴" if "fight" in str(pred_lbl).lower() else "🟢"
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("True Label", true_lbl)
    c2.metric("Predicted",  f"{pred_color} {pred_lbl}")
    c3.metric("Correct?",   f"{ok_emoji} {correct}")
    try:    c4.metric("Confidence", f"{float(conf):.1%}")
    except: c4.metric("Confidence", conf)
    c5,c6,c7,c8 = st.columns(4)
    c5.metric("Dataset", dataset); c6.metric("Onset Frame", o_frame)
    c7.metric("Onset Time", o_time); c8.metric("Total Frames", tot_fr)
    with st.expander("📋 Full pred.txt", expanded=False):
        ca,cb = st.columns(2)
        with ca:
            for k in ["model_path","model_val_acc","probs","pred_class"]:
                st.markdown(f"**{k}:** `{pred.get(k,'?')}`")
        with cb:
            for k in ["window_size","window_stride","target_layer","img_size","cam_target_class","spike_delta","onset_threshold"]:
                st.markdown(f"**{k}:** `{pred.get(k,'?')}`")


# ──────────────────────────────────────────
# Folder helpers
# ──────────────────────────────────────────
def class_root(dataset: str, cls: str) -> Path:
    return UPLOAD_ROOT / dataset / cls

def list_video_folders(dataset: str, cls: str):
    root = class_root(dataset, cls)
    if not root.exists(): return []
    folders = [x for x in root.iterdir() if x.is_dir() and not x.name.startswith("_")]
    folders.sort(key=lambda x: x.name.lower())
    return folders

def find_file(folder: Path, pattern: str):
    m = list(folder.glob(pattern))
    return m[0] if m else None

def get_files(folder: Path) -> dict:
    files = {}
    orig = find_file(folder,"*original*.mp4")
    if not orig:
        cands = [f for f in folder.glob("*.mp4") if "gradcam" not in f.name.lower()]
        if len(cands)==1: orig = cands[0]
    if orig: files["original"] = orig
    for f in sorted(folder.glob("*.mp4")):
        n = f.name.lower()
        if "gradcampp" in n or "gradcam++" in n: files["gradcampp"] = f
        elif "gradcam" in n: files["gradcam"] = f
    for key,pat in [("raw_grid","raw_grid.png"),("gradcam_grid","gradcam_grid.png"),
                    ("gradcampp_grid","gradcampp_grid.png"),("pred","pred.txt")]:
        f = find_file(folder, pat)
        if f: files[key] = f
    return files

def get_all_pred_records() -> list:
    """Walk all uploaded folders and collect pred.txt data."""
    records = []
    for ds in DATASETS:
        for cls in DATASETS[ds]:
            for folder in list_video_folders(ds, cls):
                files = get_files(folder)
                if "pred" in files:
                    p = parse_pred_txt(files["pred"])
                    p["_dataset"] = ds
                    p["_class"]   = cls
                    p["_folder"]  = folder.name
                    records.append(p)
    return records


# ──────────────────────────────────────────
# PDF Export
# ──────────────────────────────────────────
def generate_pdf_report(pred: dict, scores, fps, folder_name: str, img_paths: dict) -> bytes:
    """Generate a PDF report using matplotlib (no external PDF lib needed)."""
    fig = plt.figure(figsize=(11,8.5))
    fig.patch.set_facecolor("#0e1117")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    title_ax = fig.add_subplot(gs[0,:])
    title_ax.axis("off")
    title_ax.set_facecolor("#0e1117")
    title_ax.text(0.5, 0.7, "Violence Detection — Incident Report",
                  ha="center", va="center", fontsize=18, fontweight="bold", color="white")
    title_ax.text(0.5, 0.3, f"Video: {folder_name}   |   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                  ha="center", va="center", fontsize=10, color="#aaaaaa")

    # Timeline
    ax_tl = fig.add_subplot(gs[1,0:2])
    ax_tl.set_facecolor("#1a1a2e")
    t = np.arange(len(scores))/fps
    ax_tl.plot(t, scores, color="#e05252", linewidth=1.5)
    ax_tl.axhline(CFG.THRESH_VIOLENCE, linestyle="--", color="red", linewidth=1)
    ax_tl.axhline(CFG.THRESH_SUSPICIOUS, linestyle="--", color="orange", linewidth=1)
    ax_tl.set_xlabel("Time (s)", color="white", fontsize=8)
    ax_tl.set_ylabel("Probability", color="white", fontsize=8)
    ax_tl.set_title("Violence Timeline", color="white", fontsize=10)
    ax_tl.tick_params(colors="white"); ax_tl.spines[:].set_color("#333355")

    # Hist
    ax_h = fig.add_subplot(gs[1,2])
    ax_h.set_facecolor("#1a1a2e")
    ax_h.hist(scores, bins=15, color="#5271e0")
    ax_h.set_title("Confidence Dist.", color="white", fontsize=10)
    ax_h.tick_params(colors="white"); ax_h.spines[:].set_color("#333355")

    # Pred info
    ax_info = fig.add_subplot(gs[2,:])
    ax_info.axis("off"); ax_info.set_facecolor("#0e1117")
    info_lines = [
        f"Dataset: {pred.get('dataset','?')}",
        f"True Label: {pred.get('true_label','?')}   |   Predicted: {pred.get('pred_label','?')}   |   Correct: {pred.get('correct','?')}",
        f"Confidence: {pred.get('confidence','?')}   |   Onset Frame: {pred.get('onset_frame','?')}   |   Onset Time: {pred.get('onset_time','?')}",
        f"Model: {pred.get('model_path','?')}   |   Val Acc: {pred.get('model_val_acc','?')}",
        f"Total Frames: {pred.get('total_frames','?')}   |   Window: {pred.get('window_size','?')}   |   Stride: {pred.get('window_stride','?')}",
    ]
    for i, line in enumerate(info_lines):
        ax_info.text(0.02, 0.85-i*0.18, line, transform=ax_info.transAxes,
                     fontsize=9, color="white" if i>0 else "#e05252",
                     fontweight="bold" if i==0 else "normal")

    buf = io.BytesIO()
    plt.savefig(buf, format="pdf", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────
# Alert banner
# ──────────────────────────────────────────
def show_fight_alert(folder_name: str, confidence: str):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #ff0000 0%, #8b0000 100%);
        border: 2px solid #ff4444;
        border-radius: 12px;
        padding: 20px 28px;
        margin: 12px 0;
        animation: pulse 1s infinite alternate;
        box-shadow: 0 0 24px rgba(255,0,0,0.5);
    ">
        <div style="font-size:28px; font-weight:900; color:white; letter-spacing:2px;">
            🚨 FIGHT DETECTED — ALERT
        </div>
        <div style="font-size:15px; color:#ffcccc; margin-top:8px;">
            Video: <b>{folder_name}</b> &nbsp;|&nbsp; Confidence: <b>{confidence}</b>
        </div>
        <div style="font-size:13px; color:#ffaaaa; margin-top:4px;">
            ⚠️ Immediate review recommended
        </div>
    </div>
    <style>
    @keyframes pulse {{
        from {{ box-shadow: 0 0 12px rgba(255,0,0,0.4); }}
        to   {{ box-shadow: 0 0 32px rgba(255,0,0,0.9); }}
    }}
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────
# ZIP upload helper
# ──────────────────────────────────────────
def extract_zip_to_uploads(zip_bytes: bytes, dataset: str, cls: str) -> tuple:
    """Extract a ZIP into the correct dataset/class folder. Returns (n_folders, n_files)."""
    n_folders, n_files = 0, 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        # Group files by their immediate parent folder
        folder_map = {}
        for name in names:
            parts = Path(name).parts
            if len(parts) >= 2:
                folder_key = parts[-2]  # immediate parent folder name
                if folder_key not in folder_map:
                    folder_map[folder_key] = []
                folder_map[folder_key].append(name)
            elif len(parts) == 1 and not name.endswith("/"):
                # top-level file — put in a "misc" folder
                folder_map.setdefault("misc", []).append(name)

        for folder_name, file_list in folder_map.items():
            dest = class_root(dataset, cls) / folder_name
            dest.mkdir(parents=True, exist_ok=True)
            n_folders += 1
            for zip_path in file_list:
                if zip_path.endswith("/"): continue
                file_name = Path(zip_path).name
                try:
                    data = zf.read(zip_path)
                    with open(dest / file_name, "wb") as f:
                        f.write(data)
                    n_files += 1
                except Exception:
                    pass
    return n_folders, n_files


# ──────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────
def init_state():
    defaults = {
        "logged_in": False, "username": "",
        "scores": None, "fps": None, "video_path": None, "frames": None,
        "analyzed": False, "_ex_files": None, "_ex_pred": {},
        "_ex_folder_path": None,
        "run_id": datetime.now().strftime("run_%Y%m%d_%H%M%S"),
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

for ds in DATASETS:
    for cls in DATASETS[ds]:
        class_root(ds, cls).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════
st.set_page_config(page_title=CFG.APP_TITLE, layout="wide", page_icon="🎯")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size:17px !important; }
[data-testid="stMetricLabel"] { font-size:11px !important; }
.block-container { padding-top:0.8rem !important; }
h1  { font-size:1.4rem !important; margin-bottom:0.1rem !important; }
h2,h3 { font-size:1rem !important; margin-top:0.3rem !important; }
.stTabs [data-baseweb="tab"] { font-size:12px; padding:5px 9px; }
div[data-testid="stImage"] img { border-radius:6px; }
hr { margin:0.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# LOGIN / REGISTER WALL
# ══════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""
    <div style="text-align:center; padding:40px 0 10px 0;">
        <div style="font-size:48px;">🎯</div>
        <div style="font-size:2rem; font-weight:900; color:white;">Violence Detection Dashboard</div>
        <div style="color:#888; margin-top:6px;">Please log in or create an account to continue</div>
    </div>
    """, unsafe_allow_html=True)

    col_gap, col_form, col_gap2 = st.columns([1,1.2,1])
    with col_form:
        auth_tab = st.tabs(["🔐 Login", "📝 Create Account"])

        with auth_tab[0]:
            st.markdown("#### Login")
            lu = st.text_input("Username", key="login_user")
            lp = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", type="primary", use_container_width=True):
                if try_login(lu, lp):
                    st.session_state.logged_in = True
                    st.session_state.username  = lu
                    st.rerun()
                else:
                    st.error("❌ Incorrect username or password.")

        with auth_tab[1]:
            st.markdown("#### Create Account")
            ru = st.text_input("Choose Username", key="reg_user")
            rp = st.text_input("Choose Password", type="password", key="reg_pass")
            rp2= st.text_input("Confirm Password", type="password", key="reg_pass2")
            if st.button("Create Account", type="primary", use_container_width=True):
                if rp != rp2:
                    st.error("❌ Passwords do not match.")
                else:
                    ok, msg = try_register(ru, rp)
                    if ok:
                        st.success(f"✅ {msg} You can now log in.")
                    else:
                        st.error(f"❌ {msg}")
    st.stop()


# ══════════════════════════════════════════
# MAIN APP (only shown when logged in)
# ══════════════════════════════════════════
hdr_l, hdr_r = st.columns([4,1])
with hdr_l:
    st.title(f"🎯 {CFG.APP_TITLE}")
with hdr_r:
    st.markdown(f"<div style='text-align:right; padding-top:12px; color:#888;'>👤 {st.session_state.username}</div>", unsafe_allow_html=True)
    if st.button("Logout", key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.rerun()

st.warning("⚠️ Cloud mode: uploads reset on restart. Run locally for permanent storage.", icon="⚠️")

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

    s1,s2,s3,s4 = st.columns([1,1,2,1])
    with s1: sel_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="ex_ds")
    with s2: sel_cls = st.selectbox("Class", DATASETS[sel_ds], key="ex_cls")
    with s3:
        vfolders = list_video_folders(sel_ds, sel_cls)
        if not vfolders:
            st.info(f"No folders yet for **{sel_ds}/{sel_cls}**. Use Upload Manager ➡️")
            sel_folder = None
        else:
            sel_folder = st.selectbox(f"Video folder ({len(vfolders)} available)",
                                      [f.name for f in vfolders], key="ex_folder")
    with s4:
        label_map = st.selectbox("Label mapping", options=[1,0],
                                 format_func=lambda x:"1=Fight (default)" if x==1 else "0=Fight (flipped)",
                                 key="ex_lmap")

    analyze_btn = st.button("🔍 Analyze this folder", type="primary",
                            disabled=(sel_folder is None), key="analyze_btn")

    if analyze_btn and sel_folder:
        folder_path = class_root(sel_ds, sel_cls) / sel_folder
        files = get_files(folder_path)
        st.session_state["_ex_folder_path"] = str(folder_path)
        st.session_state["_ex_files"]       = {k:str(v) for k,v in files.items()}
        st.session_state["_ex_pred"]        = parse_pred_txt(files["pred"]) if "pred" in files else {}
        if "original" in files:
            try:
                with st.spinner("Loading video..."):
                    frames, fps = read_video_frames(files["original"], max_frames=140)
                    frames_r    = [resize_keep(f,720) for f in frames]
                    scores      = fake_model_scores(frames_r, fps)
                st.session_state.scores     = scores
                st.session_state.fps        = fps
                st.session_state.video_path = str(files["original"])
                st.session_state.frames     = frames_r
                st.session_state.analyzed   = True
            except Exception as e:
                st.error(f"Could not load video: {e}")

    if st.session_state.get("_ex_files"):
        files       = {k:Path(v) for k,v in st.session_state["_ex_files"].items()}
        pred_data   = st.session_state.get("_ex_pred",{})
        folder_path = Path(st.session_state["_ex_folder_path"])
        folder_name = folder_path.name

        # ── FIGHT ALERT ──
        if pred_data:
            pl = pred_data.get("pred_label","")
            lm = label_map
            is_fight = ("fight" in pl.lower() and lm==1) or ("nonfight" in pl.lower() and lm==0)
            if is_fight:
                show_fight_alert(folder_name, pred_data.get("confidence","?"))

        st.markdown("---")
        st.markdown("#### 📋 Prediction Results")
        render_pred_card(pred_data, label_map=label_map)

        st.markdown("---")
        st.markdown("#### 🎬 Videos")
        vid_keys   = [k for k in ["original","gradcam","gradcampp"] if k in files]
        vid_labels = {"original":"📹 Original","gradcam":"🔥 Grad-CAM","gradcampp":"🔥 Grad-CAM++"}
        if vid_keys:
            vcols = st.columns(len(vid_keys))
            for i, vk in enumerate(vid_keys):
                with vcols[i]:
                    st.markdown(f"**{vid_labels[vk]}**")
                    vpath = files[vk]
                    prev  = folder_path / f"_preview_{vk}.mp4"
                    if not prev.exists():
                        with st.spinner("Converting..."):
                            make_web_preview(vpath, prev)
                    if prev.exists(): st.video(str(prev))
                    else: st.warning("Preview unavailable")
        else:
            st.info("No .mp4 files found.")

        st.markdown("---")
        st.markdown("#### 🖼️ Frame Grids")
        img_keys   = [k for k in ["raw_grid","gradcam_grid","gradcampp_grid"] if k in files]
        img_labels = {"raw_grid":"📷 Raw Frames","gradcam_grid":"🌡️ Grad-CAM","gradcampp_grid":"🌡️ Grad-CAM++"}
        if img_keys:
            icols = st.columns(len(img_keys))
            for i, ik in enumerate(img_keys):
                with icols[i]:
                    st.markdown(f"**{img_labels[ik]}**")
                    st.image(str(files[ik]), use_container_width=True)
        else:
            st.info("No grid images found.")

        # ── Grad-CAM Side-by-Side Comparison ──
        if "gradcam_grid" in files and "gradcampp_grid" in files:
            st.markdown("---")
            st.markdown("#### 🔬 Grad-CAM vs Grad-CAM++ Comparison")
            gc1, gc2 = st.columns(2)
            with gc1:
                st.markdown("**Grad-CAM**")
                st.image(str(files["gradcam_grid"]), use_container_width=True)
            with gc2:
                st.markdown("**Grad-CAM++**")
                st.image(str(files["gradcampp_grid"]), use_container_width=True)
            st.caption("Grad-CAM++ generally produces sharper, more precise activation maps.")

        # ── Quick score summary ──
        if st.session_state.analyzed and st.session_state.scores is not None:
            st.markdown("---")
            st.markdown("#### 📊 Quick Analysis")
            scores = st.session_state.scores
            fps    = st.session_state.fps
            _, start_sec = detect_fight_start(scores, fps)
            peak = float(np.max(scores)); avg = float(np.mean(scores))
            stat = status_from_score(peak, label_map)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Status", f"{color_from_status(stat)} {stat}")
            m2.metric("Peak Prob", f"{peak:.2f}")
            m3.metric("Avg Prob",  f"{avg:.2f}")
            m4.metric("Fight Start", fmt_time(start_sec))
            c1,c2 = st.columns(2)
            with c1: st.pyplot(make_timeline_plot(scores,fps), clear_figure=True, use_container_width=False)
            with c2: st.pyplot(make_hist_plot(scores),         clear_figure=True, use_container_width=False)

            # ── PDF Export ──
            st.markdown("---")
            st.markdown("#### 📄 Export Report")
            if st.button("📄 Generate PDF Report", key="pdf_btn"):
                with st.spinner("Generating PDF..."):
                    pdf_bytes = generate_pdf_report(
                        pred_data, scores, fps, folder_name,
                        {k:str(v) for k,v in files.items()}
                    )
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"report_{folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
    else:
        st.info("👆 Select a dataset, class, and folder — then click **Analyze this folder**.")


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
        with fc1:
            search_q = st.text_input("Search folder name", placeholder="e.g. fi1")
        with fc2:
            ds_filter = st.selectbox("Dataset", ["All"] + list(DATASETS.keys()), key="sf_ds")
        with fc3:
            cls_filter = st.selectbox("Class", ["All","Fight","NonFight"], key="sf_cls")
        with fc4:
            correct_filter = st.selectbox("Correct?", ["All","True","False"], key="sf_correct")

        df = df_all.copy()
        if search_q:
            df = df[df["_folder"].str.contains(search_q, case=False, na=False)]
        if ds_filter != "All":
            df = df[df["_dataset"] == ds_filter]
        if cls_filter != "All":
            df = df[df["_class"] == cls_filter]
        if correct_filter != "All":
            df = df[df["correct"].str.lower() == correct_filter.lower()]

        st.markdown(f"**{len(df)} results found**")

        show_cols = ["_folder","_dataset","_class","true_label","pred_label","correct","confidence","onset_time"]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols].rename(columns={"_folder":"Folder","_dataset":"Dataset",
                                                     "_class":"Class"}),
                     use_container_width=True, height=350)

        csv = df[show_cols].to_csv(index=False).encode()
        st.download_button("⬇️ Download filtered results CSV", data=csv,
                           file_name="filtered_results.csv", mime="text/csv")


# ══════════════════════════════════════════
# TAB 3 — VIDEO COMPARISON
# ══════════════════════════════════════════
with tabs[2]:
    st.subheader("⚖️ Side-by-Side Video Comparison")
    st.caption("Compare two different video folders from any dataset/class.")

    left_col, right_col = st.columns(2, gap="large")

    def folder_selector(col, prefix):
        with col:
            ds  = st.selectbox("Dataset",  list(DATASETS.keys()), key=f"{prefix}_ds")
            cls = st.selectbox("Class",    DATASETS[ds],           key=f"{prefix}_cls")
            vf  = list_video_folders(ds, cls)
            if not vf:
                st.info("No folders uploaded yet.")
                return None, None, None
            fn = st.selectbox(f"Folder ({len(vf)} available)", [f.name for f in vf], key=f"{prefix}_folder")
            return ds, cls, fn

    l_ds,l_cls,l_fn = folder_selector(left_col,  "cmp_l")
    r_ds,r_cls,r_fn = folder_selector(right_col, "cmp_r")

    if st.button("⚖️ Compare these folders", type="primary",
                 disabled=(not l_fn or not r_fn)):
        for col, ds, cls, fn, side in [
            (left_col,  l_ds,l_cls,l_fn,"LEFT"),
            (right_col, r_ds,r_cls,r_fn,"RIGHT")
        ]:
            folder_path = class_root(ds,cls)/fn
            files = get_files(folder_path)
            with col:
                st.markdown(f"### {side}: `{fn}`")
                pred = parse_pred_txt(files["pred"]) if "pred" in files else {}
                if pred:
                    pl = pred.get("pred_label","?")
                    conf = pred.get("confidence","?")
                    correct = pred.get("correct","?")
                    ok = "✅" if str(correct).lower()=="true" else "❌"
                    pc = "🔴" if "fight" in pl.lower() else "🟢"
                    st.markdown(f"**Predicted:** {pc} {pl} | **Conf:** {conf} | **Correct:** {ok}")
                if "original" in files:
                    prev = folder_path/f"_preview_original.mp4"
                    if not prev.exists(): make_web_preview(files["original"], prev)
                    if prev.exists(): st.video(str(prev))
                if "gradcam_grid" in files:
                    st.image(str(files["gradcam_grid"]), use_container_width=True,
                             caption="Grad-CAM Grid")
                if "gradcampp_grid" in files:
                    st.image(str(files["gradcampp_grid"]), use_container_width=True,
                             caption="Grad-CAM++ Grid")


# ══════════════════════════════════════════
# TAB 4 — CONTROL ROOM
# ══════════════════════════════════════════
with tabs[3]:
    st.subheader("🎥 Single Camera Control Room")
    colA,colB = st.columns([1.6,1.0], gap="large")
    with colA:
        src = st.radio("Source",["Pick from uploaded folders","Use local path"],horizontal=True)
        vpath = None
        if src=="Pick from uploaded folders":
            cr_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="cr_ds")
            cr_cls = st.selectbox("Class", DATASETS[cr_ds], key="cr_cls")
            vf     = list_video_folders(cr_ds,cr_cls)
            if not vf: st.warning("No folders uploaded yet.")
            else:
                cr_fn  = st.selectbox("Folder",[f.name for f in vf],key="cr_folder")
                ff     = get_files(class_root(cr_ds,cr_cls)/cr_fn)
                if "original" in ff: vpath = str(ff["original"])
        else:
            vpath = st.text_input("Local path", value=st.session_state.video_path or "")
        run_btn = st.button("▶️ Run Preview", type="primary", disabled=not bool(vpath))
        if run_btn and vpath:
            try:
                frames,fps = read_video_frames(vpath,max_frames=140)
                frames     = [resize_keep(f,720) for f in frames]
                scores     = fake_model_scores(frames,fps)
                st.session_state.scores=scores; st.session_state.fps=fps
                st.session_state.video_path=vpath; st.session_state.frames=frames
                ph,il = st.empty(), st.empty()
                for i in range(min(len(frames),90)):
                    p=float(scores[i]); s=status_from_score(p); e=color_from_status(s)
                    ph.image(to_rgb(frames[i]),caption=f"Frame {i} | {e} {s} | p={p:.2f}",use_container_width=True)
                    il.markdown(f"**Time:** {fmt_time(i/fps)} | **{e} {s}** | p=`{p:.2f}`")
                    time.sleep(0.02)
                st.success("Done ✅")
            except Exception as e: st.error(str(e))
    with colB:
        st.write("### Indicators")
        scores=st.session_state.scores; fps=st.session_state.fps
        if scores is None: st.info("Analyze a video first.")
        else:
            peak=float(np.max(scores)); avg=float(np.mean(scores))
            _,start_sec=detect_fight_start(scores,fps)
            st.metric("Peak probability",f"{peak:.2f}")
            st.metric("Average probability",f"{avg:.2f}")
            st.metric("Fight start",fmt_time(start_sec))
            st.pyplot(make_timeline_plot(scores,fps),clear_figure=True,use_container_width=False)


# ══════════════════════════════════════════
# TAB 5 — MULTI-CAMERA
# ══════════════════════════════════════════
with tabs[4]:
    st.subheader("🟦 Multi-Camera Wall")
    use_same=st.checkbox("Use same video for all cameras",value=True)
    cams=[]
    if use_same:
        w_ds=st.selectbox("Dataset",list(DATASETS.keys()),key="wall_ds")
        w_cls=st.selectbox("Class",DATASETS[w_ds],key="wall_cls")
        wf=list_video_folders(w_ds,w_cls)
        if not wf: st.info("No folders uploaded yet.")
        else:
            wfn=st.selectbox("Folder",[f.name for f in wf],key="wall_folder")
            wfiles=get_files(class_root(w_ds,w_cls)/wfn)
            if "original" in wfiles: cams=[str(wfiles["original"])]*4
    else:
        cams=[st.text_input(f"Camera {i+1} path","") for i in range(4)]
    if st.button("🧱 Render Camera Wall",disabled=(not cams or not all(bool(x) for x in cams))):
        cols=st.columns(2,gap="large")
        for idx,vp in enumerate(cams):
            try:
                frames,fps=read_video_frames(vp,max_frames=45)
                frames=[resize_keep(f,480) for f in frames]
                scores=fake_model_scores(frames,fps)
                peak=float(np.max(scores)); s=status_from_score(peak)
                if s=="ALERT": show_fight_alert(f"Camera {idx+1}",f"{peak:.2f}")
                with cols[idx%2]:
                    st.image(to_rgb(frames[min(10,len(frames)-1)]),use_container_width=True,
                             caption=f"Cam {idx+1} | {color_from_status(s)} {s} | p={peak:.2f}")
                    st.progress(min(1.0,peak))
            except Exception as e:
                with cols[idx%2]: st.error(f"Camera {idx+1}: {e}")


# ══════════════════════════════════════════
# TAB 6 — UPLOAD MANAGER
# ══════════════════════════════════════════
with tabs[5]:
    st.subheader("📤 Upload Manager")

    upload_mode = st.radio("Upload mode", ["📁 Single folder (select files)", "🗜️ ZIP file (whole dataset)"], horizontal=True)

    if upload_mode == "📁 Single folder (select files)":
        st.markdown("""
**Steps:** Select dataset & class → type folder name → select all files from that folder → Save
        """)
        uc1,uc2,uc3=st.columns(3)
        with uc1: up_ds  = st.selectbox("Dataset",list(DATASETS.keys()),key="up_ds")
        with uc2: up_cls = st.selectbox("Class",DATASETS[up_ds],key="up_cls")
        with uc3: folder_nm = st.text_input("Folder name",placeholder="e.g. fi1_xvid")
        up_files=st.file_uploader("Select all files (mp4 + png + pred.txt)",
                                   type=["mp4","avi","mov","mkv","png","jpg","txt"],
                                   accept_multiple_files=True,key="up_files")
        if st.button("💾 Save",type="primary",disabled=(not up_files or not folder_nm.strip())):
            dest=class_root(up_ds,up_cls)/folder_nm.strip()
            dest.mkdir(parents=True,exist_ok=True)
            for uf in up_files:
                with open(dest/uf.name,"wb") as f: f.write(uf.getbuffer())
            st.success(f"✅ Saved {len(up_files)} file(s) → `{up_ds}/{up_cls}/{folder_nm.strip()}`")

    else:
        st.markdown("""
**ZIP structure expected:**
```
yourzip.zip
└── fi1_xvid/
│   ├── fi1_xvid_original.mp4
│   ├── gradcam_grid.png
│   └── pred.txt
└── fi3_xvid/
    ├── ...
```
        """)
        uc1,uc2=st.columns(2)
        with uc1: zip_ds  = st.selectbox("Dataset",list(DATASETS.keys()),key="zip_ds")
        with uc2: zip_cls = st.selectbox("Class",DATASETS[zip_ds],key="zip_cls")
        zip_file = st.file_uploader("Upload ZIP file",type=["zip"],key="zip_upload")
        if st.button("📦 Extract ZIP",type="primary",disabled=(not zip_file)):
            with st.spinner("Extracting..."):
                n_f, n_files = extract_zip_to_uploads(zip_file.read(), zip_ds, zip_cls)
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
                        st.markdown(f"📁 **{fl.name}** — {len(items)} file(s): "
                                    f"`{'`, `'.join(f.name for f in sorted(items))}`")
    if not found: st.info("Nothing uploaded yet.")


# ══════════════════════════════════════════
# TAB 7 — ANALYTICS
# ══════════════════════════════════════════
with tabs[6]:
    st.subheader("📈 Analytics")
    scores=st.session_state.scores; fps=st.session_state.fps
    if scores is None or fps is None:
        st.info("Analyze a video in **Video Explorer** first.")
    else:
        _,start_sec=detect_fight_start(scores,fps)
        step=max(1,int(0.5*fps))
        rows=[{"time":fmt_time(i/fps),"frame":i,"prob":round(float(scores[i]),3),
               "status":status_from_score(float(scores[i]))}
              for i in range(0,len(scores),step)]
        df=pd.DataFrame(rows)
        cA,cB=st.columns([1.3,1.0],gap="large")
        with cA:
            st.write("#### Detection Log")
            st.dataframe(df,use_container_width=True,height=260)
        with cB:
            st.write("#### Summary")
            st.metric("Fight start",fmt_time(start_sec))
            st.metric("Peak probability",f"{float(np.max(scores)):.2f}")
            above=scores>=CFG.THRESH_VIOLENCE
            runs,run=[],0
            for a in above:
                if a: run+=1
                else:
                    if run>0: runs.append(run)
                    run=0
            if run>0: runs.append(run)
            st.metric("Short ALERT spikes (<0.5s)",str(len([r for r in runs if r/fps<0.5])))
        c1,c2=st.columns(2)
        with c1: st.pyplot(make_timeline_plot(scores,fps),clear_figure=True,use_container_width=False)
        with c2: st.pyplot(make_hist_plot(scores),         clear_figure=True,use_container_width=False)


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
        correct_col="correct"
        n_correct=df[correct_col].str.lower().eq("true").sum() if correct_col in df.columns else 0
        acc=n_correct/total if total>0 else 0

        m1,m2,m3,m4=st.columns(4)
        m1.metric("Total Videos",      str(total))
        m2.metric("Correct Predictions",str(n_correct))
        m3.metric("Overall Accuracy",  f"{acc:.1%}")
        m4.metric("Datasets",          str(df["_dataset"].nunique()) if "_dataset" in df.columns else "?")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Per Dataset")
            if "_dataset" in df.columns:
                ds_grp=df.groupby("_dataset").apply(
                    lambda x: pd.Series({
                        "Total": len(x),
                        "Correct": x["correct"].str.lower().eq("true").sum() if "correct" in x else 0,
                        "Accuracy": f"{x['correct'].str.lower().eq('true').mean():.1%}" if "correct" in x else "?"
                    })
                ).reset_index()
                st.dataframe(ds_grp, use_container_width=True)
        with col_b:
            st.markdown("#### Per Class")
            if "_class" in df.columns:
                cls_grp=df.groupby("_class").apply(
                    lambda x: pd.Series({
                        "Total": len(x),
                        "Correct": x["correct"].str.lower().eq("true").sum() if "correct" in x else 0,
                        "Accuracy": f"{x['correct'].str.lower().eq('true').mean():.1%}" if "correct" in x else "?"
                    })
                ).reset_index()
                st.dataframe(cls_grp, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Confidence Distribution (all videos)")
        if "confidence" in df.columns:
            try:
                confs=df["confidence"].astype(float)
                fig=plt.figure(figsize=(7,2.5))
                plt.hist(confs, bins=20, color="#5271e0", edgecolor="white", linewidth=0.5)
                plt.xlabel("Confidence",fontsize=9); plt.ylabel("Count",fontsize=9)
                plt.title("Confidence Distribution — All Videos",fontsize=10)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True, use_container_width=False)
            except Exception: st.info("Could not parse confidence values.")

        st.markdown("#### All Records")
        show=["_folder","_dataset","_class","true_label","pred_label","correct","confidence","onset_time"]
        show=[c for c in show if c in df.columns]
        st.dataframe(df[show].rename(columns={"_folder":"Folder","_dataset":"Dataset","_class":"Class"}),
                     use_container_width=True, height=260)


# ══════════════════════════════════════════
# TAB 9 — FP/FN BROWSER
# ══════════════════════════════════════════
with tabs[8]:
    st.subheader("❌ False Positive / False Negative Browser")
    records=get_all_pred_records()
    if not records:
        st.info("No pred.txt files found. Upload folders first.")
    else:
        df=pd.DataFrame(records)
        if "correct" not in df.columns:
            st.info("pred.txt files don't contain 'correct' field.")
        else:
            wrong=df[df["correct"].str.lower()!="true"].copy()
            if wrong.empty:
                st.success("🎉 No false positives or negatives found! All predictions are correct.")
            else:
                st.markdown(f"**{len(wrong)} incorrect predictions found:**")

                fp=wrong[(wrong.get("true_label","").str.lower()=="nonfight") &
                         (wrong.get("pred_label","").str.lower()=="fight")] if "true_label" in wrong.columns and "pred_label" in wrong.columns else pd.DataFrame()
                fn=wrong[(wrong.get("true_label","").str.lower()=="fight") &
                         (wrong.get("pred_label","").str.lower()=="nonfight")] if "true_label" in wrong.columns and "pred_label" in wrong.columns else pd.DataFrame()

                t1,t2,t3=st.tabs([f"All Wrong ({len(wrong)})",f"False Positives ({len(fp)})",f"False Negatives ({len(fn)})"])

                for tab,data,label in [(t1,wrong,"Wrong"),(t2,fp,"False Positive"),(t3,fn,"False Negative")]:
                    with tab:
                        if data.empty:
                            st.info(f"No {label} predictions.")
                        else:
                            show=["_folder","_dataset","_class","true_label","pred_label","confidence","onset_time"]
                            show=[c for c in show if c in data.columns]
                            st.dataframe(data[show].rename(columns={"_folder":"Folder","_dataset":"Dataset","_class":"Class"}),
                                         use_container_width=True, height=300)
                            csv=data[show].to_csv(index=False).encode()
                            st.download_button(f"⬇️ Download {label} CSV",data=csv,
                                               file_name=f"{label.lower().replace(' ','_')}.csv",
                                               mime="text/csv")


# ══════════════════════════════════════════
# TAB 10 — THRESHOLD TUNER
# ══════════════════════════════════════════
with tabs[9]:
    st.subheader("🎛️ Confidence Threshold Tuner")
    records=get_all_pred_records()
    if not records:
        st.info("No pred.txt files found. Upload folders first.")
    else:
        st.markdown("Adjust thresholds and see how accuracy changes across all uploaded videos.")
        t1,t2=st.columns(2)
        with t1: new_thr_v = st.slider("Violence threshold",   0.10,0.99,CFG.THRESH_VIOLENCE,   0.01)
        with t2: new_thr_s = st.slider("Suspicious threshold", 0.10,0.99,CFG.THRESH_SUSPICIOUS, 0.01)

        df=pd.DataFrame(records)
        if "confidence" in df.columns and "true_label" in df.columns:
            try:
                df["conf_f"] = df["confidence"].astype(float)
                df["pred_fight_new"] = df["conf_f"] >= new_thr_v
                df["true_fight"]     = df["true_label"].str.lower()=="fight"
                df["correct_new"]    = df["pred_fight_new"]==df["true_fight"]
                new_acc=df["correct_new"].mean()

                m1,m2,m3=st.columns(3)
                m1.metric("New Accuracy at threshold", f"{new_acc:.1%}")
                m2.metric("Violence threshold", f"{new_thr_v:.2f}")
                m3.metric("Suspicious threshold", f"{new_thr_s:.2f}")

                # Sweep
                thresholds=np.arange(0.1,1.0,0.05)
                accs=[]
                for t in thresholds:
                    pred_f=df["conf_f"]>=t
                    accs.append((pred_f==df["true_fight"]).mean())

                fig=plt.figure(figsize=(7,3))
                plt.plot(thresholds,[a*100 for a in accs],color="#5271e0",linewidth=2,marker="o",markersize=4)
                plt.axvline(new_thr_v,color="red",linestyle="--",linewidth=1,label=f"Current={new_thr_v:.2f}")
                plt.xlabel("Threshold",fontsize=10); plt.ylabel("Accuracy (%)",fontsize=10)
                plt.title("Accuracy vs Threshold",fontsize=11); plt.legend(fontsize=9)
                plt.grid(alpha=0.3); plt.tight_layout()
                st.pyplot(fig,clear_figure=True,use_container_width=False)

                st.markdown("#### Results at current threshold")
                show=["_folder","_dataset","_class","true_label","conf_f","correct_new"]
                show=[c for c in show if c in df.columns]
                st.dataframe(df[show].rename(columns={"_folder":"Folder","_dataset":"Dataset",
                                                        "_class":"Class","conf_f":"Confidence",
                                                        "correct_new":"Correct@NewThr"}),
                             use_container_width=True,height=250)
            except Exception as e:
                st.error(f"Could not process: {e}")
        else:
            st.info("pred.txt files need 'confidence' and 'true_label' fields for this feature.")


# ══════════════════════════════════════════
# TAB 11 — CONFUSION MATRIX
# ══════════════════════════════════════════
with tabs[10]:
    st.subheader("🧩 Confusion Matrix & Per-Class Metrics")
    records=get_all_pred_records()
    if not records:
        st.info("No pred.txt files found. Upload folders first.")
    else:
        df=pd.DataFrame(records)
        if "true_label" not in df.columns or "pred_label" not in df.columns:
            st.info("pred.txt files need 'true_label' and 'pred_label' fields.")
        else:
            cm_col, metrics_col = st.columns([1,1.2], gap="large")
            with cm_col:
                fig, cm = make_confusion_matrix(records)
                st.pyplot(fig, clear_figure=True, use_container_width=False)

            with metrics_col:
                st.markdown("#### Per-Class Metrics")
                TP = int(cm[0][0]); FN = int(cm[0][1])
                FP = int(cm[1][0]); TN = int(cm[1][1])

                prec_fight  = TP/(TP+FP) if (TP+FP)>0 else 0
                rec_fight   = TP/(TP+FN) if (TP+FN)>0 else 0
                f1_fight    = 2*prec_fight*rec_fight/(prec_fight+rec_fight) if (prec_fight+rec_fight)>0 else 0

                prec_nonf   = TN/(TN+FN) if (TN+FN)>0 else 0
                rec_nonf    = TN/(TN+FP) if (TN+FP)>0 else 0
                f1_nonf     = 2*prec_nonf*rec_nonf/(prec_nonf+rec_nonf) if (prec_nonf+rec_nonf)>0 else 0

                overall_acc = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0

                metrics_df=pd.DataFrame([
                    {"Class":"Fight",   "Precision":f"{prec_fight:.1%}","Recall":f"{rec_fight:.1%}","F1":f"{f1_fight:.1%}","Support":TP+FN},
                    {"Class":"NonFight","Precision":f"{prec_nonf:.1%}", "Recall":f"{rec_nonf:.1%}", "F1":f"{f1_nonf:.1%}", "Support":FP+TN},
                ])
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                st.metric("Overall Accuracy", f"{overall_acc:.1%}")

                st.markdown("#### Raw Counts")
                raw_df=pd.DataFrame([
                    {"":""      ,"Pred Fight":TP,"Pred NonFight":FN},
                    {"":"True Fight",   "Pred Fight":TP,"Pred NonFight":FN},
                    {"":"True NonFight","Pred Fight":FP,"Pred NonFight":TN},
                ])
                st.markdown(f"- **TP** (Fight→Fight): `{TP}` &nbsp; **FN** (Fight→NonFight): `{FN}`")
                st.markdown(f"- **FP** (NonFight→Fight): `{FP}` &nbsp; **TN** (NonFight→NonFight): `{TN}`")


# ══════════════════════════════════════════
# TAB 12 — INCIDENT REPORT
# ══════════════════════════════════════════
with tabs[11]:
    st.subheader("🧾 Incident Report Generator")
    scores=st.session_state.scores; fps=st.session_state.fps; vpath=st.session_state.video_path
    col1,col2=st.columns([1.1,1.2],gap="large")
    with col1:
        st.write("### Inputs")
        cam_nm  = st.text_input("Camera name",  value="Entrance Camera")
        loc     = st.text_input("Location",      value="Main Gate / Hallway")
        mod_nm  = st.text_input("Model",         value="R3D-18")
        ds_nm   = st.text_input("Dataset",       value="RWF-2000 / HockeyFight")
        notes   = st.text_area ("Notes",         value="")
        gen_btn = st.button("🧾 Generate Report", type="primary",
                            disabled=(scores is None or fps is None or vpath is None))
    with col2:
        st.write("### Preview")
        if scores is None or fps is None or vpath is None:
            st.info("Analyze a video in Video Explorer first.")
        else:
            _,start_sec=detect_fight_start(scores,fps)
            peak=float(np.max(scores)); stat=status_from_score(peak)
            st.markdown(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"- **Video:** `{Path(vpath).name}`")
            st.markdown(f"- **Fight start:** **{fmt_time(start_sec)}**")
            st.markdown(f"- **Peak prob:** **{peak:.2f}**")
            st.markdown(f"- **Status:** {color_from_status(stat)} **{stat}**")
            c1,c2=st.columns(2)
            with c1: st.pyplot(make_timeline_plot(scores,fps),clear_figure=True,use_container_width=False)
            with c2: st.pyplot(make_hist_plot(scores),         clear_figure=True,use_container_width=False)
    if gen_btn and scores is not None:
        sf,start_sec=detect_fight_start(scores,fps)
        peak=float(np.max(scores))
        report={
            "incident_id":  f"INC-{int(time.time())}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "generated_by": st.session_state.username,
            "camera":cam_nm,"location":loc,
            "video_file":Path(vpath).name if vpath else None,
            "model":mod_nm,"dataset":ds_nm,
            "thresholds":{"suspicious":CFG.THRESH_SUSPICIOUS,"violence":CFG.THRESH_VIOLENCE},
            "results":{
                "peak_probability":round(peak,4),
                "estimated_fight_start_sec":None if start_sec is None else round(float(start_sec),3),
                "estimated_fight_start_time":fmt_time(start_sec),
                "how_started":describe_onset(st.session_state.frames,scores,fps,sf),
            },
            "notes":notes,
        }
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        path=Path(CFG.OUTPUT_DIR)/f"incident_{ts}.json"
        with open(path,"w") as f: json.dump(report,f,indent=2)
        st.success(f"Saved: {path}")
        buf=io.BytesIO(json.dumps(report,indent=2).encode())
        st.download_button("⬇️ Download JSON",data=buf,
                           file_name=path.name,mime="application/json")

st.divider()
st.caption("Replace fake_model_scores() with your real R3D-18 inference when ready.")
