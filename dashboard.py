# dashboard.py — VisionGuard Violence Detection Dashboard v6
# Design: Fixed left sidebar (logo + nav + settings + last result)
# Features: All 6 CAM videos, all 6 grids, timeline, ByteTrack fighter IDs,
#           pred_nonfight class, full pred.txt fields, new CAM method tabs

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
    APP_TITLE: str           = "VisionGuard"
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
    "hockeyfight": ["Fight", "Nonfight"],
    "rwf":         ["Fight", "NonFight", "pred_nonfight"],
}

ALL_VID_KEYS = ["original", "gradcam", "gradcampp", "smooth_gradcampp", "layercam", "combined"]
VID_LABELS   = {
    "original":          "📹 Original",
    "gradcam":           "🔥 GradCAM",
    "gradcampp":         "🔥 GradCAM++",
    "smooth_gradcampp":  "✨ Smooth GradCAM++",
    "layercam":          "🌊 LayerCAM",
    "combined":          "🎯 Combined",
}

ALL_GRID_KEYS = ["raw_grid", "gradcam_grid", "gradcampp_grid",
                 "smooth_gradcampp_grid", "layercam_grid", "combined_grid"]
GRID_LABELS = {
    "raw_grid":              "📷 Raw Frames",
    "gradcam_grid":          "🌡️ GradCAM",
    "gradcampp_grid":        "🌡️ GradCAM++",
    "smooth_gradcampp_grid": "✨ Smooth GradCAM++",
    "layercam_grid":         "🌊 LayerCAM",
    "combined_grid":         "🎯 Combined",
}


# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="VisionGuard — Violence Detection",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────
# CSS
# ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@700;900&display=swap');

html, body, .stApp { background:#080c10 !important; color:#c8d8e8 !important; }
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a0f18 0%,#080c14 100%) !important;
    border-right: 1px solid #1a2535 !important;
    min-width: 260px !important; max-width: 260px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

#MainMenu, footer, header { visibility:hidden !important; }
[data-testid="stDecoration"] { display:none !important; }
.block-container {
    padding: 1.2rem 1.8rem 2rem 1.8rem !important;
    max-width: 100% !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: #0d1520; border-bottom: 1px solid #1a2535;
    border-radius: 0; padding: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif; font-size: 12px; font-weight: 600;
    padding: 8px 16px; border-radius: 0; color: #556677 !important;
    background: transparent !important; border-bottom: 2px solid transparent;
    letter-spacing: 0.5px; text-transform: uppercase;
}
.stTabs [aria-selected="true"] {
    color: #e8f4ff !important; border-bottom: 2px solid #e05252 !important;
    background: transparent !important;
}

[data-testid="metric-container"] {
    background: #0d1520; border: 1px solid #1a2535;
    border-radius: 6px; padding: 10px 14px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.1rem !important; font-weight: 400 !important; color: #7ecfff !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 10px !important; color: #445566 !important;
    text-transform: uppercase; letter-spacing: 1px;
}

.stButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important; font-size: 13px !important;
    border-radius: 4px !important; letter-spacing: 1px; text-transform: uppercase;
    transition: all 0.15s ease;
}
.stButton > button[kind="primary"] {
    background: #e05252 !important; border: none !important; color: white !important;
    box-shadow: 0 0 12px rgba(224,82,82,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #ff6060 !important; box-shadow: 0 0 20px rgba(224,82,82,0.5) !important;
}
.stButton > button:not([kind="primary"]) {
    background: #0d1520 !important; border: 1px solid #1a2535 !important; color: #7ecfff !important;
}

.stSelectbox > div > div, .stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0d1520 !important; border: 1px solid #1a2535 !important;
    color: #c8d8e8 !important; border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
}

.streamlit-expanderHeader {
    background: #0d1520 !important; border: 1px solid #1a2535 !important;
    border-radius: 4px !important; font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important; color: #7ecfff !important;
}
.streamlit-expanderContent {
    background: #080c10 !important; border: 1px solid #1a2535 !important;
    border-top: none !important;
}

[data-testid="stDataFrame"] { border: 1px solid #1a2535 !important; border-radius: 6px; }
hr { border-color: #1a2535 !important; margin: 0.8rem 0 !important; }
div[data-testid="stAlert"] { border-radius: 4px !important; border-left-width: 3px !important; }
div[data-testid="stImage"] img { border-radius: 6px; border: 1px solid #1a2535; }
[data-testid="stSlider"] > div > div > div { background: #e05252 !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080c10; }
::-webkit-scrollbar-thumb { background: #1a2535; border-radius: 2px; }

h1 { font-family:'Orbitron',sans-serif !important; font-size:1.1rem !important;
     color:#e8f4ff !important; letter-spacing:2px; }
h2 { font-family:'Rajdhani',sans-serif !important; font-size:1.1rem !important;
     font-weight:700 !important; color:#aabbc8 !important; letter-spacing:1px; text-transform:uppercase; }
h3 { font-family:'Rajdhani',sans-serif !important; font-size:0.95rem !important;
     font-weight:600 !important; color:#7a99b0 !important; }

.stApp::before {
    content:''; position:fixed; top:0; left:0; right:0; bottom:0;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.03) 2px,rgba(0,0,0,0.03) 4px);
    pointer-events:none; z-index:9999;
}

[data-testid="stVideo"] video {
    max-height: 160px !important; width: 100% !important;
    border-radius: 4px 4px 0 0 !important; background: #000 !important;
    display: block; object-fit: contain !important;
}
[data-testid="stVideo"] {
    border: 1px solid #1a2535 !important; border-radius: 6px !important;
    overflow: visible !important; background: #000 !important;
    margin-bottom: 8px !important;
}
[data-testid="stVideo"] > div {
    overflow: visible !important;
}
/* give the native video controls bar enough height */
[data-testid="stVideo"] > div > video::-webkit-media-controls-panel {
    min-height: 36px !important;
    padding: 0 4px !important;
    font-size: 10px !important;
}

@keyframes vg-shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}

.vg-badge-fight {
    display:inline-block; padding:3px 10px; border-radius:3px;
    background:rgba(224,82,82,0.15); border:1px solid #e05252;
    color:#ff8080; font-family:'Share Tech Mono',monospace; font-size:11px; letter-spacing:1px;
}
.vg-badge-normal {
    display:inline-block; padding:3px 10px; border-radius:3px;
    background:rgba(82,224,138,0.12); border:1px solid #52e08a;
    color:#52e08a; font-family:'Share Tech Mono',monospace; font-size:11px; letter-spacing:1px;
}

/* ── Welcome page animations ── */
@keyframes pulse-shield {
    0%,100% { transform:scale(1);   opacity:1;   }
    50%      { transform:scale(1.1); opacity:0.75; }
}
@keyframes fadein-up {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0);    }
}
@keyframes fadein-up-d1 {
    0%   { opacity:0; transform:translateY(18px); }
    30%  { opacity:0; transform:translateY(18px); }
    100% { opacity:1; transform:translateY(0); }
}
@keyframes fadein-up-d2 {
    0%   { opacity:0; transform:translateY(18px); }
    50%  { opacity:0; transform:translateY(18px); }
    100% { opacity:1; transform:translateY(0); }
}
@keyframes fadein-up-d3 {
    0%   { opacity:0; transform:translateY(18px); }
    65%  { opacity:0; transform:translateY(18px); }
    100% { opacity:1; transform:translateY(0); }
}

.vg-welcome  { animation: fadein-up 0.5s ease both; }
.vg-wcard {
    background:#0d1520; border:1px solid #1a2535; border-radius:12px;
    padding:18px 20px; transition:border-color .2s, box-shadow .2s;
}
.vg-wcard:hover        { border-color:#e05252; box-shadow:0 0 22px rgba(224,82,82,.12); }
.vg-wcard-blue:hover   { border-color:#7ecfff; box-shadow:0 0 22px rgba(126,207,255,.12); }
.vg-wcard-green:hover  { border-color:#52e08a; box-shadow:0 0 22px rgba(82,224,138,.12); }
.vg-wcard-orange:hover { border-color:#f5a623; box-shadow:0 0 22px rgba(245,166,35,.12); }

/* welcome stat tiles */
.vg-stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 20px 0 0 0;
    animation: fadein-up-d1 0.8s ease both;
}
.vg-stat-tile {
    background: #0d1520;
    border: 1px solid #1a2535;
    border-radius: 10px;
    padding: 20px 14px;
    text-align: center;
    transition: border-color .2s, box-shadow .2s;
}
.vg-stat-tile:hover { border-color: #e05252; box-shadow: 0 0 20px rgba(224,82,82,.1); }
.vg-stat-tile.blue:hover   { border-color: #7ecfff; box-shadow: 0 0 20px rgba(126,207,255,.1); }
.vg-stat-tile.green:hover  { border-color: #52e08a; box-shadow: 0 0 20px rgba(82,224,138,.1); }
.vg-stat-tile.orange:hover { border-color: #f5a623; box-shadow: 0 0 20px rgba(245,166,35,.1); }
.vg-stat-icon  { font-size: 22px; margin-bottom: 8px; }
.vg-stat-val   { font-family:'Orbitron',sans-serif; font-size:20px; font-weight:900; color:#7ecfff; margin-bottom:4px; }
.vg-stat-label { font-family:'Rajdhani',sans-serif; font-size:10px; color:#2a3a4a; text-transform:uppercase; letter-spacing:1.5px; }

/* steps */
.vg-steps-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 12px 0 0 0;
    animation: fadein-up-d2 0.9s ease both;
}
.vg-step-card {
    position: relative;
    background: #0d1520;
    border: 1px solid #1a2535;
    border-radius: 12px;
    padding: 22px 20px 20px;
    overflow: hidden;
    transition: box-shadow .2s;
}
.vg-step-card:hover { box-shadow: 0 4px 28px rgba(0,0,0,.35); }
.vg-step-card.red   { border-left: 3px solid #e05252; }
.vg-step-card.blue  { border-left: 3px solid #7ecfff; }
.vg-step-card.green { border-left: 3px solid #52e08a; }
.vg-step-num {
    position:absolute; top:10px; right:14px;
    font-family:'Orbitron',sans-serif; font-size:28px; font-weight:900; color:#1a2535; line-height:1;
}
.vg-step-icon  { font-size:22px; margin-bottom:10px; }
.vg-step-title { font-family:'Rajdhani',sans-serif; font-weight:700; font-size:13px; letter-spacing:1px; margin-bottom:7px; }
.vg-step-desc  { font-family:'Rajdhani',sans-serif; font-size:12px; color:#556677; line-height:1.6; }

/* quick launch */
.vg-ql-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 12px 0 0 0;
    animation: fadein-up-d3 1.0s ease both;
}
.vg-ql-btn {
    background: #0d1520;
    border: 1px solid #1a2535;
    border-radius: 6px;
    padding: 11px 10px;
    text-align: center;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: .8px;
    color: #7ecfff;
    text-transform: uppercase;
    transition: all .15s;
}

.vg-divider-label {
    font-family:'Share Tech Mono',monospace;
    font-size:9px; color:#2a3a4a; letter-spacing:3px;
    margin: 28px 0 12px 0;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# SAFE VIDEO HELPER
# ──────────────────────────────────────────
def _safe_video(path):
    try:
        p = Path(path)
        if not p.exists(): st.warning("⚠️ Video file not found."); return
        with open(p, "rb") as f: data = f.read()
        if len(data) == 0: st.warning("⚠️ Video file is empty."); return
        st.video(data)
    except Exception as e:
        st.warning(f"⚠️ Video preview unavailable. ({type(e).__name__})")


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
    with open(USERS_FILE, "w") as f: json.dump(u, f, indent=2)

def try_login(u, p):
    return load_users().get(u) == hash_pw(p)

def try_register(u, p):
    if not u or not p: return False, "Username and password required."
    if len(p) < 4: return False, "Password must be at least 4 characters."
    users = load_users()
    if u in users: return False, "Username already exists."
    users[u] = hash_pw(p); save_users(users)
    return True, "Account created! You can now log in."


# ──────────────────────────────────────────
# Core utils
# ──────────────────────────────────────────
def is_fight_pred(pred: dict, flip: bool = False) -> bool:
    lbl = str(pred.get("pred_label", "")).lower()
    raw = "fight" in lbl and "non" not in lbl
    return (not raw) if flip else raw

def pred_label_to_status(pred_label: str) -> str:
    if "fight" in str(pred_label).lower() and "non" not in str(pred_label).lower():
        return "ALERT"
    return "NORMAL"

def color_from_status(s):
    return {"ALERT": "🔴", "SUSPICIOUS": "🟡", "NORMAL": "🟢", "UNKNOWN": "⚪"}.get(s, "⚪")

def fmt_time(sec):
    if sec is None: return "N/A"
    try: return f"{int(float(sec)//60):02d}:{int(float(sec)%60):02d}"
    except: return str(sec)

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

def scores_from_pred(pred: dict, n_frames: int, fps: float):
    conf = 0.5
    try: conf = float(pred.get("confidence", 0.5))
    except: pass
    onset_frame = 0
    try: onset_frame = int(pred.get("onset_frame", 0))
    except: pass
    is_fight = is_fight_pred(pred)
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

def ffmpeg_ok():
    try:
        subprocess.run(["ffmpeg","-version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=5)
        return True
    except: return False

def make_web_preview(src, dst):
    dst = Path(dst); src = Path(src)
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
        w, h = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        while True:
            ret, f = cap.read()
            if not ret: break
            out.write(f)
        cap.release(); out.release()
        return dst.exists()
    except: return False

def describe_onset(pred: dict) -> str:
    onset_t = pred.get("onset_time", "?")
    onset_f = pred.get("onset_frame", "?")
    spike   = pred.get("spike_delta", "?")
    thr     = pred.get("onset_threshold", "?")
    if onset_f == "N/A" or onset_t == "N/A":
        return "No clear onset detected."
    return (f"Fight onset detected at frame {onset_f} ({onset_t}). "
            f"Onset threshold: {thr}, spike delta: {spike}.")


# ──────────────────────────────────────────
# Plots
# ──────────────────────────────────────────
def make_timeline_plot(scores, fps, pred=None):
    t   = np.arange(len(scores)) / fps
    fig = plt.figure(figsize=(6, 2.5), facecolor="#080c10")
    ax  = fig.add_subplot(111)
    ax.set_facecolor("#0a0f18")
    is_fight = is_fight_pred(pred) if pred else False
    color = "#e05252" if is_fight else "#52e08a"
    ax.plot(t, scores, color=color, linewidth=1.6)
    ax.fill_between(t, scores, alpha=0.12, color=color)
    ax.axhline(CFG.THRESH_SUSPICIOUS, linestyle="--", color="#f5a623", linewidth=0.8,
               label=f"Suspicious ({CFG.THRESH_SUSPICIOUS})")
    ax.axhline(CFG.THRESH_VIOLENCE,   linestyle="--", color="#e05252", linewidth=0.8,
               label=f"Violence ({CFG.THRESH_VIOLENCE})")
    if pred:
        try:
            onset_f = int(pred.get("onset_frame", 0))
            onset_t_val = onset_f / fps
            if 0 < onset_t_val < t[-1]:
                ax.axvline(onset_t_val, color="#7ecfff", linewidth=1.2,
                           linestyle=":", label=f"Onset ({pred.get('onset_time','?')})")
        except: pass
    ax.set_xlabel("Time (s)", fontsize=8, color="#445566")
    ax.set_ylabel("P(fight)", fontsize=8, color="#445566")
    ax.tick_params(colors="#445566", labelsize=7)
    ax.spines[:].set_color("#1a2535")
    ax.legend(fontsize=6, facecolor="#0a0f18", edgecolor="#1a2535", labelcolor="#7ecfff")
    plt.tight_layout()
    return fig

def make_hist_plot(scores):
    fig = plt.figure(figsize=(4, 2.5), facecolor="#080c10")
    ax  = fig.add_subplot(111)
    ax.set_facecolor("#0a0f18")
    ax.hist(scores, bins=20, color="#5271e0", edgecolor="#080c10", linewidth=0.3)
    ax.set_xlabel("Probability", fontsize=8, color="#445566")
    ax.set_ylabel("Count", fontsize=8, color="#445566")
    ax.tick_params(colors="#445566", labelsize=7)
    ax.spines[:].set_color("#1a2535")
    plt.tight_layout()
    return fig

def make_confusion_matrix(records):
    labels = ["Fight", "NonFight"]
    cm = np.zeros((2, 2), dtype=int)
    lmap = {"fight": 0, "nonfight": 1, "Fight": 0, "NonFight": 1, "Nonfight": 1}
    for r in records:
        t = lmap.get(r.get("true_label", ""), -1)
        p = 0 if is_fight_pred(r) else 1
        if t >= 0 and p >= 0: cm[t][p] += 1
    fig, ax = plt.subplots(figsize=(4, 3), facecolor="#080c10")
    ax.set_facecolor("#0a0f18")
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color="#7ecfff", fontsize=9)
    ax.set_yticklabels(labels, color="#7ecfff", fontsize=9)
    ax.set_xlabel("Predicted", color="#445566"); ax.set_ylabel("True", color="#445566")
    ax.tick_params(colors="#445566"); ax.spines[:].set_color("#1a2535")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max()/2 else "#7ecfff",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig, cm


# ──────────────────────────────────────────
# pred.txt parser + card
# ──────────────────────────────────────────
def parse_pred_txt(path) -> dict:
    out = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    out[k.strip()] = v.strip()
    except: pass
    return out

def render_pred_card(pred: dict):
    if not pred: st.info("No pred.txt found."); return
    is_fight = is_fight_pred(pred)
    badge = '<span class="vg-badge-fight">⚠ FIGHT</span>' if is_fight \
            else '<span class="vg-badge-normal">✓ NORMAL</span>'
    st.markdown(badge, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Label",  pred.get("true_label", "?"))
    c2.metric("Predicted",   pred.get("pred_label", "?"))
    ok_emoji = "✅" if str(pred.get("correct","")).lower()=="true" else "❌"
    c3.metric("Correct",     f"{ok_emoji} {pred.get('correct','?')}")
    try:    c4.metric("Confidence", f"{float(pred.get('confidence','0')):.1%}")
    except: c4.metric("Confidence", pred.get("confidence", "?"))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Onset Frame",  pred.get("onset_frame", "N/A"))
    c6.metric("Onset Time",   pred.get("onset_time",  "N/A"))
    c7.metric("Total Frames", pred.get("total_frames","?"))
    c8.metric("Dataset",      pred.get("dataset",     "?"))
    fighter_ids = pred.get("fighter_ids", "")
    cam_focused = pred.get("cam_focused", "")
    if fighter_ids or cam_focused:
        st.markdown(f"""
        <div style="background:#0d1520;border:1px solid #1a2535;border-left:3px solid #7ecfff;
                    border-radius:4px;padding:8px 14px;margin:8px 0;
                    font-family:'Share Tech Mono',monospace;font-size:12px;">
            <span style="color:#445566;">BYTETRACK</span>&nbsp;&nbsp;
            <span style="color:#7ecfff;">🥊 {fighter_ids or 'N/A'}</span>
            &nbsp;&nbsp;<span style="color:#445566;">|</span>&nbsp;&nbsp;
            <span style="color:#445566;">CAM FOCUS</span>&nbsp;
            <span style="color:#52e08a;">{cam_focused or 'N/A'}</span>
        </div>""", unsafe_allow_html=True)
    with st.expander("📋 Full pred.txt", expanded=False):
        col_a, col_b = st.columns(2)
        left_keys  = ["model_path","model_val_acc","probs","window_size",
                      "window_stride","onset_threshold","spike_delta","smooth_window"]
        right_keys = ["cam_methods","gradcam_layers","smooth_n_passes","smooth_sigma",
                      "cam_focused","fighter_ids","bytetrack_csv","heatmap"]
        with col_a:
            for k in left_keys:
                if k in pred:
                    st.markdown(f"<span style='color:#445566;font-size:11px;font-family:monospace'>{k}:</span> "
                                f"<span style='color:#c8d8e8;font-size:11px;font-family:monospace'>{pred[k]}</span>",
                                unsafe_allow_html=True)
        with col_b:
            for k in right_keys:
                if k in pred:
                    st.markdown(f"<span style='color:#445566;font-size:11px;font-family:monospace'>{k}:</span> "
                                f"<span style='color:#c8d8e8;font-size:11px;font-family:monospace'>{pred[k]}</span>",
                                unsafe_allow_html=True)


# ──────────────────────────────────────────
# Folder helpers
# ──────────────────────────────────────────
def class_root(ds, cls): return UPLOAD_ROOT / ds / cls

def list_video_folders(ds, cls):
    root = class_root(ds, cls)
    if not root.exists(): return []
    f = [x for x in root.iterdir() if x.is_dir() and not x.name.startswith("_")]
    f.sort(key=lambda x: x.name.lower())
    return f

def find_file(folder, pattern):
    m = list(folder.glob(pattern))
    return m[0] if m else None

def get_files(folder) -> dict:
    files = {}
    def real_files(pattern):
        return [f for f in folder.glob(pattern)
                if not f.name.startswith("._") and not f.name.startswith(".")]
    for vk in ALL_VID_KEYS:
        if vk == "original":
            cands = real_files("*original*.mp4")
            if not cands:
                cands = [f for f in real_files("*.mp4")
                         if not any(x in f.name.lower() for x in
                                    ["gradcam","layercam","combined","smooth","_preview"])]
            if cands: files["original"] = cands[0]
        elif vk == "smooth_gradcampp":
            cands = real_files("*smooth_gradcampp*.mp4") + real_files("*smooth*.mp4")
            cands = [f for f in cands if not f.name.startswith("_preview")]
            if cands: files["smooth_gradcampp"] = cands[0]
        elif vk == "gradcampp":
            cands = [f for f in real_files("*gradcampp*.mp4") + real_files("*gradcam++*.mp4")
                     if "smooth" not in f.name.lower() and not f.name.startswith("_preview")]
            if cands: files["gradcampp"] = cands[0]
        elif vk == "gradcam":
            cands = [f for f in real_files("*gradcam*.mp4")
                     if "pp" not in f.name.lower() and "++" not in f.name.lower()
                     and "smooth" not in f.name.lower() and not f.name.startswith("_preview")]
            if cands: files["gradcam"] = cands[0]
        elif vk == "layercam":
            cands = [f for f in real_files("*layercam*.mp4") if not f.name.startswith("_preview")]
            if cands: files["layercam"] = cands[0]
        elif vk == "combined":
            cands = [f for f in real_files("*combined*.mp4") if not f.name.startswith("_preview")]
            if cands: files["combined"] = cands[0]
    for gk in ALL_GRID_KEYS:
        f = find_file(folder, f"{gk}.png")
        if f: files[gk] = f
    f = find_file(folder, "timeline.png")
    if f: files["timeline"] = f
    f = find_file(folder, "pred.txt")
    if f: files["pred"] = f
    return files

def get_all_pred_records():
    records = []
    for ds in DATASETS:
        for cls in DATASETS[ds]:
            for folder in list_video_folders(ds, cls):
                files = get_files(folder)
                if "pred" in files:
                    p = parse_pred_txt(files["pred"])
                    p["_dataset"] = ds; p["_class"] = cls; p["_folder"] = folder.name
                    records.append(p)
    return records

def clear_all_uploads():
    if UPLOAD_ROOT.exists(): shutil.rmtree(UPLOAD_ROOT)
    for ds in DATASETS:
        for cls in DATASETS[ds]:
            class_root(ds, cls).mkdir(parents=True, exist_ok=True)

def extract_zip_to_uploads(zip_bytes, dataset, cls):
    n_folders, n_files = 0, 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        folder_map = {}
        for name in names:
            parts = Path(name).parts
            if len(parts) >= 2:
                fk = parts[-2]; folder_map.setdefault(fk, []).append(name)
            elif len(parts) == 1 and not name.endswith("/"):
                folder_map.setdefault("misc", []).append(name)
        for fn, flist in folder_map.items():
            dest = class_root(dataset, cls) / fn
            dest.mkdir(parents=True, exist_ok=True); n_folders += 1
            for zp in flist:
                if zp.endswith("/"): continue
                try:
                    with open(dest / Path(zp).name, "wb") as f: f.write(zf.read(zp))
                    n_files += 1
                except: pass
    return n_folders, n_files


# ──────────────────────────────────────────
# PDF Report
# ──────────────────────────────────────────
def generate_pdf_report(pred, scores, fps, folder_name) -> bytes:
    fig = plt.figure(figsize=(11, 8.5), facecolor="#080c10")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)
    title_ax = fig.add_subplot(gs[0, :]); title_ax.axis("off")
    title_ax.set_facecolor("#080c10")
    title_ax.text(0.5, 0.75, "VisionGuard — Violence Detection Report",
                  ha="center", va="center", fontsize=16, fontweight="bold", color="white")
    title_ax.text(0.5, 0.3, f"Video: {folder_name}   |   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                  ha="center", va="center", fontsize=10, color="#aaaaaa")
    ax_tl = fig.add_subplot(gs[1, 0:2]); ax_tl.set_facecolor("#0a0f18")
    t = np.arange(len(scores)) / fps
    is_fight = is_fight_pred(pred)
    ax_tl.plot(t, scores, color="#e05252" if is_fight else "#52e08a", linewidth=1.5)
    ax_tl.axhline(CFG.THRESH_VIOLENCE, linestyle="--", color="red", linewidth=1)
    ax_tl.set_xlabel("Time (s)", color="white", fontsize=8)
    ax_tl.set_ylabel("P(fight)", color="white", fontsize=8)
    ax_tl.tick_params(colors="white"); ax_tl.spines[:].set_color("#333355")
    ax_h = fig.add_subplot(gs[1, 2]); ax_h.set_facecolor("#0a0f18")
    ax_h.hist(scores, bins=15, color="#5271e0")
    ax_h.tick_params(colors="white"); ax_h.spines[:].set_color("#333355")
    ax_info = fig.add_subplot(gs[2, :]); ax_info.axis("off")
    lines = [
        f"STATUS: {'FIGHT DETECTED' if is_fight else 'NO FIGHT'}   |   Confidence: {pred.get('confidence','?')}",
        f"Dataset: {pred.get('dataset','?')}   True: {pred.get('true_label','?')}   Predicted: {pred.get('pred_label','?')}   Correct: {pred.get('correct','?')}",
        f"Onset Frame: {pred.get('onset_frame','?')}   Onset Time: {pred.get('onset_time','?')}   Frames: {pred.get('total_frames','?')}",
        f"Fighters: {pred.get('fighter_ids','N/A')}   CAM Focus: {pred.get('cam_focused','N/A')}",
        f"Model: {pred.get('model_path','?')}   Val Acc: {pred.get('model_val_acc','?')}",
    ]
    for i, line in enumerate(lines):
        ax_info.text(0.02, 0.95 - i*0.19, line, transform=ax_info.transAxes, fontsize=8,
                     color="#ff6666" if i == 0 and is_fight else ("white" if i > 0 else "#66ff66"),
                     fontweight="bold" if i == 0 else "normal")
    buf = io.BytesIO()
    plt.savefig(buf, format="pdf", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig); buf.seek(0); return buf.read()


# ──────────────────────────────────────────
# Fight Alert Banner
# ──────────────────────────────────────────
def show_fight_alert(folder_name, confidence):
    st.markdown(f"""
    <div style="background:rgba(224,82,82,0.12);border:1px solid #e05252;border-left:4px solid #e05252;
                border-radius:4px;padding:14px 20px;margin:10px 0;
                box-shadow:0 0 20px rgba(224,82,82,0.2);">
        <div style="font-family:'Orbitron',sans-serif;font-size:14px;font-weight:900;
                    color:#ff8080;letter-spacing:3px;">
            🚨 FIGHT DETECTED — IMMEDIATE REVIEW REQUIRED
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:12px;color:#cc6666;margin-top:6px;">
            VIDEO: {folder_name} &nbsp;|&nbsp; CONFIDENCE: {confidence}
        </div>
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# Session state
# ──────────────────────────────────────────
def init_state():
    defaults = {
        "logged_in": False, "username": "",
        "active_pred": {}, "active_scores": None, "active_fps": None,
        "active_frames": None, "active_folder_name": "",
        "active_video_path": None, "active_dataset": "", "active_class": "",
        "run_id": datetime.now().strftime("run_%Y%m%d_%H%M%S"),
        "_confirm_clear": False,
        "nav_page": "🏠 Welcome",
        "thr_violence": CFG.THRESH_VIOLENCE,
        "thr_suspicious": CFG.THRESH_SUSPICIOUS,
        "max_frames": CFG.MAX_FRAMES,
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_state()
for ds in DATASETS:
    for cls in DATASETS[ds]:
        class_root(ds, cls).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════
# LOGIN WALL
# ══════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""
    <style>.block-container { padding-top: 0 !important; }</style>
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;padding:6vh 0 2vh 0;gap:6px;">
        <div style="font-size:48px;animation:pulse-shield 3s ease-in-out infinite;">🛡️</div>
        <div style="font-family:'Orbitron',sans-serif;font-size:2.4rem;
                    font-weight:900;color:#e8f4ff;letter-spacing:4px;">
            VISIONGUARD
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:12px;
                    color:#445566;letter-spacing:2px;margin-bottom:20px;">
            VIOLENCE DETECTION SYSTEM v6
        </div>
    </div>""", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        auth_tabs = st.tabs(["🔐  Login", "📝  Register"])
        with auth_tabs[0]:
            lu = st.text_input("Username", key="login_u", placeholder="operator id")
            lp = st.text_input("Password", type="password", key="login_p", placeholder="password")
            if st.button("AUTHENTICATE →", type="primary", use_container_width=True, key="login_btn"):
                if try_login(lu.strip(), lp):
                    st.session_state.logged_in = True
                    st.session_state.username  = lu.strip()
                    st.rerun()
                else:
                    st.error("❌ Authentication failed.")
        with auth_tabs[1]:
            ru  = st.text_input("Username",         key="reg_u",  placeholder="choose username")
            rp  = st.text_input("Password",         type="password", key="reg_p")
            rp2 = st.text_input("Confirm Password", type="password", key="reg_p2")
            if st.button("CREATE ACCOUNT →", type="primary", use_container_width=True, key="reg_btn"):
                if rp != rp2: st.error("❌ Passwords do not match.")
                else:
                    ok, msg = try_register(ru.strip(), rp)
                    st.success(f"✅ {msg}") if ok else st.error(f"❌ {msg}")
    st.stop()


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 16px 16px 16px;border-bottom:1px solid #1a2535;">
        <div style="font-family:'Orbitron',sans-serif;font-size:1.1rem;
                    font-weight:900;color:#e8f4ff;letter-spacing:3px;">
            🛡️ VISIONGUARD
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                    color:#2a3a4a;margin-top:2px;letter-spacing:1px;">
            Violence Detection v6
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding:12px 16px;border-bottom:1px solid #1a2535;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#445566;">
            👤 operator
        </div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:14px;
                    font-weight:700;color:#7ecfff;">
            {st.session_state.username}
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                    color:#2a3a4a;margin-top:2px;">
            {st.session_state.run_id}
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:10px 16px 4px 16px;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                    color:#2a3a4a;letter-spacing:2px;margin-bottom:6px;">NAVIGATION</div>
    </div>""", unsafe_allow_html=True)

    nav_pages = [
        "🏠 Welcome",
        "📁 Video Explorer",
        "🔥 GradCAM Viewer",
        "🖼️ Grid Viewer",
        "📊 Timeline",
        "🔍 Search & Filter",
        "⚖️ Compare",
        "📤 Upload Manager",
        "📈 Analytics",
        "📊 Dataset Stats",
        "❌ FP/FN Browser",
        "🧩 Confusion Matrix",
        "🧾 Incident Report",
    ]
    for np_ in nav_pages:
        is_active = st.session_state.nav_page == np_
        if st.button(np_, key=f"nav_{np_}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.nav_page = np_; st.rerun()

    st.markdown("<div style='border-top:1px solid #1a2535;margin:8px 0'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:4px 16px 4px 16px;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                    color:#2a3a4a;letter-spacing:2px;margin-bottom:6px;">⚙ SETTINGS</div>
    </div>""", unsafe_allow_html=True)

    st.session_state.thr_violence   = st.slider("Violence thr",   0.10, 0.99,
                                                  st.session_state.thr_violence,   0.01, key="sb_thr_v")
    st.session_state.thr_suspicious = st.slider("Suspicious thr", 0.10, 0.99,
                                                  st.session_state.thr_suspicious, 0.01, key="sb_thr_s")
    st.session_state.max_frames     = st.number_input("Max frames", 30, 500,
                                                       st.session_state.max_frames, 10, key="sb_maxf")

    st.markdown("<div style='border-top:1px solid #1a2535;margin:8px 0'></div>", unsafe_allow_html=True)

    pred_sb = st.session_state.active_pred
    is_f_sb = is_fight_pred(pred_sb) if pred_sb else False
    badge_color = "#e05252" if is_f_sb else "#52e08a"
    badge_text  = "⚠ FIGHT" if is_f_sb else "✓ NORMAL"
    folder_sb   = st.session_state.active_folder_name or "—"
    st.markdown(f"""
    <div style="padding:8px 16px 16px 16px;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                    color:#2a3a4a;letter-spacing:2px;margin-bottom:6px;">LAST RESULT</div>
        <div style="background:{badge_color}22;border:1px solid {badge_color};
                    border-radius:3px;padding:6px 10px;">
            <div style="font-family:'Orbitron',sans-serif;font-size:11px;
                        font-weight:700;color:{badge_color};">{badge_text}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                        color:#445566;margin-top:3px;word-break:break-all;">{folder_sb}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if st.button("🚪 LOGOUT", use_container_width=True, key="sb_logout"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()


# ══════════════════════════════════════════
# MAIN — top quick-action buttons (hidden on Welcome page)
# ══════════════════════════════════════════
page = st.session_state.nav_page

if page != "🏠 Welcome":
    qa1, qa2, qa3, qa4 = st.columns(4, gap="small")
    with qa1:
        if st.button("📁 VIDEO EXPLORER", use_container_width=True, key="qa_explorer",
                     type="primary" if page == "📁 Video Explorer" else "secondary"):
            st.session_state.nav_page = "📁 Video Explorer"; st.rerun()
    with qa2:
        if st.button("📤 UPLOAD", use_container_width=True, key="qa_upload",
                     type="primary" if page == "📤 Upload Manager" else "secondary"):
            st.session_state.nav_page = "📤 Upload Manager"; st.rerun()
    with qa3:
        if st.button("⚖️ COMPARE", use_container_width=True, key="qa_compare",
                     type="primary" if page == "⚖️ Compare" else "secondary"):
            st.session_state.nav_page = "⚖️ Compare"; st.rerun()
    with qa4:
        if st.button("📊 DATASET STATS", use_container_width=True, key="qa_stats",
                     type="primary" if page == "📊 Dataset Stats" else "secondary"):
            st.session_state.nav_page = "📊 Dataset Stats"; st.rerun()

# Active folder status bar (hidden on Welcome page)
if st.session_state.active_folder_name and page != "🏠 Welcome":
    pred_h = st.session_state.active_pred
    is_f_h = is_fight_pred(pred_h)
    sc = "#e05252" if is_f_h else "#52e08a"
    st.markdown(f"""
    <div style="background:#0d1520;border:1px solid #1a2535;border-left:3px solid {sc};
                border-radius:4px;padding:8px 16px;margin:8px 0;
                display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <span style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#2a3a4a;">ACTIVE</span>
        <span style="font-family:'Rajdhani',sans-serif;font-weight:700;color:#e8f4ff;font-size:13px;">
            {st.session_state.active_folder_name}</span>
        <span style="color:#1a2535;">|</span>
        <span style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#2a3a4a;">DATASET</span>
        <span style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#7ecfff;">
            {st.session_state.active_dataset}/{st.session_state.active_class}</span>
        <span style="color:#1a2535;">|</span>
        <span style="font-family:'Orbitron',sans-serif;font-size:11px;font-weight:700;color:{sc};">
            {'⚠ FIGHT' if is_f_h else '✓ NORMAL'}</span>
        <span style="color:#1a2535;">|</span>
        <span style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#2a3a4a;">CONF</span>
        <span style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#c8d8e8;">
            {pred_h.get('confidence','?')}</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# PAGE: WELCOME
# ══════════════════════════════════════════
if page == "🏠 Welcome":
    records_count = len(get_all_pred_records())
    username = st.session_state.username
    hour     = datetime.now().hour
    greeting = "Good morning ☀️" if hour < 12 else ("Good afternoon 🌤️" if hour < 18 else "Good evening 🌙")
    today    = datetime.now().strftime("%A, %d %b %Y")

    st.markdown(f"""
    <div style="text-align:center;padding:40px 0 28px 0;animation:fadein-up 0.5s ease both;">
        <div style="font-size:64px;display:inline-block;
                    animation:pulse-shield 3s ease-in-out infinite;
                    filter:drop-shadow(0 0 32px rgba(224,82,82,0.55));">🛡️</div>
        <div style="font-family:'Orbitron',sans-serif;font-size:clamp(1.6rem,3.5vw,2.4rem);
                    font-weight:900;color:#e8f4ff;letter-spacing:6px;margin-top:16px;
                    text-shadow:0 0 40px rgba(126,207,255,0.15);">
            WELCOME BACK
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                    color:#2a3a4a;letter-spacing:4px;margin-top:6px;">
            VISIONGUARD &nbsp;·&nbsp; VIOLENCE DETECTION v6
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:12px;
                    color:#445566;margin-top:12px;letter-spacing:1px;">
            {greeting} &nbsp;·&nbsp;
            <span style="color:#7ecfff;font-size:14px;font-weight:bold;">{username}</span>
            &nbsp;·&nbsp; {today}
        </div>
    </div>

    <div class="vg-stat-row" style="max-width:860px;margin:0 auto 0 auto;">
        <div class="vg-stat-tile">
            <div class="vg-stat-icon">📂</div>
            <div class="vg-stat-val" style="color:#7ecfff;">{records_count if records_count else "0"}</div>
            <div class="vg-stat-label">Videos Loaded</div>
        </div>
        <div class="vg-stat-tile blue">
            <div class="vg-stat-icon">🔥</div>
            <div class="vg-stat-val" style="color:#7ecfff;">4</div>
            <div class="vg-stat-label">CAM Methods</div>
        </div>
        <div class="vg-stat-tile green">
            <div class="vg-stat-icon">🥊</div>
            <div class="vg-stat-val" style="color:#52e08a;">ON</div>
            <div class="vg-stat-label">ByteTrack</div>
        </div>
        <div class="vg-stat-tile orange">
            <div class="vg-stat-icon">🧠</div>
            <div class="vg-stat-val" style="color:#f5a623;">R3D-18</div>
            <div class="vg-stat-label">Backbone</div>
        </div>
    </div>

    <div style="max-width:860px;margin:0 auto;">
        <div class="vg-divider-label" style="margin-top:36px;">── HOW TO GET STARTED ──</div>
        <div class="vg-steps-row">
            <div class="vg-step-card red">
                <div class="vg-step-num">01</div>
                <div class="vg-step-icon">📤</div>
                <div class="vg-step-title" style="color:#e05252;">UPLOAD</div>
                <div class="vg-step-desc">Use Upload Manager to add your GradCAM output folders — mp4 videos, grid PNGs, and pred.txt.</div>
            </div>
            <div class="vg-step-card blue">
                <div class="vg-step-num">02</div>
                <div class="vg-step-icon">🔍</div>
                <div class="vg-step-title" style="color:#7ecfff;">ANALYZE</div>
                <div class="vg-step-desc">Pick dataset → class → folder in Video Explorer, then hit Analyze Folder to load predictions and videos.</div>
            </div>
            <div class="vg-step-card green">
                <div class="vg-step-num">03</div>
                <div class="vg-step-icon">🔥</div>
                <div class="vg-step-title" style="color:#52e08a;">EXPLORE CAMS</div>
                <div class="vg-step-desc">Browse GradCAM Viewer for all 6 methods, Grid Viewer for frame grids, Timeline for onset detection.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="vg-divider-label" style="margin-top:32px;max-width:860px;margin-left:auto;margin-right:auto;">── QUICK LAUNCH ──</div>', unsafe_allow_html=True)

    _, c1, c2, c3, c4, _ = st.columns([0.15, 1, 1, 1, 1, 0.15], gap="small")
    quick = [
        (c1, "📁  VIDEO EXPLORER",   "📁 Video Explorer"),
        (c2, "📤  UPLOAD MANAGER",   "📤 Upload Manager"),
        (c3, "📊  DATASET STATS",    "📊 Dataset Stats"),
        (c4, "🧩  CONFUSION MATRIX", "🧩 Confusion Matrix"),
    ]
    for col, label, target in quick:
        with col:
            if st.button(label, use_container_width=True, key=f"wel_ql_{target}", type="secondary"):
                st.session_state.nav_page = target; st.rerun()

    st.markdown("""
    <div style="max-width:860px;margin:32px auto 0 auto;
                background:#0d1520;border:1px solid #1a2535;border-left:4px solid #e05252;
                border-radius:10px;padding:20px 24px;">
        <div style="font-family:'Orbitron',sans-serif;font-size:11px;font-weight:900;
                    color:#e05252;letter-spacing:3px;margin-bottom:12px;">
            🛡️ SYSTEM CAPABILITIES
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;
                    font-family:'Share Tech Mono',monospace;font-size:11px;">
            <div>
                <div style="color:#445566;font-size:9px;letter-spacing:2px;margin-bottom:6px;">DETECTION</div>
                <div style="color:#c8d8e8;line-height:1.8;">
                    Real-time violence detection<br>
                    Fight onset localization<br>
                    Confidence scoring<br>
                    Multi-window analysis
                </div>
            </div>
            <div>
                <div style="color:#445566;font-size:9px;letter-spacing:2px;margin-bottom:6px;">VISUALIZATION</div>
                <div style="color:#c8d8e8;line-height:1.8;">
                    GradCAM heatmaps<br>
                    GradCAM++ overlays<br>
                    Smooth GradCAM++<br>
                    LayerCAM · Combined
                </div>
            </div>
            <div>
                <div style="color:#445566;font-size:9px;letter-spacing:2px;margin-bottom:6px;">TRACKING</div>
                <div style="color:#c8d8e8;line-height:1.8;">
                    ByteTrack fighter IDs<br>
                    CAM-focused tracking<br>
                    Incident reporting<br>
                    PDF export
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# PAGE: VIDEO EXPLORER
# ══════════════════════════════════════════
elif page == "📁 Video Explorer":
    st.subheader("📁 Video Explorer")

    s1, s2, s3 = st.columns([1, 1, 2])
    with s1: sel_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="ex_ds")
    with s2: sel_cls = st.selectbox("Class",   DATASETS[sel_ds],      key="ex_cls")
    with s3:
        vfolders = list_video_folders(sel_ds, sel_cls)
        if not vfolders:
            st.info(f"No folders for **{sel_ds}/{sel_cls}**. Use Upload Manager.")
            sel_folder = None
        else:
            sel_folder = st.selectbox(f"Folder ({len(vfolders)} available)",
                                      [f.name for f in vfolders], key="ex_folder")

    if st.button("🔍 ANALYZE FOLDER", type="primary",
                 disabled=(sel_folder is None), key="analyze_btn"):
        folder_path = class_root(sel_ds, sel_cls) / sel_folder
        status_ph = st.empty()
        bar_ph    = st.empty()

        def _show_step(pct, msg, color="#7ecfff"):
            status_ph.markdown(
                f"<div style='font-family:Share Tech Mono,monospace;font-size:12px;"
                f"color:{color};letter-spacing:2px;padding:4px 0;'>{msg}</div>",
                unsafe_allow_html=True)
            bar_ph.progress(pct)

        _show_step(0.05, "⟳  INITIALIZING ANALYSIS..."); time.sleep(0.1)
        _show_step(0.15, "📂  READING FOLDER STRUCTURE...")
        files = get_files(folder_path); time.sleep(0.12)
        _show_step(0.30, "📋  PARSING PRED.TXT...")
        pred = parse_pred_txt(files["pred"]) if "pred" in files else {}; time.sleep(0.12)
        _show_step(0.50, "🎬  LOADING VIDEO FRAMES...")
        frames, fps = [], float(CFG.DEFAULT_FPS)
        if "original" in files:
            try:
                frames, fps = read_video_frames(files["original"],
                                                max_frames=st.session_state.max_frames)
                frames = [resize_keep(f, 640) for f in frames]
            except: pass
        time.sleep(0.08)
        _show_step(0.70, "📊  COMPUTING PROBABILITY CURVES...")
        n      = len(frames) if frames else 100
        scores = scores_from_pred(pred, n, fps); time.sleep(0.08)
        _show_step(0.85, "🎯  PRE-CONVERTING PREVIEW VIDEOS...")
        for _vk in ["original", "gradcampp", "combined"]:
            if _vk in files:
                _fp = files[_vk].parent / f"_preview_{_vk}.mp4"
                if not (_fp.exists() and _fp.stat().st_size > 5000):
                    make_web_preview(files[_vk], _fp)
        time.sleep(0.05)
        _show_step(1.0, "✅  ANALYSIS COMPLETE", color="#52e08a"); time.sleep(0.35)
        status_ph.empty(); bar_ph.empty()

        st.session_state.active_pred        = pred
        st.session_state.active_scores      = scores
        st.session_state.active_fps         = fps
        st.session_state.active_frames      = frames
        st.session_state.active_folder_name = sel_folder
        st.session_state.active_video_path  = str(files.get("original", ""))
        st.session_state.active_dataset     = sel_ds
        st.session_state.active_class       = sel_cls
        st.session_state["_active_files"]   = {k: str(v) for k, v in files.items()}
        st.rerun()

    # ── Results ────────────────────────────────────────────────
    if st.session_state.active_folder_name:
        pred        = st.session_state.active_pred
        scores      = st.session_state.active_scores
        fps         = st.session_state.active_fps
        folder_name = st.session_state.active_folder_name
        files       = {k: Path(v) for k, v in st.session_state.get("_active_files", {}).items()}
        is_fight    = is_fight_pred(pred)
        onset_f     = pred.get("onset_frame", "N/A")
        onset_t     = pred.get("onset_time",  "N/A")
        conf_val    = pred.get("confidence",  "?")
        fighters    = pred.get("fighter_ids", "N/A")

        if is_fight:
            st.markdown(f"""
            <div style="background:rgba(224,82,82,0.10);border:1px solid #e05252;
                        border-left:4px solid #e05252;border-radius:4px;
                        padding:14px 20px;margin:12px 0;">
                <div style="font-family:'Orbitron',sans-serif;font-size:15px;
                            font-weight:900;color:#ff7070;letter-spacing:3px;margin-bottom:8px;">
                    🚨 VIOLENCE DETECTED
                </div>
                <div style="display:flex;gap:28px;flex-wrap:wrap;
                            font-family:'Share Tech Mono',monospace;font-size:12px;">
                    <span style="color:#886666;">ONSET FRAME <span style="color:#ffbbbb;">{onset_f}</span></span>
                    <span style="color:#886666;">TIME <span style="color:#ffbbbb;">{onset_t}</span></span>
                    <span style="color:#886666;">CONFIDENCE <span style="color:#ffbbbb;">{conf_val}</span></span>
                    <span style="color:#886666;">FIGHTERS <span style="color:#7ecfff;">{fighters}</span></span>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(82,224,138,0.06);border:1px solid #52e08a44;
                        border-left:4px solid #52e08a;border-radius:4px;
                        padding:12px 20px;margin:12px 0;">
                <div style="font-family:'Orbitron',sans-serif;font-size:13px;
                            font-weight:700;color:#52e08a;letter-spacing:2px;margin-bottom:4px;">
                    ✓ NO VIOLENCE DETECTED
                </div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#336644;">
                    CONFIDENCE: <span style="color:#88ccaa;">{conf_val}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        render_pred_card(pred)
        st.markdown("---")

        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                        color:#2a3a4a;letter-spacing:2px;margin-bottom:8px;">
                        🎬  VIDEO ANALYSIS</div>""", unsafe_allow_html=True)

        vid_col_l, vid_col_r = st.columns(2, gap="large")
        with vid_col_l:
            st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                            color:#7ecfff;letter-spacing:1px;margin-bottom:4px;
                            padding:4px 8px;background:#0d1520;border-radius:3px;
                            display:inline-block;">📹 ORIGINAL</div>""", unsafe_allow_html=True)
            if "original" in files:
                fp = files["original"].parent / "_preview_original.mp4"
                if not (fp.exists() and fp.stat().st_size > 5000):
                    with st.spinner("Converting..."):
                        make_web_preview(files["original"], fp)
                _safe_video(fp if (fp.exists() and fp.stat().st_size > 5000) else files["original"])
            else:
                st.info("No original video found.")

        with vid_col_r:
            right_vk = next((k for k in ["gradcampp", "combined", "gradcam"] if k in files), None)
            right_label = {"gradcampp":"🔥 GRADCAM++","combined":"🎯 COMBINED",
                           "gradcam":"🔥 GRADCAM"}.get(right_vk, "CAM")
            st.markdown(f"""<div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                            color:#e05252;letter-spacing:1px;margin-bottom:4px;
                            padding:4px 8px;background:#0d1520;border-radius:3px;
                            display:inline-block;">{right_label}</div>""", unsafe_allow_html=True)
            if right_vk:
                fp = files[right_vk].parent / f"_preview_{right_vk}.mp4"
                if not (fp.exists() and fp.stat().st_size > 5000):
                    with st.spinner("Converting..."):
                        make_web_preview(files[right_vk], fp)
                _safe_video(fp if (fp.exists() and fp.stat().st_size > 5000) else files[right_vk])
            else:
                st.info("No CAM video found. Use GradCAM Viewer.")

        if scores is not None and fps:
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            stat = pred_label_to_status(pred.get("pred_label", ""))
            m1.metric("Status",     f"{color_from_status(stat)} {stat}")
            m2.metric("Confidence", conf_val)
            m3.metric("Onset Time", onset_t)
            m4.metric("Fighters",   fighters)
            c1, c2 = st.columns(2)
            with c1: st.pyplot(make_timeline_plot(scores, fps, pred), clear_figure=True)
            with c2: st.pyplot(make_hist_plot(scores), clear_figure=True)

        st.markdown("---")
        if st.button("📄 GENERATE PDF", key="ex_pdf_btn"):
            with st.spinner("Generating..."):
                pdf = generate_pdf_report(pred, scores or np.zeros(10), fps or 25.0, folder_name)
            st.download_button("⬇️ Download PDF", data=pdf,
                               file_name=f"report_{folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                               mime="application/pdf", key="ex_pdf_dl")

    else:
        # ══════════════════════════════════════════
        # ✨ WELCOME LANDING — shown after login
        # ══════════════════════════════════════════
        records_count = len(get_all_pred_records())
        username = st.session_state.username
        hour     = datetime.now().hour
        greeting = "Good morning ☀️" if hour < 12 else ("Good afternoon 🌤️" if hour < 18 else "Good evening 🌙")
        today    = datetime.now().strftime("%A, %d %b %Y")

        # ── Hero ───────────────────────────────────────────────
        st.markdown(f"""
        <div class="vg-welcome" style="text-align:center;padding:32px 0 24px 0;">
            <div style="font-size:56px;animation:pulse-shield 3s ease-in-out infinite;
                        display:inline-block;
                        filter:drop-shadow(0 0 28px rgba(224,82,82,0.45));">🛡️</div>
            <div style="font-family:'Orbitron',sans-serif;font-size:clamp(1.4rem,3vw,2rem);
                        font-weight:900;color:#e8f4ff;letter-spacing:5px;margin-top:14px;">
                WELCOME BACK
            </div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                        color:#2a3a4a;letter-spacing:3px;margin-top:4px;">
                VISIONGUARD · VIOLENCE DETECTION v6
            </div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:12px;
                        color:#445566;margin-top:10px;letter-spacing:1px;">
                {greeting} &nbsp;·&nbsp;
                <span style="color:#7ecfff;font-size:13px;">{username}</span>
                &nbsp;·&nbsp; {today}
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Stat tiles ─────────────────────────────────────────
        st.markdown(f"""
        <div class="vg-stat-row">
            <div class="vg-stat-tile">
                <div class="vg-stat-icon">📂</div>
                <div class="vg-stat-val" style="color:#7ecfff;">{records_count if records_count else "0"}</div>
                <div class="vg-stat-label">Videos Loaded</div>
            </div>
            <div class="vg-stat-tile blue">
                <div class="vg-stat-icon">🔥</div>
                <div class="vg-stat-val" style="color:#7ecfff;">4</div>
                <div class="vg-stat-label">CAM Methods</div>
            </div>
            <div class="vg-stat-tile green">
                <div class="vg-stat-icon">🥊</div>
                <div class="vg-stat-val" style="color:#52e08a;">ON</div>
                <div class="vg-stat-label">ByteTrack</div>
            </div>
            <div class="vg-stat-tile orange">
                <div class="vg-stat-icon">🧠</div>
                <div class="vg-stat-val" style="color:#f5a623;">R3D-18</div>
                <div class="vg-stat-label">Backbone</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── How to get started ─────────────────────────────────
        st.markdown("""
        <div class="vg-divider-label">── HOW TO GET STARTED ──</div>
        <div class="vg-steps-row">
            <div class="vg-step-card red">
                <div class="vg-step-num">01</div>
                <div class="vg-step-icon">📤</div>
                <div class="vg-step-title" style="color:#e05252;">UPLOAD</div>
                <div class="vg-step-desc">Use Upload Manager to add your GradCAM output folders — mp4 videos, grid PNGs, and pred.txt.</div>
            </div>
            <div class="vg-step-card blue">
                <div class="vg-step-num">02</div>
                <div class="vg-step-icon">🔍</div>
                <div class="vg-step-title" style="color:#7ecfff;">ANALYZE</div>
                <div class="vg-step-desc">Pick dataset → class → folder above, then hit Analyze Folder to load predictions and videos.</div>
            </div>
            <div class="vg-step-card green">
                <div class="vg-step-num">03</div>
                <div class="vg-step-icon">🔥</div>
                <div class="vg-step-title" style="color:#52e08a;">EXPLORE CAMS</div>
                <div class="vg-step-desc">Browse GradCAM Viewer for all 6 methods, Grid Viewer for frame grids, Timeline for onset.</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Quick launch ────────────────────────────────────────
        st.markdown("""
        <div class="vg-divider-label" style="margin-top:28px;">── QUICK LAUNCH ──</div>""",
        unsafe_allow_html=True)

        ql1, ql2, ql3, ql4 = st.columns(4, gap="small")
        quick = [
            (ql1, "📤  UPLOAD MANAGER",   "📤 Upload Manager"),
            (ql2, "🔥  GRADCAM VIEWER",   "🔥 GradCAM Viewer"),
            (ql3, "📊  DATASET STATS",    "📊 Dataset Stats"),
            (ql4, "🧩  CONFUSION MATRIX", "🧩 Confusion Matrix"),
        ]
        for col, label, target in quick:
            with col:
                if st.button(label, use_container_width=True, key=f"ql_{target}"):
                    st.session_state.nav_page = target; st.rerun()


# ══════════════════════════════════════════
# PAGE: GRADCAM VIEWER
# ══════════════════════════════════════════
elif page == "🔥 GradCAM Viewer":
    st.subheader("🔥 GradCAM Viewer — All 6 Methods")

    if not st.session_state.active_folder_name:
        st.info("Analyze a folder in **Video Explorer** first.")
    else:
        files = {k: Path(v) for k, v in st.session_state.get("_active_files", {}).items()}
        pred  = st.session_state.active_pred

        if is_fight_pred(pred):
            show_fight_alert(st.session_state.active_folder_name, pred.get("confidence","?"))

        fighter_ids = pred.get("fighter_ids", "")
        if fighter_ids and "None" not in fighter_ids:
            st.markdown(f"""
            <div style="background:#0d1520;border:1px solid #7ecfff33;border-radius:4px;
                        padding:8px 14px;margin-bottom:8px;
                        font-family:'Share Tech Mono',monospace;font-size:12px;">
                🥊 <span style="color:#7ecfff;">BYTETRACK FIGHTERS: {fighter_ids}</span>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <span style="color:#52e08a;">CAM FOCUS: {pred.get('cam_focused','N/A')}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("##### Row 1: Original · GradCAM · GradCAM++")
        r1_cols = st.columns(3)
        for i, vk in enumerate(["original", "gradcam", "gradcampp"]):
            with r1_cols[i]:
                st.markdown(f"**{VID_LABELS.get(vk, vk)}**")
                if vk in files:
                    fp = files[vk].parent / f"_preview_{vk}.mp4"
                    if not (fp.exists() and fp.stat().st_size > 5000):
                        with st.spinner(f"Converting {vk}..."):
                            make_web_preview(files[vk], fp)
                    _safe_video(fp if fp.exists() and fp.stat().st_size > 5000 else files[vk])
                else:
                    st.info(f"No {vk} video found.")

        st.markdown("---")
        st.markdown("##### Row 2: Smooth GradCAM++ · LayerCAM · Combined")
        r2_cols = st.columns(3)
        for i, vk in enumerate(["smooth_gradcampp", "layercam", "combined"]):
            with r2_cols[i]:
                st.markdown(f"**{VID_LABELS.get(vk, vk)}**")
                if vk in files:
                    fp = files[vk].parent / f"_preview_{vk}.mp4"
                    if not (fp.exists() and fp.stat().st_size > 5000):
                        with st.spinner(f"Converting {vk}..."):
                            make_web_preview(files[vk], fp)
                    _safe_video(fp if fp.exists() and fp.stat().st_size > 5000 else files[vk])
                else:
                    st.info(f"No {vk} video found.")

        st.markdown("---")
        st.markdown("#### 📋 Prediction Details")
        render_pred_card(pred)


# ══════════════════════════════════════════
# PAGE: GRID VIEWER
# ══════════════════════════════════════════
elif page == "🖼️ Grid Viewer":
    st.subheader("🖼️ Frame Grid Viewer — All 6 Methods")
    if not st.session_state.active_folder_name:
        st.info("Analyze a folder in **Video Explorer** first.")
    else:
        files = {k: Path(v) for k, v in st.session_state.get("_active_files", {}).items()}
        st.markdown("##### Row 1: Raw · GradCAM · GradCAM++")
        g1 = st.columns(3)
        for i, gk in enumerate(["raw_grid", "gradcam_grid", "gradcampp_grid"]):
            with g1[i]:
                st.markdown(f"**{GRID_LABELS.get(gk, gk)}**")
                if gk in files: st.image(str(files[gk]), use_container_width=True)
                else: st.info(f"No {gk}.png found.")
        st.markdown("---")
        st.markdown("##### Row 2: Smooth GradCAM++ · LayerCAM · Combined")
        g2 = st.columns(3)
        for i, gk in enumerate(["smooth_gradcampp_grid", "layercam_grid", "combined_grid"]):
            with g2[i]:
                st.markdown(f"**{GRID_LABELS.get(gk, gk)}**")
                if gk in files: st.image(str(files[gk]), use_container_width=True)
                else: st.info(f"No {gk}.png found.")
        if "timeline" in files:
            st.markdown("---")
            st.markdown("##### 📈 P(fight) Timeline")
            st.image(str(files["timeline"]), use_container_width=True)


# ══════════════════════════════════════════
# PAGE: TIMELINE
# ══════════════════════════════════════════
elif page == "📊 Timeline":
    st.subheader("📊 P(fight) Timeline")
    if not st.session_state.active_folder_name:
        st.info("Analyze a folder in **Video Explorer** first.")
    else:
        files  = {k: Path(v) for k, v in st.session_state.get("_active_files", {}).items()}
        pred   = st.session_state.active_pred
        scores = st.session_state.active_scores
        fps    = st.session_state.active_fps
        if "timeline" in files:
            st.image(str(files["timeline"]), use_container_width=True,
                     caption="Generated by gradcam script — exact model scores")
            st.markdown("---")
            st.markdown("*Below: dashboard-reconstructed scores from pred.txt metadata*")
        if scores is not None and fps:
            st.pyplot(make_timeline_plot(scores, fps, pred), clear_figure=True)
            st.pyplot(make_hist_plot(scores), clear_figure=True)
            step = max(1, int(0.5*fps))
            rows = [{"time": fmt_time(i/fps), "frame": i, "prob": round(float(scores[i]), 4)}
                    for i in range(0, len(scores), step)]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=220)


# ══════════════════════════════════════════
# PAGE: SEARCH & FILTER
# ══════════════════════════════════════════
elif page == "🔍 Search & Filter":
    st.subheader("🔍 Search & Filter")
    records = get_all_pred_records()
    if not records:
        st.info("No pred.txt files found. Upload folders first.")
    else:
        df_all = pd.DataFrame(records)
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1: sq   = st.text_input("Search name", placeholder="e.g. fi1", key="sf_search")
        with fc2: dsf  = st.selectbox("Dataset", ["All"]+list(DATASETS.keys()), key="sf_ds")
        with fc3: clsf = st.selectbox("Class", ["All","Fight","NonFight","Nonfight","pred_nonfight"], key="sf_cls")
        with fc4: corf = st.selectbox("Correct?", ["All","True","False"], key="sf_cor")
        df = df_all.copy()
        if sq:          df = df[df["_folder"].str.contains(sq, case=False, na=False)]
        if dsf  != "All": df = df[df["_dataset"] == dsf]
        if clsf != "All": df = df[df["_class"]   == clsf]
        if corf != "All": df = df[df["correct"].str.lower() == corf.lower()]
        st.markdown(f"**{len(df)} results**")
        show = ["_folder","_dataset","_class","true_label","pred_label",
                "correct","confidence","onset_time","fighter_ids"]
        show = [c for c in show if c in df.columns]
        st.dataframe(df[show].rename(columns={"_folder":"Folder","_dataset":"Dataset","_class":"Class"}),
                     use_container_width=True, height=380)
        st.download_button("⬇️ Download CSV", data=df[show].to_csv(index=False).encode(),
                           file_name="filtered.csv", mime="text/csv", key="sf_csv")


# ══════════════════════════════════════════
# PAGE: COMPARE
# ══════════════════════════════════════════
elif page == "⚖️ Compare":
    st.subheader("⚖️ Side-by-Side Comparison")

    def folder_sel(prefix, col):
        with col:
            ds  = st.selectbox("Dataset", list(DATASETS.keys()), key=f"{prefix}_ds")
            cls = st.selectbox("Class",   DATASETS[ds],           key=f"{prefix}_cls")
            vf  = list_video_folders(ds, cls)
            if not vf: st.info("No folders."); return None, None, None
            fn = st.selectbox(f"Folder ({len(vf)})", [f.name for f in vf], key=f"{prefix}_fn")
            return ds, cls, fn

    lc, rc = st.columns(2, gap="large")
    l_ds, l_cls, l_fn = folder_sel("cmp_l", lc)
    r_ds, r_cls, r_fn = folder_sel("cmp_r", rc)

    if st.button("⚖️ COMPARE", type="primary", disabled=(not l_fn or not r_fn), key="cmp_btn"):
        for col, ds, cls, fn, side in [(lc,l_ds,l_cls,l_fn,"LEFT"),(rc,r_ds,r_cls,r_fn,"RIGHT")]:
            fp   = class_root(ds, cls) / fn
            fls  = get_files(fp)
            pred = parse_pred_txt(fls["pred"]) if "pred" in fls else {}
            with col:
                st.markdown(f"### {side}: `{fn}`")
                is_f = is_fight_pred(pred)
                if is_f: st.error(f"🔴 FIGHT — {pred.get('confidence','?')} conf")
                else:    st.success(f"🟢 NORMAL — {pred.get('confidence','?')} conf")
                ok = "✅" if str(pred.get("correct","")).lower()=="true" else "❌"
                st.markdown(f"**True:** {pred.get('true_label','?')} | "
                            f"**Pred:** {pred.get('pred_label','?')} | **Correct:** {ok}")
                st.markdown(f"**Fighters:** {pred.get('fighter_ids','N/A')}")
                if "original" in fls:
                    pp = fp / "_preview_original.mp4"
                    if not pp.exists(): make_web_preview(fls["original"], pp)
                    _safe_video(pp if pp.exists() else fls["original"])
                for gk in ["combined_grid", "gradcam_grid"]:
                    if gk in fls:
                        st.image(str(fls[gk]), use_container_width=True, caption=GRID_LABELS.get(gk, gk))
                        break
                if "timeline" in fls:
                    st.image(str(fls["timeline"]), use_container_width=True, caption="Timeline")


# ══════════════════════════════════════════
# PAGE: UPLOAD MANAGER
# ══════════════════════════════════════════
elif page == "📤 Upload Manager":
    st.subheader("📤 Upload Manager")
    with st.expander("🗑️ Clear All Uploads", expanded=False):
        st.warning("⚠️ This will permanently delete ALL uploaded data.")
        if not st.session_state._confirm_clear:
            if st.button("🗑️ CLEAR ALL", key="clear_btn"):
                st.session_state._confirm_clear = True; st.rerun()
        else:
            st.error("Are you sure? Cannot be undone.")
            cy, cn = st.columns(2)
            with cy:
                if st.button("✅ YES DELETE", type="primary", key="confirm_yes"):
                    clear_all_uploads()
                    for k in ["active_pred","active_scores","active_fps","active_frames",
                              "active_folder_name","active_video_path","_active_files"]:
                        st.session_state[k] = {} if "pred" in k or "files" in k else None
                    st.session_state.active_folder_name = ""
                    st.session_state._confirm_clear = False
                    st.success("✅ Cleared!"); st.rerun()
            with cn:
                if st.button("❌ Cancel", key="confirm_no"):
                    st.session_state._confirm_clear = False; st.rerun()

    st.divider()
    upload_mode = st.radio("Mode", ["📁 Single folder","🗜️ ZIP file"], horizontal=True, key="up_mode")
    if upload_mode == "📁 Single folder":
        uc1, uc2, uc3 = st.columns(3)
        with uc1: up_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="up_ds")
        with uc2: up_cls = st.selectbox("Class",   DATASETS[up_ds],       key="up_cls")
        with uc3: fn_inp = st.text_input("Folder name", placeholder="e.g. fi1_xvid", key="up_fn_inp")
        up_files = st.file_uploader(
            "Files (mp4 + png + pred.txt — all 14 files per video)",
            type=["mp4","avi","mov","mkv","png","jpg","txt"],
            accept_multiple_files=True, key="up_files")
        st.caption("Expected: _original.mp4, _gradcam.mp4, _gradcampp.mp4, _smooth_gradcampp.mp4, "
                   "_layercam.mp4, _combined.mp4, raw_grid.png, gradcam_grid.png, gradcampp_grid.png, "
                   "smooth_gradcampp_grid.png, layercam_grid.png, combined_grid.png, timeline.png, pred.txt")
        if st.button("💾 SAVE", type="primary", disabled=(not up_files or not fn_inp.strip()), key="up_save"):
            dest = class_root(up_ds, up_cls) / fn_inp.strip()
            dest.mkdir(parents=True, exist_ok=True)
            for uf in up_files:
                with open(dest / uf.name, "wb") as f: f.write(uf.getbuffer())
            st.success(f"✅ Saved {len(up_files)} file(s) → `{up_ds}/{up_cls}/{fn_inp.strip()}`")
    else:
        uc1, uc2 = st.columns(2)
        with uc1: zip_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="zip_ds")
        with uc2: zip_cls = st.selectbox("Class",   DATASETS[zip_ds],      key="zip_cls")
        zf = st.file_uploader("Upload ZIP", type=["zip"], key="zip_up")
        if st.button("📦 EXTRACT", type="primary", disabled=(not zf), key="zip_extract"):
            with st.spinner("Extracting..."):
                n_f, n_files = extract_zip_to_uploads(zf.read(), zip_ds, zip_cls)
            st.success(f"✅ Extracted {n_f} folder(s), {n_files} file(s)")

    st.divider()
    st.markdown("### 📂 Uploaded Folders")
    found = False
    for ds in DATASETS:
        for cls in DATASETS[ds]:
            flist = list_video_folders(ds, cls)
            if flist:
                found = True
                with st.expander(f"**{ds}/{cls}** — {len(flist)} folder(s)", expanded=False):
                    for fl in flist:
                        ffiles = list(fl.iterdir())
                        n_mp4  = len([f for f in ffiles if f.suffix == ".mp4"])
                        n_png  = len([f for f in ffiles if f.suffix == ".png"])
                        has_pred = any(f.name == "pred.txt" for f in ffiles)
                        st.markdown(
                            f"📁 **{fl.name}** — {n_mp4} videos · {n_png} grids · "
                            f"{'✅ pred.txt' if has_pred else '❌ no pred.txt'}")
    if not found: st.info("Nothing uploaded yet.")


# ══════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════
elif page == "📈 Analytics":
    st.subheader("📈 Analytics")
    scores = st.session_state.active_scores
    fps    = st.session_state.active_fps
    pred   = st.session_state.active_pred
    fname  = st.session_state.active_folder_name
    if scores is None or fps is None:
        st.info("Analyze a folder in **Video Explorer** first.")
    else:
        st.markdown(f"Showing: **{fname}**")
        step = max(1, int(0.5*fps))
        rows = [{"time": fmt_time(i/fps), "frame": i, "prob": round(float(scores[i]), 4)}
                for i in range(0, len(scores), step)]
        cA, cB = st.columns([1.3, 1.0], gap="large")
        with cA:
            st.markdown("#### Frame Log")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)
        with cB:
            st.markdown("#### Summary")
            for label, key in [("Prediction", "pred_label"), ("Confidence", "confidence"),
                                ("Onset Frame", "onset_frame"), ("Onset Time", "onset_time"),
                                ("Total Frames", "total_frames"), ("Val Acc", "model_val_acc"),
                                ("Fighters", "fighter_ids"), ("CAM Focus", "cam_focused")]:
                st.metric(label, pred.get(key, "?"))
        c1, c2 = st.columns(2)
        with c1: st.pyplot(make_timeline_plot(scores, fps, pred), clear_figure=True)
        with c2: st.pyplot(make_hist_plot(scores), clear_figure=True)


# ══════════════════════════════════════════
# PAGE: DATASET STATS
# ══════════════════════════════════════════
elif page == "📊 Dataset Stats":
    st.markdown("""
    <div style="font-family:'Orbitron',sans-serif;font-size:1rem;font-weight:900;
                color:#e8f4ff;letter-spacing:2px;margin-bottom:16px;">
        📊 DATASET ACCURACY REPORT
    </div>""", unsafe_allow_html=True)

    records = get_all_pred_records()
    if not records:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;font-family:'Share Tech Mono',monospace;
                    font-size:12px;color:#2a3a4a;">
            NO PRED.TXT FILES FOUND<br>
            <span style="color:#445566;">Upload your GradCAM output folders first.</span>
        </div>""", unsafe_allow_html=True)
    else:
        df = pd.DataFrame(records)

        def ds_stats(sub):
            total = len(sub)
            if total == 0: return total, 0, 0.0, 0, 0, 0, 0
            correct = sub["correct"].str.lower().eq("true").sum() if "correct" in sub.columns else 0
            acc     = correct / total
            fight = sub[sub["true_label"].str.lower() == "fight"] \
                    if "true_label" in sub.columns else sub.iloc[0:0]
            non   = sub[sub["true_label"].str.lower().str.contains("non", na=False)] \
                    if "true_label" in sub.columns else sub.iloc[0:0]
            tp = int(fight["correct"].str.lower().eq("true").sum()) if len(fight) else 0
            tn = int(non["correct"].str.lower().eq("true").sum())   if len(non)   else 0
            fp = int(non["correct"].str.lower().ne("true").sum())   if len(non)   else 0
            fn = int(fight["correct"].str.lower().ne("true").sum()) if len(fight) else 0
            return total, int(correct), acc, tp, tn, fp, fn

        total_all, correct_all, acc_all, tp_all, tn_all, fp_all, fn_all = ds_stats(df)
        hf_df  = df[df["_dataset"] == "hockeyfight"] if "_dataset" in df.columns else df.iloc[0:0]
        rwf_df = df[df["_dataset"] == "rwf"]          if "_dataset" in df.columns else df.iloc[0:0]
        _, hf_c,  hf_acc,  hf_tp,  hf_tn,  hf_fp,  hf_fn  = ds_stats(hf_df)
        _, rwf_c, rwf_acc, rwf_tp, rwf_tn, rwf_fp, rwf_fn = ds_stats(rwf_df)

        om1, om2, om3, om4, om5 = st.columns(5, gap="small")
        om1.metric("Total Videos",     str(total_all))
        om2.metric("Correct",          str(correct_all))
        om3.metric("Overall Accuracy", f"{acc_all:.1%}")
        om4.metric("True Positives",   str(tp_all))
        om5.metric("True Negatives",   str(tn_all))

        st.markdown("<div style='margin:18px 0 8px 0'></div>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                        color:#2a3a4a;letter-spacing:2px;margin-bottom:10px;">
                        PER-DATASET BREAKDOWN</div>""", unsafe_allow_html=True)

        def acc_bar(pct, color):
            filled = int(pct * 20)
            return (f'<span style="font-family:monospace;color:{color};font-size:13px;">'
                    f'{"█"*filled}{"░"*(20-filled)}</span>')

        def ds_card(col, name, emoji, total, correct, acc, tp, tn, fp, fn, color):
            prec = tp/(tp+fp) if (tp+fp) > 0 else 0
            rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
            col.markdown(f"""
            <div style="background:#0d1520;border:1px solid #1a2535;border-left:4px solid {color};
                        border-radius:10px;padding:20px;">
                <div style="font-family:'Orbitron',sans-serif;font-size:13px;font-weight:900;
                            color:{color};letter-spacing:2px;margin-bottom:2px;">
                    {emoji} {name.upper()}</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                            color:#2a3a4a;margin-bottom:14px;">{total} videos analyzed</div>
                <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:6px;">
                    <div style="font-family:'Orbitron',sans-serif;font-size:2rem;
                                font-weight:900;color:#e8f4ff;">{acc:.1%}</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                                color:#445566;">ACCURACY</div>
                </div>
                <div style="margin-bottom:14px;">{acc_bar(acc, color)}</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;
                            font-family:'Share Tech Mono',monospace;font-size:11px;">
                    <div style="background:#080c10;border-radius:4px;padding:8px 10px;">
                        <div style="color:#2a3a4a;font-size:9px;margin-bottom:2px;">CORRECT</div>
                        <div style="color:#52e08a;font-size:14px;font-weight:700;">{correct}</div>
                    </div>
                    <div style="background:#080c10;border-radius:4px;padding:8px 10px;">
                        <div style="color:#2a3a4a;font-size:9px;margin-bottom:2px;">WRONG</div>
                        <div style="color:#e05252;font-size:14px;font-weight:700;">{total-correct}</div>
                    </div>
                    <div style="background:#080c10;border-radius:4px;padding:8px 10px;">
                        <div style="color:#2a3a4a;font-size:9px;margin-bottom:2px;">TRUE POS</div>
                        <div style="color:#7ecfff;font-size:14px;">{tp}</div>
                    </div>
                    <div style="background:#080c10;border-radius:4px;padding:8px 10px;">
                        <div style="color:#2a3a4a;font-size:9px;margin-bottom:2px;">TRUE NEG</div>
                        <div style="color:#7ecfff;font-size:14px;">{tn}</div>
                    </div>
                    <div style="background:#080c10;border-radius:4px;padding:8px 10px;">
                        <div style="color:#2a3a4a;font-size:9px;margin-bottom:2px;">FALSE POS</div>
                        <div style="color:#f5a623;font-size:14px;">{fp}</div>
                    </div>
                    <div style="background:#080c10;border-radius:4px;padding:8px 10px;">
                        <div style="color:#2a3a4a;font-size:9px;margin-bottom:2px;">FALSE NEG</div>
                        <div style="color:#f5a623;font-size:14px;">{fn}</div>
                    </div>
                </div>
                <div style="margin-top:12px;padding-top:12px;border-top:1px solid #1a2535;
                            display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;
                            font-family:'Share Tech Mono',monospace;font-size:10px;">
                    <div><div style="color:#2a3a4a;font-size:9px;">PRECISION</div>
                         <div style="color:#c8d8e8;">{prec:.1%}</div></div>
                    <div><div style="color:#2a3a4a;font-size:9px;">RECALL</div>
                         <div style="color:#c8d8e8;">{rec:.1%}</div></div>
                    <div><div style="color:#2a3a4a;font-size:9px;">F1 SCORE</div>
                         <div style="color:{color};">{f1:.1%}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

        card_l, card_r = st.columns(2, gap="medium")
        if len(hf_df) > 0:
            ds_card(card_l, "HockeyFight", "🏒", len(hf_df), hf_c,
                    hf_acc, hf_tp, hf_tn, hf_fp, hf_fn, "#7ecfff")
        else:
            card_l.markdown("""
            <div style="background:#0d1520;border:1px solid #1a2535;border-left:4px solid #1a2535;
                        border-radius:10px;padding:30px 20px;text-align:center;
                        font-family:'Share Tech Mono',monospace;font-size:11px;color:#2a3a4a;">
                🏒 HOCKEYFIGHT<br><br>No data uploaded yet.</div>""", unsafe_allow_html=True)

        if len(rwf_df) > 0:
            ds_card(card_r, "RWF-2000", "🥊", len(rwf_df), rwf_c,
                    rwf_acc, rwf_tp, rwf_tn, rwf_fp, rwf_fn, "#e05252")
        else:
            card_r.markdown("""
            <div style="background:#0d1520;border:1px solid #1a2535;border-left:4px solid #1a2535;
                        border-radius:10px;padding:30px 20px;text-align:center;
                        font-family:'Share Tech Mono',monospace;font-size:11px;color:#2a3a4a;">
                🥊 RWF-2000<br><br>No data uploaded yet.</div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin:20px 0 8px 0'></div>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                        color:#2a3a4a;letter-spacing:2px;margin-bottom:10px;">
                        ACCURACY COMPARISON</div>""", unsafe_allow_html=True)

        ds_names, accs, colors = [], [], []
        if len(hf_df):  ds_names.append("HockeyFight"); accs.append(hf_acc);  colors.append("#7ecfff")
        if len(rwf_df): ds_names.append("RWF-2000");    accs.append(rwf_acc); colors.append("#e05252")
        if total_all:   ds_names.append("Overall");     accs.append(acc_all); colors.append("#52e08a")

        if ds_names:
            fig, ax = plt.subplots(figsize=(7, 2.6), facecolor="#080c10")
            ax.set_facecolor("#0a0f18")
            bars = ax.barh(ds_names, [a*100 for a in accs], color=colors, height=0.45, edgecolor="none")
            for bar, acc_v in zip(bars, accs):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{acc_v:.1%}", va="center", ha="left", color="#e8f4ff",
                        fontsize=10, fontfamily="monospace")
            ax.set_xlim(0, 115)
            ax.set_xlabel("Accuracy (%)", color="#445566", fontsize=8)
            ax.tick_params(colors="#7ecfff", labelsize=9)
            ax.spines[:].set_color("#1a2535")
            ax.axvline(100, color="#1a2535", linewidth=0.6, linestyle="--")
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

        st.markdown("<div style='margin:16px 0 8px 0'></div>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                        color:#2a3a4a;letter-spacing:2px;margin-bottom:8px;">
                        PER-CLASS BREAKDOWN</div>""", unsafe_allow_html=True)
        class_rows = []
        for ds_name, ds_sub in [("HockeyFight", hf_df), ("RWF-2000", rwf_df)]:
            if len(ds_sub) == 0: continue
            for cls_name in (ds_sub["_class"].unique() if "_class" in ds_sub.columns else []):
                cls_sub = ds_sub[ds_sub["_class"] == cls_name]
                n   = len(cls_sub)
                cor = int(cls_sub["correct"].str.lower().eq("true").sum()) if "correct" in cls_sub.columns else 0
                class_rows.append({"Dataset": ds_name, "Class": cls_name, "Total": n,
                                   "Correct": cor, "Wrong": n-cor,
                                   "Accuracy": f"{cor/n:.1%}" if n else "—"})
        if class_rows:
            st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True, height=200)

        with st.expander("📋 All Records", expanded=False):
            show = ["_folder","_dataset","_class","true_label","pred_label",
                    "correct","confidence","onset_time","fighter_ids"]
            show = [c for c in show if c in df.columns]
            st.dataframe(df[show].rename(columns={"_folder":"Folder","_dataset":"Dataset","_class":"Class"}),
                         use_container_width=True, height=300)


# ══════════════════════════════════════════
# PAGE: FP/FN BROWSER
# ══════════════════════════════════════════
elif page == "❌ FP/FN Browser":
    st.subheader("❌ False Positive / False Negative Browser")
    records = get_all_pred_records()
    if not records:
        st.info("No pred.txt files found.")
    else:
        df = pd.DataFrame(records)
        if "correct" not in df.columns:
            st.info("Need 'correct' field in pred.txt.")
        else:
            wrong = df[df["correct"].str.lower() != "true"]
            if wrong.empty:
                st.success("🎉 No errors — all predictions correct!")
            else:
                tl, pl = "true_label", "pred_label"
                fp_df = wrong[(wrong[tl].str.lower().isin(["nonfight","nonfight"])) &
                              (wrong[pl].str.lower()=="fight")] \
                        if tl in wrong.columns and pl in wrong.columns else pd.DataFrame()
                fn_df = wrong[(wrong[tl].str.lower()=="fight") &
                              (wrong[pl].str.lower().str.contains("non"))] \
                        if tl in wrong.columns and pl in wrong.columns else pd.DataFrame()
                t1, t2, t3 = st.tabs([f"All Wrong ({len(wrong)})",
                                       f"False Positives ({len(fp_df)})",
                                       f"False Negatives ({len(fn_df)})"])
                show = ["_folder","_dataset","_class","true_label","pred_label",
                        "confidence","onset_time","fighter_ids"]
                for tab, data, label in [(t1,wrong,"wrong"),(t2,fp_df,"fp"),(t3,fn_df,"fn")]:
                    with tab:
                        if data.empty: st.info("None found.")
                        else:
                            s = [c for c in show if c in data.columns]
                            st.dataframe(data[s].rename(columns={"_folder":"Folder"}),
                                         use_container_width=True, height=300)
                            st.download_button("⬇️ CSV",
                                               data=data[s].to_csv(index=False).encode(),
                                               file_name=f"{label}.csv", mime="text/csv",
                                               key=f"fpfn_{label}_csv")


# ══════════════════════════════════════════
# PAGE: CONFUSION MATRIX
# ══════════════════════════════════════════
elif page == "🧩 Confusion Matrix":
    st.subheader("🧩 Confusion Matrix & Metrics")
    records = get_all_pred_records()
    if not records:
        st.info("No pred.txt files found.")
    else:
        cm_col, met_col = st.columns([1, 1.2], gap="large")
        with cm_col:
            fig, cm = make_confusion_matrix(records)
            st.pyplot(fig, clear_figure=True)
        with met_col:
            st.markdown("#### Per-Class Metrics")
            TP, FN = int(cm[0][0]), int(cm[0][1])
            FP, TN = int(cm[1][0]), int(cm[1][1])
            pf  = TP/(TP+FP) if (TP+FP) > 0 else 0
            rf  = TP/(TP+FN) if (TP+FN) > 0 else 0
            f1f = 2*pf*rf/(pf+rf) if (pf+rf) > 0 else 0
            pn  = TN/(TN+FN) if (TN+FN) > 0 else 0
            rn  = TN/(TN+FP) if (TN+FP) > 0 else 0
            f1n = 2*pn*rn/(pn+rn) if (pn+rn) > 0 else 0
            oa  = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0
            mdf = pd.DataFrame([
                {"Class":"Fight",   "Precision":f"{pf:.1%}","Recall":f"{rf:.1%}",
                 "F1":f"{f1f:.1%}","Support":TP+FN},
                {"Class":"NonFight","Precision":f"{pn:.1%}","Recall":f"{rn:.1%}",
                 "F1":f"{f1n:.1%}","Support":FP+TN},
            ])
            st.dataframe(mdf, use_container_width=True, hide_index=True)
            st.metric("Overall Accuracy", f"{oa:.1%}")
            st.markdown(f"**TP:** `{TP}` | **FN:** `{FN}` | **FP:** `{FP}` | **TN:** `{TN}`")


# ══════════════════════════════════════════
# PAGE: INCIDENT REPORT
# ══════════════════════════════════════════
elif page == "🧾 Incident Report":
    st.subheader("🧾 Incident Report Generator")
    pred   = st.session_state.active_pred
    scores = st.session_state.active_scores
    fps    = st.session_state.active_fps
    fname  = st.session_state.active_folder_name
    if not pred or scores is None:
        st.info("Analyze a folder in **Video Explorer** first.")
    else:
        st.success(f"Generating report for: **{fname}**")
        col1, col2 = st.columns([1.1, 1.2], gap="large")
        with col1:
            cam_nm = st.text_input("Camera name",  value="Entrance Camera", key="inc_cam")
            loc    = st.text_input("Location",      value="Main Gate",       key="inc_loc")
            notes  = st.text_area("Notes",          value="",                key="inc_notes")
            gen    = st.button("🧾 GENERATE REPORT", type="primary", key="inc_gen")
        with col2:
            is_f = is_fight_pred(pred)
            if is_f: st.error(f"🔴 FIGHT — {pred.get('confidence','?')} confidence")
            else:    st.success(f"🟢 NORMAL — {pred.get('confidence','?')} confidence")
            for label, key in [
                ("Video", fname), ("Dataset", pred.get("dataset","?")),
                ("True Label", pred.get("true_label","?")),
                ("Predicted",  pred.get("pred_label","?")),
                ("Correct",    pred.get("correct","?")),
                ("Onset Time", pred.get("onset_time","?")),
                ("Fighters",   pred.get("fighter_ids","N/A")),
                ("CAM Focus",  pred.get("cam_focused","N/A")),
            ]:
                st.markdown(f"- **{label}:** {key}")
            c1, c2 = st.columns(2)
            with c1: st.pyplot(make_timeline_plot(scores, fps, pred), clear_figure=True)
            with c2: st.pyplot(make_hist_plot(scores), clear_figure=True)
        if gen:
            report = {
                "incident_id":  f"INC-{int(time.time())}",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generated_by": st.session_state.username,
                "camera": cam_nm, "location": loc,
                "video_folder": fname,
                "model":   pred.get("model_path", "?"),
                "dataset": pred.get("dataset", "?"),
                "thresholds": {"violence": st.session_state.thr_violence,
                               "suspicious": st.session_state.thr_suspicious},
                "prediction": {
                    "true_label":   pred.get("true_label","?"),
                    "pred_label":   pred.get("pred_label","?"),
                    "confidence":   pred.get("confidence","?"),
                    "correct":      pred.get("correct","?"),
                    "onset_frame":  pred.get("onset_frame","?"),
                    "onset_time":   pred.get("onset_time","?"),
                    "total_frames": pred.get("total_frames","?"),
                    "fighter_ids":  pred.get("fighter_ids","N/A"),
                    "cam_focused":  pred.get("cam_focused","N/A"),
                    "how_started":  describe_onset(pred),
                },
                "notes": notes,
            }
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(CFG.OUTPUT_DIR) / f"incident_{fname}_{ts}.json"
            with open(path, "w") as f: json.dump(report, f, indent=2)
            st.success("✅ Report saved!")
            st.download_button("⬇️ Download JSON",
                               data=io.BytesIO(json.dumps(report, indent=2).encode()),
                               file_name=path.name, mime="application/json", key="inc_json_dl")
            pdf = generate_pdf_report(pred, scores, fps, fname)
            st.download_button("⬇️ Download PDF", data=pdf,
                               file_name=f"report_{fname}_{ts}.pdf",
                               mime="application/pdf", key="inc_pdf_dl")


st.divider()
st.markdown(
    "<span style='font-family:Share Tech Mono,monospace;font-size:10px;color:#1a2535;'>"
    "VISIONGUARD v6 · R3D-18 + LCM + LSTM · GradCAM | GradCAM++ | SmoothGradCAM++ | LayerCAM | Combined · ByteTrack"
    "</span>", unsafe_allow_html=True)
