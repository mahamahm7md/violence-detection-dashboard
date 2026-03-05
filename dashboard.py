# dashboard.py — Violence Detection Dashboard v2
# Improved: folder upload, pred.txt reader, gradcam/raw video display,
# dataset/class selector, compact layout, label flip fix
#
# RUN:
#   pip install streamlit opencv-python-headless numpy pandas matplotlib
#   streamlit run dashboard.py

import os
import io
import time
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    APP_TITLE: str = "Violence Detection Dashboard"
    OUTPUT_DIR: str = "outputs_dashboard"
    DEFAULT_FPS: int = 25
    MAX_FRAMES_PREVIEW: int = 220
    THRESH_VIOLENCE: float = 0.70
    THRESH_SUSPICIOUS: float = 0.45


os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

UPLOAD_ROOT = Path(CFG.OUTPUT_DIR) / "uploads"
LOGS_ROOT = Path(CFG.OUTPUT_DIR) / "logs"
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "hockeyfight": ["Fight", "NonFight"],
    "rwf": ["Fight", "NonFight"],
}


# -----------------------------
# Helper utils
# -----------------------------
def status_from_score(p: float, label_map: int = 1) -> str:
    if label_map == 0:
        p = 1.0 - p
    if p >= CFG.THRESH_VIOLENCE:
        return "ALERT"
    if p >= CFG.THRESH_SUSPICIOUS:
        return "SUSPICIOUS"
    return "NORMAL"


def color_from_status(s: str) -> str:
    return "🔴" if s == "ALERT" else ("🟡" if s == "SUSPICIOUS" else "🟢")


def fmt_time(sec):
    if sec is None:
        return "N/A"
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_keep(frame, w=640):
    h, ww = frame.shape[:2]
    if ww == w:
        return frame
    new_h = int(h * (w / ww))
    return cv2.resize(frame, (w, new_h), interpolation=cv2.INTER_AREA)


def read_video_frames(video_path, max_frames=CFG.MAX_FRAMES_PREVIEW):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = CFG.DEFAULT_FPS
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += 1
        if idx >= max_frames:
            break
    cap.release()
    return frames, float(fps)


def fake_model_scores(frames, fps):
    """Placeholder — replace with real R3D-18 inference."""
    ps, prev = [], None
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        if prev is None:
            ps.append(0.10)
        else:
            diff = cv2.absdiff(g, prev)
            motion = float(np.mean(diff)) / 255.0
            ps.append(float(np.clip(motion * 2.8, 0, 1)))
        prev = g
    return np.array(ps, dtype=np.float32)


def detect_fight_start(scores, fps, thr=None):
    if thr is None:
        thr = CFG.THRESH_VIOLENCE
    N = max(3, int(0.2 * fps))
    above = scores >= thr
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= N:
            start_frame = i - (N - 1)
            return start_frame, start_frame / fps
    return None, None


def make_timeline_plot(scores, fps):
    t = np.arange(len(scores)) / fps
    fig = plt.figure(figsize=(6, 2.5))
    plt.plot(t, scores, color="#e05252", linewidth=1.5)
    plt.axhline(CFG.THRESH_SUSPICIOUS, linestyle="--", color="orange", linewidth=1, label="Suspicious")
    plt.axhline(CFG.THRESH_VIOLENCE,   linestyle="--", color="red",    linewidth=1, label="Violence")
    plt.xlabel("Time (s)", fontsize=9)
    plt.ylabel("Prob", fontsize=9)
    plt.title("Violence Timeline", fontsize=10)
    plt.legend(fontsize=8)
    plt.tight_layout()
    return fig


def make_hist_plot(scores):
    fig = plt.figure(figsize=(4, 2.5))
    plt.hist(scores, bins=20, color="#5271e0")
    plt.xlabel("Probability", fontsize=9)
    plt.ylabel("Count", fontsize=9)
    plt.title("Confidence Distribution", fontsize=10)
    plt.tight_layout()
    return fig


def ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return True
    except Exception:
        return False


def make_web_preview(input_path: Path, preview_path: Path) -> bool:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    if preview_path.exists() and preview_path.stat().st_mtime >= input_path.stat().st_mtime:
        return True
    if ffmpeg_available():
        cmd = ["ffmpeg", "-y", "-i", str(input_path),
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-preset", "veryfast", "-crf", "23",
               "-c:a", "aac", "-b:a", "128k", str(preview_path)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=120)
            return True
        except Exception:
            pass
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or CFG.DEFAULT_FPS
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(preview_path),
                              cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
        out.release()
        return preview_path.exists()
    except Exception:
        return False


def describe_how_started(frames, scores, fps, start_frame):
    if start_frame is None:
        return "No clear onset detected (probability did not cross the violence threshold consistently)."
    win = int(max(5, fps))
    a   = max(0, start_frame - win)
    b   = min(len(frames) - 1, start_frame + 1)
    prev, motion_vals = None, []
    for i in range(a, b):
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        if prev is not None:
            motion_vals.append(float(np.mean(cv2.absdiff(g, prev))) / 255.0)
        prev = g
    if not motion_vals:
        return "Onset detected, but motion evidence is insufficient to describe the start."
    if max(motion_vals) > float(np.mean(motion_vals)) * 2.0:
        return "Onset aligns with a sudden spike in movement (rapid interaction/contact), followed by sustained activity."
    return "Onset appears gradual: movement increases steadily until the violence threshold is crossed."


def save_incident_report(report, out_dir=CFG.OUTPUT_DIR):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"incident_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


# -----------------------------
# pred.txt parser & renderer
# -----------------------------
def parse_pred_txt(path: Path) -> dict:
    result = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    k, v = line.split(":", 1)
                    result[k.strip()] = v.strip()
    except Exception:
        pass
    return result


def render_pred_card(pred: dict, label_map: int = 1):
    if not pred:
        st.info("No pred.txt found for this video.")
        return

    true_label   = pred.get("true_label",   "?")
    pred_label   = pred.get("pred_label",   "?")
    correct      = pred.get("correct",      "?")
    confidence   = pred.get("confidence",   "?")
    dataset      = pred.get("dataset",      "?")
    onset_time   = pred.get("onset_time",   "?")
    onset_frame  = pred.get("onset_frame",  "?")
    total_frames = pred.get("total_frames", "?")
    probs        = pred.get("probs",        "?")
    model_val_acc= pred.get("model_val_acc","?")
    model_path   = pred.get("model_path",   "?")

    if label_map == 0:
        flip = {"Fight":"NonFight","NonFight":"Fight","fight":"nonfight","nonfight":"fight"}
        pred_label = flip.get(pred_label, pred_label)

    is_correct    = str(correct).lower() == "true"
    correct_emoji = "✅" if is_correct else "❌"
    pred_color    = "🔴" if "fight" in str(pred_label).lower() else "🟢"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Label", true_label)
    c2.metric("Predicted",  f"{pred_color} {pred_label}")
    c3.metric("Correct?",   f"{correct_emoji} {correct}")
    try:
        c4.metric("Confidence", f"{float(confidence):.1%}")
    except Exception:
        c4.metric("Confidence", confidence)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Dataset",      dataset)
    c6.metric("Onset Frame",  onset_frame)
    c7.metric("Onset Time",   onset_time)
    c8.metric("Total Frames", total_frames)

    with st.expander("📋 Full pred.txt details", expanded=False):
        ca, cb = st.columns(2)
        with ca:
            st.markdown(f"**Model path:** `{model_path}`")
            st.markdown(f"**Model val acc:** `{model_val_acc}`")
            st.markdown(f"**Raw probs:** `{probs}`")
        with cb:
            st.markdown(f"**Window size:** `{pred.get('window_size','?')}`")
            st.markdown(f"**Window stride:** `{pred.get('window_stride','?')}`")
            st.markdown(f"**Target layer:** `{pred.get('target_layer','?')}`")
            st.markdown(f"**img_size:** `{pred.get('img_size','?')}`")
            st.markdown(f"**cam_target_class:** `{pred.get('cam_target_class','?')}`")
            st.markdown(f"**spike_delta:** `{pred.get('spike_delta','?')}`")


# -----------------------------
# Folder helpers
# -----------------------------
def get_class_root(dataset: str, cls: str) -> Path:
    return UPLOAD_ROOT / dataset / cls


def list_video_folders(dataset: str, cls: str):
    root = get_class_root(dataset, cls)
    if not root.exists():
        return []
    folders = [x for x in root.iterdir() if x.is_dir() and not x.name.startswith("_")]
    folders.sort(key=lambda x: x.name.lower())
    return folders


def find_file(folder: Path, pattern: str):
    matches = list(folder.glob(pattern))
    return matches[0] if matches else None


def get_video_files(folder: Path) -> dict:
    files = {}
    orig = find_file(folder, "*original*.mp4")
    if not orig:
        # fallback: any mp4 that isn't gradcam
        all_mp4 = [f for f in folder.glob("*.mp4") if "gradcam" not in f.name.lower()]
        if all_mp4:
            orig = all_mp4[0]
    if orig:
        files["original"] = orig

    gcam = find_file(folder, "*gradcam_onset*.mp4")
    if not gcam:
        gcam = find_file(folder, "*gradcam_on*.mp4")
    if gcam:
        files["gradcam"] = gcam

    gcampp = find_file(folder, "*gradcampp_onset*.mp4")
    if not gcampp:
        gcampp = find_file(folder, "*gradcampp*.mp4")
    if gcampp:
        files["gradcampp"] = gcampp

    for key, pat in [
        ("raw_grid",       "raw_grid.png"),
        ("gradcam_grid",   "gradcam_grid.png"),
        ("gradcampp_grid", "gradcampp_grid.png"),
        ("pred",           "pred.txt"),
    ]:
        f = find_file(folder, pat)
        if f:
            files[key] = f

    return files


# ============================================================
# Streamlit App
# ============================================================
st.set_page_config(page_title=CFG.APP_TITLE, layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 17px !important; }
[data-testid="stMetricLabel"] { font-size: 11px !important; }
.block-container { padding-top: 1rem !important; }
h1 { font-size: 1.5rem !important; margin-bottom: 0.2rem !important; }
h2, h3 { font-size: 1rem !important; margin-top: 0.4rem !important; }
.stTabs [data-baseweb="tab"] { font-size: 13px; padding: 6px 10px; }
div[data-testid="stImage"] img { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.title(f"🎯 {CFG.APP_TITLE}")
st.warning(
    "⚠️ **Cloud notice:** Uploaded files are temporary and reset on restart. "
    "For permanent storage run locally.", icon="⚠️"
)

for ds in DATASETS:
    for cls in DATASETS[ds]:
        (UPLOAD_ROOT / ds / cls).mkdir(parents=True, exist_ok=True)

tabs = st.tabs([
    "📁 Video Explorer",
    "🎥 Control Room",
    "🟦 Multi-Camera Wall",
    "📤 Upload Manager",
    "📈 Analytics",
    "📊 Dataset Comparison",
    "🧾 Incident Report",
])

for key, val in [
    ("last_scores", None), ("last_fps", None),
    ("last_video_path", None), ("last_frames", None),
    ("run_id", datetime.now().strftime("run_%Y%m%d_%H%M%S")),
]:
    if key not in st.session_state:
        st.session_state[key] = val


# ============================================================
# TAB 1 — VIDEO EXPLORER
# ============================================================
with tabs[0]:
    st.subheader("📁 Video Explorer")

    sel1, sel2, sel3, sel4 = st.columns([1, 1, 2, 1])
    with sel1:
        sel_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="ex_ds")
    with sel2:
        sel_cls = st.selectbox("Class", DATASETS[sel_ds], key="ex_cls")
    with sel3:
        vfolders = list_video_folders(sel_ds, sel_cls)
        if not vfolders:
            st.info(f"No folders yet for **{sel_ds} / {sel_cls}**. Use Upload Manager ➡️")
            sel_folder_name = None
        else:
            sel_folder_name = st.selectbox(
                f"Video folder  ({len(vfolders)} total)", [f.name for f in vfolders], key="ex_folder"
            )
    with sel4:
        label_map = st.selectbox(
            "Label mapping",
            options=[1, 0],
            format_func=lambda x: "1 = Fight (default)" if x == 1 else "0 = Fight (flipped)",
            key="ex_label_map",
            help="If predictions look flipped, switch to 0"
        )

    if sel_folder_name:
        folder_path = get_class_root(sel_ds, sel_cls) / sel_folder_name
        files = get_video_files(folder_path)

        # ── pred.txt ──
        st.markdown("---")
        st.markdown("#### 📋 Prediction Results")
        if "pred" in files:
            render_pred_card(parse_pred_txt(files["pred"]), label_map=label_map)
        else:
            st.info("No pred.txt found in this folder.")

        st.markdown("---")

        # ── Videos ──
        st.markdown("#### 🎬 Videos")
        vid_keys   = [k for k in ["original", "gradcam", "gradcampp"] if k in files]
        vid_labels = {"original": "📹 Original", "gradcam": "🔥 Grad-CAM", "gradcampp": "🔥 Grad-CAM++"}

        if vid_keys:
            vcols = st.columns(len(vid_keys))
            for i, vk in enumerate(vid_keys):
                with vcols[i]:
                    st.markdown(f"**{vid_labels[vk]}**")
                    vpath     = files[vk]
                    prev_path = folder_path / f"_preview_{vk}.mp4"
                    if not prev_path.exists():
                        with st.spinner("Preparing preview..."):
                            make_web_preview(vpath, prev_path)
                    if prev_path.exists():
                        st.video(str(prev_path))
                    else:
                        st.warning("Preview unavailable — ffmpeg not found")
        else:
            st.info("No .mp4 videos found in this folder.")

        st.markdown("---")

        # ── Grid images ──
        st.markdown("#### 🖼️ Frame Grids")
        img_keys   = [k for k in ["raw_grid", "gradcam_grid", "gradcampp_grid"] if k in files]
        img_labels = {
            "raw_grid":       "📷 Raw Frames",
            "gradcam_grid":   "🌡️ Grad-CAM Grid",
            "gradcampp_grid": "🌡️ Grad-CAM++ Grid",
        }

        if img_keys:
            icols = st.columns(len(img_keys))
            for i, ik in enumerate(img_keys):
                with icols[i]:
                    st.markdown(f"**{img_labels[ik]}**")
                    st.image(str(files[ik]), use_container_width=True)
        else:
            st.info("No grid images found in this folder.")

        # load into session for analytics / other tabs
        if "original" in files:
            try:
                frames, fps = read_video_frames(str(files["original"]), max_frames=140)
                frames_r    = [resize_keep(f, 720) for f in frames]
                scores      = fake_model_scores(frames_r, fps)
                st.session_state.last_scores     = scores
                st.session_state.last_fps        = fps
                st.session_state.last_video_path = str(files["original"])
                st.session_state.last_frames     = frames_r
            except Exception:
                pass


# ============================================================
# TAB 4 — UPLOAD MANAGER
# ============================================================
with tabs[3]:
    st.subheader("📤 Upload Manager")

    st.markdown("""
**How to use:**
1. Select **Dataset** and **Class**
2. Enter the **folder name** (video stem, e.g. `fi1_xvid` or `_q5Nwh4Z6ao_6`)
3. Select **all files** inside that folder (mp4 videos + png grids + pred.txt)
4. Click **Save** — repeat for every video folder
""")

    uc1, uc2, uc3 = st.columns(3)
    with uc1:
        up_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="up_ds")
    with uc2:
        up_cls = st.selectbox("Class", DATASETS[up_ds], key="up_cls")
    with uc3:
        folder_name_input = st.text_input(
            "Folder name (video stem)",
            placeholder="e.g. fi1_xvid  or  _q5Nwh4Z6ao_6"
        )

    up_files = st.file_uploader(
        "Select all files for this video folder  (mp4 + png + pred.txt)",
        type=["mp4", "avi", "mov", "mkv", "png", "jpg", "txt"],
        accept_multiple_files=True,
        key="up_files"
    )

    save_btn = st.button(
        "💾 Save into folder", type="primary",
        disabled=(not up_files or not folder_name_input.strip())
    )

    if save_btn and up_files and folder_name_input.strip():
        dest = UPLOAD_ROOT / up_ds / up_cls / folder_name_input.strip()
        dest.mkdir(parents=True, exist_ok=True)
        for uf in up_files:
            with open(dest / uf.name, "wb") as f:
                f.write(uf.getbuffer())
        st.success(f"✅ Saved {len(up_files)} file(s) → `{dest}`")
        st.info("Go to **Video Explorer** tab → select your dataset/class → pick this folder.")

    st.divider()
    st.markdown("### 📂 Currently uploaded folders")
    found_any = False
    for ds in DATASETS:
        for cls in DATASETS[ds]:
            folders = list_video_folders(ds, cls)
            if folders:
                found_any = True
                st.markdown(f"**{ds} / {cls}** — {len(folders)} folder(s):")
                st.code(", ".join(f.name for f in folders))
    if not found_any:
        st.info("No folders uploaded yet.")


# ============================================================
# TAB 2 — CONTROL ROOM
# ============================================================
with tabs[1]:
    st.subheader("🎥 Single Camera Control Room")
    colA, colB = st.columns([1.6, 1.0], gap="large")

    with colA:
        src = st.radio("Source", ["Pick from uploaded folders", "Use local path"], horizontal=True)
        video_path = None

        if src == "Pick from uploaded folders":
            cr_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="cr_ds")
            cr_cls = st.selectbox("Class", DATASETS[cr_ds], key="cr_cls")
            vf     = list_video_folders(cr_ds, cr_cls)
            if not vf:
                st.warning("No folders uploaded yet.")
            else:
                cr_folder = st.selectbox("Folder", [f.name for f in vf], key="cr_folder")
                fp = get_class_root(cr_ds, cr_cls) / cr_folder
                ff = get_video_files(fp)
                if "original" in ff:
                    video_path = str(ff["original"])
        else:
            video_path = st.text_input("Local path", value=st.session_state.last_video_path or "")

        run_btn = st.button("▶️ Run Preview", type="primary", disabled=not bool(video_path))

        if run_btn and video_path:
            try:
                frames, fps = read_video_frames(video_path, max_frames=140)
                frames      = [resize_keep(f, 720) for f in frames]
                scores      = fake_model_scores(frames, fps)
                st.session_state.last_scores     = scores
                st.session_state.last_fps        = fps
                st.session_state.last_video_path = video_path
                st.session_state.last_frames     = frames
                ph, il = st.empty(), st.empty()
                for i in range(min(len(frames), 90)):
                    p = float(scores[i])
                    s = status_from_score(p)
                    e = color_from_status(s)
                    ph.image(to_rgb(frames[i]),
                             caption=f"Frame {i} | {e} {s} | p={p:.2f}",
                             use_container_width=True)
                    il.markdown(f"**Time:** {fmt_time(i/fps)} | **Status:** {e} **{s}** | **p:** `{p:.2f}`")
                    time.sleep(0.02)
                st.success("Preview complete ✅")
            except Exception as e:
                st.error(str(e))

    with colB:
        st.write("### Quick Indicators")
        scores = st.session_state.last_scores
        fps    = st.session_state.last_fps
        if scores is None:
            st.info("Analyze a video first.")
        else:
            peak = float(np.max(scores))
            avg  = float(np.mean(scores))
            _, start_sec = detect_fight_start(scores, fps)
            st.metric("Peak probability",    f"{peak:.2f}")
            st.metric("Average probability", f"{avg:.2f}")
            st.metric("Fight start",         fmt_time(start_sec))
            now_s = status_from_score(float(scores[-1]))
            st.markdown(f"**Status:** {color_from_status(now_s)} **{now_s}**")
            st.pyplot(make_timeline_plot(scores, fps), clear_figure=True, use_container_width=False)


# ============================================================
# TAB 3 — MULTI-CAMERA WALL
# ============================================================
with tabs[2]:
    st.subheader("🟦 Multi-Camera Wall")
    use_same = st.checkbox("Use same video for all cameras", value=True)
    cams = []

    if use_same:
        w_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="wall_ds")
        w_cls = st.selectbox("Class",   DATASETS[w_ds],        key="wall_cls")
        wf    = list_video_folders(w_ds, w_cls)
        if not wf:
            st.info("No folders uploaded yet.")
        else:
            wfolder = st.selectbox("Folder", [f.name for f in wf], key="wall_folder")
            wfiles  = get_video_files(get_class_root(w_ds, w_cls) / wfolder)
            if "original" in wfiles:
                cams = [str(wfiles["original"])] * 4
    else:
        cams = [st.text_input(f"Camera {i+1} path", "") for i in range(4)]

    if st.button("🧱 Render Camera Wall", disabled=(not cams or not all(bool(x) for x in cams))):
        cols = st.columns(2, gap="large")
        for idx, vp in enumerate(cams):
            try:
                frames, fps = read_video_frames(vp, max_frames=45)
                frames      = [resize_keep(f, 480) for f in frames]
                scores      = fake_model_scores(frames, fps)
                peak        = float(np.max(scores))
                s           = status_from_score(peak)
                with cols[idx % 2]:
                    st.image(to_rgb(frames[min(10, len(frames)-1)]),
                             use_container_width=True,
                             caption=f"Cam {idx+1} | {color_from_status(s)} {s} | p={peak:.2f}")
                    st.progress(min(1.0, peak))
            except Exception as e:
                with cols[idx % 2]:
                    st.error(f"Camera {idx+1}: {e}")


# ============================================================
# TAB 5 — ANALYTICS
# ============================================================
with tabs[4]:
    st.subheader("📈 Analytics")
    scores = st.session_state.last_scores
    fps    = st.session_state.last_fps

    if scores is None or fps is None:
        st.info("Open a video in **Video Explorer** first.")
    else:
        _, start_sec = detect_fight_start(scores, fps)
        step = max(1, int(0.5 * fps))
        rows = [{"time": fmt_time(i/fps), "frame": i,
                 "prob": round(float(scores[i]), 3),
                 "status": status_from_score(float(scores[i]))}
                for i in range(0, len(scores), step)]
        df = pd.DataFrame(rows)

        cA, cB = st.columns([1.3, 1.0], gap="large")
        with cA:
            st.write("#### Detection Log")
            st.dataframe(df, use_container_width=True, height=260)
        with cB:
            st.write("#### Summary")
            st.metric("Fight start",      fmt_time(start_sec))
            st.metric("Peak probability", f"{float(np.max(scores)):.2f}")
            above = scores >= CFG.THRESH_VIOLENCE
            runs, run = [], 0
            for a in above:
                if a:
                    run += 1
                else:
                    if run > 0: runs.append(run)
                    run = 0
            if run > 0: runs.append(run)
            st.metric("Short ALERT spikes (<0.5s)",
                      str(len([r for r in runs if (r/fps) < 0.5])))

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(make_timeline_plot(scores, fps), clear_figure=True, use_container_width=False)
        with c2:
            st.pyplot(make_hist_plot(scores), clear_figure=True, use_container_width=False)

        log_path = LOGS_ROOT / "analysis_log.csv"
        if log_path.exists():
            st.write("#### Full Log")
            try:
                st.dataframe(pd.read_csv(log_path), use_container_width=True, height=200)
            except Exception:
                pass


# ============================================================
# TAB 6 — DATASET COMPARISON
# ============================================================
with tabs[5]:
    st.subheader("📊 Dataset Comparison")
    metrics = pd.DataFrame([
        {"Dataset":"RWF-2000",    "Model":"R3D-18","Accuracy":0.93,"Precision":0.92,"Recall":0.91,"F1":0.91},
        {"Dataset":"HockeyFight", "Model":"R3D-18","Accuracy":0.95,"Precision":0.94,"Recall":0.94,"F1":0.94},
    ])
    st.dataframe(metrics, use_container_width=True, height=150)
    met_csv = st.file_uploader("Upload your own metrics CSV", type=["csv"], key="metrics_csv")
    if met_csv:
        try:
            st.dataframe(pd.read_csv(met_csv), use_container_width=True, height=220)
        except Exception as e:
            st.error(f"CSV error: {e}")


# ============================================================
# TAB 7 — INCIDENT REPORT
# ============================================================
with tabs[6]:
    st.subheader("🧾 Incident Report Generator")
    scores     = st.session_state.last_scores
    fps        = st.session_state.last_fps
    video_path = st.session_state.last_video_path

    col1, col2 = st.columns([1.1, 1.2], gap="large")
    with col1:
        st.write("### Report Inputs")
        cam_name     = st.text_input("Camera name",       value="Entrance Camera")
        location     = st.text_input("Location",          value="Main Gate / Hallway")
        model_name   = st.text_input("Model",             value="R3D-18")
        dataset_name = st.text_input("Dataset",           value="RWF-2000 / HockeyFight")
        notes        = st.text_area ("Notes (optional)",  value="")
        gen = st.button("🧾 Generate Report", type="primary",
                        disabled=(scores is None or fps is None or video_path is None))

    with col2:
        st.write("### Preview")
        if scores is None or fps is None or video_path is None:
            st.info("Open a video in Video Explorer first.")
        else:
            _, start_sec = detect_fight_start(scores, fps)
            peak = float(np.max(scores))
            st.markdown(f"- **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"- **Video:** `{Path(video_path).name}`")
            st.markdown(f"- **Fight start:** **{fmt_time(start_sec)}**")
            st.markdown(f"- **Peak prob:** **{peak:.2f}**")
            st.markdown(
                f"- **Status:** {color_from_status(status_from_score(peak))} "
                f"**{status_from_score(peak)}**"
            )
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(make_timeline_plot(scores, fps), clear_figure=True, use_container_width=False)
            with c2:
                st.pyplot(make_hist_plot(scores), clear_figure=True, use_container_width=False)

    if gen and scores is not None:
        _, start_sec = detect_fight_start(scores, fps)
        peak         = float(np.max(scores))
        how_started  = describe_how_started(st.session_state.last_frames, scores, fps,
                                            detect_fight_start(scores, fps)[0])
        report = {
            "incident_id":  f"INC-{int(time.time())}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "camera": cam_name, "location": location,
            "video_file": Path(video_path).name if video_path else None,
            "model": model_name, "dataset": dataset_name,
            "thresholds": {"suspicious": float(CFG.THRESH_SUSPICIOUS),
                           "violence":   float(CFG.THRESH_VIOLENCE)},
            "results": {
                "peak_probability":           round(peak, 4),
                "estimated_fight_start_sec":  None if start_sec is None else round(float(start_sec), 3),
                "estimated_fight_start_time": fmt_time(start_sec),
                "how_started": how_started,
            },
            "notes": notes,
        }
        path = save_incident_report(report)
        st.success(f"Report saved: {path}")
        buf = io.BytesIO(json.dumps(report, indent=2).encode("utf-8"))
        st.download_button("⬇️ Download report JSON", data=buf,
                           file_name=os.path.basename(path), mime="application/json")

st.divider()
st.caption("Replace fake_model_scores() with your real R3D-18 inference when ready.")
