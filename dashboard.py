# dashboard.py
# ------------------------------------------------------------
# FULL Violence Detection Dashboard (All-in-One) + Organized Upload Manager + Compact Grad-CAM UI
#
# ✅ Upload Manager creates 2-class folders: uploads/fight and uploads/nonfight
# ✅ Automatically builds browser-playable previews: uploads/<class>/_preview/*.mp4
# ✅ Batch analyze multiple selected videos
# ✅ Creates organized run folders like your structure:
#    outputs_dashboard/runs/<RUN_ID>/<class>/<video_stem>/
#        original.<ext>
#        preview.mp4
#        timeline.png
#        confidence_hist.png
#        summary.json
#        onset.json          (when it started + how it started)
#        pred.txt
# ✅ Small graphs (don't take whole page)
# ✅ Grad-CAM tab compact layout:
#    Left: smaller visualization boxes
#    Right: tight info panel (status/prob/time/onset/how) + small timeline
#
# RUN:
#   pip install streamlit opencv-python-headless numpy pandas matplotlib
#   streamlit run dashboard.py
# ------------------------------------------------------------

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
    APP_TITLE: str = "Violence Detection Control Dashboard"
    OUTPUT_DIR: str = "outputs_dashboard"
    DEFAULT_FPS: int = 25
    MAX_FRAMES_PREVIEW: int = 220
    THRESH_VIOLENCE: float = 0.70
    THRESH_SUSPICIOUS: float = 0.45


os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

# Two classes like dataset folders
CLASSES = ["fight", "nonfight"]

# Organized roots
UPLOAD_ROOT = Path(CFG.OUTPUT_DIR) / "uploads"
LOGS_ROOT = Path(CFG.OUTPUT_DIR) / "logs"
LOGS_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helper utils
# -----------------------------
def status_from_score(p: float) -> str:
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

    frames = []
    idx = 0
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
    """
    Placeholder scoring (motion heuristic) so the dashboard works now.
    Replace with your real R3D-18 inference.
    Returns p[t] in [0,1].
    """
    ps = []
    prev = None
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        if prev is None:
            ps.append(0.10)
        else:
            diff = cv2.absdiff(g, prev)
            motion = float(np.mean(diff)) / 255.0
            p = np.clip(motion * 2.8, 0, 1)
            ps.append(p)
        prev = g
    return np.array(ps, dtype=np.float32)


def detect_fight_start(scores, fps, thr=CFG.THRESH_VIOLENCE):
    """
    First time score crosses thr for >= N consecutive frames.
    """
    N = max(3, int(0.2 * fps))
    above = scores >= thr
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= N:
            start_frame = i - (N - 1)
            return start_frame, start_frame / fps
    return None, None


# ✅ SMALLER GRAPHS
def make_timeline_plot(scores, fps):
    t = np.arange(len(scores)) / fps
    fig = plt.figure(figsize=(6, 3))
    plt.plot(t, scores)
    plt.axhline(CFG.THRESH_SUSPICIOUS, linestyle="--")
    plt.axhline(CFG.THRESH_VIOLENCE, linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Probability")
    plt.title("Violence Timeline")
    plt.tight_layout()
    return fig


def make_hist_plot(scores):
    fig = plt.figure(figsize=(5, 3))
    plt.hist(scores, bins=20)
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.title("Confidence Distribution")
    plt.tight_layout()
    return fig


def overlay_heatmap(frame_bgr, heatmap_01):
    h, w = frame_bgr.shape[:2]
    hm = cv2.resize(heatmap_01, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_uint8 = np.uint8(np.clip(hm * 255, 0, 255))
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(frame_bgr, 0.65, hm_color, 0.35, 0)
    return out


def fake_gradcam_heatmap(frame_bgr):
    """
    Placeholder heatmap. Replace with your real Grad-CAM heatmap per frame.
    """
    h, w = frame_bgr.shape[:2]
    y = np.linspace(0, 1, h).reshape(h, 1)
    x = np.linspace(0, 1, w).reshape(1, w)
    hm = np.exp(-((x - 0.55) ** 2 + (y - 0.55) ** 2) / 0.08)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)
    return hm.astype(np.float32)


def safe_stem(p: Path) -> str:
    s = p.stem.replace(" ", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))


def ensure_dirs():
    for c in CLASSES:
        (UPLOAD_ROOT / c).mkdir(parents=True, exist_ok=True)
        (UPLOAD_ROOT / c / "_preview").mkdir(parents=True, exist_ok=True)


def list_videos_in_class(class_name: str):
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    p = UPLOAD_ROOT / class_name
    if not p.exists():
        return []
    vids = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts]
    vids.sort(key=lambda x: x.name.lower())
    return vids


def ffmpeg_available() -> bool:
    """Check if ffmpeg is available on this system."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5  # FIX: added timeout so it never hangs on cloud
        )
        return True
    except Exception:
        return False


def make_web_preview(input_path: Path, preview_path: Path) -> bool:
    """
    Create browser-playable mp4 (H.264 + AAC).
    Uses ffmpeg if available, falls back to OpenCV.
    """
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    if preview_path.exists() and preview_path.stat().st_mtime >= input_path.stat().st_mtime:
        return True

    if ffmpeg_available():
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            str(preview_path),
        ]
        try:
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120  # FIX: added timeout to avoid hanging on cloud
            )
            return True
        except Exception:
            pass  # fall through to OpenCV fallback

    # OpenCV fallback
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1:
            fps = CFG.DEFAULT_FPS
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(preview_path), fourcc, float(fps), (w, h))
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


def save_plot_png(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def write_json(d: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_log_row(row: dict):
    log_path = LOGS_ROOT / "analysis_log.csv"
    df = pd.DataFrame([row])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


def describe_how_started(frames, scores, fps, start_frame):
    """
    Conservative explanation based on motion spike near onset.
    """
    if start_frame is None:
        return "No clear onset detected (probability did not cross the violence threshold consistently)."

    win = int(max(5, fps))
    a = max(0, start_frame - win)
    b = min(len(frames) - 1, start_frame + 1)

    prev = None
    motion_vals = []
    for i in range(a, b):
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        if prev is not None:
            diff = cv2.absdiff(g, prev)
            motion_vals.append(float(np.mean(diff)) / 255.0)
        prev = g

    if not motion_vals:
        return "Onset detected, but motion evidence is insufficient to describe the start."

    peak_motion = max(motion_vals)
    avg_motion = float(np.mean(motion_vals))

    if peak_motion > avg_motion * 2.0:
        return "Onset aligns with a sudden spike in movement (rapid interaction/contact), followed by sustained activity."
    return "Onset appears gradual: movement increases steadily until the violence threshold is crossed."


def save_incident_report(report: dict, out_dir=CFG.OUTPUT_DIR):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"incident_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title=CFG.APP_TITLE, layout="wide")
st.title(CFG.APP_TITLE)

ensure_dirs()

# FIX: Cloud storage warning banner — makes it clear to viewers that uploads are session-only
st.warning(
    "⚠️ **Cloud deployment notice:** Uploaded videos and analysis logs are stored temporarily "
    "and will be cleared if the app restarts. For permanent storage, run the app locally.",
    icon="⚠️"
)

st.caption(
    "Upload videos into class folders (fight/nonfight), preview always works (web mp4), "
    "batch analyze into organized run folders, and compact Grad-CAM dashboard layout."
)

tabs = st.tabs([
    "🎥 Control Room (Single Cam)",
    "🟦 Multi-Camera Wall",
    "📤 Upload Manager + Batch Analyzer",
    "🔥 Grad-CAM Panel",
    "📈 Analytics",
    "📊 Dataset Comparison",
    "🧾 Incident Report"
])

# Session state
if "last_scores" not in st.session_state:
    st.session_state.last_scores = None
if "last_fps" not in st.session_state:
    st.session_state.last_fps = None
if "last_video_path" not in st.session_state:
    st.session_state.last_video_path = None
if "last_frames" not in st.session_state:
    st.session_state.last_frames = None
if "last_batch_df" not in st.session_state:
    st.session_state.last_batch_df = None
if "run_id" not in st.session_state:
    st.session_state.run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")


# -----------------------------
# Tab 1: Control Room
# -----------------------------
with tabs[0]:
    st.subheader("Single Camera Control Room (Preview)")

    colA, colB = st.columns([1.6, 1.0], gap="large")

    with colA:
        src = st.radio(
            "Source type",
            ["Pick from class folders", "Use local path"],
            horizontal=True
        )

        video_path = None
        if src == "Pick from class folders":
            pick_class = st.selectbox("Class", CLASSES, index=0, key="control_class")
            vids = list_videos_in_class(pick_class)
            if not vids:
                st.warning("No videos in this folder yet. Upload some in the Upload Manager tab.")
            else:
                nm = st.selectbox("Video", [v.name for v in vids], key="control_video")
                video_path = str(UPLOAD_ROOT / pick_class / nm)
        else:
            video_path = st.text_input("Local video path", value=st.session_state.last_video_path or "")

        run_btn = st.button("▶️ Run Control Room Preview", type="primary", disabled=not bool(video_path))

        if run_btn and video_path:
            try:
                frames, fps = read_video_frames(video_path, max_frames=140)
                frames = [resize_keep(f, 720) for f in frames]
                scores = fake_model_scores(frames, fps)  # TODO replace with model inference

                st.session_state.last_scores = scores
                st.session_state.last_fps = fps
                st.session_state.last_video_path = video_path
                st.session_state.last_frames = frames

                placeholder = st.empty()
                info_line = st.empty()

                max_live = min(len(frames), 90)
                for i in range(max_live):
                    p = float(scores[i])
                    s = status_from_score(p)
                    emoji = color_from_status(s)
                    placeholder.image(
                        to_rgb(frames[i]),
                        caption=f"Frame {i} | {emoji} {s} | p={p:.2f}",
                        use_container_width=True
                    )
                    info_line.markdown(f"**Time:** {fmt_time(i/fps)} | **Status:** {emoji} **{s}** | **p:** `{p:.2f}`")
                    time.sleep(0.02)

                st.success("Preview complete ✅")
            except Exception as e:
                st.error(str(e))

    with colB:
        st.write("### Quick Indicators")
        scores = st.session_state.last_scores
        fps = st.session_state.last_fps

        if scores is None:
            st.info("Analyze a video first.")
        else:
            peak = float(np.max(scores))
            avg = float(np.mean(scores))
            start_frame, start_sec = detect_fight_start(scores, fps, thr=float(CFG.THRESH_VIOLENCE))

            st.metric("Peak probability", f"{peak:.2f}")
            st.metric("Average probability", f"{avg:.2f}")
            st.metric("Fight start", fmt_time(start_sec))

            now_p = float(scores[-1])
            now_s = status_from_score(now_p)
            st.markdown(f"**Current status:** {color_from_status(now_s)} **{now_s}** (p={now_p:.2f})")

            fig = make_timeline_plot(scores, fps)
            st.pyplot(fig, clear_figure=True, use_container_width=False)


# -----------------------------
# Tab 2: Multi-Camera Wall
# -----------------------------
with tabs[1]:
    st.subheader("Multi-Camera Wall (SOC Style)")

    use_same = st.checkbox("Use same video for all cameras", value=True)

    cams = []
    if use_same:
        csel = st.selectbox("Class", CLASSES, index=0, key="wall_class")
        vids = list_videos_in_class(csel)
        if not vids:
            st.info("No videos in that class folder yet. Upload some in the Upload Manager tab.")
        else:
            vsel = st.selectbox("Video", [v.name for v in vids], key="wall_video")
            one = str(UPLOAD_ROOT / csel / vsel)
            cams = [one, one, one, one]
    else:
        c1 = st.text_input("Camera 1 path", "")
        c2 = st.text_input("Camera 2 path", "")
        c3 = st.text_input("Camera 3 path", "")
        c4 = st.text_input("Camera 4 path", "")
        cams = [c1, c2, c3, c4]

    go = st.button("🧱 Render Camera Wall", disabled=(not cams or not all(bool(x) for x in cams)))

    if go:
        cols = st.columns(2, gap="large")
        for idx, vp in enumerate(cams):
            try:
                frames, fps = read_video_frames(vp, max_frames=45)
                frames = [resize_keep(f, 560) for f in frames]
                scores = fake_model_scores(frames, fps)
                peak = float(np.max(scores))
                s = status_from_score(peak)
                emoji = color_from_status(s)

                show_frame = to_rgb(frames[min(10, len(frames) - 1)])
                with cols[idx % 2]:
                    st.image(show_frame, use_container_width=True,
                             caption=f"Camera {idx+1} | {emoji} {s} | peak p={peak:.2f}")
                    st.progress(min(1.0, peak))
            except Exception as e:
                with cols[idx % 2]:
                    st.error(f"Camera {idx+1}: {e}")


# -----------------------------
# Tab 3: Upload Manager + Batch Analyzer
# -----------------------------
with tabs[2]:
    st.subheader("Upload Manager (2-Class Folders) + Batch Analyzer (Organized Runs)")

    topA, topB = st.columns([1.0, 1.0])
    with topA:
        st.text_input("Run ID (folder name)", key="run_id")
    with topB:
        st.caption(f"Outputs: `outputs_dashboard/runs/{st.session_state.run_id}/...`")

    RUN_ROOT = Path(CFG.OUTPUT_DIR) / "runs" / st.session_state.run_id
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    st.markdown("### 1) Upload videos → save into class folder (and create web preview mp4)")
    up_files = st.file_uploader(
        "Upload one or more videos",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True
    )
    class_choice = st.selectbox("Class", CLASSES, index=0)

    save_btn = st.button("💾 Save uploaded files into class folder", type="primary", disabled=(not up_files))

    if save_btn and up_files:
        saved = 0
        preview_ok = 0
        for uf in up_files:
            dst = UPLOAD_ROOT / class_choice / uf.name
            with open(dst, "wb") as f:
                f.write(uf.getbuffer())
            saved += 1

            prev_path = UPLOAD_ROOT / class_choice / "_preview" / (safe_stem(dst) + ".mp4")
            if make_web_preview(dst, prev_path):
                preview_ok += 1

        st.success(f"Saved {saved} file(s). Previews created: {preview_ok}/{saved} ✅")
        if preview_ok < saved:
            st.info("Some previews used the OpenCV fallback (ffmpeg not found). Videos will still play in most browsers.")

    st.divider()
    st.markdown("### 2) Pick videos (single or multiple) → preview → analyze")

    left, right = st.columns([1.4, 1.0], gap="large")

    with left:
        pick_class = st.selectbox("Browse class folder", CLASSES, index=0, key="browse_class")
        vids = list_videos_in_class(pick_class)

        if not vids:
            st.info(f"No videos found in `{pick_class}` yet. Upload some above first.")
            selected_names = []
        else:
            video_names = [v.name for v in vids]
            selected_names = st.multiselect(
                "Select videos to analyze (choose many)",
                options=video_names,
                default=video_names[: min(2, len(video_names))]
            )

            if selected_names:
                preview_name = st.selectbox("Preview one selected video", selected_names)
                original_path = UPLOAD_ROOT / pick_class / preview_name
                preview_path = UPLOAD_ROOT / pick_class / "_preview" / (safe_stem(original_path) + ".mp4")
                if not preview_path.exists():
                    make_web_preview(original_path, preview_path)

                if preview_path.exists():
                    st.video(str(preview_path))
                else:
                    st.warning("Preview mp4 not available. The video will still be analyzed correctly.")

    with right:
        st.markdown("### Analysis settings")
        thr_v = st.slider("Violence threshold", 0.10, 0.99, float(CFG.THRESH_VIOLENCE), 0.01)
        thr_s = st.slider("Suspicious threshold", 0.10, 0.99, float(CFG.THRESH_SUSPICIOUS), 0.01)
        max_frames = st.number_input("Max frames per video (speed)", 60, 3000, CFG.MAX_FRAMES_PREVIEW, 10)
        CFG.THRESH_VIOLENCE = float(thr_v)
        CFG.THRESH_SUSPICIOUS = float(thr_s)

        run_batch = st.button("🧠 Analyze selected videos", type="primary", disabled=(not selected_names))

    if run_batch and selected_names:
        progress = st.progress(0)
        results_rows = []

        for idx, name in enumerate(selected_names, start=1):
            video_path = UPLOAD_ROOT / pick_class / name
            vid_stem = safe_stem(video_path)
            vid_out_dir = RUN_ROOT / pick_class / vid_stem
            vid_out_dir.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy2(str(video_path), str(vid_out_dir / f"original{video_path.suffix.lower()}"))

                run_preview_path = vid_out_dir / "preview.mp4"
                make_web_preview(video_path, run_preview_path)

                frames, fps = read_video_frames(str(video_path), max_frames=int(max_frames))
                frames = [resize_keep(f, 720) for f in frames]

                # TODO: replace with your real model inference
                scores = fake_model_scores(frames, fps)

                start_frame, start_sec = detect_fight_start(scores, fps, thr=float(CFG.THRESH_VIOLENCE))
                peak = float(np.max(scores))
                avg = float(np.mean(scores))
                overall_status = status_from_score(peak)
                how_started = describe_how_started(frames, scores, fps, start_frame)

                save_plot_png(make_timeline_plot(scores, fps), vid_out_dir / "timeline.png")
                save_plot_png(make_hist_plot(scores), vid_out_dir / "confidence_hist.png")

                summary = {
                    "file": name,
                    "class": pick_class,
                    "fps": float(fps),
                    "frames_analyzed": int(len(frames)),
                    "thresholds": {
                        "suspicious": float(CFG.THRESH_SUSPICIOUS),
                        "violence": float(CFG.THRESH_VIOLENCE),
                    },
                    "results": {
                        "peak_probability": round(peak, 4),
                        "avg_probability": round(avg, 4),
                        "overall_status": overall_status,
                        "fight_start_frame": None if start_frame is None else int(start_frame),
                        "fight_start_sec": None if start_sec is None else round(float(start_sec), 3),
                        "fight_start_time": fmt_time(start_sec),
                        "how_started": how_started,
                    },
                    "outputs_dir": str(vid_out_dir),
                }
                write_json(summary, vid_out_dir / "summary.json")
                write_json(
                    {"fight_start_time": fmt_time(start_sec),
                     "fight_start_sec": None if start_sec is None else round(float(start_sec), 3),
                     "how_started": how_started},
                    vid_out_dir / "onset.json"
                )

                pred_text = (
                    f"Class folder: {pick_class}\n"
                    f"File: {name}\n"
                    f"Peak violence prob: {peak:.3f}\n"
                    f"Avg prob: {avg:.3f}\n"
                    f"Status: {overall_status}\n"
                    f"Fight started at: {fmt_time(start_sec)} ({'' if start_sec is None else round(float(start_sec),3)} s)\n"
                    f"How it started: {how_started}\n"
                )
                write_text(vid_out_dir / "pred.txt", pred_text)

                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "run_id": st.session_state.run_id,
                    "class": pick_class,
                    "file": name,
                    "status": overall_status,
                    "peak_p": round(peak, 4),
                    "avg_p": round(avg, 4),
                    "fight_start_time": fmt_time(start_sec),
                    "fight_start_sec": "" if start_sec is None else round(float(start_sec), 3),
                    "how_started": how_started,
                    "outputs_dir": str(vid_out_dir),
                }
                append_log_row(row)
                results_rows.append(row)

                st.session_state.last_scores = scores
                st.session_state.last_fps = fps
                st.session_state.last_video_path = str(video_path)
                st.session_state.last_frames = frames

            except Exception as e:
                results_rows.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "run_id": st.session_state.run_id,
                    "class": pick_class,
                    "file": name,
                    "status": "ERROR",
                    "error": str(e),
                    "outputs_dir": str(vid_out_dir),
                })

            progress.progress(idx / len(selected_names))

        df_res = pd.DataFrame(results_rows)
        st.session_state.last_batch_df = df_res

        st.success("Batch analysis finished ✅")
        st.dataframe(df_res, use_container_width=True, height=320)

        scores = st.session_state.last_scores
        fps = st.session_state.last_fps
        if scores is not None and fps is not None:
            g1 = make_timeline_plot(scores, fps)
            g2 = make_hist_plot(scores)
            c1, c2 = st.columns(2, gap="small")
            with c1:
                st.pyplot(g1, clear_figure=True, use_container_width=False)
            with c2:
                st.pyplot(g2, clear_figure=True, use_container_width=False)

        csv_bytes = df_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download batch results CSV",
            data=csv_bytes,
            file_name="batch_results.csv",
            mime="text/csv"
        )

        st.caption(f"Run outputs: `{RUN_ROOT}` | Log: `{LOGS_ROOT / 'analysis_log.csv'}`")


# -----------------------------
# Tab 4: Grad-CAM Panel (COMPACT layout)
# -----------------------------
with tabs[3]:
    st.subheader("Grad-CAM Panel (Compact Dashboard Layout)")

    frames = st.session_state.last_frames
    scores = st.session_state.last_scores
    fps = st.session_state.last_fps

    if frames is None or scores is None or fps is None:
        st.info("Analyze at least one video first in **Upload Manager**.")
    else:
        start_frame, start_sec = detect_fight_start(scores, fps, thr=float(CFG.THRESH_VIOLENCE))
        how_txt = describe_how_started(frames, scores, fps, start_frame)

        left, right = st.columns([1.2, 1.0], gap="large")

        with left:
            st.markdown("### Visualization")
            i = st.slider("Frame", 0, len(frames) - 1, min(10, len(frames) - 1), key="gradcam_frame")

            p = float(scores[i])
            s = status_from_score(p)
            emoji = color_from_status(s)

            hm = fake_gradcam_heatmap(frames[i])
            overlay = overlay_heatmap(frames[i], hm)

            show_w = 560
            original_small = resize_keep(frames[i], w=show_w)
            overlay_small = resize_keep(overlay, w=show_w)

            c1, c2 = st.columns(2, gap="small")
            with c1:
                st.image(to_rgb(original_small), use_container_width=True, caption="Original")
            with c2:
                st.image(to_rgb(overlay_small), use_container_width=True,
                         caption=f"Grad-CAM Overlay | {emoji} {s} | p={p:.2f}")

            with st.expander("Show Heatmap (optional)", expanded=False):
                hm_uint8 = np.uint8(hm * 255)
                hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
                hm_small = resize_keep(hm_color, w=show_w)
                st.image(to_rgb(hm_small), use_container_width=True, caption="Heatmap")

        with right:
            st.markdown("### Info Panel")

            m1, m2 = st.columns(2, gap="small")
            with m1:
                st.metric("Status", f"{emoji} {s}")
                st.metric("Frame", str(i))
                st.metric("Time", fmt_time(i / fps))
            with m2:
                st.metric("Probability", f"{p:.2f}")
                st.metric("Violence thr", f"{CFG.THRESH_VIOLENCE:.2f}")
                st.metric("Suspicious thr", f"{CFG.THRESH_SUSPICIOUS:.2f}")

            st.divider()
            st.markdown("#### Fight Onset (When + How)")

            o1, o2 = st.columns(2, gap="small")
            with o1:
                st.write("**Fight started:**")
                st.write(f"⏱️ {fmt_time(start_sec)}")
            with o2:
                detected = "YES" if start_sec is not None else "NO"
                st.write("**Violence detected?**")
                st.write(f"{'✅' if detected=='YES' else '❌'} {detected}")

            st.info(f"**How it started:** {how_txt}")

            st.divider()
            st.markdown("#### Timeline (small)")
            fig = make_timeline_plot(scores, fps)
            st.pyplot(fig, clear_figure=True, use_container_width=False)


# -----------------------------
# Tab 5: Analytics
# -----------------------------
with tabs[4]:
    st.subheader("Analytics (Logs + Distributions + False Alarms)")

    scores = st.session_state.last_scores
    fps = st.session_state.last_fps

    if scores is None or fps is None:
        st.info("Analyze a video first.")
    else:
        start_frame, start_sec = detect_fight_start(scores, fps, thr=float(CFG.THRESH_VIOLENCE))

        step = max(1, int(0.5 * fps))
        rows = []
        for i in range(0, len(scores), step):
            p = float(scores[i])
            stt = status_from_score(p)
            rows.append({"time": fmt_time(i / fps), "frame": i, "prob": round(p, 3), "status": stt})
        df = pd.DataFrame(rows)

        colA, colB = st.columns([1.3, 1.0], gap="large")
        with colA:
            st.write("### Detection Log")
            st.dataframe(df, use_container_width=True, height=300)
        with colB:
            st.write("### Summary")
            st.metric("Estimated fight start", fmt_time(start_sec))
            st.metric("Peak probability", f"{float(np.max(scores)):.2f}")

            above = scores >= CFG.THRESH_VIOLENCE
            runs = []
            run = 0
            for a in above:
                if a:
                    run += 1
                else:
                    if run > 0:
                        runs.append(run)
                    run = 0
            if run > 0:
                runs.append(run)
            short_runs = [r for r in runs if (r / fps) < 0.5]
            st.metric("Short ALERT spikes (<0.5s)", str(len(short_runs)))

        g1 = make_timeline_plot(scores, fps)
        g2 = make_hist_plot(scores)
        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.pyplot(g1, clear_figure=True, use_container_width=False)
        with c2:
            st.pyplot(g2, clear_figure=True, use_container_width=False)

        log_path = LOGS_ROOT / "analysis_log.csv"
        if log_path.exists():
            st.write("### Full Analysis Log (CSV)")
            try:
                st.dataframe(pd.read_csv(log_path), use_container_width=True, height=260)
            except Exception:
                st.info(f"Log exists at `{log_path}` but couldn't render.")


# -----------------------------
# Tab 6: Dataset Comparison
# -----------------------------
with tabs[5]:
    st.subheader("Multi-Dataset Comparison (Edit with your real results)")

    metrics = pd.DataFrame([
        {"Dataset": "RWF-2000", "Model": "R3D-18", "Accuracy": 0.93, "Precision": 0.92, "Recall": 0.91, "F1": 0.91},
        {"Dataset": "HockeyFight", "Model": "R3D-18", "Accuracy": 0.95, "Precision": 0.94, "Recall": 0.94, "F1": 0.94},
    ])
    st.dataframe(metrics, use_container_width=True, height=220)

    st.write("#### Upload your metrics CSV (optional)")
    met_csv = st.file_uploader("CSV columns: Dataset, Model, Accuracy, Precision, Recall, F1", type=["csv"], key="metrics_csv")
    if met_csv is not None:
        try:
            dfm = pd.read_csv(met_csv)
            st.dataframe(dfm, use_container_width=True, height=260)
        except Exception as e:
            st.error(f"CSV error: {e}")


# -----------------------------
# Tab 7: Incident Report
# -----------------------------
with tabs[6]:
    st.subheader("Incident Report Generator")

    scores = st.session_state.last_scores
    fps = st.session_state.last_fps
    video_path = st.session_state.last_video_path

    col1, col2 = st.columns([1.1, 1.2], gap="large")
    with col1:
        st.write("### Report Inputs")
        cam_name = st.text_input("Camera name", value="Entrance Camera")
        location = st.text_input("Location", value="Main Gate / Hallway")
        model_name = st.text_input("Model", value="R3D-18")
        dataset_name = st.text_input("Dataset", value="RWF-2000 / HockeyFight")
        notes = st.text_area("Notes (optional)", value="")
        gen = st.button("🧾 Generate Report", type="primary", disabled=(scores is None or fps is None or video_path is None))

    with col2:
        st.write("### Preview")
        if scores is None or fps is None or video_path is None:
            st.info("Analyze a video first (Upload Manager).")
        else:
            start_frame, start_sec = detect_fight_start(scores, fps, thr=float(CFG.THRESH_VIOLENCE))
            peak = float(np.max(scores))
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.markdown(f"- **Time generated:** {now}")
            st.markdown(f"- **Video:** `{Path(video_path).name}`")
            st.markdown(f"- **Estimated fight start:** **{fmt_time(start_sec)}**")
            st.markdown(f"- **Peak probability:** **{peak:.2f}**")
            st.markdown(f"- **Status:** {color_from_status(status_from_score(peak))} **{status_from_score(peak)}**")

            g1 = make_timeline_plot(scores, fps)
            g2 = make_hist_plot(scores)
            c1, c2 = st.columns(2, gap="small")
            with c1:
                st.pyplot(g1, clear_figure=True, use_container_width=False)
            with c2:
                st.pyplot(g2, clear_figure=True, use_container_width=False)

    if gen and scores is not None:
        start_frame, start_sec = detect_fight_start(scores, fps, thr=float(CFG.THRESH_VIOLENCE))
        peak = float(np.max(scores))
        how_started = describe_how_started(st.session_state.last_frames, scores, fps, start_frame)

        report = {
            "incident_id": f"INC-{int(time.time())}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "camera": cam_name,
            "location": location,
            "video_file": Path(video_path).name if video_path else None,
            "model": model_name,
            "dataset": dataset_name,
            "thresholds": {"suspicious": float(CFG.THRESH_SUSPICIOUS), "violence": float(CFG.THRESH_VIOLENCE)},
            "results": {
                "peak_probability": round(peak, 4),
                "estimated_fight_start_sec": None if start_sec is None else round(float(start_sec), 3),
                "estimated_fight_start_time": fmt_time(start_sec),
                "how_started": how_started,
            },
            "notes": notes
        }

        path = save_incident_report(report)
        st.success(f"Report saved: {path}")

        buf = io.BytesIO(json.dumps(report, indent=2).encode("utf-8"))
        st.download_button(
            "⬇️ Download report JSON",
            data=buf,
            file_name=os.path.basename(path),
            mime="application/json"
        )

st.divider()
st.caption(
    "Next: Replace fake_model_scores() with your real R3D-18 inference and fake_gradcam_heatmap() with your real Grad-CAM heatmaps."
)
