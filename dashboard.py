import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# dashboard.py — VisionGuard Violence Detection Dashboard v7
# NEW in v7: "🎬 Process Raw Video" page
#   - Upload any raw mp4/avi → pick dataset config → runs full GradCAM pipeline in background
#   - Live per-stage progress bar inside the page
#   - Results auto-load into GradCAM Viewer / Grid Viewer / Timeline / Explorer
#   - ZIP download always available after processing

import csv, io, re, time, json, shutil, hashlib, zipfile, subprocess, threading
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
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

# ── Processing constants (mirror gradcam_final.py) ──
SMOOTH_N     = 20
SMOOTH_SIGMA = 0.10
SMOOTH_K     = 2
IMG_SIZE     = 112
DISPLAY_W    = 480
ALPHA        = 0.55
EPS          = 1e-8
GRID_FRAMES  = 8
ROWS_G, COLS_G = 2, 4
CAM_METHODS  = ["gradcam", "gradcampp", "smooth_gradcampp", "layercam"]
CAM_LABELS   = {
    "gradcam":          "GradCAM (L4)",
    "gradcampp":        "GradCAM++ (L4)",
    "smooth_gradcampp": "SmoothGradCAM++ (L4)",
    "layercam":         "LayerCAM (L2+L3+L4)",
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")

# ── Dataset configs exposed in UI ──
PROC_CONFIGS = {
    "HockeyFight": {
        "name":          "hockeyfight",
        "ckpt":          "checkpoints/r3d18_best_lcm_lstm.pth",
        "window_size":   16,
        "window_stride": 2,
        "onset_thresh":  0.50,
        "spike_delta":   0.04,
        "pred_thresh":   0.50,
        "fc_dropout":    False,
        "label":         "HockeyFight",
    },
    "RWF-2000": {
        "name":          "rwf",
        "ckpt":          "checkpoints/r3d18_best_RWF_lcm_lstm.pth",
        "window_size":   32,
        "window_stride": 8,
        "onset_thresh":  0.35,
        "spike_delta":   0.04,
        "pred_thresh":   0.35,
        "fc_dropout":    True,
        "label":         "RWF-2000",
    },
}

DATASETS = {
    "hockeyfight": ["Fight", "Nonfight"],
    "rwf":         ["Fight", "NonFight", "pred_nonfight"],
}

ALL_VID_KEYS  = ["original","gradcam","gradcampp","smooth_gradcampp","layercam","combined"]
VID_LABELS    = {
    "original":         "📹 Original",
    "gradcam":          "🔥 GradCAM",
    "gradcampp":        "🔥 GradCAM++",
    "smooth_gradcampp": "✨ Smooth GradCAM++",
    "layercam":         "🌊 LayerCAM",
    "combined":         "🎯 Combined",
}
ALL_GRID_KEYS = ["raw_grid","gradcam_grid","gradcampp_grid",
                 "smooth_gradcampp_grid","layercam_grid","combined_grid"]
GRID_LABELS   = {
    "raw_grid":              "📷 Raw Frames",
    "gradcam_grid":          "🌡️ GradCAM",
    "gradcampp_grid":        "🌡️ GradCAM++",
    "smooth_gradcampp_grid": "✨ Smooth GradCAM++",
    "layercam_grid":         "🌊 LayerCAM",
    "combined_grid":         "🎯 Combined",
}


# ══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE  (exact copy from gradcam_final.py)
# ══════════════════════════════════════════════════════════════
class LCM3D(nn.Module):
    def __init__(self, channels, k_t=3, k_s=3):
        super().__init__()
        self.dw   = nn.Conv3d(channels, channels, (k_t, k_s, k_s),
                              padding=(k_t//2, k_s//2, k_s//2),
                              groups=channels, bias=False)
        self.pw   = nn.Conv3d(channels, channels, 1, bias=False)
        self.bn   = nn.BatchNorm3d(channels)
        self.act  = nn.ReLU(inplace=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels, 1, bias=True),
            nn.Sigmoid())
    def forward(self, x):
        y = self.act(self.bn(self.pw(self.dw(x))))
        return x + y * self.gate(y)


class LSTMHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(p=0.3)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.drop(out[:, -1, :])
    def forward_all_steps(self, x):
        out, _ = self.lstm(x)
        return out


class R3D18WithLCM_LSTM(nn.Module):
    def __init__(self, num_classes=2, lcm_after="layer4",
                 lstm_hidden=256, lstm_layers=1,
                 lstm_dropout=0.3, dropout_p=0.4, fc_dropout=False):
        super().__init__()
        base = r3d_18(weights=None)
        self.stem    = base.stem
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.lcm_after    = lcm_after
        self.lcm          = LCM3D(256 if lcm_after == "layer3" else 512)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.lstm_head    = LSTMHead(512, lstm_hidden, lstm_layers, lstm_dropout)
        self.fc = nn.Sequential(nn.Dropout(p=dropout_p),
                                nn.Linear(lstm_hidden, num_classes)) \
                  if fc_dropout else nn.Linear(lstm_hidden, num_classes)

    def _backbone(self, x):
        x    = self.stem(x); x = self.layer1(x)
        out2 = self.layer2(x)
        out3 = self.layer3(out2)
        if self.lcm_after == "layer3": out3 = self.lcm(out3)
        out4 = self.layer4(out3)
        if self.lcm_after == "layer4": out4 = self.lcm(out4)
        return out2, out3, out4

    def _pool_seq(self, out4):
        dev = out4.device
        p   = self.spatial_pool(out4.cpu() if dev.type == "mps" else out4)
        if dev.type == "mps": p = p.to(DEVICE)
        return p.squeeze(-1).squeeze(-1).permute(0, 2, 1)

    def forward(self, x):
        _, _, out4 = self._backbone(x)
        return self.fc(self.lstm_head(self._pool_seq(out4)))

    def forward_with_seq(self, x):
        _, _, out4 = self._backbone(x)
        seq   = self._pool_seq(out4)
        all_h = self.lstm_head.forward_all_steps(seq)
        all_h = self.lstm_head.drop(all_h)
        lin   = self.fc[-1] if isinstance(self.fc, nn.Sequential) else self.fc
        seq_p = torch.softmax(lin(all_h), dim=-1)[0, :, 1].detach().cpu().numpy()
        logits = self.fc(self.lstm_head.drop(all_h[:, -1, :]))
        return logits, seq_p


def _disable_inplace(model):
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6)):
            m.inplace = False


# ── Model cache so we don't reload on every rerun ──
_MODEL_CACHE: dict = {}

def load_model_cached(cfg: dict):
    key = cfg["ckpt"]
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    ckpt = torch.load(cfg["ckpt"], map_location="cpu")
    lafter = ckpt.get("lcm_after",   "layer4") if isinstance(ckpt, dict) else "layer4"
    lhid   = ckpt.get("lstm_hidden", 256)       if isinstance(ckpt, dict) else 256
    llyr   = ckpt.get("lstm_layers", 1)         if isinstance(ckpt, dict) else 1
    model  = R3D18WithLCM_LSTM(lcm_after=lafter, lstm_hidden=lhid,
                                lstm_layers=llyr, fc_dropout=cfg["fc_dropout"])
    sd = (ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt) \
         if isinstance(ckpt, dict) else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model = model.to(DEVICE).eval()
    _disable_inplace(model)
    val_acc = ckpt.get("best_val_acc", ckpt.get("val_acc")) if isinstance(ckpt, dict) else None
    meta = {"epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "val_acc": val_acc, "path": cfg["ckpt"]}
    _MODEL_CACHE[key] = (model, meta)
    return model, meta


# ══════════════════════════════════════════════════════════════
# CAM ENGINE  (exact copy from gradcam_final.py)
# ══════════════════════════════════════════════════════════════
class CAMEngine:
    def __init__(self, model):
        self.model  = model
        self._saved = {}
        self._hooks = [
            model.layer2[-1].conv2.register_forward_hook(
                lambda m, i, o: self._saved.update({"layer2": o})),
            model.layer3[-1].conv2.register_forward_hook(
                lambda m, i, o: self._saved.update({"layer3": o})),
            model.layer4[-1].conv2.register_forward_hook(
                lambda m, i, o: self._saved.update({"layer4": o})),
        ]

    def _fwd_grad(self, x, cls, layers=("layer2","layer3","layer4")):
        self.model.zero_grad(); self._saved.clear()
        with torch.enable_grad():
            score = self.model(x)[0, cls]
            grads = torch.autograd.grad(score,
                        [self._saved[l] for l in layers],
                        retain_graph=False, create_graph=False)
        acts  = {l: self._saved[l].detach()[0] for l in layers}
        grads = {l: grads[i].detach()[0] for i, l in enumerate(layers)}
        return acts, grads

    def _up_norm(self, cam, tgt):
        up = F.interpolate(cam.unsqueeze(0).unsqueeze(0).float(),
                           size=tgt, mode="trilinear",
                           align_corners=False).squeeze().cpu().numpy()
        mn, mx = up.min(), up.max()
        return (up - mn) / (mx - mn + EPS)

    def compute_all(self, x, cls=1):
        T, H, W = x.shape[2], x.shape[3], x.shape[4]
        tgt = (T, H, W)
        A, G = self._fwd_grad(x, cls)

        # GradCAM
        w  = G["layer4"].mean(dim=(1,2,3))
        gc = self._up_norm(F.relu((w[:,None,None,None]*A["layer4"]).sum(0)), tgt)

        # GradCAM++
        G2 = G["layer4"]**2; G3 = G["layer4"]**3
        dn = 2.0*G2 + (A["layer4"]*G3).sum(dim=(1,2,3), keepdim=True)
        al = G2 / (dn + EPS)
        wt = (al * F.relu(G["layer4"])).sum(dim=(1,2,3))
        gcpp = self._up_norm(F.relu((wt[:,None,None,None]*A["layer4"]).sum(0)), tgt)

        # LayerCAM
        lc = np.zeros((T,H,W), dtype=np.float32)
        for ln in ["layer2","layer3","layer4"]:
            lc += self._up_norm(F.relu(F.relu(G[ln])*A[ln]).sum(0), tgt)
        lc /= 3.0
        mn, mx = lc.min(), lc.max(); lc = (lc-mn)/(mx-mn+EPS)

        # SmoothGradCAM++
        sm = np.zeros((T,H,W), dtype=np.float32); n_ok = 0
        ns = SMOOTH_SIGMA * (x.max()-x.min()).item()
        for _ in range(SMOOTH_N):
            try:
                an, gn = self._fwd_grad((x+torch.randn_like(x)*ns).detach(),
                                        cls, layers=("layer4",))
                G2n = gn["layer4"]**2; G3n = gn["layer4"]**3
                dn2 = 2.0*G2n + (an["layer4"]*G3n).sum(dim=(1,2,3), keepdim=True)
                al2 = G2n/(dn2+EPS)
                wt2 = (al2*F.relu(gn["layer4"])).sum(dim=(1,2,3))
                sm += self._up_norm(F.relu((wt2[:,None,None,None]*an["layer4"]).sum(0)), tgt)
                n_ok += 1
            except Exception:
                pass
        if n_ok > 0: sm /= n_ok
        mn, mx = sm.min(), sm.max(); sm = (sm-mn)/(mx-mn+EPS)

        return {"gradcam": gc, "gradcampp": gcpp, "smooth_gradcampp": sm, "layercam": lc}

    def remove(self):
        for h in self._hooks: h.remove()


# ══════════════════════════════════════════════════════════════
# PROCESSING HELPERS  (mirror gradcam_final.py)
# ══════════════════════════════════════════════════════════════
def _win_idx(start, total, ws):
    end  = min(start+ws, total)
    idxs = list(range(start, end))
    if len(idxs) < ws: idxs = list(range(max(0, total-ws), total))
    return idxs

def _to_tensor(frames, indices):
    arr = np.stack([frames[i] for i in indices]).astype(np.float32)/255.0
    return torch.from_numpy(np.transpose(arr,(3,0,1,2))).unsqueeze(0).to(DEVICE)

def _smooth_curve(arr, k=2):
    return np.convolve(arr, np.ones(k)/k, mode="same") if k>1 else arr.copy()

def _apply_heatmap(frame, cam):
    heat = cv2.cvtColor(
        cv2.applyColorMap((np.clip(cam,0,1)*255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB)
    return np.clip(frame*(1-ALPHA)+heat*ALPHA, 0, 255).astype(np.uint8)

def _draw_info_bar(frame_rgb, ds_label, pred_lbl, conf,
                   frame_idx, total, fight_prob, onset_frame, fps,
                   method_tag, onset_thresh):
    W   = DISPLAY_W
    img = cv2.cvtColor(cv2.resize(frame_rgb, (W,W)), cv2.COLOR_RGB2BGR)
    va  = (onset_frame is not None) and (frame_idx >= onset_frame)
    GREEN=(0,210,0); RED=(0,0,210); YELLOW=(0,210,210); GREY=(150,150,150); DARK=(70,70,70)
    BAR = 120
    bar = np.zeros((BAR, W, 3), dtype=np.uint8)
    if va: bar[:,:] = (18,0,0); bar[:3,:] = RED; bar[-3:,:] = RED
    cv2.putText(bar, f"{ds_label}  Pred:{pred_lbl}  Conf:{conf*100:.1f}%",
                (8,20), FONT, 0.44, GREEN if pred_lbl=="Fight" else GREY, 1, cv2.LINE_AA)
    cv2.putText(bar, f"Frame:{frame_idx+1}/{total}  p(fight):{fight_prob:.3f}  [{method_tag}]",
                (8,42), FONT, 0.42, RED if fight_prob>onset_thresh else GREY, 1, cv2.LINE_AA)
    if va:
        cv2.putText(bar, f"VIOLENCE DETECTED  onset:frame {onset_frame} @ {onset_frame/fps:.2f}s",
                    (8,66), FONT, 0.44, RED, 1, cv2.LINE_AA)
        cv2.putText(bar, f"+{frame_idx-onset_frame} frames since onset",
                    (8,88), FONT, 0.40, YELLOW, 1, cv2.LINE_AA)
    else:
        cv2.putText(bar, "Monitoring...", (8,66), FONT, 0.44, GREY, 1, cv2.LINE_AA)
    cv2.putText(bar, "R3D-18+LCM+LSTM", (8,108), FONT, 0.36, DARK, 1, cv2.LINE_AA)
    cv2.circle(bar, (W-16,BAR//2), 7, RED if va else DARK, -1)
    return cv2.cvtColor(np.vstack([img, bar]), cv2.COLOR_BGR2RGB)

def _write_video(path, frames, fps):
    if not frames: return
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for f in frames: wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    wr.release()

def _make_grid(imgs):
    h, w = imgs[0].shape[:2]
    g = np.zeros((ROWS_G*h, COLS_G*w, 3), dtype=np.uint8)
    for k, im in enumerate(imgs[:ROWS_G*COLS_G]):
        r, c = divmod(k, COLS_G); g[r*h:(r+1)*h, c*w:(c+1)*w] = im
    return g

def _save_timeline(sfp, rfp, onset, fps, path, vid_name, pred_lbl, thresh):
    t   = [i/fps for i in range(len(sfp))]
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(t, sfp, "#2196F3", linewidth=1.8, label="P(fight) smoothed", zorder=3)
    ax.plot(t, rfp, "#90CAF9", linewidth=0.8, alpha=0.6, label="P(fight) raw", zorder=2)
    ax.axhline(thresh, color="red", linewidth=1.2, linestyle="--", label=f"Threshold ({thresh})")
    if onset is not None:
        ot = onset/fps
        ax.axvline(ot, color="green", linewidth=2.0, label=f"Onset @ {ot:.2f}s")
        ax.fill_between(t, 0, sfp, where=[x>=ot for x in t], alpha=0.18, color="green")
    ax.set_title(f"{vid_name}  pred={pred_lbl}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("P(fight)")
    ax.set_ylim(0,1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(str(path), dpi=120); plt.close()

def _lstm_onset(sfp, rfp, total, thresh, spike):
    prev = 0.0
    for i in range(total):
        fp = float(sfp[i])
        if fp > thresh and (fp-prev) > spike: return i
        prev = max(prev, fp)
    for i in range(total):
        if sfp[i] > thresh: return i
    for i in range(total):
        if rfp[i] > thresh: return i
    for i in range(total):
        if rfp[i] > 0.30: return i
    return int(np.argmax(sfp))

def _safe_name(stem):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)


# ══════════════════════════════════════════════════════════════
# MAIN PROCESSING FUNCTION  (runs in background thread)
# ══════════════════════════════════════════════════════════════
def run_processing_pipeline(
    vid_path: Path,
    cfg: dict,
    true_label: str,
    out_dir: Path,
    progress_dict: dict,   # shared dict → UI polls this
):
    """
    Runs the full gradcam_final pipeline on a single video.
    progress_dict keys:
        stage   : str   — current stage label shown in UI
        pct     : float — 0.0–1.0
        done    : bool
        error   : str | None
        pred_lbl: str
        conf    : float
        onset   : int | None
        out_dir : str   — path to output folder
    """
    def upd(pct, stage):
        progress_dict["pct"]   = pct
        progress_dict["stage"] = stage

    try:
        upd(0.02, "📂  Reading video frames...")
        cap = cv2.VideoCapture(str(vid_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        frames = []
        while True:
            ok, f = cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(cv2.resize(f,(IMG_SIZE,IMG_SIZE)), cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames: raise RuntimeError("No frames decoded.")
        total = len(frames)
        fps   = float(fps) if fps > 1 else 15.0

        WS = cfg["window_size"]
        starts = list(range(0, total-WS+1, cfg["window_stride"]))
        if not starts: starts = [0]
        if starts[-1]+WS < total: starts.append(max(0, total-WS))

        # ── Stage 1: Predict ──────────────────────────────────
        upd(0.08, "🧠  Loading model & running predictions...")
        model, meta = load_model_cached(cfg)

        best_p = None; best_fp = -1.0
        ffp    = np.zeros(total, np.float32)
        fhc    = np.zeros(total, np.float32)
        n_wins = len(starts)
        for wi, start in enumerate(starts):
            idx = _win_idx(start, total, WS)
            with torch.no_grad():
                logits, seq = model.forward_with_seq(_to_tensor(frames, idx))
                p = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)
            if p[1] > best_fp: best_fp = float(p[1]); best_p = p
            n = len(seq)
            for si, gi in enumerate(idx):
                if 0 <= gi < total:
                    ci = min(int(si*n/len(idx)), n-1)
                    ffp[gi] += seq[ci]; fhc[gi] += 1
            upd(0.08 + 0.22*(wi+1)/n_wins,
                f"🧠  Predictions... window {wi+1}/{n_wins}")

        ffp /= np.maximum(fhc, 1)
        sfp  = _smooth_curve(ffp, SMOOTH_K)
        pred = 1 if best_fp >= cfg["pred_thresh"] else 0
        conf = float(best_p[pred])
        pred_lbl = "Fight" if pred == 1 else "Nonfight"
        progress_dict["pred_lbl"] = pred_lbl
        progress_dict["conf"]     = conf

        # ── Stage 2: Onset ────────────────────────────────────
        upd(0.32, "⏱️  Detecting fight onset...")
        onset = None
        if pred == 1:
            onset = _lstm_onset(sfp, ffp, total, cfg["onset_thresh"], cfg["spike_delta"])
        progress_dict["onset"] = onset

        # ── Stage 3: CAMs ─────────────────────────────────────
        cams = None
        if pred == 1:
            acc = {m: np.zeros((total,IMG_SIZE,IMG_SIZE), np.float32) for m in CAM_METHODS}
            hc  = np.zeros(total, np.float32)
            eng = CAMEngine(model)
            for wi, start in enumerate(starts):
                idx = _win_idx(start, total, WS)
                try:
                    cam_out = eng.compute_all(_to_tensor(frames, idx), cls=1)
                except Exception as e:
                    continue
                nc = cam_out["gradcam"].shape[0]
                for li, gi in enumerate(idx):
                    if 0 <= gi < total:
                        ci = min(int(li*nc/len(idx)), nc-1)
                        for m in CAM_METHODS: acc[m][gi] += cam_out[m][ci]
                        hc[gi] += 1
                upd(0.32 + 0.38*(wi+1)/n_wins,
                    f"🔥  Computing CAMs... window {wi+1}/{n_wins}")
            eng.remove()
            def _norm(a):
                a = a / np.maximum(hc[:,None,None], 1)
                mn, mx = a.min(), a.max()
                return (a-mn)/(mx-mn+EPS)
            cams = {m: _norm(acc[m]) for m in CAM_METHODS}

        # ── Stage 4: Build frame lists ────────────────────────
        upd(0.72, "🎬  Rendering annotated frames...")
        fl = {k: [] for k in ["original"] + CAM_METHODS + ["combined"]}
        kw = dict(ds_label=cfg["label"], pred_lbl=pred_lbl, conf=conf,
                  total=total, onset_frame=onset, fps=fps,
                  onset_thresh=cfg["onset_thresh"])

        for t, fr in enumerate(frames):
            fp_t   = float(sfp[t])
            active = (pred==1) and (onset is not None) and (t >= onset)
            fl["original"].append(
                _draw_info_bar(fr.copy(), frame_idx=t, fight_prob=fp_t,
                               method_tag="Original", **kw))
            for m in CAM_METHODS:
                f_m = _apply_heatmap(fr, cams[m][t]) if active else fr.copy()
                fl[m].append(_draw_info_bar(f_m, frame_idx=t, fight_prob=fp_t,
                                            method_tag=CAM_LABELS[m], **kw))
            if active:
                combo = (cams["gradcampp"][t]+cams["smooth_gradcampp"][t]+cams["layercam"][t])/3
                mn, mx = combo.min(), combo.max()
                combo = (combo-mn)/(mx-mn+EPS)
                f_c = _apply_heatmap(fr, combo)
            else:
                f_c = fr.copy()
            fl["combined"].append(
                _draw_info_bar(f_c, frame_idx=t, fight_prob=fp_t,
                               method_tag="Combined", **kw))

        # ── Stage 5: Write videos ─────────────────────────────
        upd(0.80, "💾  Writing videos...")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_name(vid_path.stem)
        tag_map = [("original","original"),("gradcam","gradcam"),
                   ("gradcampp","gradcampp"),("smooth_gradcampp","smooth_gradcampp"),
                   ("layercam","layercam"),("combined","combined")]
        for tag, key in tag_map:
            _write_video(out_dir/f"{stem}_{tag}.mp4", fl[key], fps)
            upd(0.80 + 0.06*(tag_map.index((tag,key))+1)/len(tag_map),
                f"💾  Writing {tag}.mp4...")

        # ── Stage 6: Grids ────────────────────────────────────
        upd(0.87, "🖼️  Saving frame grids...")
        pick = np.linspace(0, total-1, GRID_FRAMES).astype(int)
        grid_map = [("raw_grid","original"),("gradcam_grid","gradcam"),
                    ("gradcampp_grid","gradcampp"),
                    ("smooth_gradcampp_grid","smooth_gradcampp"),
                    ("layercam_grid","layercam"),("combined_grid","combined")]
        for gname, key in grid_map:
            src = [frames[i] for i in pick] if gname=="raw_grid" else [fl[key][i] for i in pick]
            cv2.imwrite(str(out_dir/f"{gname}.png"),
                        cv2.cvtColor(_make_grid(src), cv2.COLOR_RGB2BGR))

        # ── Stage 7: Timeline ─────────────────────────────────
        upd(0.93, "📊  Generating timeline plot...")
        _save_timeline(sfp, ffp, onset, fps, out_dir/"timeline.png",
                       vid_path.name, pred_lbl, cfg["onset_thresh"])

        # ── Stage 8: pred.txt ─────────────────────────────────
        upd(0.96, "📋  Writing pred.txt...")
        onset_s = f"{onset/fps:.2f}s" if onset is not None else "N/A"
        with open(out_dir/"pred.txt", "w", encoding="utf-8") as f:
            f.write(f"dataset:          {cfg['name']}\n")
            f.write(f"video:            {vid_path}\n")
            f.write(f"true_label:       {true_label}\n")
            f.write(f"pred_label:       {pred_lbl}\n")
            f.write(f"correct:          {pred_lbl==true_label}\n")
            f.write(f"confidence:       {conf:.4f}\n")
            f.write(f"probs:            [nonfight={best_p[0]:.6f}  fight={best_p[1]:.6f}]\n")
            f.write(f"model_path:       {meta['path']}\n")
            f.write(f"model_epoch:      {meta.get('epoch','N/A')}\n")
            f.write(f"model_val_acc:    {meta.get('val_acc','N/A')}\n")
            f.write(f"window_size:      {WS}\n")
            f.write(f"window_stride:    {cfg['window_stride']}\n")
            f.write(f"onset_threshold:  {cfg['onset_thresh']}\n")
            f.write(f"spike_delta:      {cfg['spike_delta']}\n")
            f.write(f"total_frames:     {total}\n")
            f.write(f"cam_methods:      GradCAM|GradCAM++|SmoothGradCAM++|LayerCAM|Combined\n")
            f.write(f"smooth_passes:    {SMOOTH_N}\n")
            if pred == 1:
                f.write(f"onset_frame:      {onset}\n")
                f.write(f"onset_time:       {onset_s}\n")
                f.write(f"heatmap:          from frame {onset} ({onset_s}) onward\n")
            else:
                f.write(f"onset_frame:      N/A\n")
                f.write(f"onset_time:       N/A\n")
                f.write(f"heatmap:          NOT rendered — Nonfight prediction\n")
                f.write(f"nonfight_reason:  p(fight)={best_p[1]:.4f} < {cfg['pred_thresh']}\n")

        upd(1.0, "✅  Done!")
        progress_dict["out_dir"] = str(out_dir)
        progress_dict["done"]    = True

    except Exception as e:
        import traceback
        progress_dict["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        progress_dict["done"]  = True


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VisionGuard — Violence Detection",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
# THEME CSS
# ══════════════════════════════════════════════════════════════
def get_theme_css(theme="dark", accent="#e05252", font_size="medium"):
    font_scale  = {"small":"0.85rem","medium":"1rem","large":"1.1rem"}.get(font_size,"1rem")
    if theme == "dark":
        bg="#080c10"; bg2="#0d1520"; bg3="#0a0f18"; border="#1a2535"
        text_main="#c8d8e8"; text_dim="#445566"; text_dim2="#2a3a4a"
        text_head="#e8f4ff"; text_blue="#7ecfff"; text_sub="#aabbc8"
        text_h3="#7a99b0"; scanline="rgba(0,0,0,0.03)"
    else:
        bg="#f0f4f8"; bg2="#ffffff"; bg3="#e8edf3"; border="#cdd5df"
        text_main="#1a2535"; text_dim="#556677"; text_dim2="#778899"
        text_head="#0a1020"; text_blue="#1a6fc4"; text_sub="#334455"
        text_h3="#445566"; scanline="rgba(0,0,0,0.01)"
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@700;900&display=swap');
html,body,.stApp{{background:{bg}!important;color:{text_main}!important;font-size:{font_scale};}}
*,*::before,*::after{{box-sizing:border-box;}}
[data-testid="stSidebar"]{{background:{'linear-gradient(180deg,#0a0f18 0%,#080c14 100%)' if theme=='dark' else 'linear-gradient(180deg,#ffffff 0%,#f0f4f8 100%)'}!important;border-right:1px solid {border}!important;min-width:260px!important;max-width:260px!important;}}
[data-testid="stSidebar"]>div:first-child{{padding:0!important;}}
[data-testid="stSidebar"] *{{color:{text_main}!important;}}
#MainMenu,footer,header{{visibility:hidden!important;}}
[data-testid="stDecoration"]{{display:none!important;}}
.block-container{{padding:1.2rem 1.8rem 2rem 1.8rem!important;max-width:100%!important;}}
.stTabs [data-baseweb="tab-list"]{{gap:0;background:{bg2};border-bottom:1px solid {border};border-radius:0;padding:0;}}
.stTabs [data-baseweb="tab"]{{font-family:'Rajdhani',sans-serif;font-size:12px;font-weight:600;padding:8px 16px;border-radius:0;color:{text_dim}!important;background:transparent!important;border-bottom:2px solid transparent;letter-spacing:0.5px;text-transform:uppercase;}}
.stTabs [aria-selected="true"]{{color:{text_head}!important;border-bottom:2px solid {accent}!important;background:transparent!important;}}
[data-testid="metric-container"]{{background:{bg2};border:1px solid {border};border-radius:6px;padding:10px 14px!important;}}
[data-testid="stMetricValue"]{{font-family:'Share Tech Mono',monospace!important;font-size:1.1rem!important;font-weight:400!important;color:{text_blue}!important;}}
[data-testid="stMetricLabel"]{{font-family:'Rajdhani',sans-serif!important;font-size:10px!important;color:{text_dim}!important;text-transform:uppercase;letter-spacing:1px;}}
.stButton>button{{font-family:'Rajdhani',sans-serif!important;font-weight:700!important;font-size:13px!important;border-radius:4px!important;letter-spacing:1px;text-transform:uppercase;transition:all 0.15s ease;}}
.stButton>button[kind="primary"]{{background:{accent}!important;border:none!important;color:white!important;box-shadow:0 0 12px {accent}55!important;}}
.stButton>button[kind="primary"]:hover{{filter:brightness(1.15)!important;box-shadow:0 0 20px {accent}88!important;}}
.stButton>button:not([kind="primary"]){{background:{bg2}!important;border:1px solid {border}!important;color:{text_blue}!important;}}
.stSelectbox>div>div,.stTextInput>div>div>input,.stTextArea>div>div>textarea{{background:{bg2}!important;border:1px solid {border}!important;color:{text_main}!important;border-radius:4px!important;font-family:'Share Tech Mono',monospace!important;}}
.streamlit-expanderHeader{{background:{bg2}!important;border:1px solid {border}!important;border-radius:4px!important;font-family:'Rajdhani',sans-serif!important;font-weight:600!important;color:{text_blue}!important;}}
.streamlit-expanderContent{{background:{bg}!important;border:1px solid {border}!important;border-top:none!important;}}
[data-testid="stDataFrame"]{{border:1px solid {border}!important;border-radius:6px;}}
hr{{border-color:{border}!important;margin:0.8rem 0!important;}}
div[data-testid="stAlert"]{{border-radius:4px!important;border-left-width:3px!important;}}
div[data-testid="stImage"] img{{border-radius:6px;border:1px solid {border};}}
[data-testid="stSlider"]>div>div>div{{background:{accent}!important;}}
::-webkit-scrollbar{{width:4px;height:4px;}}
::-webkit-scrollbar-track{{background:{bg};}}
::-webkit-scrollbar-thumb{{background:{border};border-radius:2px;}}
h1{{font-family:'Orbitron',sans-serif!important;font-size:1.1rem!important;color:{text_head}!important;letter-spacing:2px;}}
h2{{font-family:'Rajdhani',sans-serif!important;font-size:1.1rem!important;font-weight:700!important;color:{text_sub}!important;letter-spacing:1px;text-transform:uppercase;}}
h3{{font-family:'Rajdhani',sans-serif!important;font-size:0.95rem!important;font-weight:600!important;color:{text_h3}!important;}}
.stApp::before{{content:'';position:fixed;top:0;left:0;right:0;bottom:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,{scanline} 2px,{scanline} 4px);pointer-events:none;z-index:9999;}}
[data-testid="stVideo"] video{{max-height:160px!important;width:100%!important;border-radius:4px 4px 0 0!important;background:#000!important;display:block;object-fit:contain!important;}}
[data-testid="stVideo"]{{border:1px solid {border}!important;border-radius:6px!important;overflow:visible!important;background:#000!important;margin-bottom:8px!important;}}
.vg-badge-fight{{display:inline-block;padding:3px 10px;border-radius:3px;background:rgba(224,82,82,0.15);border:1px solid #e05252;color:#ff8080;font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:1px;}}
.vg-badge-normal{{display:inline-block;padding:3px 10px;border-radius:3px;background:rgba(82,224,138,0.12);border:1px solid #52e08a;color:#52e08a;font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:1px;}}
@keyframes pulse-shield{{0%,100%{{transform:scale(1);opacity:1;}}50%{{transform:scale(1.1);opacity:0.75;}}}}
@keyframes fadein-up{{from{{opacity:0;transform:translateY(18px);}}to{{opacity:1;transform:translateY(0);}}}}
.vg-settings-card{{background:{bg2};border:1px solid {border};border-radius:12px;padding:20px 22px;margin-bottom:16px;transition:border-color .2s;}}
.vg-settings-section-title{{font-family:'Orbitron',sans-serif;font-size:10px;font-weight:900;color:{text_dim2};letter-spacing:3px;text-transform:uppercase;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid {border};}}

/* Processing page progress bar */
.proc-stage{{font-family:'Share Tech Mono',monospace;font-size:13px;letter-spacing:1px;padding:6px 0;}}
.proc-done{{color:#52e08a;}}
.proc-active{{color:{accent};animation:fadein-up 0.3s ease;}}
.proc-pending{{color:{text_dim2};}}
</style>"""


# ══════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()
def load_users():
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE) as f: return json.load(f)
        except: pass
    return {}
def save_users(u):
    with open(USERS_FILE,"w") as f: json.dump(u, f, indent=2)
def try_login(u, p):   return load_users().get(u) == hash_pw(p)
def try_register(u, p):
    if not u or not p: return False, "Username and password required."
    if len(p) < 4:     return False, "Password must be at least 4 characters."
    users = load_users()
    if u in users:     return False, "Username already exists."
    users[u] = hash_pw(p); save_users(users)
    return True, "Account created! You can now log in."


# ══════════════════════════════════════════════════════════════
# CORE UTILS
# ══════════════════════════════════════════════════════════════
def is_fight_pred(pred, flip=False):
    lbl = str(pred.get("pred_label","")).lower()
    raw = "fight" in lbl and "non" not in lbl
    return (not raw) if flip else raw

def pred_label_to_status(pred_label):
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

def scores_from_pred(pred, n_frames, fps):
    conf = 0.5
    try: conf = float(pred.get("confidence",0.5))
    except: pass
    onset_frame = 0
    try: onset_frame = int(pred.get("onset_frame",0))
    except: pass
    is_fight = is_fight_pred(pred)
    scores   = np.zeros(n_frames, dtype=np.float32)
    if is_fight:
        for i in range(n_frames):
            if i < onset_frame: scores[i] = max(0.05, conf*0.1)
            else:
                ramp = min(1.0, (i-onset_frame)/max(1,fps))
                scores[i] = float(np.clip(conf*(0.7+0.3*ramp),0,1))
    else:
        scores = np.clip(np.random.normal(0.15,0.05,n_frames),0,0.4).astype(np.float32)
        scores[0] = 0.10
    return scores

def ffmpeg_ok():
    try:
        subprocess.run(["ffmpeg","-version"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
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
                           check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=120)
            return True
        except: pass
    try:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened(): return False
        fps2 = cap.get(cv2.CAP_PROP_FPS) or CFG.DEFAULT_FPS
        w2, h2 = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), float(fps2), (w2,h2))
        while True:
            ret, f = cap.read()
            if not ret: break
            out.write(f)
        cap.release(); out.release()
        return dst.exists()
    except: return False

def describe_onset(pred):
    onset_t = pred.get("onset_time","?"); onset_f = pred.get("onset_frame","?")
    if onset_f == "N/A" or onset_t == "N/A": return "No clear onset detected."
    return (f"Fight onset at frame {onset_f} ({onset_t}). "
            f"Threshold: {pred.get('onset_threshold','?')}, spike: {pred.get('spike_delta','?')}.")

def _safe_video(path):
    try:
        p = Path(path)
        if not p.exists(): st.warning("⚠️ Video file not found."); return
        with open(p,"rb") as f: data = f.read()
        if len(data)==0: st.warning("⚠️ Video file is empty."); return
        st.video(data)
    except Exception as e:
        st.warning(f"⚠️ Video preview unavailable. ({type(e).__name__})")


# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════
def get_plot_colors():
    theme = st.session_state.get("ui_theme","dark")
    if theme=="dark":
        return {"bg":"#080c10","ax":"#0a0f18","spine":"#1a2535","tick":"#445566",
                "legend_bg":"#0a0f18","legend_edge":"#1a2535","legend_text":"#7ecfff","xlabel":"#445566"}
    return {"bg":"#f0f4f8","ax":"#ffffff","spine":"#cdd5df","tick":"#556677",
            "legend_bg":"#ffffff","legend_edge":"#cdd5df","legend_text":"#1a6fc4","xlabel":"#556677"}

def make_timeline_plot(scores, fps, pred=None):
    c   = get_plot_colors()
    t   = np.arange(len(scores))/fps
    fig = plt.figure(figsize=(6,2.5), facecolor=c["bg"])
    ax  = fig.add_subplot(111); ax.set_facecolor(c["ax"])
    is_fight = is_fight_pred(pred) if pred else False
    color    = "#e05252" if is_fight else "#52e08a"
    ax.plot(t, scores, color=color, linewidth=1.6)
    ax.fill_between(t, scores, alpha=0.12, color=color)
    ax.axhline(st.session_state.get("thr_suspicious",CFG.THRESH_SUSPICIOUS),
               linestyle="--",color="#f5a623",linewidth=0.8)
    ax.axhline(st.session_state.get("thr_violence",CFG.THRESH_VIOLENCE),
               linestyle="--",color="#e05252",linewidth=0.8)
    if pred:
        try:
            onset_f   = int(pred.get("onset_frame",0))
            onset_tv  = onset_f/fps
            if 0 < onset_tv < t[-1]:
                ax.axvline(onset_tv,color="#7ecfff",linewidth=1.2,linestyle=":")
        except: pass
    ax.set_xlabel("Time (s)",fontsize=8,color=c["xlabel"])
    ax.set_ylabel("P(fight)",fontsize=8,color=c["xlabel"])
    ax.tick_params(colors=c["tick"],labelsize=7); ax.spines[:].set_color(c["spine"])
    plt.tight_layout(); return fig

def make_hist_plot(scores):
    c   = get_plot_colors()
    fig = plt.figure(figsize=(4,2.5), facecolor=c["bg"])
    ax  = fig.add_subplot(111); ax.set_facecolor(c["ax"])
    ax.hist(scores,bins=20,color="#5271e0",edgecolor=c["bg"],linewidth=0.3)
    ax.set_xlabel("Probability",fontsize=8,color=c["xlabel"])
    ax.set_ylabel("Count",fontsize=8,color=c["xlabel"])
    ax.tick_params(colors=c["tick"],labelsize=7); ax.spines[:].set_color(c["spine"])
    plt.tight_layout(); return fig

def make_confusion_matrix(records):
    c  = get_plot_colors(); labels=["Fight","NonFight"]; cm=np.zeros((2,2),dtype=int)
    lmap={"fight":0,"nonfight":1,"Fight":0,"NonFight":1,"Nonfight":1}
    for r in records:
        t2=lmap.get(r.get("true_label",""),-1); p2=0 if is_fight_pred(r) else 1
        if t2>=0 and p2>=0: cm[t2][p2]+=1
    fig,ax=plt.subplots(figsize=(4,3),facecolor=c["bg"]); ax.set_facecolor(c["ax"])
    ax.imshow(cm,interpolation="nearest",cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels,color=c["legend_text"],fontsize=9)
    ax.set_yticklabels(labels,color=c["legend_text"],fontsize=9)
    ax.set_xlabel("Predicted",color=c["xlabel"]); ax.set_ylabel("True",color=c["xlabel"])
    ax.tick_params(colors=c["tick"]); ax.spines[:].set_color(c["spine"])
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i][j]),ha="center",va="center",fontsize=14,fontweight="bold",
                    color="white" if cm[i][j]>cm.max()/2 else c["legend_text"])
    plt.tight_layout(); return fig, cm


# ══════════════════════════════════════════════════════════════
# PRED.TXT PARSER + CARD
# ══════════════════════════════════════════════════════════════
def parse_pred_txt(path):
    out={}
    try:
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k,v = line.strip().split(":",1)
                    out[k.strip()]=v.strip()
    except: pass
    return out

def render_pred_card(pred):
    if not pred: st.info("No pred.txt found."); return
    is_fight = is_fight_pred(pred)
    badge = '<span class="vg-badge-fight">⚠ FIGHT</span>' if is_fight \
            else '<span class="vg-badge-normal">✓ NORMAL</span>'
    st.markdown(badge, unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("True Label",  pred.get("true_label","?"))
    c2.metric("Predicted",   pred.get("pred_label","?"))
    ok_emoji = "✅" if str(pred.get("correct","")).lower()=="true" else "❌"
    c3.metric("Correct", f"{ok_emoji} {pred.get('correct','?')}")
    try:    c4.metric("Confidence", f"{float(pred.get('confidence','0')):.1%}")
    except: c4.metric("Confidence", pred.get("confidence","?"))
    c5,c6,c7,c8 = st.columns(4)
    c5.metric("Onset Frame",  pred.get("onset_frame","N/A"))
    c6.metric("Onset Time",   pred.get("onset_time","N/A"))
    c7.metric("Total Frames", pred.get("total_frames","?"))
    c8.metric("Dataset",      pred.get("dataset","?"))
    with st.expander("📋 Full pred.txt", expanded=False):
        col_a,col_b = st.columns(2)
        left_keys  = ["model_path","model_val_acc","probs","window_size","window_stride","onset_threshold","spike_delta"]
        right_keys = ["cam_methods","smooth_passes","heatmap","nonfight_reason"]
        for col, keys in [(col_a,left_keys),(col_b,right_keys)]:
            with col:
                for k in keys:
                    if k in pred:
                        st.markdown(f"<span style='color:#445566;font-size:11px;font-family:monospace'>{k}:</span> "
                                    f"<span style='font-size:11px;font-family:monospace'>{pred[k]}</span>",
                                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FOLDER HELPERS
# ══════════════════════════════════════════════════════════════
def class_root(ds, cls):   return UPLOAD_ROOT / ds / cls
def list_video_folders(ds, cls):
    root = class_root(ds, cls)
    if not root.exists(): return []
    f = [x for x in root.iterdir() if x.is_dir() and not x.name.startswith("_")]
    f.sort(key=lambda x: x.name.lower())
    return f

def find_file(folder, pattern):
    m = list(folder.glob(pattern))
    return m[0] if m else None

def get_files(folder):
    files = {}
    def real_files(pattern):
        return [f for f in folder.glob(pattern)
                if not f.name.startswith("._") and not f.name.startswith(".")]
    for vk in ALL_VID_KEYS:
        if vk=="original":
            cands = real_files("*original*.mp4")
            if not cands:
                cands = [f for f in real_files("*.mp4")
                         if not any(x in f.name.lower() for x in
                                    ["gradcam","layercam","combined","smooth","_preview"])]
            if cands: files["original"]=cands[0]
        elif vk=="smooth_gradcampp":
            cands = [f for f in real_files("*smooth_gradcampp*.mp4")+real_files("*smooth*.mp4")
                     if not f.name.startswith("_preview")]
            if cands: files["smooth_gradcampp"]=cands[0]
        elif vk=="gradcampp":
            cands = [f for f in real_files("*gradcampp*.mp4")+real_files("*gradcam++*.mp4")
                     if "smooth" not in f.name.lower() and not f.name.startswith("_preview")]
            if cands: files["gradcampp"]=cands[0]
        elif vk=="gradcam":
            cands = [f for f in real_files("*gradcam*.mp4")
                     if "pp" not in f.name.lower() and "++" not in f.name.lower()
                     and "smooth" not in f.name.lower() and not f.name.startswith("_preview")]
            if cands: files["gradcam"]=cands[0]
        elif vk=="layercam":
            cands = [f for f in real_files("*layercam*.mp4") if not f.name.startswith("_preview")]
            if cands: files["layercam"]=cands[0]
        elif vk=="combined":
            cands = [f for f in real_files("*combined*.mp4") if not f.name.startswith("_preview")]
            if cands: files["combined"]=cands[0]
    for gk in ALL_GRID_KEYS:
        f2 = find_file(folder, f"{gk}.png")
        if f2: files[gk]=f2
    f2 = find_file(folder,"timeline.png");  
    if f2: files["timeline"]=f2
    f2 = find_file(folder,"pred.txt");      
    if f2: files["pred"]=f2
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
    if UPLOAD_ROOT.exists(): shutil.rmtree(UPLOAD_ROOT)
    for ds in DATASETS:
        for cls in DATASETS[ds]: class_root(ds,cls).mkdir(parents=True,exist_ok=True)

def extract_zip_to_uploads(zip_bytes, dataset, cls):
    n_folders=0; n_files=0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names=zf.namelist(); folder_map={}
        for name in names:
            parts=Path(name).parts
            if len(parts)>=2: fk=parts[-2]; folder_map.setdefault(fk,[]).append(name)
            elif len(parts)==1 and not name.endswith("/"): folder_map.setdefault("misc",[]).append(name)
        for fn,flist in folder_map.items():
            dest=class_root(dataset,cls)/fn; dest.mkdir(parents=True,exist_ok=True); n_folders+=1
            for zp in flist:
                if zp.endswith("/"): continue
                try:
                    with open(dest/Path(zp).name,"wb") as f2: f2.write(zf.read(zp))
                    n_files+=1
                except: pass
    return n_folders, n_files


# ══════════════════════════════════════════════════════════════
# HISTORY HELPER
# ══════════════════════════════════════════════════════════════
def push_history(folder_name, dataset, cls, pred, active_files_dict):
    """Push current analysis into the last-5 history list."""
    entry = {
        "folder":   folder_name,
        "dataset":  dataset,
        "cls":      cls,
        "pred_lbl": pred.get("pred_label", "?"),
        "conf":     pred.get("confidence", "?"),
        "onset_t":  pred.get("onset_time", "N/A"),
        "ts":       datetime.now().strftime("%H:%M:%S"),
        "_files":   active_files_dict,   # {k: str(path)} snapshot
    }
    hist = st.session_state.get("_history", [])
    # Avoid duplicate consecutive entry
    if hist and hist[0].get("folder") == folder_name:
        return
    hist.insert(0, entry)
    st.session_state["_history"] = hist[:5]   # keep last 5 only


def restore_history(entry: dict):
    """Restore a history entry as the active session."""
    files     = {k: Path(v) for k, v in entry["_files"].items()}
    pred_data = parse_pred_txt(files["pred"]) if "pred" in files else {}
    frames, fps2 = [], float(CFG.DEFAULT_FPS)
    if "original" in files:
        try:
            frames, fps2 = read_video_frames(
                files["original"], max_frames=st.session_state.get("max_frames", CFG.MAX_FRAMES))
            frames = [resize_keep(f, 640) for f in frames]
        except: pass
    n      = len(frames) if frames else 100
    scores = scores_from_pred(pred_data, n, fps2)
    st.session_state.active_pred        = pred_data
    st.session_state.active_scores      = scores
    st.session_state.active_fps         = fps2
    st.session_state.active_frames      = frames
    st.session_state.active_folder_name = entry["folder"]
    st.session_state.active_video_path  = str(files.get("original", ""))
    st.session_state.active_dataset     = entry["dataset"]
    st.session_state.active_class       = entry["cls"]
    st.session_state["_active_files"]   = {k: str(v) for k, v in files.items()}


# ══════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════
def generate_pdf_report(pred, scores, fps, folder_name):
    fig = plt.figure(figsize=(11,8.5),facecolor="#080c10")
    gs  = gridspec.GridSpec(3,3,figure=fig,hspace=0.55,wspace=0.4)
    tax = fig.add_subplot(gs[0,:]); tax.axis("off"); tax.set_facecolor("#080c10")
    tax.text(0.5,0.75,"VisionGuard — Violence Detection Report",ha="center",va="center",
             fontsize=16,fontweight="bold",color="white")
    tax.text(0.5,0.3,f"Video: {folder_name}   |   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             ha="center",va="center",fontsize=10,color="#aaaaaa")
    atl = fig.add_subplot(gs[1,0:2]); atl.set_facecolor("#0a0f18")
    t   = np.arange(len(scores))/fps
    is_fight=is_fight_pred(pred)
    atl.plot(t,scores,color="#e05252" if is_fight else "#52e08a",linewidth=1.5)
    atl.axhline(CFG.THRESH_VIOLENCE,linestyle="--",color="red",linewidth=1)
    atl.set_xlabel("Time (s)",color="white",fontsize=8); atl.set_ylabel("P(fight)",color="white",fontsize=8)
    atl.tick_params(colors="white"); atl.spines[:].set_color("#333355")
    ah = fig.add_subplot(gs[1,2]); ah.set_facecolor("#0a0f18")
    ah.hist(scores,bins=15,color="#5271e0"); ah.tick_params(colors="white"); ah.spines[:].set_color("#333355")
    ai = fig.add_subplot(gs[2,:]); ai.axis("off")
    lines=[
        f"STATUS: {'FIGHT DETECTED' if is_fight else 'NO FIGHT'}   |   Confidence: {pred.get('confidence','?')}",
        f"Dataset: {pred.get('dataset','?')}   True: {pred.get('true_label','?')}   Predicted: {pred.get('pred_label','?')}   Correct: {pred.get('correct','?')}",
        f"Onset Frame: {pred.get('onset_frame','?')}   Onset Time: {pred.get('onset_time','?')}   Frames: {pred.get('total_frames','?')}",
        f"Model: {pred.get('model_path','?')}   Val Acc: {pred.get('model_val_acc','?')}",
    ]
    for i, line in enumerate(lines):
        ai.text(0.02,0.95-i*0.22,line,transform=ai.transAxes,fontsize=8,
                color="#ff6666" if i==0 and is_fight else ("white" if i>0 else "#66ff66"),
                fontweight="bold" if i==0 else "normal")
    buf=io.BytesIO(); plt.savefig(buf,format="pdf",facecolor=fig.get_facecolor(),bbox_inches="tight")
    plt.close(fig); buf.seek(0); return buf.read()


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
def init_state():
    defaults={
        "logged_in":False,"username":"",
        "active_pred":{},"active_scores":None,"active_fps":None,
        "active_frames":None,"active_folder_name":"",
        "active_video_path":None,"active_dataset":"","active_class":"",
        "run_id":datetime.now().strftime("run_%Y%m%d_%H%M%S"),
        "_confirm_clear":False,
        "nav_page":"🏠 Welcome",
        "thr_violence":CFG.THRESH_VIOLENCE,"thr_suspicious":CFG.THRESH_SUSPICIOUS,
        "max_frames":CFG.MAX_FRAMES,
        "ui_theme":"dark","accent_color":"#e05252","font_size":"medium",
        "compact_sidebar":False,"show_scanlines":True,"video_autoplay":False,
        "default_dataset":"hockeyfight","default_class":"Fight",
        "notify_fights":True,"show_confidence_bar":True,"chart_style":"line",
        # Processing state
        "_proc_running":   False,
        "_proc_progress":  {},
        "_proc_thread":    None,
        "_proc_out_dir":   "",
        "_proc_ds":        "",
        "_proc_cls":       "",
        "_proc_folder":    "",
        # Analysis history — last 5
        # each entry: {folder, dataset, cls, pred_lbl, conf, onset_t, ts, _active_files_snap}
        "_history":        [],
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

init_state()
for ds in DATASETS:
    for cls in DATASETS[ds]: class_root(ds,cls).mkdir(parents=True,exist_ok=True)

st.markdown(get_theme_css(
    theme     = st.session_state.get("ui_theme","dark"),
    accent    = st.session_state.get("accent_color","#e05252"),
    font_size = st.session_state.get("font_size","medium"),
), unsafe_allow_html=True)

theme  = st.session_state.get("ui_theme","dark")
accent = st.session_state.get("accent_color","#e05252")
bg2    = "#0d1520" if theme=="dark" else "#ffffff"
bg3    = "#0a0f18" if theme=="dark" else "#e8edf3"
bord   = "#1a2535" if theme=="dark" else "#cdd5df"
tblue  = "#7ecfff" if theme=="dark" else "#1a6fc4"
tdim   = "#445566" if theme=="dark" else "#556677"
tdim2  = "#2a3a4a" if theme=="dark" else "#778899"
thead  = "#e8f4ff" if theme=="dark" else "#0a1020"


# ══════════════════════════════════════════════════════════════
# LOGIN WALL
# ══════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""<style>.block-container{padding-top:0!important;}</style>
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:6vh 0 2vh 0;gap:6px;">
        <div style="font-size:48px;animation:pulse-shield 3s ease-in-out infinite;">🛡️</div>
        <div style="font-family:'Orbitron',sans-serif;font-size:2.4rem;font-weight:900;letter-spacing:4px;">VISIONGUARD</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:12px;color:#445566;letter-spacing:2px;margin-bottom:20px;">VIOLENCE DETECTION SYSTEM v7</div>
    </div>""", unsafe_allow_html=True)
    _, mid, _ = st.columns([1,1,1])
    with mid:
        auth_tabs = st.tabs(["🔐  Login","📝  Register"])
        with auth_tabs[0]:
            lu = st.text_input("Username", key="login_u", placeholder="operator id")
            lp = st.text_input("Password", type="password", key="login_p", placeholder="password")
            if st.button("AUTHENTICATE →", type="primary", use_container_width=True, key="login_btn"):
                if try_login(lu.strip(), lp):
                    st.session_state.logged_in=True; st.session_state.username=lu.strip(); st.rerun()
                else: st.error("❌ Authentication failed.")
        with auth_tabs[1]:
            ru=st.text_input("Username",key="reg_u",placeholder="choose username")
            rp=st.text_input("Password",type="password",key="reg_p",placeholder="min 4 characters")
            rp2=st.text_input("Confirm Password",type="password",key="reg_p2",placeholder="repeat password")
            if st.button("CREATE ACCOUNT →",type="primary",use_container_width=True,key="reg_btn"):
                if rp != rp2: st.error("❌ Passwords do not match.")
                else:
                    ok,msg=try_register(ru.strip(),rp)
                    if ok: st.success(f"✅ {msg}")
                    else:  st.error(f"❌ {msg}")
    st.stop()


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 16px 16px 16px;border-bottom:1px solid {bord};">
        <div style="font-family:'Orbitron',sans-serif;font-size:1.1rem;font-weight:900;letter-spacing:3px;">🛡️ VISIONGUARD</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:{tdim2};margin-top:2px;letter-spacing:1px;">Violence Detection v7</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding:12px 16px;border-bottom:1px solid {bord};">
        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:{tdim};">👤 operator</div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:14px;font-weight:700;color:{tblue};">{st.session_state.username}</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:{tdim2};margin-top:2px;">{st.session_state.run_id}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="padding:10px 16px 4px 16px;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:{tdim2};letter-spacing:2px;margin-bottom:6px;">NAVIGATION</div>
    </div>""", unsafe_allow_html=True)

    nav_pages = [
        "🏠 Welcome",
        "🎬 Process Raw Video",   # ← NEW
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
        "⚙️ Settings",
    ]
    for np_ in nav_pages:
        is_active = st.session_state.nav_page == np_
        # Show running indicator on Process page
        label = np_
        if np_ == "🎬 Process Raw Video" and st.session_state._proc_running:
            label = "🎬 Process Raw Video ⟳"
        if st.button(label, key=f"nav_{np_}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.nav_page = np_; st.rerun()

    st.markdown(f"<div style='border-top:1px solid {bord};margin:8px 0'></div>", unsafe_allow_html=True)
    st.session_state.thr_violence   = st.slider("Violence thr",0.10,0.99,st.session_state.thr_violence,0.01,key="sb_thr_v")
    st.session_state.thr_suspicious = st.slider("Suspicious thr",0.10,0.99,st.session_state.thr_suspicious,0.01,key="sb_thr_s")
    st.session_state.max_frames     = st.number_input("Max frames",30,500,st.session_state.max_frames,10,key="sb_maxf")

    st.markdown(f"<div style='border-top:1px solid {bord};margin:8px 0'></div>", unsafe_allow_html=True)

    # ── History panel ──────────────────────────────────────────
    hist = st.session_state.get("_history", [])
    st.markdown(f"""
    <div style="padding:6px 16px 2px 16px;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:{tdim2};letter-spacing:2px;margin-bottom:4px;">
            🕐 RECENT ANALYSES ({len(hist)}/5)
        </div>
    </div>""", unsafe_allow_html=True)

    if not hist:
        st.markdown(f"<div style='padding:0 16px 8px 16px;font-family:Share Tech Mono,monospace;font-size:10px;color:{tdim2};'>No analyses yet.</div>", unsafe_allow_html=True)
    else:
        for hi, entry in enumerate(hist):
            is_f_h2   = entry["pred_lbl"] == "Fight"
            hcolor    = accent if is_f_h2 else "#52e08a"
            is_active_h = (st.session_state.active_folder_name == entry["folder"])
            border_style = f"border:1px solid {hcolor};" if is_active_h else f"border:1px solid {bord};"
            st.markdown(f"""
            <div style="margin:0 8px 4px 8px;background:{bg2};{border_style}border-left:3px solid {hcolor};border-radius:4px;padding:5px 8px;">
                <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:{hcolor};letter-spacing:1px;">{'⚠ FIGHT' if is_f_h2 else '✓ NORMAL'} · {entry['conf']}</div>
                <div style="font-family:'Rajdhani',sans-serif;font-size:11px;font-weight:600;word-break:break-all;margin:1px 0;">{entry['folder'][:28]}{'…' if len(entry['folder'])>28 else ''}</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:{tdim2};">{entry['ts']} · {entry['dataset']}</div>
            </div>""", unsafe_allow_html=True)
            if st.button("↩ restore", key=f"hist_restore_{hi}", use_container_width=True):
                restore_history(entry)
                st.session_state.nav_page = "📁 Video Explorer"
                st.rerun()

    if st.button("🚪 LOGOUT", use_container_width=True, key="sb_logout"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()


# ══════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════
page = st.session_state.nav_page
if page != "🏠 Welcome":
    _hcol, _tcol = st.columns([0.06, 0.94])
    with _hcol:
        if st.button("🏠", key="home_btn", help="Go to Home"):
            st.session_state.nav_page="🏠 Welcome"; st.rerun()
    with _tcol:
        _page_label = page.split(" ",1)[1] if " " in page else page
        st.markdown(f"<div style='font-family:Rajdhani,sans-serif;font-weight:700;font-size:14px;color:{tdim};letter-spacing:1px;padding-top:6px;'>{page}</div>", unsafe_allow_html=True)

if st.session_state.active_folder_name and page != "🏠 Welcome":
    pred_h = st.session_state.active_pred
    is_f_h = is_fight_pred(pred_h)
    sc     = accent if is_f_h else "#52e08a"
    st.markdown(f"""
    <div style="background:{bg2};border:1px solid {bord};border-left:3px solid {sc};border-radius:4px;padding:8px 16px;margin:8px 0;display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <span style="font-family:'Share Tech Mono',monospace;font-size:10px;color:{tdim2};">ACTIVE</span>
        <span style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:13px;">{st.session_state.active_folder_name}</span>
        <span style="color:{bord};">|</span>
        <span style="font-family:'Share Tech Mono',monospace;font-size:10px;color:{tdim2};">DATASET</span>
        <span style="font-family:'Share Tech Mono',monospace;font-size:11px;color:{tblue};">{st.session_state.active_dataset}/{st.session_state.active_class}</span>
        <span style="color:{bord};">|</span>
        <span style="font-family:'Orbitron',sans-serif;font-size:11px;font-weight:700;color:{sc};">{'⚠ FIGHT' if is_f_h else '✓ NORMAL'}</span>
        <span style="color:{bord};">|</span>
        <span style="font-family:'Share Tech Mono',monospace;font-size:10px;color:{tdim2};">CONF</span>
        <span style="font-family:'Share Tech Mono',monospace;font-size:11px;">{pred_h.get('confidence','?')}</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: WELCOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Welcome":
    records_count = len(get_all_pred_records())
    username      = st.session_state.username
    hour          = datetime.now().hour
    if "login_time" not in st.session_state: st.session_state.login_time = datetime.now().strftime("%H:%M")
    greeting_word = "Good morning" if hour<12 else ("Good afternoon" if hour<18 else "Good evening")
    greeting_icon = "☀️" if hour<12 else ("🌤️" if hour<18 else "🌙")
    today         = datetime.now().strftime("%A, %d %B %Y")
    user_initials = username[:2].upper() if username else "OP"
    _acolors      = ["#e05252","#7ecfff","#52e08a","#f5a623","#a855f7","#ec4899","#06b6d4"]
    avatar_color  = _acolors[sum(ord(c) for c in username) % len(_acolors)]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{bg2},{bg3});border:1px solid {bord};border-top:3px solid {avatar_color};border-radius:16px;padding:28px 32px;display:flex;align-items:center;gap:22px;margin-bottom:18px;">
        <div style="width:64px;height:64px;border-radius:50%;background:linear-gradient(135deg,{avatar_color}cc,{avatar_color}44);border:2px solid {avatar_color};display:flex;align-items:center;justify-content:center;font-family:'Orbitron',sans-serif;font-size:18px;font-weight:900;color:white;">{user_initials}</div>
        <div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:{tdim};letter-spacing:2px;">{greeting_icon} {greeting_word.upper()}</div>
            <div style="font-family:'Orbitron',sans-serif;font-size:1.5rem;font-weight:900;letter-spacing:3px;">{username.upper()}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:{tdim};margin-top:4px;">{today} · {st.session_state.run_id}</div>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px;">
        <div style="background:{bg2};border:1px solid {bord};border-radius:10px;padding:16px 12px;text-align:center;">
            <div style="font-size:20px;margin-bottom:6px;">📂</div>
            <div style="font-family:'Orbitron',sans-serif;font-size:18px;font-weight:900;color:{tblue};">{records_count}</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:9px;color:{tdim2};text-transform:uppercase;letter-spacing:1.5px;">Videos</div>
        </div>
        <div style="background:{bg2};border:1px solid {bord};border-radius:10px;padding:16px 12px;text-align:center;">
            <div style="font-size:20px;margin-bottom:6px;">🔥</div>
            <div style="font-family:'Orbitron',sans-serif;font-size:18px;font-weight:900;color:{tblue};">4</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:9px;color:{tdim2};text-transform:uppercase;letter-spacing:1.5px;">CAM Methods</div>
        </div>
        <div style="background:{bg2};border:1px solid {bord};border-radius:10px;padding:16px 12px;text-align:center;">
            <div style="font-size:20px;margin-bottom:6px;">🎬</div>
            <div style="font-family:'Orbitron',sans-serif;font-size:18px;font-weight:900;color:#52e08a;">NEW</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:9px;color:{tdim2};text-transform:uppercase;letter-spacing:1.5px;">Raw Processing</div>
        </div>
        <div style="background:{bg2};border:1px solid {bord};border-radius:10px;padding:16px 12px;text-align:center;">
            <div style="font-size:20px;margin-bottom:6px;">🧠</div>
            <div style="font-family:'Orbitron',sans-serif;font-size:12px;font-weight:900;color:#f5a623;">R3D-18</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:9px;color:{tdim2};text-transform:uppercase;letter-spacing:1.5px;">Backbone</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:9px;color:{tdim2};letter-spacing:3px;margin:6px 0 10px 0;'>── QUICK LAUNCH ──</div>", unsafe_allow_html=True)
    _ql = st.columns(6, gap="small")
    for _c, (_lbl, _tgt) in zip(_ql, [
        ("🎬 Process","🎬 Process Raw Video"),("📁 Explorer","📁 Video Explorer"),
        ("📤 Upload","📤 Upload Manager"),("🔥 GradCAM","🔥 GradCAM Viewer"),
        ("📊 Stats","📊 Dataset Stats"),("⚙️ Settings","⚙️ Settings"),
    ]):
        with _c:
            if st.button(_lbl, use_container_width=True, key=f"ql_{_tgt}"):
                st.session_state.nav_page=_tgt; st.rerun()


# ══════════════════════════════════════════════════════════════
# PAGE: PROCESS RAW VIDEO  ← NEW
# ══════════════════════════════════════════════════════════════
elif page == "🎬 Process Raw Video":
    st.markdown(f"""
    <div style="font-family:'Orbitron',sans-serif;font-size:1rem;font-weight:900;letter-spacing:2px;margin-bottom:4px;">🎬 PROCESS RAW VIDEO</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:{tdim2};letter-spacing:2px;margin-bottom:18px;">
        Upload any raw video → full GradCAM pipeline → results appear in all viewer pages
    </div>""", unsafe_allow_html=True)

    proc = st.session_state._proc_progress

    # ── If processing finished, show result and offer to load / download ──
    if proc.get("done") and not st.session_state._proc_running:
        if proc.get("error"):
            st.error(f"❌ Processing failed:\n\n```\n{proc['error']}\n```")
            if st.button("🔄 Try Again", key="proc_retry"):
                st.session_state._proc_progress = {}
                st.rerun()
        else:
            # ── Auto-load into session state immediately ──────
            out_dir     = Path(proc.get("out_dir",""))
            folder_name = st.session_state._proc_folder
            ds_name     = st.session_state._proc_ds
            cls_name    = st.session_state._proc_cls

            if out_dir.exists() and not st.session_state.get("_proc_loaded"):
                files_loaded    = get_files(out_dir)
                pred_data       = parse_pred_txt(files_loaded["pred"]) if "pred" in files_loaded else {}
                frames_l, fps_l = [], float(CFG.DEFAULT_FPS)
                if "original" in files_loaded:
                    try:
                        frames_l, fps_l = read_video_frames(
                            files_loaded["original"],
                            max_frames=st.session_state.max_frames)
                        frames_l = [resize_keep(f, 640) for f in frames_l]
                    except: pass
                n_l      = len(frames_l) if frames_l else 100
                scores_l = scores_from_pred(pred_data, n_l, fps_l)
                active_files_snap = {k: str(v) for k, v in files_loaded.items()}

                st.session_state.active_pred        = pred_data
                st.session_state.active_scores      = scores_l
                st.session_state.active_fps         = fps_l
                st.session_state.active_frames      = frames_l
                st.session_state.active_folder_name = folder_name
                st.session_state.active_video_path  = str(files_loaded.get("original",""))
                st.session_state.active_dataset     = ds_name
                st.session_state.active_class       = cls_name
                st.session_state["_active_files"]   = active_files_snap
                push_history(folder_name, ds_name, cls_name, pred_data, active_files_snap)
                st.session_state["_proc_loaded"]    = True   # prevent re-loading on rerun

            pred_lbl = proc.get("pred_lbl","?")
            conf     = proc.get("conf", 0)
            onset    = proc.get("onset")
            is_fight = pred_lbl == "Fight"
            color    = accent if is_fight else "#52e08a"

            st.markdown(f"""
            <div style="background:{color}18;border:1px solid {color};border-left:4px solid {color};border-radius:8px;padding:18px 24px;margin:12px 0;">
                <div style="font-family:'Orbitron',sans-serif;font-size:16px;font-weight:900;color:{color};letter-spacing:3px;margin-bottom:8px;">
                    {'🚨 FIGHT DETECTED' if is_fight else '✅ NO VIOLENCE DETECTED'}
                </div>
                <div style="display:flex;gap:28px;font-family:'Share Tech Mono',monospace;font-size:12px;flex-wrap:wrap;">
                    <span>PREDICTION <span style="color:{color};font-weight:bold;">{pred_lbl}</span></span>
                    <span>CONFIDENCE <span style="color:{color};font-weight:bold;">{conf:.1%}</span></span>
                    <span>ONSET FRAME <span style="color:{color};font-weight:bold;">{onset if onset is not None else 'N/A'}</span></span>
                </div>
                <div style="margin-top:10px;font-family:'Share Tech Mono',monospace;font-size:10px;color:{color}88;">
                    ✅ Results auto-loaded — use the sidebar to navigate to Explorer, Grid Viewer, GradCAM Viewer
                </div>
            </div>""", unsafe_allow_html=True)

            # ── Navigation shortcuts ──
            st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:{tdim2};letter-spacing:2px;margin:12px 0 8px 0;'>── OPEN IN ──</div>", unsafe_allow_html=True)
            nc1, nc2, nc3, nc4 = st.columns(4)
            with nc1:
                if st.button("📁 Video Explorer", type="primary", use_container_width=True, key="proc_go_ex"):
                    st.session_state.nav_page = "📁 Video Explorer"; st.rerun()
            with nc2:
                if st.button("🔥 GradCAM Viewer", use_container_width=True, key="proc_go_cam"):
                    st.session_state.nav_page = "🔥 GradCAM Viewer"; st.rerun()
            with nc3:
                if st.button("🖼️ Grid Viewer", use_container_width=True, key="proc_go_grid"):
                    st.session_state.nav_page = "🖼️ Grid Viewer"; st.rerun()
            with nc4:
                if st.button("📊 Timeline", use_container_width=True, key="proc_go_tl"):
                    st.session_state.nav_page = "📊 Timeline"; st.rerun()

            # ── ZIP download ──
            st.markdown("---")
            if out_dir.exists():
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in out_dir.iterdir():
                        if f.is_file(): zf.write(f, arcname=f.name)
                zip_buf.seek(0)
                dl_col, new_col = st.columns(2)
                with dl_col:
                    st.download_button("⬇️ DOWNLOAD ZIP", data=zip_buf,
                                       file_name=f"{out_dir.name}_outputs.zip",
                                       mime="application/zip", use_container_width=True,
                                       key="proc_zip_dl")
                with new_col:
                    if st.button("🔄 PROCESS ANOTHER", use_container_width=True, key="proc_new"):
                        st.session_state._proc_progress = {}
                        st.session_state._proc_running  = False
                        st.session_state["_proc_loaded"] = False
                        st.rerun()

    # ── If currently processing — show live progress ──
    elif st.session_state._proc_running:
        pct   = proc.get("pct", 0.0)
        stage = proc.get("stage", "Processing...")

        st.markdown(f"""
        <div style="background:{bg2};border:1px solid {bord};border-radius:8px;padding:24px;margin:12px 0;">
            <div style="font-family:'Orbitron',sans-serif;font-size:12px;font-weight:900;color:{accent};letter-spacing:3px;margin-bottom:16px;">
                ⟳ PROCESSING IN PROGRESS
            </div>
            <div class="proc-stage proc-active">{stage}</div>
        </div>""", unsafe_allow_html=True)

        st.progress(min(pct, 1.0))
        st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:11px;color:{tdim};'>{int(pct*100)}% complete</div>", unsafe_allow_html=True)

        # Check if thread finished
        thread = st.session_state._proc_thread
        if thread is not None and not thread.is_alive():
            st.session_state._proc_running = False

        # Auto-refresh every 1.5 s while running
        time.sleep(1.5)
        st.rerun()

    # ── Upload form (idle state) ──
    else:
        st.markdown(f"""
        <div style="background:{bg2};border:1px solid {bord};border-left:4px solid {accent};border-radius:8px;padding:16px 20px;margin-bottom:18px;font-family:'Share Tech Mono',monospace;font-size:11px;color:{tdim};">
            <b style="color:{tblue};">What happens when you click RUN:</b><br><br>
            1️⃣ &nbsp;Read all frames from your video<br>
            2️⃣ &nbsp;Sliding-window predictions → P(fight) per frame<br>
            3️⃣ &nbsp;Fight onset detection (LSTM spike analysis)<br>
            4️⃣ &nbsp;GradCAM · GradCAM++ · SmoothGradCAM++ · LayerCAM + Combined<br>
            5️⃣ &nbsp;Render 6 annotated videos with info bars<br>
            6️⃣ &nbsp;Save 6 frame grids + timeline plot + pred.txt<br>
            7️⃣ &nbsp;Auto-load results into GradCAM Viewer / Grid Viewer / Timeline
        </div>""", unsafe_allow_html=True)

        up_col, cfg_col = st.columns([1.2, 1], gap="large")

        with up_col:
            st.markdown("#### 📤 Upload Raw Video")
            raw_file = st.file_uploader(
                "Drop your .mp4 or .avi here",
                type=["mp4","avi","mov","mkv"],
                key="proc_raw_upload",
                help="Any raw video — no preprocessing needed",
            )
            if raw_file:
                st.success(f"✅ `{raw_file.name}` ready ({raw_file.size//1024} KB)")

        with cfg_col:
            st.markdown("#### ⚙️ Processing Config")
            sel_config = st.selectbox(
                "Dataset / Model",
                list(PROC_CONFIGS.keys()),
                key="proc_cfg_sel",
                help="HockeyFight uses window=16, RWF uses window=32",
            )
            cfg_info = PROC_CONFIGS[sel_config]
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:{tdim};line-height:2;">
                window={cfg_info['window_size']} · stride={cfg_info['window_stride']}<br>
                pred_thresh={cfg_info['pred_thresh']} · onset_thresh={cfg_info['onset_thresh']}<br>
                ckpt: <span style="color:{tblue};">{cfg_info['ckpt']}</span>
            </div>""", unsafe_allow_html=True)

            true_label_inp = st.selectbox(
                "True Label (optional — for pred.txt accuracy field)",
                ["Unknown","Fight","Nonfight"],
                key="proc_true_label",
            )
            cls_for_folder = "Fight" if true_label_inp in ["Fight"] else "Nonfight"

        st.markdown("---")

        ckpt_exists = Path(cfg_info["ckpt"]).exists()
        if not ckpt_exists:
            st.warning(f"⚠️ Checkpoint not found: `{cfg_info['ckpt']}`. Make sure checkpoints are in place before running.")

        run_disabled = (raw_file is None) or (not ckpt_exists)

        if st.button("▶ RUN FULL PIPELINE", type="primary",
                     disabled=run_disabled, key="proc_run_btn",
                     use_container_width=False):

            # Save uploaded file to disk
            stem       = _safe_name(Path(raw_file.name).stem)
            ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name= f"{stem}_{ts}"
            ds_name    = cfg_info["name"]
            out_dir    = class_root(ds_name, cls_for_folder) / folder_name
            out_dir.mkdir(parents=True, exist_ok=True)

            tmp_vid = out_dir / raw_file.name
            tmp_vid.write_bytes(raw_file.getbuffer())

            # Init progress dict
            progress_dict = {
                "pct": 0.0, "stage": "Initializing...",
                "done": False, "error": None,
                "pred_lbl": "?", "conf": 0.0, "onset": None,
                "out_dir": str(out_dir),
            }

            # Launch background thread
            t = threading.Thread(
                target=run_processing_pipeline,
                args=(tmp_vid, cfg_info, true_label_inp, out_dir, progress_dict),
                daemon=True,
            )
            t.start()

            st.session_state._proc_running  = True
            st.session_state._proc_progress = progress_dict
            st.session_state._proc_thread   = t
            st.session_state._proc_out_dir  = str(out_dir)
            st.session_state._proc_ds       = ds_name
            st.session_state._proc_cls      = cls_for_folder
            st.session_state._proc_folder   = folder_name
            st.session_state["_proc_loaded"] = False
            st.rerun()


# ══════════════════════════════════════════════════════════════
# PAGE: VIDEO EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📁 Video Explorer":
    st.subheader("📁 Video Explorer")
    s1,s2,s3 = st.columns([1,1,2])
    with s1: sel_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="ex_ds")
    with s2: sel_cls = st.selectbox("Class",   DATASETS[sel_ds],      key="ex_cls")
    with s3:
        vfolders = list_video_folders(sel_ds, sel_cls)
        if not vfolders:
            st.info(f"No folders for **{sel_ds}/{sel_cls}**. Use Upload Manager or Process Raw Video.")
            sel_folder = None
        else:
            sel_folder = st.selectbox(f"Folder ({len(vfolders)} available)",
                                      [f.name for f in vfolders], key="ex_folder")

    if st.button("🔍 ANALYZE FOLDER", type="primary", disabled=(sel_folder is None), key="analyze_btn"):
        folder_path = class_root(sel_ds, sel_cls) / sel_folder
        status_ph = st.empty(); bar_ph = st.empty()
        def _show_step(pct, msg, color=tblue):
            status_ph.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:12px;color:{color};letter-spacing:2px;padding:4px 0;'>{msg}</div>", unsafe_allow_html=True)
            bar_ph.progress(pct)
        _show_step(0.05,"⟳  INITIALIZING..."); time.sleep(0.1)
        _show_step(0.20,"📂  READING FOLDER...")
        files = get_files(folder_path); time.sleep(0.1)
        _show_step(0.40,"📋  PARSING PRED.TXT...")
        pred = parse_pred_txt(files["pred"]) if "pred" in files else {}; time.sleep(0.1)
        _show_step(0.60,"🎬  LOADING FRAMES...")
        frames, fps2 = [], float(CFG.DEFAULT_FPS)
        if "original" in files:
            try:
                frames, fps2 = read_video_frames(files["original"], max_frames=st.session_state.max_frames)
                frames = [resize_keep(f,640) for f in frames]
            except: pass
        _show_step(0.80,"📊  COMPUTING SCORES...")
        n      = len(frames) if frames else 100
        scores = scores_from_pred(pred, n, fps2); time.sleep(0.05)
        _show_step(1.0,"✅  DONE", color="#52e08a"); time.sleep(0.3)
        status_ph.empty(); bar_ph.empty()
        st.session_state.active_pred        = pred
        st.session_state.active_scores      = scores
        st.session_state.active_fps         = fps2
        st.session_state.active_frames      = frames
        st.session_state.active_folder_name = sel_folder
        st.session_state.active_video_path  = str(files.get("original",""))
        st.session_state.active_dataset     = sel_ds
        st.session_state.active_class       = sel_cls
        st.session_state["_active_files"]   = {k: str(v) for k,v in files.items()}
        push_history(sel_folder, sel_ds, sel_cls, pred, st.session_state["_active_files"])
        st.rerun()

    if st.session_state.active_folder_name:
        pred        = st.session_state.active_pred
        scores      = st.session_state.active_scores
        fps2        = st.session_state.active_fps
        folder_name = st.session_state.active_folder_name
        files       = {k: Path(v) for k,v in st.session_state.get("_active_files",{}).items()}
        is_fight    = is_fight_pred(pred)
        conf_val    = pred.get("confidence","?")
        onset_f     = pred.get("onset_frame","N/A")
        onset_t     = pred.get("onset_time","N/A")
        color_banner = accent if is_fight else "#52e08a"

        st.markdown(f"""
        <div style="background:{color_banner}10;border:1px solid {color_banner}44;border-left:4px solid {color_banner};border-radius:4px;padding:12px 20px;margin:12px 0;">
            <div style="font-family:'Orbitron',sans-serif;font-size:14px;font-weight:900;color:{color_banner};letter-spacing:2px;">
                {'🚨 VIOLENCE DETECTED' if is_fight else '✓ NO VIOLENCE DETECTED'}
            </div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:11px;margin-top:5px;display:flex;gap:20px;flex-wrap:wrap;">
                <span>ONSET {onset_f} ({onset_t})</span>
                <span>CONF {conf_val}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        render_pred_card(pred)
        st.markdown("---")

        vid_l, vid_r = st.columns(2, gap="large")
        with vid_l:
            st.markdown(f"<span style='font-family:Share Tech Mono,monospace;font-size:10px;color:{tblue};'>📹 ORIGINAL</span>", unsafe_allow_html=True)
            if "original" in files:
                fp = files["original"].parent/"_preview_original.mp4"
                if not (fp.exists() and fp.stat().st_size>5000):
                    with st.spinner("Converting..."): make_web_preview(files["original"], fp)
                _safe_video(fp if (fp.exists() and fp.stat().st_size>5000) else files["original"])
            else: st.info("No original video found.")
        with vid_r:
            rvk = next((k for k in ["gradcampp","combined","gradcam"] if k in files), None)
            rlbl = {"gradcampp":"🔥 GRADCAM++","combined":"🎯 COMBINED","gradcam":"🔥 GRADCAM"}.get(rvk,"CAM")
            st.markdown(f"<span style='font-family:Share Tech Mono,monospace;font-size:10px;color:{accent};'>{rlbl}</span>", unsafe_allow_html=True)
            if rvk:
                fp = files[rvk].parent/f"_preview_{rvk}.mp4"
                if not (fp.exists() and fp.stat().st_size>5000):
                    with st.spinner("Converting..."): make_web_preview(files[rvk], fp)
                _safe_video(fp if (fp.exists() and fp.stat().st_size>5000) else files[rvk])
            else: st.info("No CAM video found.")

        if scores is not None and fps2:
            st.markdown("---")
            m1,m2,m3,m4 = st.columns(4)
            stat=pred_label_to_status(pred.get("pred_label",""))
            m1.metric("Status", f"{color_from_status(stat)} {stat}")
            m2.metric("Confidence", conf_val)
            m3.metric("Onset Time", onset_t)
            m4.metric("Total Frames", pred.get("total_frames","?"))
            c1,c2 = st.columns(2)
            with c1: st.pyplot(make_timeline_plot(scores, fps2, pred), clear_figure=True)
            with c2: st.pyplot(make_hist_plot(scores), clear_figure=True)

        st.markdown("---")
        if st.button("📄 GENERATE PDF", key="ex_pdf_btn"):
            with st.spinner("Generating..."):
                pdf = generate_pdf_report(pred, scores or np.zeros(10), fps2 or 25.0, folder_name)
            st.download_button("⬇️ Download PDF", data=pdf,
                               file_name=f"report_{folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                               mime="application/pdf", key="ex_pdf_dl")
    else:
        st.info(f"📂 Select a dataset, class and folder above, then click **Analyze Folder**.")


# ══════════════════════════════════════════════════════════════
# PAGE: GRADCAM VIEWER
# ══════════════════════════════════════════════════════════════
elif page == "🔥 GradCAM Viewer":
    st.subheader("🔥 GradCAM Viewer — All 6 Methods")
    if not st.session_state.active_folder_name:
        st.info("Analyze a folder in **Video Explorer** first, or run **Process Raw Video**.")
    else:
        files = {k: Path(v) for k,v in st.session_state.get("_active_files",{}).items()}
        pred  = st.session_state.active_pred
        if is_fight_pred(pred):
            st.markdown(f"""
            <div style="background:rgba(224,82,82,0.10);border:1px solid {accent};border-left:4px solid {accent};border-radius:4px;padding:12px 20px;margin:10px 0;">
                <div style="font-family:'Orbitron',sans-serif;font-size:14px;font-weight:900;color:#ff7070;letter-spacing:3px;">🚨 FIGHT DETECTED — REVIEWING CAM OVERLAYS</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:12px;color:#cc6666;margin-top:4px;">Confidence: {pred.get('confidence','?')} | Onset: {pred.get('onset_time','?')}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("##### Row 1 — Original · GradCAM · GradCAM++")
        r1 = st.columns(3)
        for i, vk in enumerate(["original","gradcam","gradcampp"]):
            with r1[i]:
                st.markdown(f"**{VID_LABELS.get(vk,vk)}**")
                if vk in files:
                    fp = files[vk].parent/f"_preview_{vk}.mp4"
                    if not (fp.exists() and fp.stat().st_size>5000):
                        with st.spinner(f"Converting {vk}..."): make_web_preview(files[vk], fp)
                    _safe_video(fp if fp.exists() and fp.stat().st_size>5000 else files[vk])
                else: st.info(f"No {vk} video.")

        st.markdown("---")
        st.markdown("##### Row 2 — Smooth GradCAM++ · LayerCAM · Combined")
        r2 = st.columns(3)
        for i, vk in enumerate(["smooth_gradcampp","layercam","combined"]):
            with r2[i]:
                st.markdown(f"**{VID_LABELS.get(vk,vk)}**")
                if vk in files:
                    fp = files[vk].parent/f"_preview_{vk}.mp4"
                    if not (fp.exists() and fp.stat().st_size>5000):
                        with st.spinner(f"Converting {vk}..."): make_web_preview(files[vk], fp)
                    _safe_video(fp if fp.exists() and fp.stat().st_size>5000 else files[vk])
                else: st.info(f"No {vk} video.")

        st.markdown("---")
        render_pred_card(pred)


# ══════════════════════════════════════════════════════════════
# PAGE: GRID VIEWER
# ══════════════════════════════════════════════════════════════
elif page == "🖼️ Grid Viewer":
    st.subheader("🖼️ Frame Grid Viewer — All 6 Methods")
    if not st.session_state.active_folder_name:
        st.info("Analyze a folder in **Video Explorer** first, or run **Process Raw Video**.")
    else:
        files = {k: Path(v) for k,v in st.session_state.get("_active_files",{}).items()}
        st.markdown("##### Row 1 — Raw · GradCAM · GradCAM++")
        g1 = st.columns(3)
        for i, gk in enumerate(["raw_grid","gradcam_grid","gradcampp_grid"]):
            with g1[i]:
                st.markdown(f"**{GRID_LABELS.get(gk,gk)}**")
                if gk in files: st.image(str(files[gk]), use_container_width=True)
                else: st.info(f"No {gk}.png")
        st.markdown("---")
        st.markdown("##### Row 2 — Smooth GradCAM++ · LayerCAM · Combined")
        g2 = st.columns(3)
        for i, gk in enumerate(["smooth_gradcampp_grid","layercam_grid","combined_grid"]):
            with g2[i]:
                st.markdown(f"**{GRID_LABELS.get(gk,gk)}**")
                if gk in files: st.image(str(files[gk]), use_container_width=True)
                else: st.info(f"No {gk}.png")
        if "timeline" in files:
            st.markdown("---")
            st.markdown("##### 📈 P(fight) Timeline")
            st.image(str(files["timeline"]), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: TIMELINE
# ══════════════════════════════════════════════════════════════
elif page == "📊 Timeline":
    st.subheader("📊 P(fight) Timeline")
    if not st.session_state.active_folder_name:
        st.info("Analyze a folder in **Video Explorer** first, or run **Process Raw Video**.")
    else:
        files  = {k: Path(v) for k,v in st.session_state.get("_active_files",{}).items()}
        pred   = st.session_state.active_pred
        scores = st.session_state.active_scores
        fps2   = st.session_state.active_fps
        if "timeline" in files:
            st.image(str(files["timeline"]), use_container_width=True,
                     caption="Generated by pipeline — exact model scores")
            st.markdown("---")
        if scores is not None and fps2:
            st.pyplot(make_timeline_plot(scores, fps2, pred), clear_figure=True)
            st.pyplot(make_hist_plot(scores), clear_figure=True)
            step = max(1, int(0.5*fps2))
            rows = [{"time":fmt_time(i/fps2),"frame":i,"prob":round(float(scores[i]),4)}
                    for i in range(0, len(scores), step)]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=220)


# ══════════════════════════════════════════════════════════════
# PAGE: SEARCH & FILTER
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Search & Filter":
    st.subheader("🔍 Search & Filter")
    records = get_all_pred_records()
    if not records:
        st.info("No pred.txt files found.")
    else:
        df_all = pd.DataFrame(records)
        fc1,fc2,fc3,fc4 = st.columns(4)
        with fc1: sq   = st.text_input("Search name", placeholder="e.g. fi1", key="sf_search")
        with fc2: dsf  = st.selectbox("Dataset", ["All"]+list(DATASETS.keys()), key="sf_ds")
        with fc3: clsf = st.selectbox("Class", ["All","Fight","NonFight","Nonfight","pred_nonfight"], key="sf_cls")
        with fc4: corf = st.selectbox("Correct?", ["All","True","False"], key="sf_cor")
        df = df_all.copy()
        if sq:           df=df[df["_folder"].str.contains(sq,case=False,na=False)]
        if dsf  !="All": df=df[df["_dataset"]==dsf]
        if clsf !="All": df=df[df["_class"]==clsf]
        if corf !="All": df=df[df["correct"].str.lower()==corf.lower()]
        st.markdown(f"**{len(df)} results**")
        show=["_folder","_dataset","_class","true_label","pred_label","correct","confidence","onset_time"]
        show=[c for c in show if c in df.columns]
        st.dataframe(df[show].rename(columns={"_folder":"Folder","_dataset":"Dataset","_class":"Class"}),
                     use_container_width=True, height=380)
        st.download_button("⬇️ Download CSV", data=df[show].to_csv(index=False).encode(),
                           file_name="filtered.csv", mime="text/csv", key="sf_csv")


# ══════════════════════════════════════════════════════════════
# PAGE: COMPARE
# ══════════════════════════════════════════════════════════════
elif page == "⚖️ Compare":
    st.subheader("⚖️ Side-by-Side Comparison")
    def folder_sel(prefix, col):
        with col:
            ds2  = st.selectbox("Dataset", list(DATASETS.keys()), key=f"{prefix}_ds")
            cls2 = st.selectbox("Class",   DATASETS[ds2],         key=f"{prefix}_cls")
            vf2  = list_video_folders(ds2, cls2)
            if not vf2: st.info("No folders."); return None,None,None
            fn2  = st.selectbox(f"Folder ({len(vf2)})",[f.name for f in vf2],key=f"{prefix}_fn")
            return ds2, cls2, fn2
    lc,rc = st.columns(2, gap="large")
    l_ds,l_cls,l_fn = folder_sel("cmp_l", lc)
    r_ds,r_cls,r_fn = folder_sel("cmp_r", rc)
    if st.button("⚖️ COMPARE", type="primary", disabled=(not l_fn or not r_fn), key="cmp_btn"):
        for col, ds2, cls2, fn2, side in [(lc,l_ds,l_cls,l_fn,"LEFT"),(rc,r_ds,r_cls,r_fn,"RIGHT")]:
            fp2  = class_root(ds2,cls2)/fn2; fls2=get_files(fp2)
            pred2= parse_pred_txt(fls2["pred"]) if "pred" in fls2 else {}
            with col:
                st.markdown(f"### {side}: `{fn2}`")
                is_f2 = is_fight_pred(pred2)
                if is_f2: st.error(f"🔴 FIGHT — {pred2.get('confidence','?')} conf")
                else:     st.success(f"🟢 NORMAL — {pred2.get('confidence','?')} conf")
                ok2 = "✅" if str(pred2.get("correct","")).lower()=="true" else "❌"
                st.markdown(f"**True:** {pred2.get('true_label','?')} | **Pred:** {pred2.get('pred_label','?')} | **Correct:** {ok2}")
                if "original" in fls2:
                    pp2=fp2/"_preview_original.mp4"
                    if not pp2.exists(): make_web_preview(fls2["original"],pp2)
                    _safe_video(pp2 if pp2.exists() else fls2["original"])
                for gk2 in ["combined_grid","gradcam_grid"]:
                    if gk2 in fls2:
                        st.image(str(fls2[gk2]),use_container_width=True,caption=GRID_LABELS.get(gk2,gk2))
                        break
                if "timeline" in fls2:
                    st.image(str(fls2["timeline"]),use_container_width=True,caption="Timeline")


# ══════════════════════════════════════════════════════════════
# PAGE: UPLOAD MANAGER
# ══════════════════════════════════════════════════════════════
elif page == "📤 Upload Manager":
    st.subheader("📤 Upload Manager")

    with st.expander("🗑️ Clear All Uploads", expanded=False):
        st.warning("⚠️ This will permanently delete ALL uploaded data.")
        if not st.session_state._confirm_clear:
            if st.button("🗑️ CLEAR ALL", key="clear_btn"):
                st.session_state._confirm_clear=True; st.rerun()
        else:
            st.error("Are you sure? Cannot be undone.")
            cy,cn = st.columns(2)
            with cy:
                if st.button("✅ YES DELETE", type="primary", key="confirm_yes"):
                    clear_all_uploads()
                    for k in ["active_pred","active_scores","active_fps","active_frames",
                              "active_folder_name","active_video_path","_active_files"]:
                        st.session_state[k]={} if "pred" in k or "files" in k else None
                    st.session_state.active_folder_name=""
                    st.session_state._confirm_clear=False
                    st.success("✅ Cleared!"); st.rerun()
            with cn:
                if st.button("❌ Cancel", key="confirm_no"):
                    st.session_state._confirm_clear=False; st.rerun()

    st.divider()

    # ── 3 tabs: pre-computed outputs | raw single video | raw dataset ──
    up_tab1, up_tab2, up_tab3 = st.tabs([
        "📁 Pre-computed Outputs",
        "🎬 Raw Single Video",
        "📦 Raw Dataset",
    ])

    # ── TAB 1: Pre-computed outputs (original behaviour) ──────
    with up_tab1:
        upload_mode = st.radio("Mode", ["📁 Single folder","🗜️ ZIP file"],
                               horizontal=True, key="up_mode")
        if upload_mode == "📁 Single folder":
            uc1,uc2,uc3 = st.columns(3)
            with uc1: up_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="up_ds")
            with uc2: up_cls = st.selectbox("Class", DATASETS[up_ds], key="up_cls")
            with uc3: fn_inp = st.text_input("Folder name", placeholder="e.g. fi1_xvid", key="up_fn_inp")
            up_files = st.file_uploader(
                "Files (mp4 + png + pred.txt — all outputs per video)",
                type=["mp4","avi","mov","mkv","png","jpg","txt"],
                accept_multiple_files=True, key="up_files")
            if st.button("💾 SAVE", type="primary",
                         disabled=(not up_files or not fn_inp.strip()), key="up_save"):
                dest = class_root(up_ds, up_cls) / fn_inp.strip()
                dest.mkdir(parents=True, exist_ok=True)
                for uf in up_files:
                    with open(dest/uf.name,"wb") as f2: f2.write(uf.getbuffer())
                st.success(f"✅ Saved {len(up_files)} file(s) → `{up_ds}/{up_cls}/{fn_inp.strip()}`")
        else:
            uc1,uc2 = st.columns(2)
            with uc1: zip_ds  = st.selectbox("Dataset", list(DATASETS.keys()), key="zip_ds")
            with uc2: zip_cls = st.selectbox("Class", DATASETS[zip_ds], key="zip_cls")
            zf2 = st.file_uploader("Upload ZIP of pre-computed folder", type=["zip"], key="zip_up")
            if st.button("📦 EXTRACT", type="primary", disabled=(not zf2), key="zip_extract"):
                with st.spinner("Extracting..."):
                    n_f2, n_files2 = extract_zip_to_uploads(zf2.read(), zip_ds, zip_cls)
                st.success(f"✅ Extracted {n_f2} folder(s), {n_files2} file(s)")

    # ── TAB 2: Raw single video (redirect to Process page) ────
    with up_tab2:
        st.markdown(f"""
        <div style="background:{bg2};border:1px solid {bord};border-left:4px solid {accent};border-radius:8px;padding:16px 20px;margin-bottom:14px;font-family:'Share Tech Mono',monospace;font-size:11px;color:{tdim};">
            Upload a single raw video and run the full GradCAM pipeline on it.<br>
            Results auto-load into all viewer pages.
        </div>""", unsafe_allow_html=True)
        if st.button("🎬 GO TO PROCESS RAW VIDEO", type="primary", key="up_goto_proc"):
            st.session_state.nav_page = "🎬 Process Raw Video"; st.rerun()

    # ── TAB 3: Raw Dataset ─────────────────────────────────────
    with up_tab3:
        st.markdown(f"""
        <div style="background:{bg2};border:1px solid {bord};border-left:4px solid {tblue};border-radius:8px;padding:14px 18px;margin-bottom:16px;font-family:'Share Tech Mono',monospace;font-size:11px;color:{tdim};">
            Upload a folder of raw videos or a ZIP of raw videos.<br>
            Each video will be processed in sequence — full GradCAM pipeline per video.<br>
            <b style="color:{tblue};">All results are saved and appear in Video Explorer.</b>
        </div>""", unsafe_allow_html=True)

        ds_col, cls_col, cfg_col = st.columns(3)
        with ds_col:  raw_ds_sel  = st.selectbox("Dataset / Model", list(PROC_CONFIGS.keys()), key="raw_ds_sel")
        with cls_col: raw_cls_sel = st.selectbox("True Label", ["Unknown","Fight","Nonfight"], key="raw_cls_sel")
        with cfg_col:
            raw_cfg = PROC_CONFIGS[raw_ds_sel]
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:{tdim};line-height:2;margin-top:22px;">
                window={raw_cfg['window_size']} · stride={raw_cfg['window_stride']}<br>
                thresh={raw_cfg['pred_thresh']}
            </div>""", unsafe_allow_html=True)

        raw_mode = st.radio("Upload format", ["🎞️ Multiple raw videos", "🗜️ ZIP of raw videos"],
                            horizontal=True, key="raw_mode")

        raw_videos_up = None
        raw_zip_up    = None
        if raw_mode == "🎞️ Multiple raw videos":
            raw_videos_up = st.file_uploader(
                "Upload raw video files (.mp4 / .avi)",
                type=["mp4","avi","mov","mkv"],
                accept_multiple_files=True,
                key="raw_vids_up")
            if raw_videos_up:
                st.info(f"📋 {len(raw_videos_up)} video(s) queued: {', '.join(f.name for f in raw_videos_up[:5])}{'…' if len(raw_videos_up)>5 else ''}")
        else:
            raw_zip_up = st.file_uploader(
                "Upload ZIP of raw videos (flat ZIP — videos at top level or in subfolders)",
                type=["zip"], key="raw_zip_up")
            if raw_zip_up:
                # peek inside ZIP to count videos
                try:
                    with zipfile.ZipFile(io.BytesIO(raw_zip_up.getvalue())) as zcheck:
                        vid_names = [n for n in zcheck.namelist()
                                     if not n.endswith("/") and
                                     Path(n).suffix.lower() in [".mp4",".avi",".mov",".mkv"]]
                    st.info(f"📋 ZIP contains {len(vid_names)} video(s)")
                except Exception as e:
                    st.warning(f"⚠️ Could not read ZIP: {e}")
                    vid_names = []

        ckpt_ok = Path(raw_cfg["ckpt"]).exists()
        if not ckpt_ok:
            st.warning(f"⚠️ Checkpoint not found: `{raw_cfg['ckpt']}`")

        # Collect list of (filename, bytes) to process
        has_input = (raw_videos_up and len(raw_videos_up) > 0) or raw_zip_up is not None

        if st.button("▶ PROCESS ALL VIDEOS", type="primary",
                     disabled=(not has_input or not ckpt_ok), key="raw_ds_run"):

            # Build list of (name, bytes)
            vid_items = []
            if raw_mode == "🎞️ Multiple raw videos" and raw_videos_up:
                vid_items = [(f.name, f.getbuffer().tobytes()) for f in raw_videos_up]
            elif raw_zip_up:
                with zipfile.ZipFile(io.BytesIO(raw_zip_up.getvalue())) as zf_raw:
                    for zname in zf_raw.namelist():
                        if not zname.endswith("/") and \
                           Path(zname).suffix.lower() in [".mp4",".avi",".mov",".mkv"]:
                            vid_items.append((Path(zname).name, zf_raw.read(zname)))

            if not vid_items:
                st.warning("No valid video files found.")
            else:
                cls_for_raw = "Fight" if raw_cls_sel == "Fight" else "Nonfight"
                total_vids  = len(vid_items)
                st.markdown(f"**Processing {total_vids} video(s) sequentially...**")
                overall_bar  = st.progress(0.0)
                status_label = st.empty()
                results_log  = []

                for vi, (vname, vbytes) in enumerate(vid_items):
                    stem_r      = _safe_name(Path(vname).stem)
                    ts_r        = datetime.now().strftime("%Y%m%d_%H%M%S")
                    folder_r    = f"{stem_r}_{ts_r}"
                    out_dir_r   = class_root(raw_cfg["name"], cls_for_raw) / folder_r
                    out_dir_r.mkdir(parents=True, exist_ok=True)
                    tmp_r       = out_dir_r / vname
                    tmp_r.write_bytes(vbytes)

                    status_label.markdown(
                        f"<div style='font-family:Share Tech Mono,monospace;font-size:12px;color:{tblue};'>"
                        f"[{vi+1}/{total_vids}] Processing: {vname}</div>",
                        unsafe_allow_html=True)

                    # Run synchronously (one at a time — no thread needed here)
                    prog = {"pct":0.0,"stage":"","done":False,"error":None,
                            "pred_lbl":"?","conf":0.0,"onset":None,"out_dir":str(out_dir_r)}
                    run_processing_pipeline(tmp_r, raw_cfg, raw_cls_sel, out_dir_r, prog)

                    ok_r = prog.get("error") is None
                    results_log.append({
                        "video":   vname,
                        "folder":  folder_r,
                        "pred":    prog.get("pred_lbl","?"),
                        "conf":    f"{prog.get('conf',0):.1%}" if ok_r else "ERR",
                        "onset":   str(prog.get("onset","N/A")),
                        "status":  "✅" if ok_r else "❌",
                    })
                    overall_bar.progress((vi+1) / total_vids)

                status_label.markdown(
                    f"<div style='font-family:Share Tech Mono,monospace;font-size:12px;color:#52e08a;'>"
                    f"✅ All {total_vids} video(s) processed!</div>",
                    unsafe_allow_html=True)

                # Show results table
                st.markdown("#### Results")
                st.dataframe(pd.DataFrame(results_log), use_container_width=True, hide_index=True)
                st.success(f"✅ {sum(1 for r in results_log if r['status']=='✅')}/{total_vids} succeeded. Open **Video Explorer** to view.")
                if st.button("📁 Open Video Explorer", type="primary", key="raw_ds_goto_ex"):
                    st.session_state.nav_page = "📁 Video Explorer"; st.rerun()

    # ── Uploaded folders listing ───────────────────────────────
    st.divider()
    st.markdown("### 📂 Uploaded Folders")
    found = False
    for ds in DATASETS:
        for cls in DATASETS[ds]:
            flist = list_video_folders(ds, cls)
            if flist:
                found = True
                with st.expander(f"**{ds}/{cls}** — {len(flist)} folder(s)", expanded=False):
                    for fl2 in flist:
                        ffiles2  = list(fl2.iterdir())
                        n_mp4    = len([f2 for f2 in ffiles2 if f2.suffix==".mp4"])
                        n_png    = len([f2 for f2 in ffiles2 if f2.suffix==".png"])
                        has_pred = any(f2.name=="pred.txt" for f2 in ffiles2)
                        st.markdown(
                            f"📁 **{fl2.name}** — {n_mp4} videos · {n_png} grids · "
                            f"{'✅ pred.txt' if has_pred else '❌ no pred.txt'}")
    if not found:
        st.info("Nothing uploaded yet.")


# ══════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Analytics":
    st.subheader("📈 Analytics")
    scores=st.session_state.active_scores; fps2=st.session_state.active_fps
    pred=st.session_state.active_pred; fname=st.session_state.active_folder_name
    if scores is None or fps2 is None:
        st.info("Analyze a folder in **Video Explorer** first.")
    else:
        st.markdown(f"Showing: **{fname}**")
        step=max(1,int(0.5*fps2))
        rows=[{"time":fmt_time(i/fps2),"frame":i,"prob":round(float(scores[i]),4)}
              for i in range(0,len(scores),step)]
        cA,cB=st.columns([1.3,1.0],gap="large")
        with cA: st.markdown("#### Frame Log"); st.dataframe(pd.DataFrame(rows),use_container_width=True,height=280)
        with cB:
            st.markdown("#### Summary")
            for label, key in [("Prediction","pred_label"),("Confidence","confidence"),
                                ("Onset Frame","onset_frame"),("Onset Time","onset_time"),
                                ("Total Frames","total_frames"),("Val Acc","model_val_acc")]:
                st.metric(label, pred.get(key,"?"))
        c1,c2=st.columns(2)
        with c1: st.pyplot(make_timeline_plot(scores,fps2,pred),clear_figure=True)
        with c2: st.pyplot(make_hist_plot(scores),clear_figure=True)


# ══════════════════════════════════════════════════════════════
# PAGE: DATASET STATS
# ══════════════════════════════════════════════════════════════
elif page == "📊 Dataset Stats":
    st.markdown(f"<div style='font-family:Orbitron,sans-serif;font-size:1rem;font-weight:900;letter-spacing:2px;margin-bottom:16px;'>📊 DATASET ACCURACY REPORT</div>", unsafe_allow_html=True)
    records=get_all_pred_records()
    if not records:
        st.info("No pred.txt files found.")
    else:
        df=pd.DataFrame(records)
        def ds_stats(sub):
            total=len(sub)
            if total==0: return total,0,0.0,0,0,0,0
            correct=sub["correct"].str.lower().eq("true").sum() if "correct" in sub.columns else 0
            acc=correct/total
            fight=sub[sub["true_label"].str.lower()=="fight"] if "true_label" in sub.columns else sub.iloc[0:0]
            non=sub[sub["true_label"].str.lower().str.contains("non",na=False)] if "true_label" in sub.columns else sub.iloc[0:0]
            tp=int(fight["correct"].str.lower().eq("true").sum()) if len(fight) else 0
            tn=int(non["correct"].str.lower().eq("true").sum())   if len(non)   else 0
            fp2=int(non["correct"].str.lower().ne("true").sum())   if len(non)   else 0
            fn2=int(fight["correct"].str.lower().ne("true").sum()) if len(fight) else 0
            return total,int(correct),acc,tp,tn,fp2,fn2
        total_all,correct_all,acc_all,tp_all,tn_all,fp_all,fn_all=ds_stats(df)
        hf_df =df[df["_dataset"]=="hockeyfight"] if "_dataset" in df.columns else df.iloc[0:0]
        rwf_df=df[df["_dataset"]=="rwf"]          if "_dataset" in df.columns else df.iloc[0:0]
        _,hf_c, hf_acc, hf_tp, hf_tn, hf_fp, hf_fn =ds_stats(hf_df)
        _,rwf_c,rwf_acc,rwf_tp,rwf_tn,rwf_fp,rwf_fn=ds_stats(rwf_df)
        om1,om2,om3,om4,om5=st.columns(5,gap="small")
        om1.metric("Total Videos",str(total_all))
        om2.metric("Correct",str(correct_all))
        om3.metric("Overall Accuracy",f"{acc_all:.1%}")
        om4.metric("True Positives",str(tp_all))
        om5.metric("True Negatives",str(tn_all))
        card_l,card_r=st.columns(2,gap="medium")
        for col, name, emoji, total2, correct2, acc2, tp2, tn2, fp2, fn2, color in [
            (card_l,"HockeyFight","🏒",len(hf_df),hf_c,hf_acc,hf_tp,hf_tn,hf_fp,hf_fn,tblue),
            (card_r,"RWF-2000","🥊",len(rwf_df),rwf_c,rwf_acc,rwf_tp,rwf_tn,rwf_fp,rwf_fn,accent),
        ]:
            if total2==0:
                col.markdown(f"<div style='background:{bg2};border:1px solid {bord};border-radius:10px;padding:30px 20px;text-align:center;font-family:Share Tech Mono,monospace;font-size:11px;color:{tdim2};'>{emoji} {name}<br><br>No data yet.</div>",unsafe_allow_html=True)
                continue
            prec2=tp2/(tp2+fp2) if (tp2+fp2)>0 else 0
            rec2=tp2/(tp2+fn2)  if (tp2+fn2)>0 else 0
            f1_2=2*prec2*rec2/(prec2+rec2) if (prec2+rec2)>0 else 0
            col.markdown(f"""
            <div style="background:{bg2};border:1px solid {bord};border-left:4px solid {color};border-radius:10px;padding:20px;">
                <div style="font-family:'Orbitron',sans-serif;font-size:13px;font-weight:900;color:{color};letter-spacing:2px;margin-bottom:12px;">{emoji} {name.upper()}<br><span style="font-family:Share Tech Mono,monospace;font-size:10px;color:{tdim2};font-weight:400;">{total2} videos</span></div>
                <div style="font-family:'Orbitron',sans-serif;font-size:2rem;font-weight:900;margin-bottom:12px;">{acc2:.1%}</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-family:Share Tech Mono,monospace;font-size:11px;">
                    <div style="background:#080c10;border-radius:4px;padding:8px;"><div style="color:{tdim2};font-size:9px;">CORRECT</div><div style="color:#52e08a;">{correct2}</div></div>
                    <div style="background:#080c10;border-radius:4px;padding:8px;"><div style="color:{tdim2};font-size:9px;">WRONG</div><div style="color:{accent};">{total2-correct2}</div></div>
                    <div style="background:#080c10;border-radius:4px;padding:8px;"><div style="color:{tdim2};font-size:9px;">TP/TN</div><div style="color:{tblue};">{tp2}/{tn2}</div></div>
                    <div style="background:#080c10;border-radius:4px;padding:8px;"><div style="color:{tdim2};font-size:9px;">FP/FN</div><div style="color:#f5a623;">{fp2}/{fn2}</div></div>
                    <div style="background:#080c10;border-radius:4px;padding:8px;"><div style="color:{tdim2};font-size:9px;">PRECISION</div><div>{prec2:.1%}</div></div>
                    <div style="background:#080c10;border-radius:4px;padding:8px;"><div style="color:{tdim2};font-size:9px;">F1</div><div style="color:{color};">{f1_2:.1%}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: FP/FN BROWSER
# ══════════════════════════════════════════════════════════════
elif page == "❌ FP/FN Browser":
    st.subheader("❌ False Positive / False Negative Browser")
    records=get_all_pred_records()
    if not records: st.info("No pred.txt files found.")
    else:
        df=pd.DataFrame(records)
        if "correct" not in df.columns: st.info("Need 'correct' field in pred.txt.")
        else:
            wrong=df[df["correct"].str.lower()!="true"]
            if wrong.empty: st.success("🎉 No errors — all predictions correct!")
            else:
                tl,pl="true_label","pred_label"
                fp_df=wrong[(wrong[tl].str.lower().isin(["nonfight","nonfight"]))&(wrong[pl].str.lower()=="fight")] if tl in wrong.columns and pl in wrong.columns else pd.DataFrame()
                fn_df=wrong[(wrong[tl].str.lower()=="fight")&(wrong[pl].str.lower().str.contains("non"))] if tl in wrong.columns and pl in wrong.columns else pd.DataFrame()
                t1,t2,t3=st.tabs([f"All Wrong ({len(wrong)})",f"False Positives ({len(fp_df)})",f"False Negatives ({len(fn_df)})"])
                show=["_folder","_dataset","_class","true_label","pred_label","confidence","onset_time"]
                for tab, data, label in [(t1,wrong,"wrong"),(t2,fp_df,"fp"),(t3,fn_df,"fn")]:
                    with tab:
                        if data.empty: st.info("None found.")
                        else:
                            s2=[c for c in show if c in data.columns]
                            st.dataframe(data[s2].rename(columns={"_folder":"Folder"}),use_container_width=True,height=300)
                            st.download_button("⬇️ CSV",data=data[s2].to_csv(index=False).encode(),file_name=f"{label}.csv",mime="text/csv",key=f"fpfn_{label}_csv")


# ══════════════════════════════════════════════════════════════
# PAGE: CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════
elif page == "🧩 Confusion Matrix":
    st.subheader("🧩 Confusion Matrix & Metrics")
    records=get_all_pred_records()
    if not records: st.info("No pred.txt files found.")
    else:
        cm_col,met_col=st.columns([1,1.2],gap="large")
        with cm_col:
            fig2,cm2=make_confusion_matrix(records); st.pyplot(fig2,clear_figure=True)
        with met_col:
            st.markdown("#### Per-Class Metrics")
            TP,FN=int(cm2[0][0]),int(cm2[0][1]); FP,TN=int(cm2[1][0]),int(cm2[1][1])
            pf=TP/(TP+FP) if (TP+FP)>0 else 0; rf=TP/(TP+FN) if (TP+FN)>0 else 0
            f1f=2*pf*rf/(pf+rf) if (pf+rf)>0 else 0
            pn=TN/(TN+FN) if (TN+FN)>0 else 0; rn=TN/(TN+FP) if (TN+FP)>0 else 0
            f1n=2*pn*rn/(pn+rn) if (pn+rn)>0 else 0
            oa=(TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0
            mdf=pd.DataFrame([
                {"Class":"Fight","Precision":f"{pf:.1%}","Recall":f"{rf:.1%}","F1":f"{f1f:.1%}","Support":TP+FN},
                {"Class":"NonFight","Precision":f"{pn:.1%}","Recall":f"{rn:.1%}","F1":f"{f1n:.1%}","Support":FP+TN},
            ])
            st.dataframe(mdf,use_container_width=True,hide_index=True)
            st.metric("Overall Accuracy",f"{oa:.1%}")
            st.markdown(f"**TP:** `{TP}` | **FN:** `{FN}` | **FP:** `{FP}` | **TN:** `{TN}`")


# ══════════════════════════════════════════════════════════════
# PAGE: INCIDENT REPORT
# ══════════════════════════════════════════════════════════════
elif page == "🧾 Incident Report":
    st.subheader("🧾 Incident Report Generator")
    pred=st.session_state.active_pred; scores=st.session_state.active_scores
    fps2=st.session_state.active_fps;  fname=st.session_state.active_folder_name
    if not pred or scores is None:
        st.info("Analyze a folder in **Video Explorer** first.")
    else:
        col1,col2=st.columns([1.1,1.2],gap="large")
        with col1:
            cam_nm=st.text_input("Camera name",value="Entrance Camera",key="inc_cam")
            loc   =st.text_input("Location",value="Main Gate",key="inc_loc")
            notes =st.text_area("Notes",value="",key="inc_notes")
            gen   =st.button("🧾 GENERATE REPORT",type="primary",key="inc_gen")
        with col2:
            is_f3=is_fight_pred(pred)
            if is_f3: st.error(f"🔴 FIGHT — {pred.get('confidence','?')} confidence")
            else:     st.success(f"🟢 NORMAL — {pred.get('confidence','?')} confidence")
            for label, key in [("Video",fname),("Dataset",pred.get("dataset","?")),
                                ("True Label",pred.get("true_label","?")),
                                ("Predicted",pred.get("pred_label","?")),
                                ("Onset Time",pred.get("onset_time","?"))]:
                st.markdown(f"- **{label}:** {key}")
        if gen:
            report={"incident_id":f"INC-{int(time.time())}","generated_at":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "generated_by":st.session_state.username,"camera":cam_nm,"location":loc,
                    "video_folder":fname,"dataset":pred.get("dataset","?"),
                    "prediction":{"true_label":pred.get("true_label","?"),"pred_label":pred.get("pred_label","?"),
                                  "confidence":pred.get("confidence","?"),"onset_frame":pred.get("onset_frame","?"),
                                  "onset_time":pred.get("onset_time","?"),"how_started":describe_onset(pred)},"notes":notes}
            ts3=datetime.now().strftime("%Y%m%d_%H%M%S")
            rpath=Path(CFG.OUTPUT_DIR)/f"incident_{fname}_{ts3}.json"
            with open(rpath,"w") as f2: json.dump(report,f2,indent=2)
            st.success("✅ Report saved!")
            st.download_button("⬇️ Download JSON",data=io.BytesIO(json.dumps(report,indent=2).encode()),
                               file_name=rpath.name,mime="application/json",key="inc_json_dl")
            pdf=generate_pdf_report(pred,scores,fps2,fname)
            st.download_button("⬇️ Download PDF",data=pdf,
                               file_name=f"report_{fname}_{ts3}.pdf",mime="application/pdf",key="inc_pdf_dl")


# ══════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown(f"<div style='font-family:Orbitron,sans-serif;font-size:1rem;font-weight:900;letter-spacing:2px;margin-bottom:20px;'>⚙️ SETTINGS</div>", unsafe_allow_html=True)
    left_col,right_col=st.columns(2,gap="large")
    with left_col:
        st.markdown('<div class="vg-settings-card"><div class="vg-settings-section-title">🎨 APPEARANCE</div>', unsafe_allow_html=True)
        new_theme=st.radio("Theme",["dark","light"],index=0 if st.session_state.ui_theme=="dark" else 1,horizontal=True,format_func=lambda x:"🌙 Dark" if x=="dark" else "☀️ Light",key="settings_theme")
        new_accent=st.selectbox("Accent Color",["#e05252","#7ecfff","#52e08a","#f5a623","#a855f7","#ec4899","#06b6d4"],
                                format_func=lambda x:{"#e05252":"🔴 Alert Red","#7ecfff":"🔵 Cyber Blue","#52e08a":"🟢 Matrix Green","#f5a623":"🟠 Orange","#a855f7":"🟣 Purple","#ec4899":"🩷 Pink","#06b6d4":"🩵 Cyan"}.get(x,x),key="settings_accent")
        new_font=st.select_slider("Font Size",["small","medium","large"],value=st.session_state.font_size,key="settings_font")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="vg-settings-card"><div class="vg-settings-section-title">🔬 ANALYSIS DEFAULTS</div>', unsafe_allow_html=True)
        new_thr_v=st.slider("Violence Threshold",0.10,0.99,value=st.session_state.thr_violence,step=0.01,key="settings_thr_v")
        new_thr_s=st.slider("Suspicious Threshold",0.10,0.99,value=st.session_state.thr_suspicious,step=0.01,key="settings_thr_s")
        new_max_f=st.number_input("Max Frames",30,1000,value=st.session_state.max_frames,step=10,key="settings_max_frames")
        st.markdown('</div>', unsafe_allow_html=True)
    with right_col:
        st.markdown('<div class="vg-settings-card"><div class="vg-settings-section-title">👤 ACCOUNT</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:12px;margin-bottom:12px;'>Logged in as <b style='color:{tblue};'>{st.session_state.username}</b></div>", unsafe_allow_html=True)
        cp_old=st.text_input("Current Password",type="password",key="cp_old")
        cp_new=st.text_input("New Password",type="password",key="cp_new")
        cp_new2=st.text_input("Confirm New Password",type="password",key="cp_new2")
        if st.button("🔒 CHANGE PASSWORD",key="change_pw_btn"):
            if not try_login(st.session_state.username,cp_old): st.error("❌ Wrong current password.")
            elif cp_new!=cp_new2: st.error("❌ Passwords don't match.")
            elif len(cp_new)<4:   st.error("❌ Min 4 characters.")
            else:
                users=load_users(); users[st.session_state.username]=hash_pw(cp_new); save_users(users)
                st.success("✅ Password changed!")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="vg-settings-card"><div class="vg-settings-section-title">ℹ️ ABOUT</div><div style="font-family:Share Tech Mono,monospace;font-size:11px;line-height:1.8;color:{tdim};">VisionGuard v7<br>Model: R3D-18 + LCM + LSTM<br>CAM: GradCAM | GradCAM++ | SmoothGradCAM++ | LayerCAM<br>Datasets: HockeyFight · RWF-2000<br>NEW: Full in-browser processing pipeline</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sc2,rc2,_=st.columns([1,1,3])
    with sc2:
        if st.button("💾 SAVE SETTINGS",type="primary",use_container_width=True,key="save_settings"):
            st.session_state.ui_theme=new_theme; st.session_state.accent_color=new_accent
            st.session_state.font_size=new_font; st.session_state.thr_violence=new_thr_v
            st.session_state.thr_suspicious=new_thr_s; st.session_state.max_frames=new_max_f
            st.success("✅ Saved!"); time.sleep(0.3); st.rerun()
    with rc2:
        if st.button("↩️ RESET",use_container_width=True,key="reset_settings"):
            for k in ["ui_theme","accent_color","font_size"]: 
                if k in st.session_state: del st.session_state[k]
            st.session_state.thr_violence=CFG.THRESH_VIOLENCE
            st.session_state.thr_suspicious=CFG.THRESH_SUSPICIOUS
            st.session_state.max_frames=CFG.MAX_FRAMES
            st.rerun()


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.divider()
st.markdown(f"<span style='font-family:Share Tech Mono,monospace;font-size:10px;color:{tdim2};'>VISIONGUARD v7 · R3D-18 + LCM + LSTM · GradCAM | GradCAM++ | SmoothGradCAM++ | LayerCAM | Combined · Device: {DEVICE}</span>", unsafe_allow_html=True)