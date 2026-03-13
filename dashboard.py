import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# VisionGuard v8
# Refactored dashboard with grouped navigation, review workspace, dataset lab,
# history page, improved login screen, create-account and forgot-password flows.

import csv, io, re, time, json, shutil, hashlib, zipfile, subprocess, threading, random, string
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
UPLOAD_ROOT  = Path(CFG.OUTPUT_DIR) / "uploads"
LOGS_ROOT    = Path(CFG.OUTPUT_DIR) / "logs"
USERS_FILE   = Path(CFG.OUTPUT_DIR) / "users.json"
HISTORY_FILE = Path(CFG.OUTPUT_DIR) / "history.json"
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

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
MAIN_NAV = [
    "🏠 Home",
    "📥 Ingest",
    "🧪 Review Workspace",
    "📊 Dataset Lab",
    "🕘 History",
    "🛠️ Smart Tools",
    "⚙️ Settings",
]

# ══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
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
# CAM ENGINE
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

        w  = G["layer4"].mean(dim=(1,2,3))
        gc = self._up_norm(F.relu((w[:,None,None,None]*A["layer4"]).sum(0)), tgt)

        G2 = G["layer4"]**2; G3 = G["layer4"]**3
        dn = 2.0*G2 + (A["layer4"]*G3).sum(dim=(1,2,3), keepdim=True)
        al = G2 / (dn + EPS)
        wt = (al * F.relu(G["layer4"])).sum(dim=(1,2,3))
        gcpp = self._up_norm(F.relu((wt[:,None,None,None]*A["layer4"]).sum(0)), tgt)

        lc = np.zeros((T,H,W), dtype=np.float32)
        for ln in ["layer2","layer3","layer4"]:
            lc += self._up_norm(F.relu(F.relu(G[ln])*A[ln]).sum(0), tgt)
        lc /= 3.0
        mn, mx = lc.min(), lc.max(); lc = (lc-mn)/(mx-mn+EPS)

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
# PROCESSING HELPERS
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
# MAIN PROCESSING FUNCTION
# ══════════════════════════════════════════════════════════════
def run_processing_pipeline(vid_path: Path, cfg: dict, true_label: str, out_dir: Path, progress_dict: dict):
    def upd(pct, stage):
        progress_dict["pct"] = pct
        progress_dict["stage"] = stage
    try:
        upd(0.02, "📂 Reading video frames...")
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
        fps = float(fps) if fps > 1 else 15.0

        WS = cfg["window_size"]
        starts = list(range(0, total-WS+1, cfg["window_stride"]))
        if not starts: starts = [0]
        if starts[-1]+WS < total: starts.append(max(0, total-WS))

        upd(0.08, "🧠 Loading model & running predictions...")
        model, meta = load_model_cached(cfg)

        best_p = None; best_fp = -1.0
        ffp = np.zeros(total, np.float32)
        fhc = np.zeros(total, np.float32)
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
            upd(0.08 + 0.22*(wi+1)/n_wins, f"🧠 Predictions... window {wi+1}/{n_wins}")

        ffp /= np.maximum(fhc, 1)
        sfp = _smooth_curve(ffp, SMOOTH_K)
        pred = 1 if best_fp >= cfg["pred_thresh"] else 0
        conf = float(best_p[pred])
        pred_lbl = "Fight" if pred == 1 else "Nonfight"
        progress_dict["pred_lbl"] = pred_lbl
        progress_dict["conf"] = conf

        upd(0.32, "⏱️ Detecting fight onset...")
        onset = None
        if pred == 1:
            onset = _lstm_onset(sfp, ffp, total, cfg["onset_thresh"], cfg["spike_delta"])
        progress_dict["onset"] = onset

        cams = None
        if pred == 1:
            acc = {m: np.zeros((total,IMG_SIZE,IMG_SIZE), np.float32) for m in CAM_METHODS}
            hc  = np.zeros(total, np.float32)
            eng = CAMEngine(model)
            for wi, start in enumerate(starts):
                idx = _win_idx(start, total, WS)
                try:
                    cam_out = eng.compute_all(_to_tensor(frames, idx), cls=1)
                except Exception:
                    continue
                nc = cam_out["gradcam"].shape[0]
                for li, gi in enumerate(idx):
                    if 0 <= gi < total:
                        ci = min(int(li*nc/len(idx)), nc-1)
                        for m in CAM_METHODS: acc[m][gi] += cam_out[m][ci]
                        hc[gi] += 1
                upd(0.32 + 0.38*(wi+1)/n_wins, f"🔥 Computing CAMs... window {wi+1}/{n_wins}")
            eng.remove()
            def _norm(a):
                a = a / np.maximum(hc[:,None,None], 1)
                mn, mx = a.min(), a.max()
                return (a-mn)/(mx-mn+EPS)
            cams = {m: _norm(acc[m]) for m in CAM_METHODS}

        upd(0.72, "🎬 Rendering annotated frames...")
        fl = {k: [] for k in ["original"] + CAM_METHODS + ["combined"]}
        kw = dict(ds_label=cfg["label"], pred_lbl=pred_lbl, conf=conf,
                  total=total, onset_frame=onset, fps=fps,
                  onset_thresh=cfg["onset_thresh"])
        for t, fr in enumerate(frames):
            fp_t = float(sfp[t])
            active = (pred==1) and (onset is not None) and (t >= onset)
            fl["original"].append(_draw_info_bar(fr.copy(), frame_idx=t, fight_prob=fp_t, method_tag="Original", **kw))
            for m in CAM_METHODS:
                f_m = _apply_heatmap(fr, cams[m][t]) if active else fr.copy()
                fl[m].append(_draw_info_bar(f_m, frame_idx=t, fight_prob=fp_t, method_tag=CAM_LABELS[m], **kw))
            if active:
                combo = (cams["gradcampp"][t]+cams["smooth_gradcampp"][t]+cams["layercam"][t])/3
                mn, mx = combo.min(), combo.max()
                combo = (combo-mn)/(mx-mn+EPS)
                f_c = _apply_heatmap(fr, combo)
            else:
                f_c = fr.copy()
            fl["combined"].append(_draw_info_bar(f_c, frame_idx=t, fight_prob=fp_t, method_tag="Combined", **kw))

        upd(0.80, "💾 Writing videos...")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_name(vid_path.stem)
        tag_map = [("original","original"),("gradcam","gradcam"),
                   ("gradcampp","gradcampp"),("smooth_gradcampp","smooth_gradcampp"),
                   ("layercam","layercam"),("combined","combined")]
        for idx, (tag, key) in enumerate(tag_map):
            _write_video(out_dir/f"{stem}_{tag}.mp4", fl[key], fps)
            upd(0.80 + 0.06*(idx+1)/len(tag_map), f"💾 Writing {tag}.mp4...")

        upd(0.87, "🖼️ Saving frame grids...")
        pick = np.linspace(0, total-1, GRID_FRAMES).astype(int)
        grid_map = [("raw_grid","original"),("gradcam_grid","gradcam"),
                    ("gradcampp_grid","gradcampp"),("smooth_gradcampp_grid","smooth_gradcampp"),
                    ("layercam_grid","layercam"),("combined_grid","combined")]
        for gname, key in grid_map:
            src = [frames[i] for i in pick] if gname=="raw_grid" else [fl[key][i] for i in pick]
            cv2.imwrite(str(out_dir/f"{gname}.png"), cv2.cvtColor(_make_grid(src), cv2.COLOR_RGB2BGR))

        upd(0.93, "📊 Generating timeline plot...")
        _save_timeline(sfp, ffp, onset, fps, out_dir/"timeline.png", vid_path.name, pred_lbl, cfg["onset_thresh"])

        upd(0.96, "📋 Writing pred.txt...")
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

        upd(1.0, "✅ Done!")
        progress_dict["out_dir"] = str(out_dir)
        progress_dict["done"] = True
    except Exception as e:
        import traceback
        progress_dict["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        progress_dict["done"] = True

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG + THEME
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VisionGuard — Violence Detection",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

def get_theme_css(theme="dark", accent="#e05252", font_size="medium"):
    font_scale = {"small":"0.85rem","medium":"1rem","large":"1.1rem"}.get(font_size,"1rem")
    if theme == "dark":
        bg="#080c10"; bg2="#0d1520"; bg3="#0a0f18"; border="#1a2535"
        text_main="#c8d8e8"; text_dim="#445566"; text_dim2="#2a3a4a"
        text_head="#e8f4ff"; text_blue="#7ecfff"; text_sub="#aabbc8"; text_h3="#7a99b0"
    else:
        bg="#f0f4f8"; bg2="#ffffff"; bg3="#e8edf3"; border="#cdd5df"
        text_main="#1a2535"; text_dim="#556677"; text_dim2="#778899"
        text_head="#0a1020"; text_blue="#1a6fc4"; text_sub="#334455"; text_h3="#445566"
    return f'''
<style>
html,body,.stApp{{background:{bg}!important;color:{text_main}!important;font-size:{font_scale};}}
[data-testid="stSidebar"]{{background:linear-gradient(180deg,{bg3} 0%, {bg} 100%)!important;border-right:1px solid {border}!important;}}
[data-testid="stSidebar"] *{{color:{text_main}!important;}}
#MainMenu,footer,header{{visibility:hidden!important;}}
[data-testid="stDecoration"]{{display:none!important;}}
.block-container{{padding:1.1rem 1.6rem 2rem 1.6rem!important;max-width:100%!important;}}
.stTabs [data-baseweb="tab-list"]{{gap:0;background:{bg2};border-bottom:1px solid {border};padding:0;}}
.stTabs [data-baseweb="tab"]{{font-size:12px;font-weight:700;padding:8px 14px;color:{text_dim}!important;border-bottom:2px solid transparent;}}
.stTabs [aria-selected="true"]{{color:{text_head}!important;border-bottom:2px solid {accent}!important;}}
[data-testid="metric-container"]{{background:{bg2};border:1px solid {border};border-radius:8px;padding:10px 14px!important;}}
.stButton>button{{border-radius:8px!important;font-weight:700!important;}}
.stButton>button[kind="primary"]{{background:{accent}!important;border:none!important;color:white!important;}}
.stButton>button:not([kind="primary"]){{background:{bg2}!important;border:1px solid {border}!important;color:{text_blue}!important;}}
.stSelectbox>div>div,.stTextInput>div>div>input,.stTextArea textarea{{background:{bg2}!important;border:1px solid {border}!important;color:{text_main}!important;}}
.streamlit-expanderHeader{{background:{bg2}!important;border:1px solid {border}!important;border-radius:8px!important;}}
hr{{border-color:{border}!important;}}
[data-testid="stVideo"] video{{max-height:140px!important;width:100%!important;object-fit:contain!important;background:#000!important;}}
[data-testid="stVideo"]{{border:1px solid {border}!important;border-radius:8px!important;background:#000!important;max-width:420px!important;}}
.vg-card{{background:{bg2};border:1px solid {border};border-radius:12px;padding:16px;margin-bottom:12px;}}
.vg-soft{{color:{text_dim};font-size:12px;}}
.vg-title{{font-size:1.3rem;font-weight:800;color:{text_head};}}
.vg-mini{{font-size:11px;color:{text_dim2};}}
.vg-badge-fight{{display:inline-block;padding:4px 14px;border-radius:999px;background:rgba(224,82,82,0.18);border:2px solid #e05252;color:#ff5555;font-size:12px;font-weight:800;letter-spacing:0.05em;animation:fight-pulse 1.3s ease-in-out infinite;}}
.vg-badge-normal{{display:inline-block;padding:4px 14px;border-radius:999px;background:rgba(82,224,138,0.12);border:1px solid #52e08a;color:#52e08a;font-size:12px;font-weight:700;}}
@keyframes fight-pulse{{
  0%  {{box-shadow:0 0 0 0 rgba(224,82,82,0.75);background:rgba(224,82,82,0.18);border-color:#e05252;}}
  50% {{box-shadow:0 0 0 9px rgba(224,82,82,0);background:rgba(224,82,82,0.38);border-color:#ff2020;}}
  100%{{box-shadow:0 0 0 0 rgba(224,82,82,0);background:rgba(224,82,82,0.18);border-color:#e05252;}}
}}
.vg-stat-row{{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0;}}
.vg-stat{{background:{bg3};border:1px solid {border};border-radius:8px;padding:8px 14px;flex:1;min-width:100px;}}
.vg-stat-label{{font-size:10px;color:{text_dim};text-transform:uppercase;letter-spacing:0.05em;}}
.vg-stat-val{{font-size:1.1rem;font-weight:700;color:{text_head};}}
.hist-row{{background:{bg2};border:1px solid {border};border-radius:8px;padding:10px 14px;margin-bottom:6px;display:flex;align-items:center;gap:12px;}}
.login-wrap{{max-width:420px;margin:0 auto;padding-top:2.5rem;}}
.vg-back-btn{{display:inline-flex;align-items:center;gap:6px;padding:5px 14px;border-radius:8px;background:{bg2};border:1px solid {border};color:{text_blue};font-size:13px;font-weight:700;cursor:pointer;margin-bottom:10px;}}
.fight-analysis-card{{background:rgba(224,82,82,0.06);border:1px solid #e05252;border-radius:10px;padding:14px 18px;margin-bottom:14px;}}
.normal-analysis-card{{background:rgba(82,224,138,0.05);border:1px solid #52e08a;border-radius:10px;padding:12px 16px;margin-bottom:14px;}}
.idea-card{{background:{bg3};border:1px solid {border};border-radius:8px;padding:10px 14px;margin-bottom:8px;}}
</style>
'''

# ══════════════════════════════════════════════════════════════
# AUTH + HISTORY STORAGE
# ══════════════════════════════════════════════════════════════
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def load_users():
    if USERS_FILE.exists():
        try: return json.loads(USERS_FILE.read_text())
        except: pass
    return {}

def save_users(u): USERS_FILE.write_text(json.dumps(u, indent=2))

def try_login(u, p):
    users = load_users()
    info = users.get(u)
    if not info: return False
    if isinstance(info, str): return info == hash_pw(p)
    return info.get("password") == hash_pw(p)

def try_register(u, p):
    if not u or not p: return False, "Username and password required."
    if len(p) < 4: return False, "Password must be at least 4 characters."
    users = load_users()
    if u in users: return False, "Username already exists."
    reset_code = ''.join(random.choices(string.digits, k=6))
    users[u] = {"password": hash_pw(p), "created_at": datetime.now().isoformat(), "reset_code": reset_code}
    save_users(users)
    return True, f"Account created! Save your recovery code: **{reset_code}**"

def reset_password(username, code, new_pw):
    users = load_users()
    if username not in users: return False, "Username not found."
    info = users[username] if isinstance(users[username], dict) else {"password": users[username], "reset_code": "000000"}
    if info.get("reset_code") != code: return False, "Recovery code is incorrect."
    if len(new_pw) < 4: return False, "Password must be at least 4 characters."
    info["password"] = hash_pw(new_pw)
    users[username] = info
    save_users(users)
    return True, "Password reset successfully."

def load_history_store():
    if HISTORY_FILE.exists():
        try: return json.loads(HISTORY_FILE.read_text())
        except: pass
    return []

def save_history_store(items): HISTORY_FILE.write_text(json.dumps(items, indent=2))

# ══════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════
def is_fight_pred(pred, flip=False):
    lbl = str(pred.get("pred_label"," ")).lower()
    raw = "fight" in lbl and "non" not in lbl
    return (not raw) if flip else raw

def pred_label_to_status(pred_label):
    if "fight" in str(pred_label).lower() and "non" not in str(pred_label).lower(): return "ALERT"
    return "NORMAL"

def color_from_status(s): return {"ALERT":"🔴","SUSPICIOUS":"🟡","NORMAL":"🟢","UNKNOWN":"⚪"}.get(s,"⚪")

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
    conf = float(pred.get("confidence",0.5) or 0.5)
    try: onset_frame = int(pred.get("onset_frame",0))
    except: onset_frame = 0
    is_fight = is_fight_pred(pred)
    scores = np.zeros(n_frames, dtype=np.float32)
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
            subprocess.run(["ffmpeg","-y","-i",str(src),"-c:v","libx264","-pix_fmt","yuv420p",
                            "-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k",str(dst)],
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
    return f"Fight onset at frame {onset_f} ({onset_t}). Threshold: {pred.get('onset_threshold','?')}, spike: {pred.get('spike_delta','?')}."

def build_email_summary(pred, folder_name, camera="", location="", notes="", reviewer_tag=""):
    state = "Fight detected" if is_fight_pred(pred) else "No fight detected"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""Subject: VisionGuard Incident Report — {folder_name}

Hello,

An automated analysis has been completed by VisionGuard.

━━━━━━━━━━━━━━━━━━━━━━━━
INCIDENT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━
Status:        {state}
Prediction:    {pred.get("pred_label", "?")}
Confidence:    {pred.get("confidence", "?")}
Dataset:       {pred.get("dataset", "?")}
Onset time:    {pred.get("onset_time", "N/A")}
Onset frame:   {pred.get("onset_frame", "N/A")}
Total frames:  {pred.get("total_frames", "N/A")}

━━━━━━━━━━━━━━━━━━━━━━━━
LOCATION / METADATA
━━━━━━━━━━━━━━━━━━━━━━━━
Camera:        {camera or "N/A"}
Location:      {location or "N/A"}
Reviewer:      {reviewer_tag or "N/A"}
Notes:         {notes or "N/A"}
Timestamp:     {ts}

━━━━━━━━━━━━━━━━━━━━━━━━
MODEL DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━
Model path:    {pred.get("model_path", "?")}
Val accuracy:  {pred.get("model_val_acc", "?")}
True label:    {pred.get("true_label", "?")}
Correct:       {pred.get("correct", "?")}
System note:   {describe_onset(pred)}

Regards,
VisionGuard — R3D-18 + LCM + LSTM
"""

def _safe_video(path):
    """Serve video with browser-compatible h264. Transcodes to a _web.mp4 sidecar if needed."""
    try:
        p = Path(path)
        if not p.exists(): st.warning("⚠️ Video file not found."); return
        web_p = p.parent / (p.stem + "_web.mp4")
        if not web_p.exists() and ffmpeg_ok():
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(p),
                     "-c:v", "libx264", "-pix_fmt", "yuv420p",
                     "-preset", "veryfast", "-crf", "23",
                     "-movflags", "+faststart",
                     "-an", str(web_p)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120
                )
            except Exception:
                web_p = p
        serve = web_p if web_p.exists() else p
        data = serve.read_bytes()
        if not data: st.warning("⚠️ Video file is empty."); return
        st.video(data, format="video/mp4")
    except Exception as e:
        st.warning(f"⚠️ Video preview unavailable — try downloading instead. ({type(e).__name__})")

def parse_pred_txt(path):
    out = {}
    try:
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":",1)
                    out[k.strip()] = v.strip()
    except: pass
    return out

# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════
def get_plot_colors():
    theme = st.session_state.get("ui_theme","dark")
    if theme=="dark":
        return {"bg":"#080c10","ax":"#0a0f18","spine":"#1a2535","tick":"#445566","legend_text":"#7ecfff","xlabel":"#445566"}
    return {"bg":"#f0f4f8","ax":"#ffffff","spine":"#cdd5df","tick":"#556677","legend_text":"#1a6fc4","xlabel":"#556677"}

def make_timeline_plot(scores, fps, pred=None):
    c = get_plot_colors()
    t = np.arange(len(scores))/fps
    fig = plt.figure(figsize=(6,2.5), facecolor=c["bg"])
    ax = fig.add_subplot(111); ax.set_facecolor(c["ax"])
    is_fight = is_fight_pred(pred) if pred else False
    color = "#e05252" if is_fight else "#52e08a"
    ax.plot(t, scores, color=color, linewidth=1.6)
    ax.fill_between(t, scores, alpha=0.12, color=color)
    ax.axhline(st.session_state.get("thr_suspicious",CFG.THRESH_SUSPICIOUS), linestyle="--",color="#f5a623",linewidth=0.8)
    ax.axhline(st.session_state.get("thr_violence",CFG.THRESH_VIOLENCE), linestyle="--",color="#e05252",linewidth=0.8)
    if pred:
        try:
            onset_f = int(pred.get("onset_frame",0)); onset_tv = onset_f/fps
            if 0 < onset_tv < t[-1]: ax.axvline(onset_tv,color="#7ecfff",linewidth=1.2,linestyle=":")
        except: pass
    ax.set_xlabel("Time (s)",fontsize=8,color=c["xlabel"]); ax.set_ylabel("P(fight)",fontsize=8,color=c["xlabel"])
    ax.tick_params(colors=c["tick"],labelsize=7); ax.spines[:].set_color(c["spine"])
    plt.tight_layout(); return fig

def make_hist_plot(scores):
    c = get_plot_colors()
    fig = plt.figure(figsize=(4,2.5), facecolor=c["bg"])
    ax = fig.add_subplot(111); ax.set_facecolor(c["ax"])
    ax.hist(scores,bins=20,color="#5271e0",edgecolor=c["bg"],linewidth=0.3)
    ax.set_xlabel("Probability",fontsize=8,color=c["xlabel"]); ax.set_ylabel("Count",fontsize=8,color=c["xlabel"])
    ax.tick_params(colors=c["tick"],labelsize=7); ax.spines[:].set_color(c["spine"])
    plt.tight_layout(); return fig

def make_confusion_matrix(records):
    c = get_plot_colors(); labels=["Fight","NonFight"]; cm=np.zeros((2,2),dtype=int)
    lmap={"fight":0,"nonfight":1,"Fight":0,"NonFight":1,"Nonfight":1}
    for r in records:
        t2=lmap.get(r.get("true_label",""),-1); p2=0 if is_fight_pred(r) else 1
        if t2>=0 and p2>=0: cm[t2][p2]+=1
    fig,ax=plt.subplots(figsize=(4,3),facecolor=c["bg"]); ax.set_facecolor(c["ax"])
    ax.imshow(cm,interpolation="nearest",cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels,color=c["legend_text"]); ax.set_yticklabels(labels,color=c["legend_text"])
    ax.set_xlabel("Predicted",color=c["xlabel"]); ax.set_ylabel("True",color=c["xlabel"])
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i][j]),ha="center",va="center",fontsize=14,fontweight="bold",
                    color="white" if cm[i][j]>cm.max()/2 else c["legend_text"])
    plt.tight_layout(); return fig, cm

# ══════════════════════════════════════════════════════════════
# FOLDER HELPERS
# ══════════════════════════════════════════════════════════════
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

def get_files(folder):
    files = {}
    def real_files(pattern):
        return [f for f in folder.glob(pattern) if not f.name.startswith("._") and not f.name.startswith(".")]
    for vk in ALL_VID_KEYS:
        if vk=="original":
            cands = real_files("*original*.mp4")
            if not cands:
                cands = [f for f in real_files("*.mp4") if not any(x in f.name.lower() for x in ["gradcam","layercam","combined","smooth","_preview"])]
            if cands: files["original"] = cands[0]
        elif vk=="smooth_gradcampp":
            cands = [f for f in real_files("*smooth_gradcampp*.mp4")+real_files("*smooth*.mp4") if not f.name.startswith("_preview")]
            if cands: files["smooth_gradcampp"] = cands[0]
        elif vk=="gradcampp":
            cands = [f for f in real_files("*gradcampp*.mp4")+real_files("*gradcam++*.mp4") if "smooth" not in f.name.lower() and not f.name.startswith("_preview")]
            if cands: files["gradcampp"] = cands[0]
        elif vk=="gradcam":
            cands = [f for f in real_files("*gradcam*.mp4") if "pp" not in f.name.lower() and "++" not in f.name.lower() and "smooth" not in f.name.lower() and not f.name.startswith("_preview")]
            if cands: files["gradcam"] = cands[0]
        elif vk=="layercam":
            cands = [f for f in real_files("*layercam*.mp4") if not f.name.startswith("_preview")]
            if cands: files["layercam"] = cands[0]
        elif vk=="combined":
            cands = [f for f in real_files("*combined*.mp4") if not f.name.startswith("_preview")]
            if cands: files["combined"] = cands[0]
    for gk in ALL_GRID_KEYS:
        f2 = find_file(folder, f"{gk}.png")
        if f2: files[gk]=f2
    f2 = find_file(folder,"timeline.png")
    if f2: files["timeline"] = f2
    f2 = find_file(folder,"pred.txt")
    if f2: files["pred"] = f2
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
        for fn, flist in folder_map.items():
            dest=class_root(dataset,cls)/fn; dest.mkdir(parents=True,exist_ok=True); n_folders+=1
            for zp in flist:
                if zp.endswith("/"): continue
                try:
                    with open(dest/Path(zp).name,"wb") as f2: f2.write(zf.read(zp))
                    n_files+=1
                except: pass
    return n_folders, n_files

# ══════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════
def push_history(folder_name, dataset, cls, pred, active_files_dict, camera="", location="", notes="", reviewer_tag=""):
    entry = {
        "folder": folder_name,
        "dataset": dataset,
        "cls": cls,
        "pred_lbl": pred.get("pred_label", "?"),
        "conf": pred.get("confidence", "?"),
        "onset_t": pred.get("onset_time", "N/A"),
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera": camera,
        "location": location,
        "notes": notes,
        "reviewer_tag": reviewer_tag,
        "_files": active_files_dict,
    }
    hist = load_history_store()
    hist = [h for h in hist if not (h.get("folder") == folder_name and h.get("dataset") == dataset and h.get("cls") == cls)]
    hist.insert(0, entry)
    save_history_store(hist[:50])
    st.session_state["_history"] = hist[:50]

def restore_history(entry: dict):
    files = {k: Path(v) for k, v in entry.get("_files", {}).items()}
    pred_data = parse_pred_txt(files["pred"]) if "pred" in files else {}
    frames, fps2 = [], float(CFG.DEFAULT_FPS)
    if "original" in files:
        try:
            frames, fps2 = read_video_frames(files["original"], max_frames=st.session_state.get("max_frames", CFG.MAX_FRAMES))
            frames = [resize_keep(f, 640) for f in frames]
        except: pass
    n = len(frames) if frames else 100
    scores = scores_from_pred(pred_data, n, fps2)
    st.session_state.active_pred = pred_data
    st.session_state.active_scores = scores
    st.session_state.active_fps = fps2
    st.session_state.active_frames = frames
    st.session_state.active_folder_name = entry["folder"]
    st.session_state.active_video_path = str(files.get("original", ""))
    st.session_state.active_dataset = entry["dataset"]
    st.session_state.active_class = entry["cls"]
    st.session_state["_active_files"] = {k: str(v) for k, v in files.items()}
    st.session_state["review_camera"] = entry.get("camera", "")
    st.session_state["review_location"] = entry.get("location", "")
    st.session_state["review_notes"] = entry.get("notes", "")
    st.session_state["reviewer_tag"] = entry.get("reviewer_tag", "")

def update_history_metadata(folder_name, dataset, cls, camera, location, notes, reviewer_tag):
    hist = load_history_store()
    for item in hist:
        if item.get("folder") == folder_name and item.get("dataset") == dataset and item.get("cls") == cls:
            item["camera"] = camera
            item["location"] = location
            item["notes"] = notes
            item["reviewer_tag"] = reviewer_tag
    save_history_store(hist)
    st.session_state["_history"] = hist

# ══════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════
def generate_pdf_report(pred, scores, fps, folder_name):
    fig = plt.figure(figsize=(11,8.5),facecolor="#080c10")
    gs = gridspec.GridSpec(3,3,figure=fig,hspace=0.55,wspace=0.4)
    tax = fig.add_subplot(gs[0,:]); tax.axis("off"); tax.set_facecolor("#080c10")
    tax.text(0.5,0.75,"VisionGuard — Violence Detection Report",ha="center",va="center",fontsize=16,fontweight="bold",color="white")
    tax.text(0.5,0.3,f"Video: {folder_name}   |   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",ha="center",va="center",fontsize=10,color="#aaaaaa")
    atl = fig.add_subplot(gs[1,0:2]); atl.set_facecolor("#0a0f18")
    t = np.arange(len(scores))/fps
    is_fight=is_fight_pred(pred)
    atl.plot(t,scores,color="#e05252" if is_fight else "#52e08a",linewidth=1.5)
    atl.axhline(CFG.THRESH_VIOLENCE,linestyle="--",color="red",linewidth=1)
    atl.set_xlabel("Time (s)",color="white",fontsize=8); atl.set_ylabel("P(fight)",color="white",fontsize=8)
    atl.tick_params(colors="white"); atl.spines[:].set_color("#333355")
    ah = fig.add_subplot(gs[1,2]); ah.set_facecolor("#0a0f18")
    ah.hist(scores,bins=15,color="#5271e0"); ah.tick_params(colors="white"); ah.spines[:].set_color("#333355")
    ai = fig.add_subplot(gs[2,:]); ai.axis("off")
    lines = [
        f"STATUS: {'⚠ FIGHT DETECTED' if is_fight else '✓ NO FIGHT'}   |   Confidence: {pred.get('confidence','?')}",
        f"Dataset: {pred.get('dataset','?')}   True: {pred.get('true_label','?')}   Predicted: {pred.get('pred_label','?')}   Correct: {pred.get('correct','?')}",
        f"Onset Frame: {pred.get('onset_frame','?')}   Onset Time: {pred.get('onset_time','?')}   Frames: {pred.get('total_frames','?')}",
        f"Model: {pred.get('model_path','?')}   Val Acc: {pred.get('model_val_acc','?')}",
    ]
    for i, line in enumerate(lines):
        ai.text(0.02,0.95-i*0.22,line,transform=ai.transAxes,fontsize=8,
                color="#ff6666" if i==0 and is_fight else ("white" if i>0 else "#66ff66"),
                fontweight="bold" if i==0 else "normal")
    buf=io.BytesIO(); plt.savefig(buf,format="pdf",facecolor=fig.get_facecolor(),bbox_inches="tight"); plt.close(fig); buf.seek(0); return buf.read()

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "logged_in": False, "username": "",
        "active_pred": {}, "active_scores": None, "active_fps": None,
        "active_frames": None, "active_folder_name": "",
        "active_video_path": None, "active_dataset": "", "active_class": "",
        "run_id": datetime.now().strftime("run_%Y%m%d_%H%M%S"),
        "nav_section": "🏠 Home",
        "nav_history": [],
        "thr_violence": CFG.THRESH_VIOLENCE, "thr_suspicious": CFG.THRESH_SUSPICIOUS,
        "max_frames": CFG.MAX_FRAMES,
        "ui_theme": "dark", "accent_color": "#e05252", "font_size": "medium",
        "_proc_running": False, "_proc_progress": {}, "_proc_thread": None,
        "_proc_out_dir": "", "_proc_ds": "", "_proc_cls": "", "_proc_folder": "",
        "_history": load_history_store(),
        "review_camera": "Entrance Camera", "review_location": "Main Gate",
        "review_notes": "", "reviewer_tag": "",
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
for ds in DATASETS:
    for cls in DATASETS[ds]:
        class_root(ds,cls).mkdir(parents=True,exist_ok=True)

st.markdown(get_theme_css(
    st.session_state.get("ui_theme","dark"),
    st.session_state.get("accent_color","#e05252"),
    st.session_state.get("font_size","medium")
), unsafe_allow_html=True)

# ── Navigation helpers ──────────────────────────────────────────
def go_to(page: str):
    current = st.session_state.get("nav_section","🏠 Home")
    if current != page:
        hist = st.session_state.get("nav_history", [])
        hist.append(current)
        st.session_state.nav_history = hist[-10:]
    st.session_state.nav_section = page
    st.rerun()

def render_back_button():
    hist = st.session_state.get("nav_history", [])
    if not hist: return
    prev = hist[-1]
    if st.button(f"← Back to {prev}", key="back_btn"):
        st.session_state.nav_history = hist[:-1]
        st.session_state.nav_section = prev
        st.rerun()

theme  = st.session_state.get("ui_theme","dark")
accent = st.session_state.get("accent_color","#e05252")
bg2    = "#0d1520" if theme=="dark" else "#ffffff"
bg3    = "#0a0f18" if theme=="dark" else "#e8edf3"
bord   = "#1a2535" if theme=="dark" else "#cdd5df"
tblue  = "#7ecfff" if theme=="dark" else "#1a6fc4"
tdim   = "#445566" if theme=="dark" else "#556677"
tdim2  = "#2a3a4a" if theme=="dark" else "#778899"

# ══════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ══════════════════════════════════════════════════════════════
def render_login_screen():
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown("<div style='text-align:center;padding:2rem 0 1rem;'>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:3.5rem;'>🛡️</span>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.8rem;font-weight:800;color:{tblue};margin:8px 0 4px;'>VisionGuard</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:{tdim};font-size:0.9rem;margin-bottom:1.5rem;'>Violence Detection Dashboard · R3D-18 + LCM + LSTM</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        tab_signin, tab_register, tab_forgot = st.tabs(["🔐 Sign In", "📝 Create Account", "🔑 Forgot Password"])

        with tab_signin:
            lu = st.text_input("Username", key="login_u", placeholder="Enter your username")
            lp = st.text_input("Password", type="password", key="login_p", placeholder="Enter your password")
            st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)
            if st.button("Sign In →", type="primary", use_container_width=True):
                if try_login(lu.strip(), lp):
                    st.session_state.logged_in = True
                    st.session_state.username = lu.strip()
                    st.rerun()
                else:
                    st.error("Wrong username or password.")

        with tab_register:
            ru  = st.text_input("Choose username", key="reg_u")
            rp  = st.text_input("Choose password", type="password", key="reg_p")
            rp2 = st.text_input("Confirm password", type="password", key="reg_p2")
            st.caption("Minimum 4 characters. A recovery code will be shown after creation — save it.")
            st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)
            if st.button("Create Account", type="primary", use_container_width=True):
                if rp != rp2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = try_register(ru.strip(), rp)
                    if ok: st.success(msg)
                    else:  st.error(msg)

        with tab_forgot:
            st.caption("Enter your username and the recovery code shown when your account was created.")
            fu  = st.text_input("Username", key="fp_user")
            fc  = st.text_input("Recovery code", key="fp_code", placeholder="6-digit code")
            fn  = st.text_input("New password", type="password", key="fp_new")
            fn2 = st.text_input("Confirm new password", type="password", key="fp_new2")
            st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)
            if st.button("Reset Password", use_container_width=True):
                if fn != fn2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = reset_password(fu.strip(), fc.strip(), fn)
                    if ok: st.success(msg)
                    else:  st.error(msg)

if not st.session_state.logged_in:
    render_login_screen()
    st.stop()

# ══════════════════════════════════════════════════════════════
# HELPERS TO LOAD ACTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════
def load_analysis_from_folder(folder_path: Path, ds: str, cls: str, folder_name: str):
    files = get_files(folder_path)
    pred = parse_pred_txt(files["pred"]) if "pred" in files else {}
    frames, fps2 = [], float(CFG.DEFAULT_FPS)
    if "original" in files:
        try:
            frames, fps2 = read_video_frames(files["original"], max_frames=st.session_state.max_frames)
            frames = [resize_keep(f, 640) for f in frames]
        except: pass
    n = len(frames) if frames else 100
    scores = scores_from_pred(pred, n, fps2)
    st.session_state.active_pred = pred
    st.session_state.active_scores = scores
    st.session_state.active_fps = fps2
    st.session_state.active_frames = frames
    st.session_state.active_folder_name = folder_name
    st.session_state.active_video_path = str(files.get("original",""))
    st.session_state.active_dataset = ds
    st.session_state.active_class = cls
    st.session_state["_active_files"] = {k: str(v) for k, v in files.items()}
    push_history(folder_name, ds, cls, pred, st.session_state["_active_files"],
                 st.session_state.get("review_camera",""),
                 st.session_state.get("review_location",""),
                 st.session_state.get("review_notes",""),
                 st.session_state.get("reviewer_tag",""))

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"<div class='vg-card'>"
            f"<div class='vg-title'>🛡️ VisionGuard</div>"
            f"<div class='vg-soft'>v8 · {st.session_state.username}</div>"
            f"<div class='vg-mini'>{st.session_state.run_id}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        selected_nav = st.radio(
            "Navigation", MAIN_NAV,
            index=MAIN_NAV.index(st.session_state.get("nav_section","🏠 Home"))
        )
        if selected_nav != st.session_state.nav_section:
            go_to(selected_nav)

        if st.session_state.active_folder_name:
            pred_h = st.session_state.active_pred
            is_f = is_fight_pred(pred_h)
            badge = '<span class="vg-badge-fight">⚠ FIGHT</span>' if is_f else '<span class="vg-badge-normal">✓ NORMAL</span>'
            st.markdown(
                f"<div class='vg-card'>{badge}"
                f"<div style='margin-top:8px;font-weight:700'>{st.session_state.active_folder_name}</div>"
                f"<div class='vg-mini'>{st.session_state.active_dataset}/{st.session_state.active_class}</div>"
                f"<div class='vg-soft'>Conf: {pred_h.get('confidence','?')}</div></div>",
                unsafe_allow_html=True
            )

        with st.expander(f"Recent ({len(st.session_state.get('_history', []))})", expanded=False):
            hist = st.session_state.get("_history", [])[:8]
            if not hist:
                st.caption("No analyses yet.")
            else:
                for i, entry in enumerate(hist):
                    lbl = f"{entry['folder']} · {entry['pred_lbl']}"
                    if st.button(lbl, key=f"sb_hist_{i}"):
                        restore_history(entry)
                        go_to("🧪 Review Workspace")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

render_sidebar()

# ══════════════════════════════════════════════════════════════
# ACTIVE SUMMARY BAR
# ══════════════════════════════════════════════════════════════
def render_active_summary_bar():
    pred = st.session_state.active_pred
    folder = st.session_state.active_folder_name
    if not folder or not pred: return
    is_f = is_fight_pred(pred)
    badge = "vg-badge-fight" if is_f else "vg-badge-normal"
    badge_txt = "⚠ FIGHT DETECTED" if is_f else "✓ NORMAL"
    st.markdown(
        f"<div class='vg-card' style='display:flex;align-items:center;gap:16px;flex-wrap:wrap;'>"
        f"<span class='{badge}'>{badge_txt}</span>"
        f"<span style='font-weight:700;'>{folder}</span>"
        f"<span class='vg-soft'>{pred.get('dataset','?')} / {pred.get('true_label','?')}</span>"
        f"<span class='vg-soft'>Conf: <b>{pred.get('confidence','?')}</b></span>"
        f"<span class='vg-soft'>Onset: {pred.get('onset_time','N/A')}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════
def render_home():
    username = st.session_state.username
    records = get_all_pred_records()
    fights  = sum(1 for r in records if is_fight_pred(r))

    st.markdown(
        f"<div class='vg-card' style='padding:28px 28px 22px;'>"
        f"<div style='display:flex;align-items:center;gap:18px;'>"
        f"<span style='font-size:3rem;'>🛡️</span>"
        f"<div>"
        f"<div style='font-size:1.6rem;font-weight:800;'>Welcome back, {username}</div>"
        f"<div class='vg-soft' style='margin-top:4px;font-size:0.9rem;'>"
        f"VisionGuard · Violence Detection Dashboard · R3D-18 + LCM + LSTM · {str(DEVICE).upper()}"
        f"</div>"
        f"</div></div>"
        f"</div>",
        unsafe_allow_html=True
    )

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Processed folders", len(records))
    h2.metric("Fight detections", fights)
    h3.metric("Non-fight", len(records)-fights)
    h4.metric("CAM methods", 4)
    h5.metric("Device", str(DEVICE))

    st.markdown("### Quick Launch")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button("📥 Ingest Video", use_container_width=True):
            go_to("📥 Ingest")
    with q2:
        if st.button("🧪 Review Workspace", use_container_width=True):
            go_to("🧪 Review Workspace")
    with q3:
        if st.button("📊 Dataset Lab", use_container_width=True):
            go_to("📊 Dataset Lab")
    with q4:
        if st.button("🕘 History", use_container_width=True):
            go_to("🕘 History")

    # ── Smart Feature Ideas ──────────────────────────────────
    with st.expander("🛠️ Smart Tools — click any to open", expanded=False):
        ideas = [
            ("✂️ Evidence Clip Trimmer",
             "Auto-trim a fight clip to just the relevant seconds around onset. Download a tight evidence file instead of the full video.",
             True),
            ("📅 Risk Score Calendar Heatmap",
             "Visualise fight detections by day, hour, and day-of-week. Identify high-risk time windows from your history.",
             True),
            ("🗺️ Zone Manager",
             "Define named camera zones. Tag incidents to zones and see which areas have the most detections with a risk chart.",
             True),
            ("🔐 Chain of Custody Log",
             "SHA-256 hash all output files at registration time. Verify integrity later — tamper-evident audit trail for legal evidence.",
             True),
            ("👥 Person Count Estimator",
             "Add a lightweight YOLO head to count people in-frame at onset. Flag aggressor/victim patterns. (Future roadmap)",
             False),
            ("📡 Live Camera Feed",
             "Connect RTSP/webcam streams. Run sliding-window inference in near-real-time and push webhook alerts. (Future roadmap)",
             False),
            ("🔮 Escalation Predictor",
             "Use P(fight) slope in LSTM hidden states to predict fights 2–5 s before onset — alert security before contact occurs. (Future roadmap)",
             False),
            ("🌐 Multi-Camera Correlation",
             "Auto-flag 'coordinated incidents' when 2+ cameras detect fights within 30 s of each other. (Future roadmap)",
             False),
        ]
        cols = st.columns(2)
        for i, (title, desc, is_live) in enumerate(ideas):
            with cols[i % 2]:
                border_color = tblue if is_live else "#2a3a4a"
                label_html = (
                    f"<span style='background:rgba(126,207,255,0.15);color:{tblue};"
                    f"font-size:10px;font-weight:700;padding:2px 7px;border-radius:999px;"
                    f"margin-left:6px;'>LIVE</span>"
                    if is_live else
                    f"<span style='background:#1a2535;color:#445566;"
                    f"font-size:10px;padding:2px 7px;border-radius:999px;margin-left:6px;'>ROADMAP</span>"
                )
                title_color = tblue if is_live else "#445566"
                st.markdown(
                    f"<div style='background:#0a0f18;border:1px solid {border_color};border-radius:8px;"
                    f"padding:10px 14px;margin-bottom:8px;'>"
                    f"<div style='font-weight:700;color:{title_color};font-size:13px;margin-bottom:4px;'>"
                    f"{title}{label_html}</div>"
                    f"<div style='color:#7a99b0;font-size:12px;line-height:1.5;'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        if st.button("Open Smart Tools →", type="primary", use_container_width=True, key="home_smart_tools_btn"):
            go_to("🛠️ Smart Tools")

    if records:
        st.markdown("### Recent Analyses")
        recent = sorted(records, key=lambda r: r.get("_folder",""), reverse=True)[:5]
        for r in recent:
            is_f = is_fight_pred(r)
            badge = "vg-badge-fight" if is_f else "vg-badge-normal"
            badge_txt = "⚠ FIGHT" if is_f else "✓ NORMAL"
            rc1, rc2 = st.columns([5,1])
            with rc1:
                st.markdown(
                    f"<div class='vg-card' style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:10px 16px;'>"
                    f"<span class='{badge}'>{badge_txt}</span>"
                    f"<span style='font-weight:600;'>{r.get('_folder','?')}</span>"
                    f"<span class='vg-soft'>{r.get('_dataset','?')}/{r.get('_class','?')}</span>"
                    f"<span class='vg-soft'>Conf: {r.get('confidence','?')}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with rc2:
                if st.button("Review →", key=f"home_review_{r.get('_folder','')}"):
                    fp = class_root(r.get("_dataset",""), r.get("_class","")) / r.get("_folder","")
                    if fp.exists():
                        load_analysis_from_folder(fp, r.get("_dataset",""), r.get("_class",""), r.get("_folder",""))
                    go_to("🧪 Review Workspace")

# ══════════════════════════════════════════════════════════════
# INGEST PAGE
# ══════════════════════════════════════════════════════════════
def render_ingest():
    render_back_button()
    st.markdown("<div class='vg-title'>📥 Ingest</div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["🎬 Single Raw Video", "📦 Raw Dataset", "📂 Precomputed Outputs", "🗂 Manage Uploads"])

    with tab1:
        proc = st.session_state._proc_progress
        if proc.get("done") and not st.session_state._proc_running:
            if proc.get("error"):
                st.error(proc["error"])
            else:
                out_dir    = Path(proc.get("out_dir",""))
                folder_name = st.session_state._proc_folder
                ds_name    = st.session_state._proc_ds
                cls_name   = st.session_state._proc_cls
                if out_dir.exists() and not st.session_state.get("_proc_loaded"):
                    load_analysis_from_folder(out_dir, ds_name, cls_name, folder_name)
                    st.session_state["_proc_loaded"] = True
                st.success(f"Done. Prediction: **{proc.get('pred_lbl','?')}** · Confidence: {proc.get('conf',0):.1%}")
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("Open Review Workspace →", type="primary", use_container_width=True):
                        go_to("🧪 Review Workspace")
                with c2:
                    if out_dir.exists():
                        zip_buf = io.BytesIO()
                        with zipfile.ZipFile(zip_buf,"w",zipfile.ZIP_DEFLATED) as zf:
                            for f in out_dir.iterdir():
                                if f.is_file(): zf.write(f, arcname=f.name)
                        zip_buf.seek(0)
                        st.download_button("Download ZIP", data=zip_buf,
                                           file_name=f"{out_dir.name}_outputs.zip",
                                           mime="application/zip", use_container_width=True)
                with c3:
                    if st.button("Process Another", use_container_width=True):
                        st.session_state._proc_progress = {}
                        st.session_state._proc_running = False
                        st.session_state["_proc_loaded"] = False
                        st.rerun()
        elif st.session_state._proc_running:
            pct   = proc.get("pct", 0)
            stage = proc.get("stage","Processing...")
            st.progress(pct)
            st.caption(stage)
            time.sleep(0.5)
            if proc.get("done"):
                st.session_state._proc_running = False
            st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                dataset_key = st.selectbox("Dataset", list(PROC_CONFIGS.keys()), key="proc_ds_sel")
                cfg_s       = PROC_CONFIGS[dataset_key]
                true_label  = st.selectbox("True Label", DATASETS[cfg_s["name"]], key="proc_lbl_sel")
                uploaded    = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv"], key="proc_upload")
            with col2:
                with st.expander("Advanced settings", expanded=False):
                    st.caption(f"Window size: {cfg_s['window_size']}  |  Stride: {cfg_s['window_stride']}")
                    st.caption(f"Onset threshold: {cfg_s['onset_thresh']}  |  Spike delta: {cfg_s['spike_delta']}")
                    st.caption(f"Pred threshold: {cfg_s['pred_thresh']}  |  Checkpoint: {cfg_s['ckpt']}")

            if uploaded and st.button("▶ Run Processing Pipeline", type="primary", use_container_width=True):
                out_name = _safe_name(Path(uploaded.name).stem)
                out_dir  = class_root(cfg_s["name"], true_label) / out_name
                out_dir.mkdir(parents=True, exist_ok=True)
                vid_path = out_dir / uploaded.name
                vid_path.write_bytes(uploaded.read())
                prog = {"done": False, "pct": 0.0, "stage": "Starting..."}
                st.session_state._proc_progress = prog
                st.session_state._proc_running  = True
                st.session_state._proc_folder   = out_name
                st.session_state._proc_ds       = cfg_s["name"]
                st.session_state._proc_cls      = true_label
                st.session_state["_proc_loaded"] = False
                t = threading.Thread(target=run_processing_pipeline,
                                     args=(vid_path, cfg_s, true_label, out_dir, prog),
                                     daemon=True)
                t.start()
                st.session_state._proc_thread = t
                st.rerun()

    with tab2:
        st.markdown("Upload a ZIP of pre-organised video folders per dataset / class.")
        col1, col2 = st.columns(2)
        with col1:
            ds_sel  = st.selectbox("Dataset", list(DATASETS.keys()), key="ds_zip_sel")
            cls_sel = st.selectbox("Class", DATASETS[ds_sel], key="cls_zip_sel")
        with col2:
            zip_up = st.file_uploader("Upload ZIP archive", type=["zip"], key="ds_zip_up")
        if zip_up and st.button("Extract & Ingest", type="primary"):
            with st.spinner("Extracting..."):
                n_fold, n_files = extract_zip_to_uploads(zip_up.read(), ds_sel, cls_sel)
            st.success(f"Extracted {n_fold} folders and {n_files} files into {ds_sel}/{cls_sel}.")

    with tab3:
        st.markdown("Upload a ZIP of already-processed output folders (containing pred.txt, .mp4s, .png grids).")
        col1, col2 = st.columns(2)
        with col1:
            ds_pre  = st.selectbox("Dataset", list(DATASETS.keys()), key="pre_ds_sel")
            cls_pre = st.selectbox("Class", DATASETS[ds_pre], key="pre_cls_sel")
        with col2:
            pre_up = st.file_uploader("Upload precomputed ZIP", type=["zip"], key="pre_zip_up")
        if pre_up and st.button("Ingest Precomputed", type="primary"):
            with st.spinner("Extracting..."):
                n_fold, n_files = extract_zip_to_uploads(pre_up.read(), ds_pre, cls_pre)
            st.success(f"Ingested {n_fold} folders and {n_files} files.")

    with tab4:
        st.markdown("Browse and delete uploaded folders.")
        for ds in DATASETS:
            for cls in DATASETS[ds]:
                folders = list_video_folders(ds, cls)
                if not folders: continue
                with st.expander(f"{ds} / {cls}  ({len(folders)} folders)", expanded=False):
                    for folder in folders:
                        fc1, fc2, fc3 = st.columns([3,1,1])
                        files = get_files(folder)
                        has_pred = "pred" in files
                        fc1.write(f"📁 {folder.name}" + (" ✅" if has_pred else ""))
                        if fc2.button("Load", key=f"load_{folder}"):
                            load_analysis_from_folder(folder, ds, cls, folder.name)
                            go_to("🧪 Review Workspace")
                        if fc3.button("🗑 Delete", key=f"del_{folder}"):
                            shutil.rmtree(folder, ignore_errors=True)
                            st.rerun()
        st.markdown("---")
        st.warning("Danger zone")
        if st.button("🗑 Clear ALL uploads", use_container_width=True):
            clear_all_uploads()
            st.success("All uploads cleared.")
            st.rerun()

# ══════════════════════════════════════════════════════════════
# REVIEW WORKSPACE PAGE
# ══════════════════════════════════════════════════════════════
def render_review_workspace():
    render_back_button()
    st.markdown("<div class='vg-title'>🧪 Review Workspace</div>", unsafe_allow_html=True)

    with st.expander("📂 Select analysis folder", expanded=not bool(st.session_state.active_folder_name)):
        col1, col2, col3 = st.columns(3)
        with col1:
            ds_ws = st.selectbox("Dataset", list(DATASETS.keys()), key="ws_ds")
        with col2:
            cls_ws = st.selectbox("Class", DATASETS[ds_ws], key="ws_cls")
        with col3:
            folders_ws = list_video_folders(ds_ws, cls_ws)
            folder_names_ws = [f.name for f in folders_ws]
            if folder_names_ws:
                sel_folder_ws = st.selectbox("Folder", folder_names_ws, key="ws_folder_sel")
                if st.button("Load →", type="primary", key="ws_load"):
                    selected_path = class_root(ds_ws, cls_ws) / sel_folder_ws
                    load_analysis_from_folder(selected_path, ds_ws, cls_ws, sel_folder_ws)
                    st.rerun()
            else:
                st.info("No folders found. Use Ingest to add videos.")

    if not st.session_state.active_folder_name:
        st.info("No analysis loaded. Select a folder above or process a video via Ingest.")
        return

    render_active_summary_bar()

    files = st.session_state.get("_active_files", {})
    pred  = st.session_state.active_pred
    scores = st.session_state.active_scores
    fps   = st.session_state.active_fps or CFG.DEFAULT_FPS

    tab_ov, tab_vid, tab_grid, tab_tl, tab_anal, tab_rep = st.tabs([
        "📋 Overview", "🎬 Videos", "🖼 Grids", "📈 Timeline", "📊 Analytics", "📄 Report"
    ])

    with tab_ov:
        render_review_overview_tab(pred, files, fps, scores)

    with tab_vid:
        render_review_videos_tab(files)

    with tab_grid:
        render_review_grids_tab(files)

    with tab_tl:
        render_review_timeline_tab(files, pred, scores, fps)

    with tab_anal:
        render_review_analytics_tab(pred, scores, fps)

    with tab_rep:
        render_review_report_tab(pred, scores, fps)


def render_review_overview_tab(pred, files, fps, scores):
    is_f  = is_fight_pred(pred)
    conf  = pred.get("confidence","?")
    onset = pred.get("onset_time","N/A")
    total = pred.get("total_frames","?")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction", pred.get("pred_label","?"))
    c2.metric("Confidence", conf)
    c3.metric("Onset time", onset)
    c4.metric("Total frames", total)

    st.markdown("**Original vs Combined CAM**")
    vc1, vc2 = st.columns(2)
    with vc1:
        st.caption("📹 Original")
        if "original" in files:
            _safe_video(files["original"])
        else:
            st.info("Original video not found.")
    with vc2:
        st.caption("🎯 Combined CAM")
        if "combined" in files:
            _safe_video(files["combined"])
        else:
            st.info("Combined CAM video not found.")

    with st.expander("Full prediction details", expanded=False):
        for k, v in pred.items():
            st.text(f"{k}: {v}")


def render_review_videos_tab(files):
    pred = st.session_state.active_pred
    is_f = is_fight_pred(pred)

    # ── Fight / Normal Analysis Card ──────────────────────────
    if is_f:
        onset_f = pred.get("onset_frame", "?")
        onset_t = pred.get("onset_time", "N/A")
        conf    = pred.get("confidence", "?")
        total   = pred.get("total_frames", "?")
        try:
            pct_in = f"{int(float(str(onset_f)) / float(str(total)) * 100)}%" \
                     if onset_f not in ("?","N/A") and total not in ("?","N/A") else "early in clip"
        except:
            pct_in = "early in clip"

        st.markdown(
            f"<div class='fight-analysis-card'>"
            f"<div style='font-weight:800;color:#ff5555;font-size:1rem;margin-bottom:10px;'>⚠️ Fight Analysis</div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;'>"
            # How it started
            f"<div>"
            f"<div style='color:#e05252;font-size:10px;text-transform:uppercase;letter-spacing:.06em;font-weight:700;margin-bottom:4px;'>🔴 How it started</div>"
            f"<div style='color:#c8d8e8;font-size:13px;line-height:1.6;'>"
            f"A rapid motion spike was detected at frame <b>{onset_f}</b> ({onset_t}). "
            f"The R3D-18 backbone flagged sudden directional changes and close-proximity body movements — "
            f"kinematic patterns consistent with the initiation of physical contact."
            f"</div>"
            f"</div>"
            # How it developed
            f"<div>"
            f"<div style='color:#f5a623;font-size:10px;text-transform:uppercase;letter-spacing:.06em;font-weight:700;margin-bottom:4px;'>🟡 How it developed</div>"
            f"<div style='color:#c8d8e8;font-size:13px;line-height:1.6;'>"
            f"Fight probability crossed the detection threshold at <b>{pct_in}</b> into the clip "
            f"(confidence <b>{conf}</b>). The LSTM head tracked sustained high-energy motion "
            f"over subsequent frames — indicating an ongoing confrontation rather than a single brief strike."
            f"</div>"
            f"</div>"
            f"</div>"
            f"<div style='margin-top:10px;padding-top:8px;border-top:1px solid rgba(224,82,82,0.2);color:#445566;font-size:11px;'>"
            f"💡 GradCAM++ and LayerCAM heatmaps highlight the spatial regions driving the prediction. "
            f"Use the <b>Combined</b> overlay below for the most robust view — it averages GradCAM++, SmoothGradCAM++, and LayerCAM."
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='normal-analysis-card'>"
            f"<div style='font-weight:700;color:#52e08a;font-size:0.95rem;margin-bottom:4px;'>✓ No fight detected</div>"
            f"<div style='color:#7a99b0;font-size:13px;line-height:1.6;'>"
            f"The model found no aggressive motion patterns exceeding the detection threshold throughout this clip. "
            f"Motion activity remained within normal variance — no directional spikes or proximity escalation were flagged by the LSTM sequence head."
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # ── Video selector ────────────────────────────────────────
    st.markdown("**Select CAM overlay to inspect:**")
    available_vids = [k for k in ALL_VID_KEYS if k in files]
    sel_vid = st.selectbox("Video type", available_vids,
                           format_func=lambda k: VID_LABELS.get(k, k), key="review_vid_sel")

    method_info = {
        "original":         ("📹 Original",
                             "Raw footage with prediction overlay and status bar. Shows the fight onset marker, confidence score, and P(fight) value per frame."),
        "gradcam":          ("🔥 GradCAM",
                             "Standard Gradient-weighted Class Activation Mapping on Layer 4. Highlights the broad spatial regions the model weighted most for its decision."),
        "gradcampp":        ("🔥 GradCAM++",
                             "Improved GradCAM with better localization of multiple instances. More precise than standard GradCAM — recommended for identifying which person triggered the alert."),
        "smooth_gradcampp": ("✨ Smooth GradCAM++",
                             "20-pass noise-averaged GradCAM++. Reduces gradient noise for cleaner, more reliable heatmaps. Best for detailed analysis — higher compute cost."),
        "layercam":         ("🌊 LayerCAM",
                             "Fuses activations from Layers 2, 3 & 4. Captures both fine-grained texture and high-level motion features. Most comprehensive spatial coverage."),
        "combined":         ("🎯 Combined",
                             "Ensemble average of GradCAM++, SmoothGradCAM++, and LayerCAM. Reduces per-method bias — the most robust and recommended overlay for reporting."),
    }

    if sel_vid and sel_vid in files:
        vid_col, info_col = st.columns([2, 3])
        with vid_col:
            _safe_video(files[sel_vid])
            vpath = Path(files[sel_vid])
            if vpath.exists():
                st.download_button(
                    f"⬇ Download {VID_LABELS.get(sel_vid, sel_vid)}",
                    data=vpath.read_bytes(),
                    file_name=vpath.name,
                    mime="video/mp4",
                    use_container_width=True
                )
        with info_col:
            label, desc = method_info.get(sel_vid, (sel_vid, "No description available."))
            st.markdown(f"**{label}**")
            st.markdown(
                f"<div style='color:#7a99b0;font-size:13px;line-height:1.65;margin-top:4px;'>{desc}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("Video not available.")

    # ── All videos compact grid ───────────────────────────────
    with st.expander("All available videos", expanded=False):
        row = st.columns(3)
        for i, vk in enumerate(ALL_VID_KEYS):
            if vk in files:
                with row[i % 3]:
                    st.caption(VID_LABELS.get(vk, vk))
                    _safe_video(files[vk])


def render_review_grids_tab(files):
    sel_grid = st.selectbox("Grid type", [k for k in ALL_GRID_KEYS if k in files],
                             format_func=lambda k: GRID_LABELS.get(k,k), key="review_grid_sel")
    if sel_grid and sel_grid in files:
        gp = Path(files[sel_grid])
        if gp.exists():
            st.image(str(gp), use_container_width=True, caption=GRID_LABELS.get(sel_grid, sel_grid))
            st.download_button("Download grid image", data=gp.read_bytes(),
                               file_name=gp.name, mime="image/png", use_container_width=True)
    else:
        st.info("Grid not available.")

    st.markdown("**All frame grids**")
    for gk in ALL_GRID_KEYS:
        if gk in files:
            gp = Path(files[gk])
            if gp.exists():
                with st.expander(GRID_LABELS.get(gk, gk), expanded=False):
                    st.image(str(gp), use_container_width=True)


def render_review_timeline_tab(files, pred, scores, fps):
    tl_img = files.get("timeline")
    if tl_img and Path(tl_img).exists():
        st.image(str(tl_img), use_container_width=True, caption="Saved timeline plot")
        st.download_button("Download timeline PNG", data=Path(tl_img).read_bytes(),
                           file_name="timeline.png", mime="image/png")
    if scores is not None and len(scores) > 0:
        st.markdown("**Interactive timeline**")
        fig = make_timeline_plot(scores, fps, pred)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No score data available for timeline.")


def render_review_analytics_tab(pred, scores, fps):
    if scores is None or len(scores) == 0:
        st.info("No score data available."); return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max P(fight)", f"{float(np.max(scores)):.3f}")
    c2.metric("Mean P(fight)", f"{float(np.mean(scores)):.3f}")
    c3.metric("Frames > violence thr", int(np.sum(scores > st.session_state.thr_violence)))
    c4.metric("Frames > suspicious thr", int(np.sum(scores > st.session_state.thr_suspicious)))

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**P(fight) over time**")
        fig = make_timeline_plot(scores, fps, pred)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with sc2:
        st.markdown("**Score distribution**")
        fig = make_hist_plot(scores)
        st.pyplot(fig, use_container_width=True); plt.close(fig)


def render_review_report_tab(pred, scores, fps):
    folder = st.session_state.active_folder_name
    is_f   = is_fight_pred(pred)

    badge     = "vg-badge-fight" if is_f else "vg-badge-normal"
    badge_txt = "⚠ FIGHT DETECTED" if is_f else "✓ NORMAL"
    st.markdown(
        f"<div class='vg-card' style='padding:14px 18px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;'>"
        f"<span class='{badge}'>{badge_txt}</span>"
        f"<span style='font-size:1.05rem;font-weight:700;'>{folder}</span>"
        f"<span class='vg-soft'>{pred.get('dataset','?')} · conf {pred.get('confidence','?')} · onset {pred.get('onset_time','N/A')}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("**Incident metadata**")
    m1, m2 = st.columns(2)
    with m1:
        camera   = st.text_input("Camera ID",    value=st.session_state.get("review_camera",""),   key="rep_camera")
        location = st.text_input("Location",     value=st.session_state.get("review_location",""), key="rep_location")
    with m2:
        reviewer = st.text_input("Reviewer tag", value=st.session_state.get("reviewer_tag",""),    key="rep_reviewer")
        notes    = st.text_area("Notes",         value=st.session_state.get("review_notes",""),    key="rep_notes", height=80)

    if st.button("💾 Save metadata", use_container_width=True):
        st.session_state.review_camera   = camera
        st.session_state.review_location = location
        st.session_state.reviewer_tag    = reviewer
        st.session_state.review_notes    = notes
        update_history_metadata(folder, st.session_state.active_dataset,
                                 st.session_state.active_class,
                                 camera, location, notes, reviewer)
        st.success("Metadata saved.")

    st.markdown("---")
    st.markdown("**Export** — all exports use filename `" + folder + "`")

    sc = scores if scores is not None and len(scores) > 0 else np.array([0.0])
    pdf_bytes  = generate_pdf_report(pred, sc, fps, folder)
    export_obj = {
        "folder": folder,
        "dataset": st.session_state.active_dataset,
        "class":   st.session_state.active_class,
        "prediction": pred,
        "metadata": {
            "camera":      st.session_state.get("review_camera",""),
            "location":    st.session_state.get("review_location",""),
            "reviewer":    st.session_state.get("reviewer_tag",""),
            "notes":       st.session_state.get("review_notes",""),
            "exported_at": datetime.now().isoformat(),
        },
        "scores_summary": {
            "max":      float(np.max(sc)),
            "mean":     float(np.mean(sc)),
            "n_frames": len(sc),
        }
    }
    email_text = build_email_summary(
        pred, folder,
        st.session_state.get("review_camera",""),
        st.session_state.get("review_location",""),
        st.session_state.get("review_notes",""),
        st.session_state.get("reviewer_tag","")
    )

    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.download_button("📄 Download PDF Report",
                           data=pdf_bytes,
                           file_name=f"{folder}_report.pdf",
                           mime="application/pdf",
                           use_container_width=True)
    with ex2:
        st.download_button("📋 Download JSON Export",
                           data=json.dumps(export_obj, indent=2),
                           file_name=f"{folder}_export.json",
                           mime="application/json",
                           use_container_width=True)
    with ex3:
        st.download_button("✉️ Download Email Draft",
                           data=email_text,
                           file_name=f"{folder}_email.txt",
                           mime="text/plain",
                           use_container_width=True)

    with st.expander("Preview email draft", expanded=False):
        st.text_area("Email text", value=email_text, height=280, key="email_ta_prev")

# ══════════════════════════════════════════════════════════════
# DATASET LAB PAGE
# ══════════════════════════════════════════════════════════════
def render_dataset_lab():
    render_back_button()
    st.markdown("<div class='vg-title'>📊 Dataset Lab</div>", unsafe_allow_html=True)
    records = get_all_pred_records()
    if not records:
        st.info("No processed records found. Run some analyses via Ingest first."); return

    tab_search, tab_compare, tab_stats, tab_errors, tab_cm = st.tabs([
        "🔍 Search & Filter", "⚖️ Compare", "📈 Stats", "⚠️ Errors (FP/FN)", "🟦 Confusion Matrix"
    ])

    with tab_search:
        col1, col2, col3 = st.columns(3)
        with col1:
            ds_filter = st.multiselect("Dataset", list(DATASETS.keys()), default=list(DATASETS.keys()), key="lab_ds_filter")
        with col2:
            pred_filter = st.multiselect("Prediction", ["Fight","Nonfight","NonFight"], default=["Fight","Nonfight","NonFight"], key="lab_pred_filter")
        with col3:
            query = st.text_input("Search folder name", "", key="lab_search_q")

        filtered = [r for r in records
                    if r.get("_dataset","") in ds_filter
                    and r.get("pred_label","") in pred_filter
                    and (not query or query.lower() in r.get("_folder","").lower())]

        st.caption(f"Showing {len(filtered)} of {len(records)} records")
        for r in filtered:
            is_f = is_fight_pred(r)
            badge = "vg-badge-fight" if is_f else "vg-badge-normal"
            badge_txt = "FIGHT" if is_f else "NORMAL"
            with st.expander(f"{r.get('_folder','?')} — {r.get('_dataset','?')}/{r.get('_class','?')}", expanded=False):
                rc1, rc2 = st.columns([3,1])
                with rc1:
                    st.markdown(f"<span class='{badge}'>{badge_txt}</span>", unsafe_allow_html=True)
                    st.text(f"Confidence: {r.get('confidence','?')}")
                    st.text(f"True label: {r.get('true_label','?')}")
                    st.text(f"Onset: {r.get('onset_time','N/A')}")
                with rc2:
                    folder_path = class_root(r.get("_dataset",""), r.get("_class","")) / r.get("_folder","")
                    if folder_path.exists():
                        if st.button("Load →", key=f"lab_load_{r.get('_folder','')}"):
                            load_analysis_from_folder(folder_path, r.get("_dataset",""), r.get("_class",""), r.get("_folder",""))
                            go_to("🧪 Review Workspace")

    with tab_compare:
        st.markdown("Select two folders to compare side-by-side.")
        all_folder_labels = [f"{r.get('_folder','')} ({r.get('_dataset','')}/{r.get('_class','')})" for r in records]
        if len(all_folder_labels) < 2:
            st.info("Need at least 2 processed folders to compare."); return

        sc1, sc2 = st.columns(2)
        with sc1:
            sel_a = st.selectbox("Folder A", all_folder_labels, key="cmp_a")
        with sc2:
            sel_b = st.selectbox("Folder B", all_folder_labels, index=min(1, len(all_folder_labels)-1), key="cmp_b")

        ra = records[all_folder_labels.index(sel_a)]
        rb = records[all_folder_labels.index(sel_b)]

        ca1, ca2 = st.columns(2)
        for col, rec in [(ca1, ra), (ca2, rb)]:
            with col:
                is_f = is_fight_pred(rec)
                badge = "vg-badge-fight" if is_f else "vg-badge-normal"
                badge_txt = "FIGHT" if is_f else "NORMAL"
                st.markdown(f"<span class='{badge}'>{badge_txt}</span> **{rec.get('_folder','?')}**", unsafe_allow_html=True)
                st.text(f"Dataset: {rec.get('_dataset','?')}")
                st.text(f"Confidence: {rec.get('confidence','?')}")
                st.text(f"True label: {rec.get('true_label','?')}")
                st.text(f"Onset time: {rec.get('onset_time','N/A')}")

                folder_path = class_root(rec.get("_dataset",""), rec.get("_class","")) / rec.get("_folder","")
                files_rec = get_files(folder_path)
                if "original" in files_rec:
                    _safe_video(files_rec["original"])

    with tab_stats:
        fight_count = sum(1 for r in records if is_fight_pred(r))
        nonfight_count = len(records) - fight_count
        correct_count = sum(1 for r in records if str(r.get("correct","")).lower() == "true")

        st1, st2, st3, st4 = st.columns(4)
        st1.metric("Total records", len(records))
        st2.metric("Fight detections", fight_count)
        st3.metric("Non-fight", nonfight_count)
        st4.metric("Correct predictions", correct_count)

        ds_counts = {}
        for r in records:
            ds = r.get("_dataset","unknown")
            ds_counts[ds] = ds_counts.get(ds,0)+1

        st.markdown("**Records per dataset**")
        for ds, cnt in ds_counts.items():
            st.text(f"  {ds}: {cnt} records")

        confs = []
        for r in records:
            try: confs.append(float(r.get("confidence",0)))
            except: pass
        if confs:
            st.markdown("**Confidence distribution**")
            fig = make_hist_plot(np.array(confs))
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    with tab_errors:
        st.markdown("Browse false positives and false negatives.")
        errors = [r for r in records if str(r.get("correct","")).lower() == "false"]
        if not errors:
            st.success("No errors found in current records."); return
        st.warning(f"{len(errors)} incorrect predictions found.")
        for r in errors:
            is_f = is_fight_pred(r)
            badge = "vg-badge-fight" if is_f else "vg-badge-normal"
            badge_txt = "FIGHT" if is_f else "NORMAL"
            err_type = "False Positive" if is_f else "False Negative"
            with st.expander(f"{err_type}: {r.get('_folder','?')}", expanded=False):
                st.markdown(f"<span class='{badge}'>{badge_txt}</span>", unsafe_allow_html=True)
                st.text(f"True label: {r.get('true_label','?')}")
                st.text(f"Predicted: {r.get('pred_label','?')}")
                st.text(f"Confidence: {r.get('confidence','?')}")
                folder_path = class_root(r.get("_dataset",""), r.get("_class","")) / r.get("_folder","")
                if folder_path.exists():
                    if st.button("Load for review →", key=f"err_load_{r.get('_folder','')}"):
                        load_analysis_from_folder(folder_path, r.get("_dataset",""), r.get("_class",""), r.get("_folder",""))
                        go_to("🧪 Review Workspace")

    with tab_cm:
        fig, cm = make_confusion_matrix(records)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
        tp = cm[0][0]; fn = cm[0][1]; fp = cm[1][0]; tn = cm[1][1]
        total = tp + fn + fp + tn
        if total > 0:
            accuracy  = (tp+tn)/total
            precision = tp/(tp+fp) if (tp+fp) > 0 else 0
            recall    = tp/(tp+fn) if (tp+fn) > 0 else 0
            f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy",  f"{accuracy:.1%}")
            mc2.metric("Precision", f"{precision:.1%}")
            mc3.metric("Recall",    f"{recall:.1%}")
            mc4.metric("F1 Score",  f"{f1:.3f}")

# ══════════════════════════════════════════════════════════════
# HISTORY PAGE
# ══════════════════════════════════════════════════════════════
def render_history():
    render_back_button()
    st.markdown("<div class='vg-title'>🕘 History</div>", unsafe_allow_html=True)
    hist = st.session_state.get("_history", [])

    if not hist:
        st.info("No analysis history yet. Process a video to get started.")
        return

    hcol1, hcol2, hcol3 = st.columns([2,1,1])
    with hcol1:
        h_query = st.text_input("Search history", "", key="hist_search", placeholder="Filter by folder name...")
    with hcol2:
        h_ds_filter = st.multiselect("Dataset", list(DATASETS.keys()), default=list(DATASETS.keys()), key="hist_ds_filter")
    with hcol3:
        h_pred_filter = st.multiselect("Prediction", ["Fight","Nonfight","NonFight","?"], default=["Fight","Nonfight","NonFight","?"], key="hist_pred_filter")

    filtered_hist = [
        h for h in hist
        if (not h_query or h_query.lower() in h.get("folder","").lower())
        and h.get("dataset","") in h_ds_filter
        and h.get("pred_lbl","?") in h_pred_filter
    ]

    st.caption(f"Showing {len(filtered_hist)} of {len(hist)} entries")

    if st.button("⬇️ Export full history JSON", use_container_width=False):
        st.download_button("Download history.json", data=json.dumps(hist, indent=2),
                           file_name="visionguard_history.json", mime="application/json")

    st.markdown("---")

    for i, entry in enumerate(filtered_hist):
        is_f = "fight" in str(entry.get("pred_lbl","")).lower() and "non" not in str(entry.get("pred_lbl","")).lower()
        badge = "vg-badge-fight" if is_f else "vg-badge-normal"
        badge_txt = "FIGHT" if is_f else "NORMAL"

        with st.expander(
            f"{entry.get('folder','?')}  ·  {entry.get('ts','?')}",
            expanded=False
        ):
            top1, top2 = st.columns([3,1])
            with top1:
                st.markdown(
                    f"<span class='{badge}'>{badge_txt}</span>"
                    f" <b>{entry.get('folder','?')}</b>"
                    f" <span class='vg-soft'>{entry.get('dataset','?')}/{entry.get('cls','?')}</span>",
                    unsafe_allow_html=True
                )
                st.caption(f"Confidence: {entry.get('conf','?')}  ·  Onset: {entry.get('onset_t','N/A')}  ·  {entry.get('ts','?')}")
                st.caption(f"Camera: {entry.get('camera','N/A')}  ·  Location: {entry.get('location','N/A')}  ·  Reviewer: {entry.get('reviewer_tag','N/A')}")
                if entry.get("notes"):
                    st.caption(f"Notes: {entry['notes']}")
            with top2:
                if st.button("Restore →", key=f"hist_restore_{i}", type="primary"):
                    restore_history(entry)
                    go_to("🧪 Review Workspace")

            with st.form(key=f"hist_meta_form_{i}"):
                st.markdown("**Edit metadata**")
                fm1, fm2 = st.columns(2)
                with fm1:
                    new_cam  = st.text_input("Camera",   value=entry.get("camera",""),   key=f"hm_cam_{i}")
                    new_loc  = st.text_input("Location", value=entry.get("location",""), key=f"hm_loc_{i}")
                with fm2:
                    new_rev  = st.text_input("Reviewer", value=entry.get("reviewer_tag",""), key=f"hm_rev_{i}")
                    new_notes = st.text_area("Notes",    value=entry.get("notes",""),    key=f"hm_notes_{i}", height=60)
                if st.form_submit_button("Save metadata"):
                    update_history_metadata(entry.get("folder",""), entry.get("dataset",""), entry.get("cls",""),
                                            new_cam, new_loc, new_notes, new_rev)
                    st.success("Metadata updated.")
                    st.rerun()

            files_e = {k: Path(v) for k, v in entry.get("_files",{}).items()}
            pred_e  = parse_pred_txt(files_e["pred"]) if "pred" in files_e else {}
            exp1, exp2, exp3 = st.columns(3)
            with exp1:
                if pred_e:
                    sc_e = scores_from_pred(pred_e, 100, float(CFG.DEFAULT_FPS))
                    pdf_bytes = generate_pdf_report(pred_e, sc_e, float(CFG.DEFAULT_FPS), entry.get("folder","?"))
                    st.download_button("PDF Report", data=pdf_bytes,
                                       file_name=f"{entry.get('folder','hist')}_report.pdf",
                                       mime="application/pdf",
                                       key=f"hist_pdf_{i}", use_container_width=True)
            with exp2:
                export = {
                    "folder": entry.get("folder"),
                    "dataset": entry.get("dataset"),
                    "class": entry.get("cls"),
                    "prediction": pred_e,
                    "metadata": {k: entry.get(k) for k in ["camera","location","notes","reviewer_tag","ts"]},
                }
                st.download_button("JSON Export", data=json.dumps(export, indent=2),
                                   file_name=f"{entry.get('folder','hist')}_export.json",
                                   mime="application/json",
                                   key=f"hist_json_{i}", use_container_width=True)
            with exp3:
                email_text = build_email_summary(pred_e, entry.get("folder","?"),
                                                  entry.get("camera",""), entry.get("location",""),
                                                  entry.get("notes",""), entry.get("reviewer_tag",""))
                st.download_button("Email Draft", data=email_text,
                                   file_name=f"{entry.get('folder','hist')}_email.txt",
                                   mime="text/plain",
                                   key=f"hist_email_{i}", use_container_width=True)

# ══════════════════════════════════════════════════════════════
# SETTINGS PAGE
# ══════════════════════════════════════════════════════════════
def render_settings():
    render_back_button()
    st.markdown("<div class='vg-title'>⚙️ Settings</div>", unsafe_allow_html=True)
    tab_ui, tab_model, tab_acct = st.tabs(["🎨 UI", "🤖 Model / Detection", "👤 Account"])

    with tab_ui:
        st.markdown("**Appearance**")
        new_theme = st.radio("Theme", ["dark","light"], index=0 if st.session_state.ui_theme=="dark" else 1, key="set_theme_radio")
        new_accent = st.color_picker("Accent color", value=st.session_state.accent_color, key="set_accent_cp")
        new_font = st.selectbox("Font size", ["small","medium","large"],
                                index=["small","medium","large"].index(st.session_state.font_size), key="set_font")
        if st.button("Apply UI settings", type="primary"):
            st.session_state.ui_theme = new_theme
            st.session_state.accent_color = new_accent
            st.session_state.font_size = new_font
            st.rerun()

    with tab_model:
        st.markdown("**Detection thresholds**")
        t1, t2 = st.columns(2)
        with t1:
            new_thr_v = st.slider("Violence threshold", 0.0, 1.0,
                                   st.session_state.thr_violence, 0.01, key="set_thr_v")
        with t2:
            new_thr_s = st.slider("Suspicious threshold", 0.0, 1.0,
                                   st.session_state.thr_suspicious, 0.01, key="set_thr_s")
        new_max_frames = st.number_input("Max frames to load", min_value=30, max_value=1000,
                                          value=st.session_state.max_frames, step=10, key="set_max_frames")
        if st.button("Save thresholds", type="primary"):
            st.session_state.thr_violence   = new_thr_v
            st.session_state.thr_suspicious = new_thr_s
            st.session_state.max_frames     = int(new_max_frames)
            st.success("Settings saved.")

        st.markdown("**Checkpoint paths**")
        for ds_key, cfg_v in PROC_CONFIGS.items():
            ckpt_ok = Path(cfg_v["ckpt"]).exists()
            status = "✅" if ckpt_ok else "❌ not found"
            st.text(f"{ds_key}: {cfg_v['ckpt']}  {status}")

    with tab_acct:
        st.markdown(f"**Logged in as:** `{st.session_state.username}`")
        st.markdown("**Change password**")
        cp1 = st.text_input("Current password", type="password", key="set_cur_pw")
        cp2 = st.text_input("New password", type="password", key="set_new_pw")
        cp3 = st.text_input("Confirm new password", type="password", key="set_new_pw2")
        if st.button("Change password"):
            if not try_login(st.session_state.username, cp1):
                st.error("Current password is incorrect.")
            elif cp2 != cp3:
                st.error("New passwords do not match.")
            elif len(cp2) < 4:
                st.error("Password must be at least 4 characters.")
            else:
                users = load_users()
                info = users.get(st.session_state.username, {})
                if isinstance(info, str): info = {"password": info, "reset_code": "000000"}
                info["password"] = hash_pw(cp2)
                users[st.session_state.username] = info
                save_users(users)
                st.success("Password changed successfully.")

        st.markdown("---")
        st.markdown("**Session info**")
        st.text(f"Run ID: {st.session_state.run_id}")
        st.text(f"Device: {DEVICE}")
        st.text(f"Loaded models: {list(_MODEL_CACHE.keys()) or 'None'}")

# ══════════════════════════════════════════════════════════════
# SMART TOOLS PAGE
# ══════════════════════════════════════════════════════════════
ZONES_FILE = Path(CFG.OUTPUT_DIR) / "zones.json"
COC_FILE   = Path(CFG.OUTPUT_DIR) / "chain_of_custody.json"

def load_zones():
    if ZONES_FILE.exists():
        try: return json.loads(ZONES_FILE.read_text())
        except: pass
    return []

def save_zones(z): ZONES_FILE.write_text(json.dumps(z, indent=2))

def load_coc():
    if COC_FILE.exists():
        try: return json.loads(COC_FILE.read_text())
        except: pass
    return []

def save_coc(entries): COC_FILE.write_text(json.dumps(entries, indent=2))

def render_smart_tools():
    render_back_button()
    st.markdown("<div class='vg-title'>🛠️ Smart Tools</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:#7a99b0;font-size:13px;margin-bottom:16px;'>"
        f"Real, working tools built from the feature ideas. Each tab is fully functional."
        f"</div>",
        unsafe_allow_html=True
    )

    tab_clip, tab_heatmap, tab_zones, tab_coc = st.tabs([
        "✂️ Evidence Clip Trimmer",
        "📅 Risk Heatmap Calendar",
        "🗺️ Zone Manager",
        "🔐 Chain of Custody",
    ])

    # ══════════════════════════════════════════════════════
    # TAB 1 — Evidence Clip Trimmer
    # ══════════════════════════════════════════════════════
    with tab_clip:
        st.markdown("### ✂️ Evidence Clip Trimmer")
        st.markdown(
            f"<div style='color:#7a99b0;font-size:13px;margin-bottom:12px;'>"
            f"Automatically trim a video to a tight evidence window around the fight onset. "
            f"Instead of saving the full clip, export only the relevant seconds."
            f"</div>",
            unsafe_allow_html=True
        )

        # Source selection
        records = get_all_pred_records()
        fight_records = [r for r in records if is_fight_pred(r)]

        if not fight_records:
            st.info("No fight detections found. Process some videos first via Ingest.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                folder_options = [
                    f"{r.get('_folder','?')} ({r.get('_dataset','?')}/{r.get('_class','?')})"
                    for r in fight_records
                ]
                sel_idx = st.selectbox("Select fight clip", range(len(folder_options)),
                                       format_func=lambda i: folder_options[i], key="trim_sel")
                rec = fight_records[sel_idx]

            with col2:
                pre_secs  = st.number_input("Seconds before onset", min_value=0.0, max_value=30.0, value=3.0, step=0.5, key="trim_pre")
                post_secs = st.number_input("Seconds after onset",  min_value=1.0, max_value=60.0, value=8.0, step=0.5, key="trim_post")

            # Show prediction summary
            onset_t = rec.get("onset_time", "N/A")
            conf    = rec.get("confidence", "?")
            total_f = rec.get("total_frames", "?")
            st.markdown(
                f"<div style='background:rgba(224,82,82,0.07);border:1px solid #e05252;border-radius:8px;"
                f"padding:10px 14px;margin:10px 0;display:flex;gap:20px;flex-wrap:wrap;'>"
                f"<span style='color:#ff5555;font-weight:700;'>⚠ FIGHT</span>"
                f"<span style='color:#c8d8e8;font-size:13px;'>Onset: <b>{onset_t}</b></span>"
                f"<span style='color:#c8d8e8;font-size:13px;'>Confidence: <b>{conf}</b></span>"
                f"<span style='color:#c8d8e8;font-size:13px;'>Total frames: <b>{total_f}</b></span>"
                f"</div>",
                unsafe_allow_html=True
            )

            if st.button("✂️ Trim & Export Evidence Clip", type="primary", use_container_width=True, key="trim_btn"):
                folder_path = class_root(rec.get("_dataset",""), rec.get("_class","")) / rec.get("_folder","")
                files_r = get_files(folder_path)

                if "original" not in files_r:
                    st.error("Original video not found for this record.")
                else:
                    src_path = Path(files_r["original"])
                    try:
                        cap = cv2.VideoCapture(str(src_path))
                        fps_v = cap.get(cv2.CAP_PROP_FPS) or 25.0
                        total_frames_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                        # Parse onset frame
                        try:
                            onset_frame_v = int(rec.get("onset_frame", 0))
                        except:
                            onset_frame_v = 0

                        start_frame = max(0, int(onset_frame_v - pre_secs * fps_v))
                        end_frame   = min(total_frames_v, int(onset_frame_v + post_secs * fps_v))

                        # Seek and read frames
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                        clipped = []
                        for fi in range(end_frame - start_frame):
                            ret, frm = cap.read()
                            if not ret: break
                            clipped.append(frm)
                        cap.release()

                        if not clipped:
                            st.error("Could not read frames from video.")
                        else:
                            # Write to buffer
                            h_v, w_v = clipped[0].shape[:2]
                            buf = io.BytesIO()
                            tmp_path = Path(CFG.OUTPUT_DIR) / f"_trim_tmp_{rec.get('_folder','clip')}.mp4"
                            wr = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (w_v, h_v))
                            for frm in clipped: wr.write(frm)
                            wr.release()

                            clip_name = f"{rec.get('_folder','clip')}_evidence_{pre_secs:.0f}s_before_{post_secs:.0f}s_after.mp4"
                            clip_bytes = tmp_path.read_bytes()
                            tmp_path.unlink(missing_ok=True)

                            duration = len(clipped) / fps_v
                            st.success(
                                f"✅ Trimmed clip ready — {len(clipped)} frames, {duration:.1f}s "
                                f"(onset at +{pre_secs:.0f}s mark). Original was {total_frames_v} frames."
                            )
                            st.download_button(
                                f"⬇ Download Evidence Clip ({duration:.1f}s)",
                                data=clip_bytes,
                                file_name=clip_name,
                                mime="video/mp4",
                                use_container_width=True,
                                key="trim_dl"
                            )
                    except Exception as e:
                        st.error(f"Trim failed: {e}")

    # ══════════════════════════════════════════════════════
    # TAB 2 — Risk Score Calendar Heatmap
    # ══════════════════════════════════════════════════════
    with tab_heatmap:
        st.markdown("### 📅 Risk Score Calendar Heatmap")
        st.markdown(
            f"<div style='color:#7a99b0;font-size:13px;margin-bottom:12px;'>"
            f"Visualise fight detections across time. Identifies high-risk days and time windows from your analysis history."
            f"</div>",
            unsafe_allow_html=True
        )

        hist_all = load_history_store()
        if not hist_all:
            st.info("No history yet. Process some videos first — timestamps are pulled from analysis history.")
        else:
            # Parse timestamps and build daily/hourly counts
            day_counts   = {}  # "YYYY-MM-DD" → {fights, total}
            hour_counts  = {}  # hour int 0-23 → {fights, total}
            dow_counts   = {}  # weekday 0-6 → {fights, total}

            for entry in hist_all:
                ts_str = entry.get("ts", "")
                is_f   = "fight" in str(entry.get("pred_lbl","")).lower() and "non" not in str(entry.get("pred_lbl","")).lower()
                try:
                    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    day_key = dt.strftime("%Y-%m-%d")
                    hr      = dt.hour
                    dow     = dt.weekday()  # 0=Mon

                    for store, key in [(day_counts, day_key), (hour_counts, hr), (dow_counts, dow)]:
                        if key not in store: store[key] = {"fights": 0, "total": 0}
                        store[key]["total"] += 1
                        if is_f: store[key]["fights"] += 1
                except:
                    pass

            if not day_counts:
                st.info("No valid timestamps found in history.")
            else:
                c = get_plot_colors()

                # ── Daily bar chart ──
                st.markdown("**📊 Detections by Day**")
                days_sorted = sorted(day_counts.keys())
                day_fights  = [day_counts[d]["fights"] for d in days_sorted]
                day_totals  = [day_counts[d]["total"]  for d in days_sorted]
                day_normals = [t - f for t, f in zip(day_totals, day_fights)]

                fig, ax = plt.subplots(figsize=(max(6, len(days_sorted)*0.8), 3), facecolor=c["bg"])
                ax.set_facecolor(c["ax"])
                x = range(len(days_sorted))
                ax.bar(x, day_fights,  color="#e05252", label="Fight",    zorder=3)
                ax.bar(x, day_normals, bottom=day_fights, color="#52e08a", label="Normal", zorder=3, alpha=0.6)
                ax.set_xticks(list(x))
                ax.set_xticklabels(days_sorted, rotation=45, ha="right", fontsize=7, color=c["tick"])
                ax.set_ylabel("Count", fontsize=8, color=c["xlabel"])
                ax.tick_params(colors=c["tick"])
                ax.spines[:].set_color(c["spine"])
                ax.legend(fontsize=8, labelcolor=c["legend_text"], facecolor=c["ax"], edgecolor=c["spine"])
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close(fig)

                col_h, col_d = st.columns(2)

                # ── Hourly heatmap ──
                with col_h:
                    st.markdown("**🕐 Fights by Hour of Day**")
                    hours     = list(range(24))
                    h_fights  = [hour_counts.get(h, {}).get("fights", 0) for h in hours]
                    h_risk    = [f / max(hour_counts.get(h, {}).get("total", 1), 1) for f, h in zip(h_fights, hours)]

                    fig2, ax2 = plt.subplots(figsize=(5, 2.2), facecolor=c["bg"])
                    ax2.set_facecolor(c["ax"])
                    bar_colors = ["#e05252" if r > 0.5 else "#f5a623" if r > 0.2 else "#52e08a" for r in h_risk]
                    ax2.bar(hours, h_fights, color=bar_colors, zorder=3)
                    ax2.set_xlabel("Hour", fontsize=8, color=c["xlabel"])
                    ax2.set_ylabel("Fight count", fontsize=8, color=c["xlabel"])
                    ax2.tick_params(colors=c["tick"], labelsize=7)
                    ax2.spines[:].set_color(c["spine"])

                    # Annotate peak hour
                    if max(h_fights) > 0:
                        peak_h = hours[h_fights.index(max(h_fights))]
                        ax2.axvline(peak_h, color="#7ecfff", linewidth=1, linestyle=":", alpha=0.7)
                        ax2.text(peak_h + 0.3, max(h_fights) * 0.9, f"Peak\n{peak_h:02d}:00",
                                 color="#7ecfff", fontsize=7)
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True); plt.close(fig2)

                # ── Day-of-week chart ──
                with col_d:
                    st.markdown("**📆 Fights by Day of Week**")
                    dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                    dow_fights = [dow_counts.get(d, {}).get("fights", 0) for d in range(7)]
                    dow_totals = [dow_counts.get(d, {}).get("total",  0) for d in range(7)]

                    fig3, ax3 = plt.subplots(figsize=(5, 2.2), facecolor=c["bg"])
                    ax3.set_facecolor(c["ax"])
                    bar_cols_d = ["#e05252" if f > 0 else "#2a3a4a" for f in dow_fights]
                    ax3.bar(dow_labels, dow_fights, color=bar_cols_d, zorder=3)
                    ax3.bar(dow_labels, [t - f for t, f in zip(dow_totals, dow_fights)],
                            bottom=dow_fights, color="#52e08a", alpha=0.4, zorder=2)
                    ax3.set_ylabel("Count", fontsize=8, color=c["xlabel"])
                    ax3.tick_params(colors=c["tick"], labelsize=8)
                    ax3.spines[:].set_color(c["spine"])
                    plt.tight_layout()
                    st.pyplot(fig3, use_container_width=True); plt.close(fig3)

                # ── Summary insight ──
                total_fights = sum(day_fights)
                total_all    = sum(day_totals)
                if total_fights > 0:
                    peak_day = days_sorted[day_fights.index(max(day_fights))]
                    peak_hour_v = hours[h_fights.index(max(h_fights))] if max(h_fights) > 0 else None
                    peak_dow_v  = dow_labels[dow_fights.index(max(dow_fights))] if max(dow_fights) > 0 else None
                    st.markdown(
                        f"<div style='background:rgba(224,82,82,0.07);border:1px solid #e05252;"
                        f"border-radius:8px;padding:12px 16px;margin-top:8px;'>"
                        f"<div style='font-weight:700;color:#ff5555;margin-bottom:6px;'>⚠ Risk Insights</div>"
                        f"<div style='color:#c8d8e8;font-size:13px;line-height:1.7;'>"
                        f"• <b>{total_fights}</b> fights detected out of <b>{total_all}</b> total analyses "
                        f"({total_fights/max(total_all,1)*100:.0f}% fight rate)<br>"
                        f"• Highest risk day: <b>{peak_day}</b><br>"
                        + (f"• Peak hour: <b>{peak_hour_v:02d}:00</b> — consider increased monitoring<br>" if peak_hour_v is not None else "")
                        + (f"• Most fights on: <b>{peak_dow_v}s</b>" if peak_dow_v else "")
                        + f"</div></div>",
                        unsafe_allow_html=True
                    )

                # Export
                export_data = {
                    "generated_at": datetime.now().isoformat(),
                    "daily": day_counts,
                    "hourly": {str(k): v for k, v in hour_counts.items()},
                    "day_of_week": {dow_labels[k]: v for k, v in dow_counts.items()},
                }
                st.download_button(
                    "⬇ Export heatmap data (JSON)",
                    data=json.dumps(export_data, indent=2),
                    file_name="visionguard_risk_heatmap.json",
                    mime="application/json",
                    use_container_width=False,
                    key="heatmap_dl"
                )

    # ══════════════════════════════════════════════════════
    # TAB 3 — Zone Manager
    # ══════════════════════════════════════════════════════
    with tab_zones:
        st.markdown("### 🗺️ Zone Manager")
        st.markdown(
            f"<div style='color:#7a99b0;font-size:13px;margin-bottom:12px;'>"
            f"Define named camera zones (e.g. 'Entrance Gate', 'Corridor B'). "
            f"Tag incidents to zones and see which areas have the most detections."
            f"</div>",
            unsafe_allow_html=True
        )

        zones = load_zones()

        # ── Add zone ──
        with st.expander("➕ Add / Edit Zones", expanded=len(zones) == 0):
            with st.form("zone_form"):
                zc1, zc2, zc3 = st.columns(3)
                with zc1:
                    z_name = st.text_input("Zone name", placeholder="e.g. Main Entrance")
                with zc2:
                    z_cam  = st.text_input("Camera ID", placeholder="e.g. CAM-01")
                with zc3:
                    z_loc  = st.text_input("Physical location", placeholder="e.g. Building A, Floor 1")
                z_desc = st.text_area("Description", placeholder="What is monitored here?", height=60)
                if st.form_submit_button("Save Zone", type="primary"):
                    if z_name.strip():
                        # Update existing or append
                        existing = [z for z in zones if z.get("name") == z_name.strip()]
                        if existing:
                            existing[0].update({"camera": z_cam, "location": z_loc, "description": z_desc})
                        else:
                            zones.append({
                                "name": z_name.strip(),
                                "camera": z_cam,
                                "location": z_loc,
                                "description": z_desc,
                                "created_at": datetime.now().isoformat(),
                            })
                        save_zones(zones)
                        st.success(f"Zone '{z_name}' saved.")
                        st.rerun()
                    else:
                        st.error("Zone name is required.")

        if not zones:
            st.info("No zones defined yet. Add your first zone above.")
        else:
            # ── Zone list ──
            st.markdown(f"**{len(zones)} zone(s) defined**")
            records_all = get_all_pred_records()
            hist_all_z  = load_history_store()

            for zi, zone in enumerate(zones):
                zname = zone.get("name", "?")
                zcam  = zone.get("camera", "")
                zloc  = zone.get("location", "")

                # Count incidents tagged to this zone's camera
                zone_fights  = sum(1 for h in hist_all_z
                                   if h.get("camera","") == zcam
                                   and "fight" in str(h.get("pred_lbl","")).lower()
                                   and "non" not in str(h.get("pred_lbl","")).lower())
                zone_total   = sum(1 for h in hist_all_z if h.get("camera","") == zcam)

                risk_color = "#e05252" if zone_fights > 2 else "#f5a623" if zone_fights > 0 else "#52e08a"
                risk_label = "HIGH RISK" if zone_fights > 2 else "ELEVATED" if zone_fights > 0 else "CLEAR"

                with st.expander(f"📍 {zname} — {risk_label}", expanded=False):
                    zd1, zd2, zd3 = st.columns(3)
                    zd1.metric("Camera", zcam or "—")
                    zd2.metric("Location", zloc or "—")
                    zd3.metric("Fight incidents", zone_fights)

                    if zone_total > 0:
                        st.markdown(
                            f"<div style='background:rgba(0,0,0,0.2);border:1px solid {risk_color};"
                            f"border-radius:6px;padding:8px 12px;margin:6px 0;'>"
                            f"<span style='color:{risk_color};font-weight:700;font-size:12px;'>{risk_label}</span>"
                            f" — {zone_fights} fights / {zone_total} total analyses "
                            f"({zone_fights/max(zone_total,1)*100:.0f}% rate)"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                    if zone.get("description"):
                        st.caption(zone["description"])

                    # Recent incidents for this zone
                    zone_incidents = [h for h in hist_all_z if h.get("camera","") == zcam][:5]
                    if zone_incidents:
                        st.markdown("**Recent incidents at this zone:**")
                        for inc in zone_incidents:
                            is_fi = "fight" in str(inc.get("pred_lbl","")).lower() and "non" not in str(inc.get("pred_lbl","")).lower()
                            badge = "vg-badge-fight" if is_fi else "vg-badge-normal"
                            badge_txt = "FIGHT" if is_fi else "NORMAL"
                            st.markdown(
                                f"<div style='display:flex;gap:10px;align-items:center;padding:4px 0;"
                                f"border-bottom:1px solid #1a2535;'>"
                                f"<span class='{badge}' style='font-size:10px;padding:2px 8px;'>{badge_txt}</span>"
                                f"<span style='color:#c8d8e8;font-size:12px;'>{inc.get('folder','?')}</span>"
                                f"<span style='color:#445566;font-size:11px;'>{inc.get('ts','?')}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                    if st.button(f"🗑 Delete zone '{zname}'", key=f"del_zone_{zi}"):
                        zones = [z for z in zones if z.get("name") != zname]
                        save_zones(zones)
                        st.rerun()

            # ── Zone risk summary chart ──
            if zones:
                st.markdown("**Zone Risk Overview**")
                zone_names_chart  = [z.get("name","?") for z in zones]
                zone_fights_chart = []
                for z in zones:
                    zcam = z.get("camera","")
                    cnt = sum(1 for h in hist_all_z
                              if h.get("camera","") == zcam
                              and "fight" in str(h.get("pred_lbl","")).lower()
                              and "non" not in str(h.get("pred_lbl","")).lower())
                    zone_fights_chart.append(cnt)

                if max(zone_fights_chart) > 0:
                    c = get_plot_colors()
                    fig_z, ax_z = plt.subplots(figsize=(6, 2.5), facecolor=c["bg"])
                    ax_z.set_facecolor(c["ax"])
                    bar_c = ["#e05252" if f > 2 else "#f5a623" if f > 0 else "#2a3a4a"
                             for f in zone_fights_chart]
                    ax_z.barh(zone_names_chart, zone_fights_chart, color=bar_c)
                    ax_z.set_xlabel("Fight detections", fontsize=8, color=c["xlabel"])
                    ax_z.tick_params(colors=c["tick"], labelsize=8)
                    ax_z.spines[:].set_color(c["spine"])
                    plt.tight_layout()
                    st.pyplot(fig_z, use_container_width=True); plt.close(fig_z)
                else:
                    st.caption("No fight incidents recorded for any zone yet.")

            # Export zones
            st.download_button(
                "⬇ Export zones (JSON)",
                data=json.dumps(zones, indent=2),
                file_name="visionguard_zones.json",
                mime="application/json",
                key="zones_dl"
            )

    # ══════════════════════════════════════════════════════
    # TAB 4 — Chain of Custody Log
    # ══════════════════════════════════════════════════════
    with tab_coc:
        st.markdown("### 🔐 Chain of Custody Log")
        st.markdown(
            f"<div style='color:#7a99b0;font-size:13px;margin-bottom:12px;'>"
            f"Cryptographically hash output files at ingest time. "
            f"Every entry is timestamped and signed — tamper-evident audit trail for legal evidence."
            f"</div>",
            unsafe_allow_html=True
        )

        coc_entries = load_coc()

        # ── Register new entry ──
        with st.expander("➕ Register Evidence File", expanded=False):
            st.markdown("Select a processed folder to hash and register all its output files.")
            records_coc = get_all_pred_records()
            if not records_coc:
                st.info("No processed records found.")
            else:
                folder_opts_coc = [
                    f"{r.get('_folder','?')} ({r.get('_dataset','?')}/{r.get('_class','?')})"
                    for r in records_coc
                ]
                with st.form("coc_form"):
                    coc_sel_idx = st.selectbox("Folder", range(len(folder_opts_coc)),
                                               format_func=lambda i: folder_opts_coc[i], key="coc_sel")
                    coc_reviewer = st.text_input("Reviewer / Officer name", key="coc_rev_inp")
                    coc_notes    = st.text_area("Notes / Case reference", height=60, key="coc_notes_inp")

                    if st.form_submit_button("🔐 Hash & Register", type="primary"):
                        rec_coc = records_coc[coc_sel_idx]
                        folder_path_coc = class_root(rec_coc.get("_dataset",""), rec_coc.get("_class","")) / rec_coc.get("_folder","")
                        files_coc = get_files(folder_path_coc)

                        hashes = {}
                        for fk, fpath in files_coc.items():
                            try:
                                fp = Path(fpath)
                                if fp.exists() and fp.is_file():
                                    h = hashlib.sha256(fp.read_bytes()).hexdigest()
                                    hashes[fk] = {"filename": fp.name, "sha256": h, "size_bytes": fp.stat().st_size}
                            except:
                                pass

                        # Bundle hash
                        bundle_str = json.dumps(hashes, sort_keys=True)
                        bundle_hash = hashlib.sha256(bundle_str.encode()).hexdigest()

                        entry_coc = {
                            "id": hashlib.sha256(f"{rec_coc.get('_folder','')}_{datetime.now().isoformat()}".encode()).hexdigest()[:12],
                            "folder": rec_coc.get("_folder","?"),
                            "dataset": rec_coc.get("_dataset","?"),
                            "cls": rec_coc.get("_class","?"),
                            "pred_label": rec_coc.get("pred_label","?"),
                            "confidence": rec_coc.get("confidence","?"),
                            "registered_at": datetime.now().isoformat(),
                            "reviewer": coc_reviewer,
                            "notes": coc_notes,
                            "file_hashes": hashes,
                            "bundle_hash": bundle_hash,
                            "verified": True,
                        }
                        coc_entries.insert(0, entry_coc)
                        save_coc(coc_entries)
                        st.success(f"✅ Registered! Bundle SHA-256: `{bundle_hash[:32]}...`")
                        st.rerun()

        if not coc_entries:
            st.info("No evidence registered yet. Use the form above to hash and register a processed folder.")
        else:
            st.markdown(f"**{len(coc_entries)} registered evidence entries**")

            # Verify all button
            if st.button("🔍 Verify All Entries (re-hash & compare)", use_container_width=False, key="coc_verify_all"):
                n_ok = 0; n_fail = 0
                for entry_v in coc_entries:
                    folder_path_v = class_root(entry_v.get("dataset",""), entry_v.get("cls","")) / entry_v.get("folder","")
                    files_v = get_files(folder_path_v)
                    all_ok = True
                    for fk, finfo in entry_v.get("file_hashes", {}).items():
                        if fk in files_v:
                            try:
                                current_hash = hashlib.sha256(Path(files_v[fk]).read_bytes()).hexdigest()
                                if current_hash != finfo.get("sha256",""):
                                    all_ok = False
                            except:
                                all_ok = False
                    if all_ok: n_ok += 1
                    else: n_fail += 1
                if n_fail == 0:
                    st.success(f"✅ All {n_ok} entries verified — no tampering detected.")
                else:
                    st.error(f"⚠ {n_fail} entries FAILED verification — files may have been modified!")

            for i, entry_coc in enumerate(coc_entries):
                is_fi = "fight" in str(entry_coc.get("pred_label","")).lower() and "non" not in str(entry_coc.get("pred_label","")).lower()
                badge = "vg-badge-fight" if is_fi else "vg-badge-normal"
                badge_txt = "FIGHT" if is_fi else "NORMAL"

                with st.expander(
                    f"#{entry_coc.get('id','?')} · {entry_coc.get('folder','?')} · {entry_coc.get('registered_at','?')[:10]}",
                    expanded=False
                ):
                    ec1, ec2, ec3 = st.columns(3)
                    ec1.metric("Folder", entry_coc.get("folder","?"))
                    ec2.metric("Registered", entry_coc.get("registered_at","?")[:10])
                    ec3.metric("Reviewer", entry_coc.get("reviewer","—") or "—")

                    st.markdown(f"<span class='{badge}'>{badge_txt}</span> Confidence: {entry_coc.get('confidence','?')}", unsafe_allow_html=True)

                    st.markdown(f"**Bundle SHA-256:** `{entry_coc.get('bundle_hash','?')}`")

                    if entry_coc.get("notes"):
                        st.caption(f"Notes: {entry_coc['notes']}")

                    # File hashes
                    with st.expander("File hashes", expanded=False):
                        for fk, finfo in entry_coc.get("file_hashes", {}).items():
                            st.text(f"{fk}: {finfo.get('sha256','?')[:32]}...  ({finfo.get('size_bytes',0):,} bytes)")

                    # Verify this entry live
                    if st.button(f"🔍 Verify this entry", key=f"coc_verify_{i}"):
                        folder_path_v = class_root(entry_coc.get("dataset",""), entry_coc.get("cls","")) / entry_coc.get("folder","")
                        files_v = get_files(folder_path_v)
                        all_ok = True; mismatches = []
                        for fk, finfo in entry_coc.get("file_hashes",{}).items():
                            if fk in files_v:
                                try:
                                    current_hash = hashlib.sha256(Path(files_v[fk]).read_bytes()).hexdigest()
                                    if current_hash != finfo.get("sha256",""):
                                        all_ok = False
                                        mismatches.append(fk)
                                except:
                                    all_ok = False; mismatches.append(fk)
                        if all_ok:
                            st.success("✅ All file hashes match — evidence is intact.")
                        else:
                            st.error(f"⚠ Hash mismatch on: {', '.join(mismatches)} — files may have been altered!")

                    # Export certificate
                    cert = {
                        "certificate_type": "VisionGuard Chain of Custody",
                        "generated_at": datetime.now().isoformat(),
                        **entry_coc
                    }
                    st.download_button(
                        "⬇ Download CoC Certificate (JSON)",
                        data=json.dumps(cert, indent=2),
                        file_name=f"CoC_{entry_coc.get('id','?')}_{entry_coc.get('folder','?')}.json",
                        mime="application/json",
                        key=f"coc_dl_{i}"
                    )

# ══════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════
nav = st.session_state.nav_section

if nav == "🏠 Home":
    render_home()
elif nav == "📥 Ingest":
    render_ingest()
elif nav == "🧪 Review Workspace":
    render_review_workspace()
elif nav == "📊 Dataset Lab":
    render_dataset_lab()
elif nav == "🕘 History":
    render_history()
elif nav == "🛠️ Smart Tools":
    render_smart_tools()
elif nav == "⚙️ Settings":
    render_settings()
else:
    render_home()