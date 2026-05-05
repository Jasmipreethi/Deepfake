"""
Multi-Model Accuracy Comparison Script — AV Deepfake Detection

Compares any number of trained models on the same test dataset.
Computes AUC, accuracy, precision, recall, F1, per-type breakdown,
generates comparison plots, and saves a full CSV report.

Usage:
    # Compare two models
    python compare_models.py \
        --models best_model_1.pth best_model_2.pth \
        --names "Run 1" "Run 2" \
        --video_dir ./test/ \
        --output_dir comparison_results/

    # Compare three or more models
    python compare_models.py \
        --models model_a.pth model_b.pth model_c.pth \
        --names "10 epochs" "20 epochs" "30 epochs" \
        --video_dir ./test/ \
        --threshold 0.5

    # Use GPU
    python compare_models.py \
        --models best_model_1.pth best_model_2.pth \
        --video_dir ./test/ \
        --device cuda

Test folder structure required:
    test/
        real/    ← videos where true label = REAL
        fake/    ← videos where true label = FAKE

Requirements:
    pip install torch torchvision torchaudio
    pip install opencv-python-headless torchaudio
    pip install scikit-learn pandas matplotlib numpy tqdm
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#1e293b',
    'axes.labelcolor': '#1e293b',
    'text.color': '#1e293b',
    'xtick.color': '#334155',
    'ytick.color': '#334155',
    'grid.color': '#e2e8f0',
    'grid.alpha': 0.6,
    'axes.grid': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchaudio

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

try:
    from config import FEATURE_CONFIG
except ImportError:
    FEATURE_CONFIG = {
        'sr': 16000, 'fps': 25, 'num_frames': 50,
        'img_size': 224, 'audio_samples': 32000,
        'n_fft': 1024, 'hop_length': 512,
        'target_t': 63, 'n_mels': 128, 'top_db': 80,
    }

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

from torchvision.models import resnet18
from torchvision.models.video import r3d_18


class _AudioEncoder(nn.Module):
    def __init__(self, feature_dim=256, dropout=0.4, intermediate_dim=512):
        super().__init__()
        b = resnet18(weights=None)
        b.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        b.fc = nn.Sequential(
            nn.Linear(b.fc.in_features, intermediate_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(intermediate_dim, feature_dim))
        self.backbone = b

    def forward(self, x):
        return self.backbone(
            F.interpolate(x, (224, 224), mode='bilinear', align_corners=False))


class _VideoEncoder(nn.Module):
    def __init__(self, feature_dim=256, dropout=0.4, intermediate_dim=512):
        super().__init__()
        b = r3d_18(weights=None)
        b.fc = nn.Sequential(
            nn.Linear(b.fc.in_features, intermediate_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(intermediate_dim, feature_dim))
        self.backbone = b

    def forward(self, x):
        return self.backbone(x.permute(0, 2, 1, 3, 4))


class _TransformerFusion(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=512,
                 num_heads=8, num_layers=2, dropout=0.4):
        super().__init__()
        self.audio_proj    = nn.Linear(feature_dim, hidden_dim)
        self.video_proj    = nn.Linear(feature_dim, hidden_dim)
        self.cls_token     = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim))
        self.audio_classifier = nn.Linear(hidden_dim, 1)
        self.video_classifier = nn.Linear(hidden_dim, 1)
        self.joint_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, vf, af):
        B = vf.shape[0]
        tokens = (torch.cat([
            self.cls_token.expand(B, -1, -1),
            self.video_proj(vf).unsqueeze(1),
            self.audio_proj(af).unsqueeze(1)], dim=1)
            + self.pos_embedding)
        c = self.transformer(tokens)[:, 0, :]
        return {'audio_pred': torch.sigmoid(self.audio_classifier(c)),
                'video_pred': torch.sigmoid(self.video_classifier(c)),
                'joint_pred': torch.sigmoid(self.joint_classifier(c))}


class _MLPFusion(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=512, dropout=0.4):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout / 2))
        self.audio_classifier = nn.Linear(hidden_dim, 1)
        self.video_classifier = nn.Linear(hidden_dim, 1)
        self.joint_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, vf, af):
        f = self.fusion(torch.cat([vf, af], dim=1))
        return {'audio_pred': torch.sigmoid(self.audio_classifier(f)),
                'video_pred': torch.sigmoid(self.video_classifier(f)),
                'joint_pred': torch.sigmoid(self.joint_classifier(f))}


class AVDetector(nn.Module):
    def __init__(self, fusion_type='transformer', feature_dim=256,
                 hidden_dim=512, dropout=0.4):
        super().__init__()
        self.video_encoder = _VideoEncoder(feature_dim, dropout)
        self.audio_encoder = _AudioEncoder(feature_dim, dropout)
        self.fusion_module = (
            _TransformerFusion(feature_dim, hidden_dim, dropout=dropout)
            if fusion_type == 'transformer'
            else _MLPFusion(feature_dim, hidden_dim, dropout))

    def forward(self, video, audio):
        return self.fusion_module(
            self.video_encoder(video), self.audio_encoder(audio))


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path, device):
    """Load a checkpoint, auto-detect fusion type, return (model, info)."""
    ck    = torch.load(path, map_location=device, weights_only=False)
    state = ck.get('model_state_dict', ck)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    fusion = ('transformer'
              if any('transformer' in k for k in state) else 'pretrained')
    model  = AVDetector(fusion_type=fusion)
    model.load_state_dict(state)
    model.to(device).eval()
    info = {
        'epoch':    ck.get('epoch', -1) + 1,
        'best_auc': round(float(ck.get('best_val_auc', 0)), 4),
        'fusion':   fusion,
        'history':  ck.get('history', {}),
    }
    return model, info


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_window(path, start_sec, cfg=FEATURE_CONFIG):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * cfg['fps']))
    frames = []
    for _ in range(cfg['num_frames']):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (cfg['img_size'], cfg['img_size']))
        frames.append(frame / 255.0)
    cap.release()
    if not frames:
        return None, None
    while len(frames) < cfg['num_frames']:
        frames.append(frames[-1])
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    vt = torch.FloatTensor(
        (np.array(frames) - mean) / std).permute(0, 3, 1, 2)

    try:
        wav, sr = torchaudio.load(path)
        if sr != cfg['sr']:
            wav = torchaudio.transforms.Resample(sr, cfg['sr'])(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        s   = int(start_sec * cfg['sr'])
        wav = wav[:, s:s + cfg['audio_samples']]
        if wav.shape[1] < cfg['audio_samples']:
            wav = F.pad(wav, (0, cfg['audio_samples'] - wav.shape[1]))
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg['sr'], n_mels=cfg.get('n_mels', 128),
            n_fft=cfg['n_fft'], hop_length=cfg['hop_length'])(wav)
        at = torchaudio.transforms.AmplitudeToDB(
            top_db=cfg.get('top_db', 80))(mel)
        std_v = at.std()
        if std_v > 0:
            at = (at - at.mean()) / (std_v + 1e-6)
        t = at.shape[2]
        tt = cfg['target_t']
        if t < tt:
            at = F.pad(at, (0, tt - t))
        elif t > tt:
            at = at[:, :, :tt]
    except Exception:
        at = torch.zeros(1, cfg.get('n_mels', 128), cfg['target_t'])

    return vt, at


def predict_video(model, path, device, n_windows=3, threshold=0.5):
    """Run multi-window inference and return averaged scores."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or FEATURE_CONFIG['num_frames']
    fps   = cap.get(cv2.CAP_PROP_FPS) or FEATURE_CONFIG['fps']
    cap.release()
    total_sec = total / fps
    max_start = max(0.0, total_sec - FEATURE_CONFIG.get('duration', 2.0))
    starts = ([max_start * i / (n_windows - 1) for i in range(n_windows)]
              if n_windows > 1 and max_start > 0 else [0.0])

    preds = {'audio': [], 'video': [], 'joint': []}
    for s in starts:
        v, a = extract_window(path, s)
        if v is None:
            continue
        with torch.no_grad():
            out = model(v.unsqueeze(0).to(device), a.unsqueeze(0).to(device))
        preds['audio'].append(out['audio_pred'].item())
        preds['video'].append(out['video_pred'].item())
        preds['joint'].append(out['joint_pred'].item())

    if not preds['joint']:
        return None

    j = sum(preds['joint']) / len(preds['joint'])
    a = sum(preds['audio']) / len(preds['audio'])
    v = sum(preds['video']) / len(preds['video'])
    return {
        'joint_score': round(j, 4),
        'audio_score': round(a, 4),
        'video_score': round(v, 4),
        'verdict':     'REAL' if j >= threshold else 'FAKE',
    }


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_videos(video_dir):
    """Find all labelled videos. Returns list of (path, true_label, type)."""
    videos = []
    real_dir = os.path.join(video_dir, 'real')
    fake_dir = os.path.join(video_dir, 'fake')

    if os.path.isdir(real_dir) and os.path.isdir(fake_dir):
        for f in sorted(os.listdir(real_dir)):
            if Path(f).suffix.lower() in VIDEO_EXTS:
                videos.append((os.path.join(real_dir, f), 1, 'real'))
        for f in sorted(os.listdir(fake_dir)):
            if Path(f).suffix.lower() in VIDEO_EXTS:
                # Try to infer sub-type from filename
                fname = f.lower()
                if 'audio' in fname:
                    vtype = 'audio_modified'
                elif 'video' in fname or 'visual' in fname:
                    vtype = 'visual_modified'
                elif 'both' in fname:
                    vtype = 'both_modified'
                else:
                    vtype = 'fake'
                videos.append((os.path.join(fake_dir, f), 0, vtype))
    else:
        # Flat folder — infer from filename
        for f in sorted(os.listdir(video_dir)):
            if Path(f).suffix.lower() not in VIDEO_EXTS:
                continue
            fname = f.lower()
            if 'real' in fname:
                videos.append((os.path.join(video_dir, f), 1, 'real'))
            elif 'audio' in fname:
                videos.append((os.path.join(video_dir, f), 0, 'audio_modified'))
            elif 'visual' in fname or 'video' in fname:
                videos.append((os.path.join(video_dir, f), 0, 'visual_modified'))
            elif 'both' in fname:
                videos.append((os.path.join(video_dir, f), 0, 'both_modified'))
            elif 'fake' in fname:
                videos.append((os.path.join(video_dir, f), 0, 'fake'))

    if not videos:
        print("ERROR: No labelled videos found.")
        sys.exit(1)

    n_real = sum(1 for _, l, _ in videos if l == 1)
    n_fake = sum(1 for _, l, _ in videos if l == 0)
    print(f"Found {len(videos)} videos — {n_real} real, {n_fake} fake")
    return videos


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df, threshold=0.5):
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, confusion_matrix,
        precision_score, recall_score, f1_score, roc_curve)

    y_true  = df['true_label'].values
    y_score = df['joint_score'].values
    y_pred  = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Best threshold by F1
    f1s = [f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
           for t in thresholds]
    best_t   = float(thresholds[np.argmax(f1s)])
    y_pred_b = (y_score >= best_t).astype(int)

    # Per-type AUC
    per_type = {}
    for t in df['modify_type'].unique():
        mask = df['modify_type'] == t
        if mask.sum() < 2:
            continue
        gt = df.loc[mask, 'true_label'].values
        sc = df.loc[mask, 'joint_score'].values
        if len(np.unique(gt)) < 2:
            continue
        per_type[t] = {
            'auc':      round(float(roc_auc_score(gt, sc)), 4),
            'accuracy': round(float(accuracy_score(
                gt, (sc >= threshold).astype(int))), 4),
            'count':    int(mask.sum()),
        }

    return {
        'auc':           round(float(roc_auc_score(y_true, y_score)), 4),
        'accuracy':      round(float(accuracy_score(y_true, y_pred)), 4),
        'precision':     round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall':        round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'f1':            round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'best_threshold': round(best_t, 3),
        'best_f1':        round(float(max(f1s)), 4),
        'best_accuracy':  round(float(accuracy_score(y_true, y_pred_b)), 4),
        'n_real':         int((y_true == 1).sum()),
        'n_fake':         int((y_true == 0).sum()),
        'n_total':        len(y_true),
        'mean_real_score': round(float(df.loc[df['true_label']==1,'joint_score'].mean()), 4),
        'mean_fake_score': round(float(df.loc[df['true_label']==0,'joint_score'].mean()), 4),
        'fpr': fpr, 'tpr': tpr,
        'per_type': per_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette — one colour per model
PALETTE = ['#00d4ff', '#f97316', '#22c55e', '#a855f7',
           '#ef4444', '#eab308', '#06b6d4', '#ec4899']


def plot_all(results, output_dir, threshold=0.5):
    """Generate a comprehensive multi-panel comparison figure."""
    os.makedirs(output_dir, exist_ok=True)
    n = len(results)
    names   = [r['name'] for r in results]
    metrics = [r['metrics'] for r in results]
    colors  = PALETTE[:n]

    fig = plt.figure(figsize=(22, 24))
    fig.patch.set_facecolor('#050a14')
    gs  = gridspec.GridSpec(4, 3, figure=fig,
                            hspace=0.5, wspace=0.35,
                            top=0.93, bottom=0.04,
                            left=0.06, right=0.97)
    title_style = dict(color='#e2e8f0', fontsize=11, fontweight='bold',
                       pad=10)

    def ax_style(ax):
        ax.set_facecolor('#0c1425')
        ax.tick_params(colors='#64748b', labelsize=8)
        ax.spines[:].set_color('#1e3352')
        ax.xaxis.label.set_color('#64748b')
        ax.yaxis.label.set_color('#64748b')
        return ax

    # ── Row 0: Summary bar charts ────────────────────────────────────────────
    ax0 = ax_style(fig.add_subplot(gs[0, :]))
    metric_keys  = ['auc', 'accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(metric_keys))
    w = 0.8 / n
    for i, (m, name, color) in enumerate(zip(metrics, names, colors)):
        vals = [m[k] for k in metric_keys]
        bars = ax0.bar(x + i * w - (n-1)*w/2, vals, w,
                       label=name, color=color, alpha=0.85)
        for b, v in zip(bars, vals):
            ax0.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                     f'{v:.3f}', ha='center', va='bottom',
                     fontsize=7, color='#94a3b8')
    ax0.set_xticks(x)
    ax0.set_xticklabels(metric_labels, fontsize=10, color='#e2e8f0')
    ax0.set_ylim(0, 1.15)
    ax0.set_title('Overall Metrics Comparison', **title_style)
    ax0.legend(fontsize=9, facecolor='#0c1425', edgecolor='#1e3352',
               labelcolor='#e2e8f0')
    ax0.axhline(0.5, color='#1e3352', linestyle='--', linewidth=0.8)
    ax0.grid(axis='y', color='#1e3352', alpha=0.5)

    # ── Row 1, Col 0: ROC curves ─────────────────────────────────────────────
    ax_roc = ax_style(fig.add_subplot(gs[1, 0]))
    ax_roc.plot([0,1],[0,1], '--', color='#1e3352', linewidth=1)
    for m, name, color in zip(metrics, names, colors):
        ax_roc.plot(m['fpr'], m['tpr'], color=color, linewidth=2,
                    label=f'{name}  AUC={m["auc"]:.3f}')
        ax_roc.fill_between(m['fpr'], m['tpr'], alpha=0.05, color=color)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves', **title_style)
    ax_roc.legend(fontsize=7, facecolor='#0c1425',
                  edgecolor='#1e3352', labelcolor='#e2e8f0')
    ax_roc.grid(color='#1e3352', alpha=0.4)

    # ── Row 1, Col 1: Score distributions overlay ────────────────────────────
    ax_dist = ax_style(fig.add_subplot(gs[1, 1]))
    bins = np.linspace(0, 1, 30)
    for r, color in zip(results, colors):
        df = r['df']
        real_s = df[df['true_label']==1]['joint_score']
        fake_s = df[df['true_label']==0]['joint_score']
        ax_dist.hist(real_s, bins=bins, alpha=0.4, color='#22c55e',
                     label=f'{r["name"]} Real' if len(results) == 1 else None)
        ax_dist.hist(fake_s, bins=bins, alpha=0.4, color=color,
                     label=r['name'])
    ax_dist.axvline(threshold, color='#ffffff', linestyle='--',
                    linewidth=1, alpha=0.5, label=f'Threshold {threshold}')
    ax_dist.set_xlabel('Joint Score  (0=fake, 1=real)')
    ax_dist.set_ylabel('Count')
    ax_dist.set_title('Score Distributions (green=real, colours=fake per model)',
                      **title_style)
    ax_dist.legend(fontsize=7, facecolor='#0c1425',
                   edgecolor='#1e3352', labelcolor='#e2e8f0')
    ax_dist.grid(color='#1e3352', alpha=0.4)

    # ── Row 1, Col 2: Confusion matrices (stacked as text) ───────────────────
    ax_cm = ax_style(fig.add_subplot(gs[1, 2]))
    ax_cm.axis('off')
    ax_cm.set_title('Confusion Matrix Summary', **title_style)
    col_labels = ['Model', 'TP', 'TN', 'FP', 'FN', 'Acc']
    rows = [[r['name'],
             str(r['metrics']['tp']), str(r['metrics']['tn']),
             str(r['metrics']['fp']), str(r['metrics']['fn']),
             f'{r["metrics"]["accuracy"]:.1%}']
            for r in results]
    table = ax_cm.table(
        cellText=rows, colLabels=col_labels,
        cellLoc='center', loc='center',
        bbox=[0, 0.1, 1, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor('#0c1425' if row > 0 else '#1e3352')
        cell.set_edgecolor('#1e3352')
        cell.set_text_props(color='#e2e8f0' if row > 0 else '#00d4ff')

    # ── Row 2: Per-type AUC breakdown ────────────────────────────────────────
    ax_type = ax_style(fig.add_subplot(gs[2, :]))
    type_order = ['real', 'audio_modified', 'visual_modified', 'both_modified']
    x2 = np.arange(len(type_order))
    for i, (r, color) in enumerate(zip(results, colors)):
        pt = r['metrics']['per_type']
        vals = [pt.get(t, {}).get('auc', 0) for t in type_order]
        bars = ax_type.bar(x2 + i*w - (n-1)*w/2, vals, w,
                           label=r['name'], color=color, alpha=0.85)
        for b, v in zip(bars, vals):
            if v > 0:
                ax_type.text(b.get_x() + b.get_width()/2, b.get_height()+0.005,
                             f'{v:.3f}', ha='center', va='bottom',
                             fontsize=7, color='#94a3b8')
    ax_type.set_xticks(x2)
    type_labels = ['Real', 'Audio Modified', 'Visual Modified', 'Both Modified']
    ax_type.set_xticklabels(type_labels, fontsize=10, color='#e2e8f0')
    ax_type.set_ylim(0, 1.15)
    ax_type.set_ylabel('AUC')
    ax_type.set_title('AUC by Manipulation Type', **title_style)
    ax_type.legend(fontsize=9, facecolor='#0c1425',
                   edgecolor='#1e3352', labelcolor='#e2e8f0')
    ax_type.axhline(0.5, color='#1e3352', linestyle='--', linewidth=0.8)
    ax_type.grid(axis='y', color='#1e3352', alpha=0.5)

    # ── Row 3: Audio vs Video scatter — one per model ────────────────────────
    for idx, (r, color) in enumerate(zip(results[:3], colors)):
        ax_s = ax_style(fig.add_subplot(gs[3, idx]))
        df = r['df']
        for vtype, marker, tc in [
            ('real',            'o', '#22c55e'),
            ('audio_modified',  's', '#f97316'),
            ('visual_modified', '^', '#a855f7'),
            ('both_modified',   'D', '#ef4444'),
        ]:
            sub = df[df['modify_type'] == vtype]
            if len(sub):
                ax_s.scatter(sub['audio_score'], sub['video_score'],
                             c=tc, marker=marker, alpha=0.6, s=20,
                             label=vtype.replace('_', ' '))
        ax_s.axvline(0.5, color='#1e3352', linestyle='--', linewidth=0.8)
        ax_s.axhline(0.5, color='#1e3352', linestyle='--', linewidth=0.8)
        ax_s.set_xlim(-0.02, 1.02)
        ax_s.set_ylim(-0.02, 1.02)
        ax_s.set_xlabel('Audio Score')
        ax_s.set_ylabel('Video Score')
        ax_s.set_title(f'Audio vs Video — {r["name"]}', **title_style)
        ax_s.legend(fontsize=6, facecolor='#0c1425',
                    edgecolor='#1e3352', labelcolor='#e2e8f0',
                    loc='lower right')
        ax_s.grid(color='#1e3352', alpha=0.3)

    # Title
    fig.text(0.5, 0.97, 'Model Accuracy Comparison — AV Deepfake Detection',
             ha='center', va='top', color='#e2e8f0',
             fontsize=16, fontweight='bold')

    out = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor='#050a14')
    plt.close()
    print(f"\n  Plot saved: {out}")
    return out


def plot_training_history(results, output_dir):
    """Plot training AUC/loss from checkpoint history if available."""
    has_history = any(r['info'].get('history', {}).get('val_auc_joint')
                      for r in results)
    if not has_history:
        print("  No training history in checkpoints — skipping history plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('#050a14')

    for ax in axes:
        ax.set_facecolor('#0c1425')
        ax.tick_params(colors='#64748b')
        ax.spines[:].set_color('#1e3352')
        ax.xaxis.label.set_color('#64748b')
        ax.yaxis.label.set_color('#64748b')

    colors = PALETTE[:len(results)]
    markers = ['o', '^', 's', 'D', 'P', 'X', '*', 'v']

    for r, color, marker in zip(results, colors, markers):
        h    = r['info'].get('history', {})
        name = r['name']
        if h.get('val_auc_joint'):
            eps = range(1, len(h['val_auc_joint']) + 1)
            axes[0].plot(eps, h['val_auc_joint'], color=color,
                         linewidth=2, marker=marker, markersize=5,
                         label=name)
            best = max(h['val_auc_joint'])
            best_ep = h['val_auc_joint'].index(best) + 1
            axes[0].axvline(best_ep, color=color, linestyle='--',
                            alpha=0.3, linewidth=1)
            axes[0].annotate(f'{best:.3f}',
                             xy=(best_ep, best),
                             xytext=(best_ep + 0.1, best - 0.02),
                             color=color, fontsize=8)
        if h.get('train_loss'):
            eps = range(1, len(h['train_loss']) + 1)
            axes[1].plot(eps, h['train_loss'], color=color,
                         linewidth=2, marker=marker, markersize=5,
                         label=f'{name} train', linestyle='-')
        if h.get('val_loss'):
            eps = range(1, len(h['val_loss']) + 1)
            axes[1].plot(eps, h['val_loss'], color=color,
                         linewidth=1.5, marker=marker, markersize=4,
                         label=f'{name} val', linestyle='--', alpha=0.7)

    for ax, title, ylabel in [
        (axes[0], 'Validation AUC over Epochs', 'Joint AUC'),
        (axes[1], 'Loss over Epochs',            'Loss'),
    ]:
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title, color='#e2e8f0', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, facecolor='#0c1425',
                  edgecolor='#1e3352', labelcolor='#e2e8f0')
        ax.grid(color='#1e3352', alpha=0.4)
        if 'AUC' in ylabel:
            ax.set_ylim(0, 1.05)

    fig.suptitle('Training History', color='#e2e8f0',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(output_dir, 'training_history.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#050a14')
    plt.close()
    print(f"  History plot saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results, threshold):
    print(f"\n{'═'*65}")
    print(f"  MULTI-MODEL ACCURACY COMPARISON")
    print(f"  Threshold: {threshold}  |  Videos: {results[0]['metrics']['n_total']}")
    print(f"{'═'*65}")

    # Per-model metrics
    for r in results:
        m = r['metrics']
        print(f"\n  {'─'*55}")
        print(f"  {r['name']}")
        print(f"  Checkpoint: Ep {r['info']['epoch']} | "
              f"Train AUC {r['info']['best_auc']}")
        print(f"  {'─'*55}")
        print(f"  AUC:          {m['auc']:.4f}   ← primary metric")
        print(f"  Accuracy:     {m['accuracy']:.1%}")
        print(f"  Precision:    {m['precision']:.4f}")
        print(f"  Recall:       {m['recall']:.4f}")
        print(f"  F1:           {m['f1']:.4f}")
        print(f"  TP:{m['tp']:3d}  TN:{m['tn']:3d}  FP:{m['fp']:3d}  FN:{m['fn']:3d}")
        print(f"  Mean real score: {m['mean_real_score']:.3f} | "
              f"Mean fake score: {m['mean_fake_score']:.3f}")
        print(f"  Best threshold by F1: {m['best_threshold']} "
              f"→ F1={m['best_f1']:.4f}  Acc={m['best_accuracy']:.1%}")
        if m['per_type']:
            print(f"\n  Per-type AUC:")
            for t, v in m['per_type'].items():
                bar = '█' * int(v['auc'] * 20)
                print(f"    {t:20s}: {v['auc']:.4f}  {bar}")

    # Winner table
    print(f"\n{'─'*65}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'─'*65}")
    metrics_to_compare = [
        ('AUC',       'auc',       'higher'),
        ('Accuracy',  'accuracy',  'higher'),
        ('F1',        'f1',        'higher'),
        ('Precision', 'precision', 'higher'),
        ('Recall',    'recall',    'higher'),
        ('False Neg', 'fn',        'lower'),
        ('False Pos', 'fp',        'lower'),
    ]
    wins = {r['name']: 0 for r in results}
    for label, key, better in metrics_to_compare:
        vals = [(r['name'], r['metrics'][key]) for r in results]
        best_val = min(v for _, v in vals) if better == 'lower' \
                   else max(v for _, v in vals)
        winners  = [n for n, v in vals if v == best_val]
        row = f"  {label:12s}"
        for n, v in vals:
            marker = ' ✓' if n in winners else '  '
            row += f"  {str(round(v,4)):>8}{marker}"
            if n in winners and len(winners) == 1:
                wins[n] += 1
        print(row)

    print(f"\n  {'─'*55}")
    sorted_wins = sorted(wins.items(), key=lambda x: -x[1])
    for name, w in sorted_wins:
        print(f"  {name}: {w}/{len(metrics_to_compare)} metrics won")
    champion = sorted_wins[0][0]
    print(f"\n  ✓ Best overall model: {champion}")
    print(f"{'═'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Compare multiple AV Deepfake Detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument('--models',    nargs='+', required=True,
                   help='Paths to .pth checkpoint files')
    p.add_argument('--names',     nargs='+', default=None,
                   help='Display names for each model (optional)')
    p.add_argument('--video_dir', required=True,
                   help='Test video folder (needs real/ and fake/ subfolders)')
    p.add_argument('--output_dir', default='comparison_results',
                   help='Where to save plots and CSV')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Decision threshold (default 0.5)')
    p.add_argument('--n_windows', type=int, default=3,
                   help='Windows averaged per video (default 3)')
    p.add_argument('--device',    default='auto',
                   choices=['auto', 'cuda', 'cpu'])
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\nDevice: {device}")

    # Names
    names = args.names or [f"Model {i+1}" for i in range(len(args.models))]
    if len(names) != len(args.models):
        print("ERROR: --names count must match --models count")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover videos once — shared across all models
    videos = discover_videos(args.video_dir)

    # Load models and run inference
    results = []
    for path, name in zip(args.models, names):
        print(f"\n{'─'*55}")
        print(f"Loading: {name}  ({path})")

        if not os.path.exists(path):
            print(f"  ✗ File not found: {path}")
            continue

        try:
            model, info = load_model(path, device)
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue

        print(f"  ✓ Epoch {info['epoch']} | Train AUC {info['best_auc']}")
        print(f"  Running inference on {len(videos)} videos...")

        rows = []
        for vpath, true_label, vtype in tqdm(
                videos, desc=f"  {name}", unit="video"):
            pred = predict_video(model, vpath, device,
                                 n_windows=args.n_windows,
                                 threshold=args.threshold)
            if pred is None:
                continue
            rows.append({
                'file':        os.path.basename(vpath),
                'true_label':  true_label,
                'true_verdict':'REAL' if true_label == 1 else 'FAKE',
                'modify_type': vtype,
                'joint_score': pred['joint_score'],
                'audio_score': pred['audio_score'],
                'video_score': pred['video_score'],
                'verdict':     pred['verdict'],
                'correct':     pred['verdict'] == ('REAL' if true_label==1 else 'FAKE'),
            })

        df = pd.DataFrame(rows)
        m  = compute_metrics(df, args.threshold)
        results.append({'name': name, 'path': path,
                        'model': model, 'info': info,
                        'df': df, 'metrics': m})

        # Save per-model CSV
        csv_path = os.path.join(args.output_dir,
                                f"{name.replace(' ', '_')}_predictions.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Predictions saved: {csv_path}")

    if not results:
        print("ERROR: No models loaded successfully.")
        sys.exit(1)

    # Print report
    print_report(results, args.threshold)

    # Save combined metrics JSON
    summary = {r['name']: {
        k: v for k, v in r['metrics'].items()
        if not isinstance(v, np.ndarray)
    } for r in results}
    summary['config'] = {
        'threshold': args.threshold,
        'n_windows': args.n_windows,
        'device':    str(device),
    }
    json_path = os.path.join(args.output_dir, 'metrics_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Metrics saved: {json_path}")

    # Plots
    print("\nGenerating plots...")
    plot_all(results, args.output_dir, args.threshold)
    plot_training_history(results, args.output_dir)

    print(f"\n✓ All results saved to: {os.path.abspath(args.output_dir)}\n")


if __name__ == '__main__':
    main()