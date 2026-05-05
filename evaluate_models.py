"""
Model Evaluation Script — AV Deepfake Detection
[DEPRECATED — use compare_models.py instead]

This script has been superseded by compare_models.py, which is a strict
superset: it supports any number of models (not just two), adds per-type
AUC breakdown, a confidence field, a richer N-model comparison figure, and
an overlaid training history plot.

Equivalent commands:

  # Old (two-model comparison via evaluate_models.py):
  python evaluate_models.py \\
      --model1 run1_best_model.pth \\
      --model2 run2_best_model.pth \\
      --name1 "Run 1" --name2 "Run 2" \\
      --video_dir test_videos/

  # New (same comparison via compare_models.py):
  python compare_models.py \\
      --models run1_best_model.pth run2_best_model.pth \\
      --names "Run 1" "Run 2" \\
      --video_dir test_videos/

  # Five-model comparison (not possible in evaluate_models.py):
  python compare_models.py \\
      --models logs/logs_1/best_model.pth logs/logs_2/best_model.pth \\
                logs/logs_3/best_model.pth logs/logs_4/best_model.pth \\
                logs/logs_5/best_model.pth \\
      --names "Run 1" "Run 2" "Run 3" "Run 4" "Run 5" \\
      --video_dir test/ \\
      --output_dir comparison_results/
"""

import sys

def main():
    print("[DEPRECATED] evaluate_models.py has been superseded by compare_models.py.")
    print("See the docstring at the top of this file for equivalent commands.")
    print("Run: python compare_models.py --help")
    sys.exit(0)

if __name__ == '__main__':
    main()


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
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchaudio
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CONFIG (mirrors training pipeline)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from config import FEATURE_CONFIG
except ImportError:
    FEATURE_CONFIG = {
        'sr': 16000, 'fps': 25, 'duration': 2.0,
        'num_frames': 50, 'img_size': 224,
        'audio_samples': 32000, 'n_fft': 1024,
        'hop_length': 512, 'target_t': 63, 'n_mels': 128, 'top_db': 80,
    }

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_at(video_path, start_sec, cfg=FEATURE_CONFIG):
    """Extract one 2-second window starting at start_sec."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    start_frame = int(start_sec * cfg['fps'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(cfg['num_frames']):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (cfg['img_size'], cfg['img_size']))
        frame = frame / 255.0
        frames.append(frame)
    cap.release()

    if not frames:
        return None, None
    while len(frames) < cfg['num_frames']:
        frames.append(frames[-1])

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    frames_arr = (np.array(frames) - mean) / std
    video_tensor = torch.FloatTensor(frames_arr).permute(0, 3, 1, 2)

    try:
        waveform, orig_sr = torchaudio.load(video_path)
        if orig_sr != cfg['sr']:
            waveform = torchaudio.transforms.Resample(orig_sr, cfg['sr'])(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        s = int(start_sec * cfg['sr'])
        waveform = waveform[:, s:s + cfg['audio_samples']]
        if waveform.shape[1] < cfg['audio_samples']:
            waveform = F.pad(waveform, (0, cfg['audio_samples'] - waveform.shape[1]))
        n_mels = cfg.get('n_mels', 128)
        top_db  = cfg.get('top_db', 80)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg['sr'], n_mels=n_mels,
            n_fft=cfg['n_fft'], hop_length=cfg['hop_length']
        )(waveform)
        audio_tensor = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(mel)
        std_v = audio_tensor.std()
        if std_v > 0:
            audio_tensor = (audio_tensor - audio_tensor.mean()) / (std_v + 1e-6)
        t = audio_tensor.shape[2]
        target_t = cfg['target_t']
        if t < target_t:
            audio_tensor = F.pad(audio_tensor, (0, target_t - t))
        elif t > target_t:
            audio_tensor = audio_tensor[:, :, :target_t]
    except Exception:
        audio_tensor = torch.zeros(1, cfg.get('n_mels', 128), cfg['target_t'])

    return video_tensor, audio_tensor


def extract_windows(video_path, n_windows=3, cfg=FEATURE_CONFIG):
    """Extract n evenly-spaced windows from a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or cfg['num_frames']
    cap.release()

    total_sec = total_frames / cfg['fps']
    max_start = max(0.0, total_sec - cfg['duration'])
    starts = (
        [max_start * i / (n_windows - 1) for i in range(n_windows)]
        if n_windows > 1 and max_start > 0 else [0.0]
    )

    windows = []
    for s in starts:
        v, a = _extract_at(video_path, s, cfg)
        if v is not None:
            windows.append((v, a))
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (must match training)
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
            nn.Linear(intermediate_dim, feature_dim)
        )
        self.backbone = b

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)


class _VideoEncoder(nn.Module):
    def __init__(self, feature_dim=256, dropout=0.4, intermediate_dim=512):
        super().__init__()
        b = r3d_18(weights=None)
        b.fc = nn.Sequential(
            nn.Linear(b.fc.in_features, intermediate_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(intermediate_dim, feature_dim)
        )
        self.backbone = b

    def forward(self, x):
        return self.backbone(x.permute(0, 2, 1, 3, 4))


class _TransformerFusion(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=512,
                 num_heads=8, num_layers=2, dropout=0.4, ff_multiplier=4):
        super().__init__()
        self.audio_proj    = nn.Linear(feature_dim, hidden_dim)
        self.video_proj    = nn.Linear(feature_dim, hidden_dim)
        self.cls_token     = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * ff_multiplier,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.audio_cls = nn.Linear(hidden_dim, 1)
        self.video_cls = nn.Linear(hidden_dim, 1)
        self.joint_cls = nn.Linear(hidden_dim, 1)

    def forward(self, vf, af):
        B = vf.shape[0]
        v   = self.video_proj(vf).unsqueeze(1)
        a   = self.audio_proj(af).unsqueeze(1)
        cls = self.cls_token.expand(B, -1, -1)
        out = self.transformer(torch.cat([cls, v, a], dim=1) + self.pos_embedding)
        c   = out[:, 0, :]
        return {
            'audio_pred': torch.sigmoid(self.audio_cls(c)),
            'video_pred': torch.sigmoid(self.video_cls(c)),
            'joint_pred': torch.sigmoid(self.joint_cls(c)),
        }


class _MLPFusion(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=512, dropout=0.4):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout / 2)
        )
        self.audio_cls = nn.Linear(hidden_dim, 1)
        self.video_cls = nn.Linear(hidden_dim, 1)
        self.joint_cls = nn.Linear(hidden_dim, 1)

    def forward(self, vf, af):
        f = self.fusion(torch.cat([vf, af], dim=1))
        return {
            'audio_pred': torch.sigmoid(self.audio_cls(f)),
            'video_pred': torch.sigmoid(self.video_cls(f)),
            'joint_pred': torch.sigmoid(self.joint_cls(f)),
        }


class AVDetector(nn.Module):
    def __init__(self, fusion_type='transformer', feature_dim=256,
                 hidden_dim=512, dropout=0.4, intermediate_dim=512,
                 num_heads=8, num_layers=2, ff_multiplier=4):
        super().__init__()
        self.video_encoder = _VideoEncoder(feature_dim, dropout, intermediate_dim)
        self.audio_encoder = _AudioEncoder(feature_dim, dropout, intermediate_dim)
        if fusion_type == 'transformer':
            self.fusion_module = _TransformerFusion(
                feature_dim, hidden_dim, num_heads, num_layers, dropout, ff_multiplier)
        else:
            self.fusion_module = _MLPFusion(feature_dim, hidden_dim, dropout)

    def forward(self, video, audio):
        return self.fusion_module(self.video_encoder(video), self.audio_encoder(audio))


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path, device):
    """Load a checkpoint, auto-detect architecture, return (model, info_dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck.get('model_state_dict', ck)

    # Unwrap DataParallel prefix
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}

    # Detect fusion type
    fusion_type = 'transformer' if any('transformer' in k for k in state.keys()) else 'pretrained'

    model = AVDetector(fusion_type=fusion_type)
    model.load_state_dict(state)
    model.to(device).eval()

    info = {
        'epoch':      ck.get('epoch', -1) + 1,
        'best_auc':   ck.get('best_val_auc', None),
        'history':    ck.get('history', {}),
        'fusion':     fusion_type,
    }
    return model, info


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, video_path, device, n_windows=3, threshold=0.5):
    """Run inference on one video, return result dict."""
    windows = extract_windows(video_path, n_windows=n_windows)
    if not windows:
        return None

    preds = {'audio': [], 'video': [], 'joint': []}
    with torch.no_grad():
        for v, a in windows:
            out = model(v.unsqueeze(0).to(device), a.unsqueeze(0).to(device))
            preds['audio'].append(out['audio_pred'].item())
            preds['video'].append(out['video_pred'].item())
            preds['joint'].append(out['joint_pred'].item())

    j = sum(preds['joint']) / len(preds['joint'])
    a = sum(preds['audio']) / len(preds['audio'])
    v = sum(preds['video']) / len(preds['video'])

    return {
        'file':        os.path.basename(video_path),
        'audio_score': round(a, 4),
        'video_score': round(v, 4),
        'joint_score': round(j, 4),
        'verdict':     'REAL' if j >= threshold else 'FAKE',
        'confidence':  round(abs(j - threshold) * 2, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_videos(video_dir):
    """
    Discover videos and infer ground-truth labels.

    Supports two folder structures:
    1. Subfolders named 'real' and 'fake'
    2. Flat folder — labels inferred from filename containing 'real' or 'fake'

    Returns list of (path, true_label) where true_label is 1=real, 0=fake.
    """
    videos = []

    real_dir = os.path.join(video_dir, 'real')
    fake_dir = os.path.join(video_dir, 'fake')

    if os.path.isdir(real_dir) and os.path.isdir(fake_dir):
        # Structure 1: subfolders
        for f in sorted(os.listdir(real_dir)):
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                videos.append((os.path.join(real_dir, f), 1))
        for f in sorted(os.listdir(fake_dir)):
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                videos.append((os.path.join(fake_dir, f), 0))
        print(f"Found {sum(l==1 for _,l in videos)} real + "
              f"{sum(l==0 for _,l in videos)} fake videos (subfolder structure)")
    else:
        # Structure 2: flat folder, infer from filename
        for f in sorted(os.listdir(video_dir)):
            if os.path.splitext(f)[1].lower() not in VIDEO_EXTS:
                continue
            fname_lower = f.lower()
            if 'real' in fname_lower:
                videos.append((os.path.join(video_dir, f), 1))
            elif 'fake' in fname_lower:
                videos.append((os.path.join(video_dir, f), 0))
            else:
                print(f"  ⚠ Skipping {f} — cannot infer label (name must contain 'real' or 'fake')")

        if not videos:
            print("ERROR: No labelled videos found.")
            print("Either use real/ and fake/ subfolders, or include 'real'/'fake' in filenames.")
            sys.exit(1)

        print(f"Found {sum(l==1 for _,l in videos)} real + "
              f"{sum(l==0 for _,l in videos)} fake videos (flat folder)")

    if not videos:
        print("ERROR: No videos found.")
        sys.exit(1)

    return videos


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df, threshold=0.5):
    """Compute full metrics from a results DataFrame."""
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, confusion_matrix,
        precision_score, recall_score, f1_score, roc_curve
    )

    y_true = df['true_label'].values
    y_score = df['joint_score'].values
    y_pred  = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Best threshold by F1
    f1s = [f1_score([1 if s >= t else 0 for s in y_score], y_true, zero_division=0)
           for t in thresholds]
    best_thresh = float(thresholds[np.argmax(f1s)])
    y_pred_best = (y_score >= best_thresh).astype(int)

    return {
        'accuracy':       float(accuracy_score(y_true, y_pred)),
        'auc':            float(roc_auc_score(y_true, y_score)),
        'precision':      float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':         float(recall_score(y_true, y_pred, zero_division=0)),
        'f1':             float(f1_score(y_true, y_pred, zero_division=0)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'fpr': fpr, 'tpr': tpr,
        'best_threshold': best_thresh,
        'best_f1':        float(max(f1s)),
        'best_accuracy':  float(accuracy_score(y_true, y_pred_best)),
        'n_real':         int((y_true == 1).sum()),
        'n_fake':         int((y_true == 0).sum()),
        'n_total':        len(y_true),
        'confidence_mean': float(np.mean(np.abs(y_score - threshold) * 2)),
    }


def print_metrics(metrics, name, threshold=0.5):
    """Print a formatted metrics report."""
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Videos:       {metrics['n_real']} real + {metrics['n_fake']} fake = {metrics['n_total']} total")
    print(f"  Threshold:    {threshold}")
    print()
    print(f"  AUC:          {metrics['auc']:.4f}   (primary metric — threshold independent)")
    print(f"  Accuracy:     {metrics['accuracy']:.1%}")
    print(f"  Precision:    {metrics['precision']:.4f}  (of predicted real, how many are real)")
    print(f"  Recall:       {metrics['recall']:.4f}  (of actual real, how many detected)")
    print(f"  F1 Score:     {metrics['f1']:.4f}")
    print(f"  Confidence:   {metrics['confidence_mean']:.4f}  (0=uncertain, 1=certain)")
    print()
    print(f"  Confusion Matrix (threshold={threshold}):")
    print(f"    True  Positives (real→real):  {metrics['tp']:4d}  ✓")
    print(f"    True  Negatives (fake→fake):  {metrics['tn']:4d}  ✓")
    print(f"    False Positives (fake→real):  {metrics['fp']:4d}  ✗ over-trusting")
    print(f"    False Negatives (real→fake):  {metrics['fn']:4d}  ✗ missed deepfakes")
    print()
    print(f"  Best threshold by F1: {metrics['best_threshold']:.3f} "
          f"→ F1={metrics['best_f1']:.4f}, Acc={metrics['best_accuracy']:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(df1, m1, name1, df2, m2, name2, output_dir, threshold=0.5):
    """Generate a comprehensive 3×3 comparison figure."""
    os.makedirs(output_dir, exist_ok=True)

    REAL_COL  = '#2ecc71'
    FAKE_COL  = '#e74c3c'
    M1_COL    = '#3498db'
    M2_COL    = '#e67e22'
    GRID_ALPHA = 0.25

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle('Model Comparison — AV Deepfake Detection',
                 fontsize=18, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.93, bottom=0.06)

    # ── Row 0: Summary bar charts ──────────────────────────────────────────
    ax_summary = fig.add_subplot(gs[0, :])
    metrics_names  = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
    m1_vals = [m1['auc'], m1['accuracy'], m1['precision'], m1['recall'], m1['f1']]
    m2_vals = [m2['auc'], m2['accuracy'], m2['precision'], m2['recall'], m2['f1']]
    x = np.arange(len(metrics_names))
    w = 0.35
    b1 = ax_summary.bar(x - w/2, m1_vals, w, label=name1, color=M1_COL, alpha=0.85)
    b2 = ax_summary.bar(x + w/2, m2_vals, w, label=name2, color=M2_COL, alpha=0.85)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax_summary.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(metrics_names, fontsize=11)
    ax_summary.set_ylim(0, 1.12)
    ax_summary.set_ylabel('Score')
    ax_summary.set_title('Overall Metrics Comparison', fontsize=13, fontweight='bold')
    ax_summary.legend(fontsize=10)
    ax_summary.axhline(0.5, color='grey', linestyle='--', alpha=0.4, linewidth=0.8)
    ax_summary.grid(axis='y', alpha=GRID_ALPHA)

    # ── Row 1, Col 0: ROC curves ───────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[1, 0])
    ax_roc.plot(m1['fpr'], m1['tpr'],
                color=M1_COL, linewidth=2, label=f'{name1}  AUC={m1["auc"]:.3f}')
    ax_roc.plot(m2['fpr'], m2['tpr'],
                color=M2_COL, linewidth=2, label=f'{name2}  AUC={m2["auc"]:.3f}')
    ax_roc.plot([0,1],[0,1], 'k--', alpha=0.3, linewidth=1)
    ax_roc.fill_between(m1['fpr'], m1['tpr'], alpha=0.08, color=M1_COL)
    ax_roc.fill_between(m2['fpr'], m2['tpr'], alpha=0.08, color=M2_COL)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves', fontsize=11, fontweight='bold')
    ax_roc.legend(fontsize=8)
    ax_roc.grid(alpha=GRID_ALPHA)
    ax_roc.text(0.6, 0.1, 'Random\nbaseline', ha='center', fontsize=7, color='grey')

    # ── Row 1, Col 1 & 2: Score distributions ─────────────────────────────
    for col, (df, name, color) in enumerate(
        [(df1, name1, M1_COL), (df2, name2, M2_COL)]
    ):
        ax = fig.add_subplot(gs[1, col + 1])
        real_scores = df[df['true_label'] == 1]['joint_score']
        fake_scores = df[df['true_label'] == 0]['joint_score']
        bins = np.linspace(0, 1, 25)
        ax.hist(real_scores, bins=bins, alpha=0.65, color=REAL_COL,
                label=f'Real (n={len(real_scores)})', edgecolor='white')
        ax.hist(fake_scores, bins=bins, alpha=0.65, color=FAKE_COL,
                label=f'Fake (n={len(fake_scores)})', edgecolor='white')
        ax.axvline(threshold, color='black', linestyle='--',
                   linewidth=1.2, label=f'Threshold={threshold}')
        ax.set_xlabel('Joint Score  (0=fake, 1=real)')
        ax.set_ylabel('Count')
        ax.set_title(f'{name}\nScore Distribution', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=GRID_ALPHA)

        # Annotation: bimodal = good
        overlap = ((real_scores < threshold) | (fake_scores >= threshold)).sum() \
            if len(real_scores) and len(fake_scores) else 0
        sep = "Well separated ✓" if (real_scores.mean() - fake_scores.mean()) > 0.3 \
            else "Poorly separated ✗"
        ax.text(0.5, 0.95, sep, transform=ax.transAxes,
                ha='center', va='top', fontsize=8,
                color='green' if '✓' in sep else 'red',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # ── Row 2, Col 0 & 1: Confusion matrices ──────────────────────────────
    for col, (m, name) in enumerate([(m1, name1), (m2, name2)]):
        ax = fig.add_subplot(gs[2, col])
        cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
        im = ax.imshow(cm, cmap='Blues', vmin=0)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred Fake', 'Pred Real'])
        ax.set_yticklabels(['Actual Fake', 'Actual Real'])
        ax.set_title(f'{name}\nConfusion Matrix (t={threshold})',
                     fontsize=10, fontweight='bold')
        total = cm.sum()
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                pct = val / total * 100
                ax.text(c, r, f'{val}\n({pct:.1f}%)',
                        ha='center', va='center', fontsize=11,
                        color='white' if val > cm.max() * 0.5 else 'black',
                        fontweight='bold')
        labels = [['TN', 'FP'], ['FN', 'TP']]
        label_colors = [['green', 'red'], ['orange', 'green']]
        for r in range(2):
            for c in range(2):
                ax.text(c, r + 0.38, labels[r][c],
                        ha='center', va='center', fontsize=8,
                        color=label_colors[r][c], alpha=0.8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 2, Col 2: Audio vs Video scatter ──────────────────────────────
    ax_scatter = fig.add_subplot(gs[2, 2])
    for df, name, marker, color in [
        (df1, name1, 'o', M1_COL),
        (df2, name2, '^', M2_COL)
    ]:
        real = df[df['true_label'] == 1]
        fake = df[df['true_label'] == 0]
        ax_scatter.scatter(real['audio_score'], real['video_score'],
                           c=REAL_COL, marker=marker, alpha=0.5, s=25,
                           label=f'{name} Real')
        ax_scatter.scatter(fake['audio_score'], fake['video_score'],
                           c=FAKE_COL, marker=marker, alpha=0.5, s=25,
                           label=f'{name} Fake')
    ax_scatter.axvline(0.5, color='grey', linestyle='--', alpha=0.4, linewidth=0.8)
    ax_scatter.axhline(0.5, color='grey', linestyle='--', alpha=0.4, linewidth=0.8)
    ax_scatter.set_xlabel('Audio Score')
    ax_scatter.set_ylabel('Video Score')
    ax_scatter.set_title('Audio vs Video Scores\n(both models)', fontsize=10, fontweight='bold')
    ax_scatter.set_xlim(-0.02, 1.02)
    ax_scatter.set_ylim(-0.02, 1.02)
    ax_scatter.legend(fontsize=7, ncol=2)
    ax_scatter.grid(alpha=GRID_ALPHA)

    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")
    return plot_path


def plot_training_history(info1, name1, info2, name2, output_dir):
    """Plot training AUC and loss curves if checkpoint history is available."""
    h1 = info1.get('history', {})
    h2 = info2.get('history', {})

    if not h1.get('val_auc_joint') and not h2.get('val_auc_joint'):
        print("  No training history available — skipping history plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History Comparison', fontsize=14, fontweight='bold')

    M1_COL = '#3498db'
    M2_COL = '#e67e22'

    for ax, key, ylabel, title in [
        (axes[0], 'val_auc_joint', 'Val Joint AUC', 'Validation AUC over Epochs'),
        (axes[1], 'train_loss',    'Train Loss',     'Training Loss over Epochs'),
    ]:
        if key in h1 and h1[key]:
            epochs1 = range(1, len(h1[key]) + 1)
            ax.plot(epochs1, h1[key], color=M1_COL, linewidth=2,
                    marker='o', markersize=4, label=name1)
            best1 = max(h1[key]) if 'auc' in key else min(h1[key])
            best1_ep = h1[key].index(best1) + 1
            ax.axvline(best1_ep, color=M1_COL, linestyle='--', alpha=0.4)

        if key in h2 and h2[key]:
            epochs2 = range(1, len(h2[key]) + 1)
            ax.plot(epochs2, h2[key], color=M2_COL, linewidth=2,
                    marker='^', markersize=4, label=name2)
            best2 = max(h2[key]) if 'auc' in key else min(h2[key])
            best2_ep = h2[key].index(best2) + 1
            ax.axvline(best2_ep, color=M2_COL, linestyle='--', alpha=0.4)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        if 'auc' in key:
            ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  History plot saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Compare two AV Deepfake Detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument('--model1',      required=True,  help='Path to first  best_model.pth')
    p.add_argument('--model2',      required=True,  help='Path to second best_model.pth')
    p.add_argument('--name1',       default=None,   help='Display name for model 1')
    p.add_argument('--name2',       default=None,   help='Display name for model 2')
    p.add_argument('--video_dir',   required=True,  help='Folder of test videos (see usage)')
    p.add_argument('--output_dir',  default='eval_results', help='Where to save results')
    p.add_argument('--threshold',   type=float, default=0.5,
                   help='Decision threshold for real/fake (default 0.5)')
    p.add_argument('--n_windows',   type=int, default=3,
                   help='Windows averaged per video (default 3)')
    p.add_argument('--device',      default='auto',
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

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("LOADING MODELS")
    print(f"{'─'*55}")

    model1, info1 = load_model(args.model1, device)
    model2, info2 = load_model(args.model2, device)

    name1 = args.name1 or f"Model 1 (ep {info1['epoch']}, AUC {info1['best_auc']:.3f})"
    name2 = args.name2 or f"Model 2 (ep {info2['epoch']}, AUC {info2['best_auc']:.3f})"

    print(f"\n  {name1}")
    print(f"    Epoch: {info1['epoch']}  |  Best train AUC: {info1['best_auc']}  |  Fusion: {info1['fusion']}")
    print(f"\n  {name2}")
    print(f"    Epoch: {info2['epoch']}  |  Best train AUC: {info2['best_auc']}  |  Fusion: {info2['fusion']}")

    # ── Discover test videos ──────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("TEST VIDEOS")
    print(f"{'─'*55}")
    videos = discover_videos(args.video_dir)

    # ── Run inference ─────────────────────────────────────────────────────────
    results1 = []
    results2 = []

    for path, true_label in tqdm(videos, desc="Running inference", unit="video"):
        for model, results, name in [
            (model1, results1, name1),
            (model2, results2, name2),
        ]:
            r = predict(model, path, device,
                        n_windows=args.n_windows, threshold=args.threshold)
            if r is None:
                print(f"  ⚠ Skipped (unreadable): {os.path.basename(path)}")
                continue
            r['true_label'] = true_label
            r['true_verdict'] = 'REAL' if true_label == 1 else 'FAKE'
            r['correct'] = (r['verdict'] == r['true_verdict'])
            results.append(r)

    if not results1 or not results2:
        print("ERROR: No results — check video paths.")
        sys.exit(1)

    df1 = pd.DataFrame(results1)
    df2 = pd.DataFrame(results2)

    # ── Compute metrics ───────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("METRICS")
    print(f"{'─'*55}")

    m1 = compute_metrics(df1, threshold=args.threshold)
    m2 = compute_metrics(df2, threshold=args.threshold)

    print_metrics(m1, name1, threshold=args.threshold)
    print_metrics(m2, name2, threshold=args.threshold)

    # ── Winner summary ────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("WINNER")
    print(f"{'─'*55}")

    comparisons = [
        ('AUC',       m1['auc'],       m2['auc'],       'higher'),
        ('Accuracy',  m1['accuracy'],  m2['accuracy'],  'higher'),
        ('F1 Score',  m1['f1'],        m2['f1'],        'higher'),
        ('False Neg', m1['fn'],        m2['fn'],        'lower'),
    ]

    wins1 = wins2 = 0
    for metric, v1, v2, better in comparisons:
        if better == 'higher':
            winner = name1 if v1 > v2 else name2 if v2 > v1 else 'TIE'
            v1s, v2s = f'{v1:.4f}', f'{v2:.4f}'
        else:
            winner = name1 if v1 < v2 else name2 if v2 < v1 else 'TIE'
            v1s, v2s = str(v1), str(v2)
        sym = '→' if winner != 'TIE' else '='
        print(f"  {metric:12s}  {v1s} vs {v2s}  {sym}  {winner}")
        if winner == name1:
            wins1 += 1
        elif winner == name2:
            wins2 += 1

    print()
    if wins1 > wins2:
        print(f"  ✓ Overall winner: {name1} ({wins1}/{len(comparisons)} metrics)")
    elif wins2 > wins1:
        print(f"  ✓ Overall winner: {name2} ({wins2}/{len(comparisons)} metrics)")
    else:
        print(f"  = Models tied ({wins1}/{len(comparisons)} each) — use AUC as tiebreaker")

    # ── Save results ──────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("SAVING RESULTS")
    print(f"{'─'*55}")

    csv1 = os.path.join(args.output_dir, 'model1_predictions.csv')
    csv2 = os.path.join(args.output_dir, 'model2_predictions.csv')
    df1.to_csv(csv1, index=False)
    df2.to_csv(csv2, index=False)
    print(f"  Predictions saved: {csv1}")
    print(f"  Predictions saved: {csv2}")

    # Save metrics summary
    summary = {
        name1: {k: v for k, v in m1.items() if not isinstance(v, np.ndarray)},
        name2: {k: v for k, v in m2.items() if not isinstance(v, np.ndarray)},
        'config': {
            'threshold': args.threshold,
            'n_windows': args.n_windows,
            'device':    str(device),
            'model1':    args.model1,
            'model2':    args.model2,
        }
    }
    summary_path = os.path.join(args.output_dir, 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Metrics summary:   {summary_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_comparison(df1, m1, name1, df2, m2, name2, args.output_dir, args.threshold)
    plot_training_history(info1, name1, info2, name2, args.output_dir)

    print(f"\n{'='*55}")
    print(f"  Done. All results in: {os.path.abspath(args.output_dir)}")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()