"""
Standalone Inference Script for AV Deepfake Detection

Loads a trained best_model.pth and runs inference on a single video
or a folder of videos. No training pipeline dependencies required —
just the model weights and this file.

Usage:
    # Single video
    python inference.py --model best_model.pth --video path/to/video.mp4

    # Folder of videos
    python inference.py --model best_model.pth --video_dir path/to/videos/

    # Save results to CSV
    python inference.py --model best_model.pth --video_dir path/to/videos/ --output results.csv

    # Use CPU explicitly
    python inference.py --model best_model.pth --video path/to/video.mp4 --device cpu

Requirements:
    pip install torch torchvision torchaudio opencv-python-headless
"""

import os
import sys
import argparse
import json
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

warnings.filterwarnings('ignore')


# =============================================================================
# FEATURE EXTRACTION (self-contained, no data_utils dependency)
# =============================================================================

FEATURE_CONFIG = {
    'sr':            16000,
    'fps':           25,
    'duration':      2.0,
    'num_frames':    50,
    'img_size':      224,
    'audio_samples': 32000,
    'target_t':      63,     # fixed mel time dimension
}


def extract_features(video_path, cfg=FEATURE_CONFIG):
    """Extract audio and video features from a 2-second window.

    Tries to use the middle 2 seconds of the video.
    Returns (video_tensor, audio_tensor) or (None, None) on failure.

    video_tensor: (50, 3, 224, 224)
    audio_tensor: (1, 128, 63)
    """
    # ---- Video ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or cfg['num_frames']
    total_sec    = total_frames / cfg['fps']
    start_sec    = max(0, (total_sec / 2) - (cfg['duration'] / 2))
    start_frame  = int(start_sec * cfg['fps'])

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

    if len(frames) == 0:
        return None, None

    # Pad if short
    while len(frames) < cfg['num_frames']:
        frames.append(frames[-1])

    # ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    frames_arr = (np.array(frames) - mean) / std
    video_tensor = torch.FloatTensor(frames_arr).permute(0, 3, 1, 2)  # (50, 3, 224, 224)

    # ---- Audio ----
    try:
        waveform, orig_sr = torchaudio.load(video_path)

        if orig_sr != cfg['sr']:
            waveform = torchaudio.transforms.Resample(orig_sr, cfg['sr'])(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        start_sample = int(start_sec * cfg['sr'])
        end_sample   = start_sample + cfg['audio_samples']
        waveform     = waveform[:, start_sample:end_sample]

        if waveform.shape[1] < cfg['audio_samples']:
            waveform = F.pad(waveform, (0, cfg['audio_samples'] - waveform.shape[1]))

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg['sr'], n_mels=128, n_fft=1024, hop_length=512
        )(waveform)
        audio_tensor = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel)

        # Per-sample normalisation
        std_val = audio_tensor.std()
        if std_val > 0:
            audio_tensor = (audio_tensor - audio_tensor.mean()) / (std_val + 1e-6)

        # Force fixed time dimension
        t = audio_tensor.shape[2]
        if t < cfg['target_t']:
            audio_tensor = F.pad(audio_tensor, (0, cfg['target_t'] - t))
        elif t > cfg['target_t']:
            audio_tensor = audio_tensor[:, :, :cfg['target_t']]

    except Exception:
        audio_tensor = torch.zeros(1, 128, cfg['target_t'])

    return video_tensor, audio_tensor


def extract_multiple_windows(video_path, n_windows=3, cfg=FEATURE_CONFIG):
    """Extract n evenly-spaced windows and return list of (video, audio) tensors."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or cfg['num_frames']
    cap.release()

    total_sec = total_frames / cfg['fps']
    max_start = max(0.0, total_sec - cfg['duration'])

    if n_windows == 1 or max_start == 0:
        starts = [0.0]
    else:
        starts = [max_start * i / (n_windows - 1) for i in range(n_windows)]

    windows = []
    for start_sec in starts:
        v, a = extract_features.__wrapped__(video_path, cfg) \
            if hasattr(extract_features, '__wrapped__') \
            else _extract_at(video_path, start_sec, cfg)
        if v is not None:
            windows.append((v, a))
    return windows


def _extract_at(video_path, start_sec, cfg=FEATURE_CONFIG):
    """Extract a single window starting at start_sec."""
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

    if len(frames) == 0:
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
        start_sample = int(start_sec * cfg['sr'])
        end_sample   = start_sample + cfg['audio_samples']
        waveform     = waveform[:, start_sample:end_sample]
        if waveform.shape[1] < cfg['audio_samples']:
            waveform = F.pad(waveform, (0, cfg['audio_samples'] - waveform.shape[1]))
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg['sr'], n_mels=128, n_fft=1024, hop_length=512
        )(waveform)
        audio_tensor = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel)
        std_val = audio_tensor.std()
        if std_val > 0:
            audio_tensor = (audio_tensor - audio_tensor.mean()) / (std_val + 1e-6)
        t = audio_tensor.shape[2]
        if t < cfg['target_t']:
            audio_tensor = F.pad(audio_tensor, (0, cfg['target_t'] - t))
        elif t > cfg['target_t']:
            audio_tensor = audio_tensor[:, :, :cfg['target_t']]
    except Exception:
        audio_tensor = torch.zeros(1, 128, cfg['target_t'])

    return video_tensor, audio_tensor


# =============================================================================
# MODEL ARCHITECTURE (mirrors main.py — must match training exactly)
# =============================================================================

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.video import r3d_18, R3D_18_Weights


class PretrainedAudioEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)


class PretrainedVideoEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.backbone = r3d_18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # (B,T,C,H,W) -> (B,C,T,H,W)
        return self.backbone(x)


class TransformerFusion(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=512, num_heads=8, num_layers=2, dropout=0.4):
        super().__init__()
        self.audio_proj  = nn.Linear(feature_dim, hidden_dim)
        self.video_proj  = nn.Linear(feature_dim, hidden_dim)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.audio_classifier = nn.Linear(hidden_dim, 1)
        self.video_classifier = nn.Linear(hidden_dim, 1)
        self.joint_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, video_feat, audio_feat):
        B = video_feat.shape[0]
        v   = self.video_proj(video_feat).unsqueeze(1)
        a   = self.audio_proj(audio_feat).unsqueeze(1)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, v, a], dim=1) + self.pos_embedding
        fused  = self.transformer(tokens)
        cls_out = fused[:, 0, :]
        return {
            'audio_pred': torch.sigmoid(self.audio_classifier(cls_out)),
            'video_pred': torch.sigmoid(self.video_classifier(cls_out)),
            'joint_pred': torch.sigmoid(self.joint_classifier(cls_out)),
        }


class PretrainedFusion(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=512, dropout=0.4):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout / 2)
        )
        self.audio_classifier = nn.Linear(hidden_dim, 1)
        self.video_classifier = nn.Linear(hidden_dim, 1)
        self.joint_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, video_feat, audio_feat):
        fused = self.fusion(torch.cat([video_feat, audio_feat], dim=1))
        return {
            'audio_pred': torch.sigmoid(self.audio_classifier(fused)),
            'video_pred': torch.sigmoid(self.video_classifier(fused)),
            'joint_pred': torch.sigmoid(self.joint_classifier(fused)),
        }


class AVDeepfakeDetector(nn.Module):
    def __init__(self, fusion_type='transformer', feature_dim=256, hidden_dim=512, dropout=0.4):
        super().__init__()
        self.video_encoder = PretrainedVideoEncoder(feature_dim)
        self.audio_encoder = PretrainedAudioEncoder(feature_dim)
        if fusion_type == 'transformer':
            self.fusion_module = TransformerFusion(feature_dim, hidden_dim, dropout=dropout)
        else:
            self.fusion_module = PretrainedFusion(feature_dim, hidden_dim, dropout)

    def forward(self, video, audio):
        return self.fusion_module(self.video_encoder(video), self.audio_encoder(audio))


# =============================================================================
# INFERENCE
# =============================================================================

def load_model(model_path, device):
    """Load model from .pth checkpoint."""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Detect fusion type from checkpoint keys
    state = checkpoint.get('model_state_dict', checkpoint)

    # Unwrap DataParallel prefix if saved with DataParallel
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}

    # Detect fusion type from state dict keys
    if any('transformer' in k for k in state.keys()):
        fusion_type = 'transformer'
    else:
        fusion_type = 'pretrained'

    model = AVDeepfakeDetector(fusion_type=fusion_type)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    auc   = checkpoint.get('best_val_auc', 'unknown')
    print(f"  ✓ Loaded checkpoint — epoch: {epoch}, best val AUC: {auc}")
    return model


def predict_video(model, video_path, device, n_windows=3):
    """Run inference on a single video.

    Extracts n_windows evenly-spaced windows and averages predictions.

    Returns dict with keys: audio_pred, video_pred, joint_pred, verdict
    """
    windows = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 50
    cap.release()

    total_sec = total_frames / FEATURE_CONFIG['fps']
    max_start = max(0.0, total_sec - FEATURE_CONFIG['duration'])
    starts = (
        [max_start * i / (n_windows - 1) for i in range(n_windows)]
        if n_windows > 1 and max_start > 0
        else [0.0]
    )

    for start_sec in starts:
        v, a = _extract_at(video_path, start_sec)
        if v is not None:
            windows.append((v, a))

    if not windows:
        return None

    preds = {'audio': [], 'video': [], 'joint': []}
    with torch.no_grad():
        for v, a in windows:
            out = model(
                v.unsqueeze(0).to(device),
                a.unsqueeze(0).to(device)
            )
            preds['audio'].append(out['audio_pred'].item())
            preds['video'].append(out['video_pred'].item())
            preds['joint'].append(out['joint_pred'].item())

    audio_score = sum(preds['audio']) / len(preds['audio'])
    video_score = sum(preds['video']) / len(preds['video'])
    joint_score = sum(preds['joint']) / len(preds['joint'])

    # Score interpretation:
    # Close to 1.0 = REAL, close to 0.0 = FAKE
    verdict = 'REAL' if joint_score >= 0.5 else 'FAKE'

    return {
        'file':        os.path.basename(video_path),
        'audio_score': round(audio_score, 4),
        'video_score': round(video_score, 4),
        'joint_score': round(joint_score, 4),
        'verdict':     verdict,
        'confidence':  round(abs(joint_score - 0.5) * 2, 4),  # 0=uncertain, 1=certain
    }


def print_result(result):
    """Pretty-print a single prediction result."""
    verdict_color = '\033[92m' if result['verdict'] == 'REAL' else '\033[91m'
    reset = '\033[0m'
    print(f"\n{'─' * 50}")
    print(f"  File     : {result['file']}")
    print(f"  Verdict  : {verdict_color}{result['verdict']}{reset}  "
          f"(confidence: {result['confidence']:.0%})")
    print(f"  Scores   : audio={result['audio_score']:.4f}  "
          f"video={result['video_score']:.4f}  "
          f"joint={result['joint_score']:.4f}")
    print(f"  (1.0 = real, 0.0 = fake)")
    print(f"{'─' * 50}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='AV Deepfake Detection — Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--model',     required=True,
                        help='Path to best_model.pth')
    parser.add_argument('--video',     default=None,
                        help='Path to a single video file')
    parser.add_argument('--video_dir', default=None,
                        help='Path to a folder of video files')
    parser.add_argument('--output',    default=None,
                        help='Save results to this CSV path')
    parser.add_argument('--device',    default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to run on (default: auto)')
    parser.add_argument('--n_windows', type=int, default=3,
                        help='Number of windows to average per video (default: 3)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Joint score threshold for REAL/FAKE (default: 0.5)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    if not os.path.exists(args.model):
        print(f"ERROR: model file not found: {args.model}")
        sys.exit(1)
    model = load_model(args.model, device)

    # Collect video paths
    video_paths = []
    if args.video:
        if not os.path.exists(args.video):
            print(f"ERROR: video not found: {args.video}")
            sys.exit(1)
        video_paths = [args.video]
    elif args.video_dir:
        if not os.path.exists(args.video_dir):
            print(f"ERROR: directory not found: {args.video_dir}")
            sys.exit(1)
        exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_paths = [
            os.path.join(args.video_dir, f)
            for f in sorted(os.listdir(args.video_dir))
            if os.path.splitext(f)[1].lower() in exts
        ]
        print(f"Found {len(video_paths)} videos in {args.video_dir}")
    else:
        print("ERROR: provide --video or --video_dir")
        sys.exit(1)

    # Run inference
    results = []
    for path in tqdm(video_paths, desc="Inference", unit="video"):
        result = predict_video(model, path, device, n_windows=args.n_windows)
        if result is None:
            print(f"  ⚠ Skipped (could not read): {path}")
            continue

        # Apply custom threshold
        result['verdict'] = 'REAL' if result['joint_score'] >= args.threshold else 'FAKE'

        if len(video_paths) == 1:
            print_result(result)
        else:
            verdict_sym = '✓' if result['verdict'] == 'REAL' else '✗'
            print(f"  {verdict_sym} {result['file']:50s} "
                  f"joint={result['joint_score']:.4f} → {result['verdict']}")
        results.append(result)

    # Summary for multi-video runs
    if len(results) > 1:
        fakes = sum(1 for r in results if r['verdict'] == 'FAKE')
        reals = sum(1 for r in results if r['verdict'] == 'REAL')
        print(f"\n{'='*50}")
        print(f"  Total : {len(results)} videos")
        print(f"  REAL  : {reals}")
        print(f"  FAKE  : {fakes}")
        print(f"{'='*50}")

    # Save CSV
    if args.output and results:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()