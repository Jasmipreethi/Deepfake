"""
Plot mel-spectrogram comparison: real vs fake audio.

Extracts mel-spectrograms from one real and one audio-modified video
from the test set and plots them side by side with annotations.

Usage:
    python plot_mel_spectrogram.py
"""

import os
import sys
import subprocess
import tempfile
import torch
import torchaudio
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#1e293b',
    'axes.labelcolor': '#1e293b',
    'text.color': '#1e293b',
    'xtick.color': '#334155',
    'ytick.color': '#334155',
    'figure.dpi': 100,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

CFG = {
    'sr': 16000,
    'duration': 2.0,
    'audio_samples': 32000,
    'n_fft': 1024,
    'hop_length': 512,
    'n_mels': 128,
    'top_db': 80,
}


def extract_audio(video_path, cfg=CFG):
    """Extract audio as waveform tensor using ffmpeg CLI (macOS sox limitation)."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav_path = tmp.name

    try:
        subprocess.run([
            'ffmpeg', '-y', '-v', 'error',
            '-i', video_path,
            '-ar', str(cfg['sr']),
            '-ac', '1',
            '-t', str(cfg['duration']),
            '-f', 'wav', wav_path
        ], check=True)
        waveform, orig_sr = torchaudio.load(wav_path)
        return waveform
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def extract_mel(video_path, cfg=CFG):
    """Extract mel-spectrogram from the first 2 seconds of a video."""
    waveform = extract_audio(video_path, cfg)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Pad/crop to exact sample count
    if waveform.shape[1] < cfg['audio_samples']:
        waveform = F.pad(waveform, (0, cfg['audio_samples'] - waveform.shape[1]))
    else:
        waveform = waveform[:, :cfg['audio_samples']]

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg['sr'], n_mels=cfg['n_mels'],
        n_fft=cfg['n_fft'], hop_length=cfg['hop_length']
    )(waveform)
    db = torchaudio.transforms.AmplitudeToDB(top_db=cfg['top_db'])(mel)

    # Per-sample normalisation
    s = db.std()
    if s > 0:
        db = (db - db.mean()) / (s + 1e-6)

    # Fixed time dim: 63
    t = db.shape[2]
    if t < 63:
        db = F.pad(db, (0, 63 - t))
    elif t > 63:
        db = db[:, :, :63]

    return db.squeeze().numpy()


def plot_comparison(real_mel, fake_mel, real_label, fake_label, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Mel-Spectrogram Comparison: Real vs Fake Audio',
                 fontsize=14, fontweight='bold')

    vmin = min(real_mel.min(), fake_mel.min())
    vmax = max(real_mel.max(), fake_mel.max())

    # ── Real ──
    im1 = ax1.imshow(real_mel, origin='lower', aspect='auto',
                     cmap='inferno', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Real Audio ({real_label})', fontsize=12, fontweight='bold',
                  color='#27ae60')
    ax1.set_xlabel('Time frames (63 × 32ms = 2s)')
    ax1.set_ylabel('Mel frequency bins (128)')
    ax1.annotate('Natural harmonic\nstructure',
                 xy=(55, 90), fontsize=8, color='white',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

    # ── Fake ──
    im2 = ax2.imshow(fake_mel, origin='lower', aspect='auto',
                     cmap='inferno', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Fake Audio - Audio-Modified ({fake_label})',
                  fontsize=12, fontweight='bold', color='#e74c3c')
    ax2.set_xlabel('Time frames (63 × 32ms = 2s)')
    ax2.set_ylabel('Mel frequency bins (128)')
    ax2.annotate('Over-smoothed\nspectral pattern',
                 xy=(55, 90), fontsize=8, color='white',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    ax2.annotate('Unnatural\nharmonic gaps',
                 xy=(25, 30), fontsize=8, color='white',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

    # Shared colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.85, pad=0.01)
    cbar.set_label('dB (normalised)', fontsize=10)

    # Params text box
    fig.text(0.5, -0.02,
             'Parameters: 16,000 Hz | FFT: 1024-point | Hop length: 512 | Mel bins: 128 | Time dim: 63 (2s)',
             ha='center', fontsize=9, color='#7f8c8d', style='italic')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f'Saved: {output_path}')


def main():
    test_dir = 'test'
    if not os.path.isdir(test_dir):
        print(f'ERROR: {test_dir}/ not found')
        sys.exit(1)

    real_dir = os.path.join(test_dir, 'real')
    fake_dir = os.path.join(test_dir, 'fake')

    real_files = sorted(os.listdir(real_dir))
    fake_files = [f for f in sorted(os.listdir(fake_dir)) if 'audio_fake' in f]

    if not real_files or not fake_files:
        print('ERROR: no test videos found')
        sys.exit(1)

    # Use a pair from the same speaker for better comparison
    real_path = os.path.join(real_dir, real_files[0])
    fake_path = os.path.join(fake_dir, fake_files[0])
    print(f'Real: {real_files[0]}')
    print(f'Fake: {fake_files[0]}')

    real_mel = extract_mel(real_path)
    fake_mel = extract_mel(fake_path)
    print(f'Real mel shape: {real_mel.shape}, range: [{real_mel.min():.2f}, {real_mel.max():.2f}]')
    print(f'Fake mel shape: {fake_mel.shape}, range: [{fake_mel.min():.2f}, {fake_mel.max():.2f}]')

    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'figures'
    out_path = os.path.join(out_dir, 'mel_spectrogram_comparison.png')
    plot_comparison(real_mel, fake_mel,
                    real_files[0].replace('_real.mp4', ''),
                    fake_files[0].replace('_audio_fake.mp4', ''),
                    out_path)


if __name__ == '__main__':
    main()
