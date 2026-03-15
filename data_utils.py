"""
Data utilities for AV Deepfake Detection
Optimized for large-scale datasets with lazy-loading from disk.
"""

import sys
#sys.path.insert(0, '/content/drive/MyDrive/Colab Notebooks/Deepfake') --Colab only

import os
import json
import random
import numpy as np
import pandas as pd
import cv2
import torchaudio
import torch
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from config import VAL_DIR, FEATURE_CONFIG

_MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
    sample_rate=FEATURE_CONFIG['sr'],
    n_mels=128,
    n_fft=1024,
    hop_length=512
)
_DB_TRANSFORM = torchaudio.transforms.AmplitudeToDB(top_db=80)

def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(val_dir=VAL_DIR):
    """Load and return metadata DataFrame"""
    metadata_paths = [
        os.path.join(val_dir, 'val_metadata.json'),
        os.path.join(val_dir, '..', 'val_metadata.json'),
        os.path.join(os.path.dirname(val_dir), 'val_metadata.json'),
    ]
    
    metadata_path = None
    for path in metadata_paths:
        if os.path.exists(path):
            metadata_path = path
            print(f"✓ Found metadata at: {path}")
            break
    
    if metadata_path is None:
        raise FileNotFoundError("Could not find val_metadata.json")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    df = pd.DataFrame(metadata)
    print(f"✓ Loaded {len(df):,} entries")
    print(f"\nModification types:")
    print(df['modify_type'].value_counts())
    
    return df

def spec_augment(audio_tensor, freq_mask_param=20, time_mask_param=15):
    """Apply SpecAugment: random frequency and time masking."""
    _, n_mels, n_time = audio_tensor.shape
    
    # Frequency masking: zero out a random band of mel bins
    f = random.randint(0, freq_mask_param)
    f0 = random.randint(0, max(1, n_mels - f))
    audio_tensor[:, f0:f0 + f, :] = 0.0
    
    # Time masking: zero out a random band of time steps
    t = random.randint(0, time_mask_param)
    t0 = random.randint(0, max(1, n_time - t))
    audio_tensor[:, :, t0:t0 + t] = 0.0
    
    return audio_tensor

def sample_videos(df, samples_per_type, val_split=0.2, seed=42, use_all=False):
    """Sample balanced dataset with train/val split BY SPEAKER
    
    Args:
        df: metadata DataFrame
        samples_per_type: dict of {modify_type: count} for subset mode
        val_split: fraction of data for validation
        seed: random seed
        use_all: if True, use ALL videos with audio (ignore samples_per_type)
    
    The split is done by speaker ID so that no speaker appears in both
    train and val, preventing identity leakage.
    """
    set_seeds(seed)
    
    df_valid = df[(df['audio_frames'] > 0) & (df['video_frames'] > 0)].copy()
    dropped = len(df) - len(df_valid)
    print(f"\nVideos with both audio and video: {len(df_valid):,} (dropped {dropped:,})")
    
    if use_all:
        # Use all videos with audio
        mini_df = df_valid.reset_index(drop=True)
        print(f"\nUsing ALL {len(mini_df):,} videos:")
        for mod_type in mini_df['modify_type'].unique():
            count = len(mini_df[mini_df['modify_type'] == mod_type])
            print(f"  ✓ {mod_type:20s}: {count:,} videos")
    else:
        # Subset mode: sample fixed count per type
        samples = []
        for mod_type, count in samples_per_type.items():
            subset = df_valid[df_valid['modify_type'] == mod_type]
            if len(subset) >= count:
                sampled = subset.sample(count, random_state=seed)
                samples.append(sampled)
                print(f"✓ {mod_type:20s}: {count} samples")
            else:
                print(f"⚠ {mod_type:20s}: only {len(subset)} available, taking all")
                samples.append(subset)
        mini_df = pd.concat(samples).reset_index(drop=True)
    
    # Extract speaker IDs from file paths (e.g. "source/speaker_id/video.mp4")
    mini_df['speaker'] = mini_df['file'].apply(
        lambda f: f.split('/')[1] if '/' in f and len(f.split('/')) > 1 else 'unknown'
    )
    
    n_speakers = mini_df['speaker'].nunique()
    print(f"\nUnique speakers: {n_speakers:,}")
    
    # Speaker-based split: all videos from one speaker stay in the same split
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(gss.split(mini_df, groups=mini_df['speaker']))
    
    train_df = mini_df.iloc[train_idx].reset_index(drop=True)
    val_df = mini_df.iloc[val_idx].reset_index(drop=True)
    
    train_speakers = train_df['speaker'].nunique()
    val_speakers = val_df['speaker'].nunique()
    overlap = set(train_df['speaker'].unique()) & set(val_df['speaker'].unique())
    
    print(f"\n{'='*60}")
    print(f"Speaker-Based Split Summary:")
    print(f"  Train: {len(train_df):,} videos from {train_speakers:,} speakers")
    print(f"  Val:   {len(val_df):,} videos from {val_speakers:,} speakers")
    print(f"  Speaker overlap: {len(overlap)} (should be 0)")
    
    # Show per-type distribution in each split
    print(f"\n  Train distribution:")
    for t in sorted(train_df['modify_type'].unique()):
        print(f"    {t:20s}: {len(train_df[train_df['modify_type'] == t]):,}")
    print(f"  Val distribution:")
    for t in sorted(val_df['modify_type'].unique()):
        print(f"    {t:20s}: {len(val_df[val_df['modify_type'] == t]):,}")
    
    return train_df, val_df


def extract_av_features(video_path, fake_segments=None, total_frames=0, cfg=FEATURE_CONFIG,augment=False, start_sec_override=None):
    """Extract synchronized audio and video features from video.
    
    Returns tensors with FIXED shapes regardless of input:
      video: (num_frames, 3, img_size, img_size) = (50, 3, 224, 224)
      audio: (1, 128, 63)  — fixed via n_fft=1024, hop_length=512
    """
    # Determine which 2-second window to extract
    if start_sec_override is not None:
        start_sec = start_sec_override
    elif fake_segments and len(fake_segments) > 0:
        start_sec = fake_segments[0][0]
    else:
        total_sec = total_frames / cfg['fps']
        start_sec = max(0, (total_sec / 2) - (cfg['duration'] / 2))
    
    # Video extraction
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
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std
        frames.append(frame)
    cap.release()
    
    # Pad if needed
    if len(frames) < cfg['num_frames']:
        while len(frames) < cfg['num_frames']:
            frames.append(frames[-1] if frames else np.zeros((cfg['img_size'], cfg['img_size'], 3)))
    
    video_tensor = torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
    
    if augment:
        audio_tensor = spec_augment(audio_tensor)
        # Random horizontal flip applied consistently across all 50 frames
        if random.random() < 0.5:
            video_tensor = torch.flip(video_tensor, dims=[3])  # flip width dimension
        
        # Random brightness/contrast jitter applied to the whole clip
        brightness = random.uniform(-0.2, 0.2)
        contrast   = random.uniform(0.8, 1.2)
        video_tensor = torch.clamp(video_tensor * contrast + brightness, 0.0, 1.0)
    D
    # Audio extraction using torchaudio
    try:
        # Load full audio, then slice to the 2-second window
        waveform, orig_sr = torchaudio.load(video_path)
        
        # Resample to target sample rate if needed
        if orig_sr != cfg['sr']:
            resampler = torchaudio.transforms.Resample(orig_sr, cfg['sr'])
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Slice to the 2-second window
        start_sample = int(start_sec * cfg['sr'])
        end_sample = start_sample + cfg['audio_samples']
        waveform = waveform[:, start_sample:end_sample]
        
        # Pad if too short
        if waveform.shape[1] < cfg['audio_samples']:
            pad_size = cfg['audio_samples'] - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        
        # Compute mel-spectrogram
        mel_spec = _MEL_TRANSFORM(waveform)
        audio_tensor = _DB_TRANSFORM(mel_spec)
        
    except Exception as e:
        # Fallback for corrupted/unreadable audio: use zero tensor
        # Per-sample mean/std normalisation
        mean = audio_tensor.mean()
        std  = audio_tensor.std()
        if std > 0:
            audio_tensor = (audio_tensor - mean) / (std + 1e-6)
    
    return video_tensor, audio_tensor

def extract_multiple_windows(video_path, fake_segments=None, total_frames=0,
                              cfg=FEATURE_CONFIG, n_windows=3):
    """Extract n windows, prioritising fake segments when known."""
    total_sec = total_frames / cfg['fps']
    duration = cfg['duration']
    max_start = max(0, total_sec - duration)

    starts = []

    if fake_segments and len(fake_segments) > 0:
        # Always add a window at the start of each known fake segment
        for seg in fake_segments:
            seg_start = seg[0]
            # Clamp so the window doesn't run past the end of the video
            clamped = min(seg_start, max_start)
            starts.append(clamped)
            if len(starts) >= n_windows:
                break

        # If we still have room, fill remaining slots with evenly spaced windows
        remaining = n_windows - len(starts)
        if remaining > 0 and max_start > 0:
            for i in range(remaining):
                evenly_spaced = max_start * i / (remaining - 1) if remaining > 1 else 0.0
                # Only add if it's not already covered by a fake segment window
                if not any(abs(evenly_spaced - s) < duration for s in starts):
                    starts.append(evenly_spaced)
    else:
        # Real video — spread evenly across the full duration
        if n_windows == 1 or max_start == 0:
            starts = [0.0]
        else:
            starts = [max_start * i / (n_windows - 1) for i in range(n_windows)]

    # Extract each window
    windows = []
    for start_sec in starts:
        v, a = extract_av_features(
            video_path, fake_segments, total_frames, cfg,
            augment=False,
            start_sec_override=start_sec
        )
        if v is not None:
            windows.append((v, a))

    return windows

# =============================================================================
# DISK-BASED FEATURE STORAGE (saves each video as individual .pt file)
# =============================================================================

def process_split_to_disk(split_df, split_name, feature_dir, val_dir=VAL_DIR):
    """Extract features for a split and savmel_te each as an individual .pt file.
    
    Saves:
      feature_dir/{split_name}/0.pt, 1.pt, 2.pt, ...
      feature_dir/{split_name}_manifest.json  (metadata for each sample)
    
    Supports resuming — skips videos whose .pt files already exist.
    """
    split_dir = os.path.join(feature_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    manifest_path = os.path.join(feature_dir, f'{split_name}_manifest.json')
    failed_path = os.path.join(feature_dir, f'{split_name}_failed.json')
    
    # Load existing manifest for resume support
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    else:
        manifest = []
    
    # Load previously failed indices to skip them
    if os.path.exists(failed_path):
        with open(failed_path, 'r') as f:
            failed_indices = set(json.load(f))
    else:
        failed_indices = set()
    
    existing_indices = {m['idx'] for m in manifest}
    
    success = 0
    skipped = 0
    failed = 0
    
    print(f"\nProcessing {split_name} set ({len(split_df):,} videos)...")
    if existing_indices:
        print(f"  Resuming: {len(existing_indices):,} already extracted")
    if failed_indices:
        print(f"  Skipping: {len(failed_indices):,} previously failed")
    
    for idx, row in split_df.iterrows():
        pt_path = os.path.join(split_dir, f'{idx}.pt')
        
        # Skip previously failed files
        if idx in failed_indices:
            failed += 1
            continue
        
        # Skip if .pt file already exists on disk
        if idx in existing_indices and os.path.exists(pt_path):
            skipped += 1
            continue
        
        # Also skip if .pt file exists but wasn't in manifest (rebuild manifest entry)
        if os.path.exists(pt_path):
            mt = row['modify_type']
            manifest.append({
                'idx': idx,
                'file': row['file'],
                'type': mt,
                'speaker': row['file'].split('/')[1] if '/' in row['file'] else 'unknown',
                'pt_file': f'{idx}.pt',
                'fake_segments': row.get('fake_segments', []),
                'total_frames':  int(row['video_frames'])
            })
            skipped += 1
            continue
        
        # Progress logging (every 100 videos)
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  [{split_name} {idx+1}/{len(split_df)}] "
                  f"done={success + skipped} failed={failed}")
        
        video_path = os.path.join(val_dir, row['file'])
        
        if not os.path.exists(video_path):
            failed += 1
            failed_indices.add(idx)
            continue
        
        video_tensor, audio_tensor = extract_av_features(
            video_path=video_path,
            fake_segments=row.get('fake_segments'),
            total_frames=row['video_frames'],
            augment=(split_name == 'train')
        )
        
        if video_tensor is None:
            failed += 1
            failed_indices.add(idx)
            continue
        
        # Create labels
        mt = row['modify_type']
        label_map = {
            'real': (1, 1),
            'both_modified': (0, 0),
            'audio_modified': (0, 1),
            'visual_modified': (1, 0)
        }
        audio_label, video_label = label_map.get(mt, (0, 0))
        
        # Save individual .pt file
        feature_data = {
            'video': video_tensor,
            'audio': audio_tensor,
            'labels': torch.FloatTensor([audio_label, video_label]),
        }
        pt_path = os.path.join(split_dir, f'{idx}.pt')
        torch.save(feature_data, pt_path)
        
        # Update manifest
        manifest.append({
            'idx': idx,
            'file': row['file'],
            'type': mt,
            'speaker': row['file'].split('/')[1] if '/' in row['file'] else 'unknown',
            'pt_file': f'{idx}.pt'
        })
        success += 1
        
        # Save manifest and failed list periodically (every 500 videos) for crash safety
        if success % 500 == 0:
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)
            with open(failed_path, 'w') as f:
                json.dump(list(failed_indices), f)
    
    # Final save
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    with open(failed_path, 'w') as f:
        json.dump(list(failed_indices), f)
    
    total = success + skipped
    print(f"\n  {split_name} Summary: {total:,} extracted "
          f"({skipped:,} cached, {success:,} new, {failed:,} failed)")
    
    return split_dir, manifest_path


def extract_av_features(video_path, fake_segments=None, total_frames=0, cfg=FEATURE_CONFIG, augment=False):
    """Extract features for both splits, saving to disk.
    
    Returns (train_dir, train_manifest, val_dir, val_manifest) paths.
    """
    train_manifest = os.path.join(feature_dir, 'train_manifest.json')
    val_manifest = os.path.join(feature_dir, 'val_manifest.json')
    
    # Check if extraction is already complete
    if use_cache and os.path.exists(train_manifest) and os.path.exists(val_manifest):
        with open(train_manifest, 'r') as f:
            t_man = json.load(f)
        with open(val_manifest, 'r') as f:
            v_man = json.load(f)
        
        # Only skip if extraction covered most of the data
        if len(t_man) >= len(train_df) * 0.95 and len(v_man) >= len(val_df) * 0.95:
            print("=" * 60)
            print(f"Features already extracted on disk.")
            print(f"  Train: {len(t_man):,} features")
            print(f"  Val:   {len(v_man):,} features")
            print("=" * 60)
            return (os.path.join(feature_dir, 'train'), train_manifest,
                    os.path.join(feature_dir, 'val'), val_manifest)
    
    print("=" * 60)
    print("FEATURE EXTRACTION (saving to disk)")
    print("=" * 60)
    
    train_dir, train_manifest = process_split_to_disk(
        train_df, "train", feature_dir, val_dir
    )
    val_dir_out, val_manifest = process_split_to_disk(
        val_df, "val", feature_dir, val_dir
    )
    
    return train_dir, train_manifest, val_dir_out, val_manifest


# =============================================================================
# LAZY-LOADING DATASET (loads one .pt at a time from disk)
# =============================================================================

class AVDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that lazy-loads features from individual .pt files.
    
    Only one video's tensors are in memory at a time per worker,
    so RAM usage stays constant regardless of dataset size.
    """
    def __init__(self, feature_dir, manifest_path):
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        self.feature_dir = feature_dir
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        entry = self.manifest[idx]
        pt_path = os.path.join(self.feature_dir, entry['pt_file'])
        data = torch.load(pt_path, map_location='cpu', weights_only=True)
        return {
            'video': data['video'],
            'audio': data['audio'],
            'labels': data['labels'],
            'type': entry['type'],
            'file': entry['file'],
            'fake_segments': entry.get('fake_segments', []),
            'total_frames':  entry.get('total_frames', 0)   
        }


def create_dataloaders(train_dir, train_manifest, val_dir, val_manifest, 
                       batch_size=16, num_workers=2):
    """Create DataLoaders with lazy-loading datasets."""
    train_loader = torch.utils.data.DataLoader(
        AVDataset(train_dir, train_manifest),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        AVDataset(val_dir, val_manifest),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
