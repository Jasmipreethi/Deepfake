"""
Data utilities for AV Deepfake Detection
Optimized for large-scale datasets with lazy-loading from disk.
"""

#import sys
#sys.path.insert(0, '/content/drive/MyDrive/Colab Notebooks/Deepfake') --Colab only

import os
import json
import random
import numpy as np
import pandas as pd
import cv2
import torchaudio
import torch
import multiprocessing
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from config import VAL_DIR, FEATURE_CONFIG

# Fix 9: transforms moved inside extract_av_features to avoid pickle
# errors when num_workers > 0 on Windows / some macOS versions.

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
    audio_tensor = audio_tensor.clone()  # Fix 7: avoid mutating the original tensor
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
    
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            f"Speaker-based split produced an empty split "
            f"(train={len(train_df)}, val={len(val_df)}). "
            f"Increase samples_per_type or reduce val_split."
        )

    return train_df, val_df


def extract_av_features(video_path, fake_segments=None, total_frames=0, cfg=FEATURE_CONFIG, augment=False, start_sec_override=None):
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
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return None, None

    if len(frames) < cfg['num_frames']:
        while len(frames) < cfg['num_frames']:
            frames.append(frames[-1])

    frames_arr = np.array(frames)  # (T, H, W, 3) float32 in [0, 1]

    # Fix 3: augmentation applied BEFORE ImageNet normalisation so that
    #         clamp(0, 1) operates on the correct [0, 1] pixel range.
    if augment:
        # Random horizontal flip — consistent across all frames
        if random.random() < 0.5:
            frames_arr = frames_arr[:, :, ::-1, :].copy()

        # Random brightness/contrast jitter — stays in [0, 1] before normalisation
        brightness = random.uniform(-0.2, 0.2)
        contrast   = random.uniform(0.8, 1.2)
        frames_arr = np.clip(frames_arr * contrast + brightness, 0.0, 1.0)

    # ImageNet normalisation (applied after augmentation)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    frames_arr = (frames_arr - mean) / std

    video_tensor = torch.FloatTensor(frames_arr).permute(0, 3, 1, 2)

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
        
        # Compute mel-spectrogram and convert to dB
        # Fix 9: instantiated locally so workers can pickle them safely
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg['sr'],
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )
        db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
        mel_spec = mel_transform(waveform)
        audio_tensor = db_transform(mel_spec)
        
        # Per-sample normalisation
        mean = audio_tensor.mean()
        std  = audio_tensor.std()
        if std > 0:
            audio_tensor = (audio_tensor - mean) / (std + 1e-6)
        
        # Audio augmentation — only after audio_tensor is fully built
        if augment:
            audio_tensor = spec_augment(audio_tensor)

        # Force fixed time dimension — mel output can vary by 1-2 frames
        # depending on exact waveform length. Pad or trim to exactly 63.
        target_t = 63
        t = audio_tensor.shape[2]
        if t < target_t:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, target_t - t))
        elif t > target_t:
            audio_tensor = audio_tensor[:, :, :target_t]

    except Exception as e:
        # Fallback for corrupted/unreadable audio: use zero tensor
        audio_tensor = torch.zeros(1, 128, 63)
    
    return video_tensor, audio_tensor
        

def extract_multiple_windows(video_path, fake_segments=None, total_frames=0,
                              cfg=FEATURE_CONFIG, n_windows=3):
    """Extract n windows, prioritising fake segments when known."""
    # Guard against total_frames=0 causing division by zero
    if total_frames <= 0:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or cfg['num_frames']
        cap.release()
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

    # Fix 6: warn if fewer windows were extracted than requested
    if len(windows) < n_windows:
        import warnings
        warnings.warn(
            f"extract_multiple_windows: requested {n_windows} windows but "
            f"only {len(windows)} could be extracted from {video_path}",
            RuntimeWarning,
            stacklevel=2
        )
    return windows

# =============================================================================
# DISK-BASED FEATURE STORAGE (saves each video as individual .pt file)
# =============================================================================

def _extract_one_video(args):
    """Top-level worker for multiprocessing.Pool (must be module-level for pickle)."""
    idx, row_dict, split_dir, val_dir, split_name = args
    pt_path = os.path.join(split_dir, f'{idx}.pt')

    # Already extracted — rebuild manifest entry
    if os.path.exists(pt_path):
        mt = row_dict['modify_type']
        entry = {
            'idx': idx, 'file': row_dict['file'], 'type': mt,
            'speaker': row_dict['file'].split('/')[1] if '/' in row_dict['file'] else 'unknown',
            'pt_file': f'{idx}.pt',
            'fake_segments': row_dict.get('fake_segments', []),
            'total_frames': int(row_dict['video_frames'])
        }
        return idx, 'skipped', entry, None

    video_path = os.path.join(val_dir, row_dict['file'])
    if not os.path.exists(video_path):
        return idx, 'failed', None, ('File not found: ' + video_path)

    try:
        augment = (split_name == 'train')
        video_tensor, audio_tensor = extract_av_features(
            video_path=video_path,
            fake_segments=row_dict.get('fake_segments'),
            total_frames=row_dict['video_frames'],
            augment=augment
        )
        if video_tensor is None:
            return idx, 'failed', None, ('extract_av_features returned None: ' + video_path)

        mt = row_dict['modify_type']
        label_map = {
            'real': (1, 1), 'both_modified': (0, 0),
            'audio_modified': (0, 1), 'visual_modified': (1, 0)
        }
        audio_label, video_label = label_map.get(mt, (0, 0))
        torch.save({
            'video': video_tensor,
            'audio': audio_tensor,
            'labels': torch.FloatTensor([audio_label, video_label]),
        }, pt_path)

        entry = {
            'idx': idx, 'file': row_dict['file'], 'type': mt,
            'speaker': row_dict['file'].split('/')[1] if '/' in row_dict['file'] else 'unknown',
            'pt_file': f'{idx}.pt',
            'fake_segments': row_dict.get('fake_segments', []),
            'total_frames': int(row_dict['video_frames'])
        }
        return idx, 'success', entry, None

    except Exception as exc:
        return idx, 'failed', None, ('Exception idx=' + str(idx) + ': ' + str(exc))


def process_split_to_disk(split_df, split_name, feature_dir, val_dir=VAL_DIR,
                          num_workers=None):
    """Extract features for a split in parallel and save each as an individual .pt file.
    
    Saves:
      feature_dir/{split_name}/0.pt, 1.pt, 2.pt, ...
      feature_dir/{split_name}_manifest.json  (metadata for each sample)

    Supports resuming — skips videos whose .pt files already exist.
    Uses fork-based multiprocessing for parallel extraction on Linux.
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 4)  # leave 4 for OS + main

    split_dir = os.path.join(feature_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    manifest_path = os.path.join(feature_dir, f'{split_name}_manifest.json')
    failed_path   = os.path.join(feature_dir, f'{split_name}_failed.json')

    # Load existing manifest and failed list for resume support
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    else:
        manifest = []
    if os.path.exists(failed_path):
        with open(failed_path, 'r') as f:
            failed_indices = set(json.load(f))
    else:
        failed_indices = set()
    existing_indices = {m['idx'] for m in manifest}

    # Build work list — only unprocessed, non-failed videos
    work_items = [
        (idx, row.to_dict(), split_dir, val_dir, split_name)
        for idx, row in split_df.iterrows()
        if idx not in failed_indices
        and not (idx in existing_indices and os.path.exists(os.path.join(split_dir, f'{idx}.pt')))
    ]

    already_done = len(existing_indices)
    print(f"\nProcessing {split_name} set ({len(split_df):,} videos) with {num_workers} workers")
    print(f"  Already extracted : {already_done:,}")
    print(f"  Previously failed : {len(failed_indices):,}")
    print(f"  To process now    : {len(work_items):,}")

    success = 0
    failed  = 0
    first_errors = []  # capture first 5 failure reasons for diagnosis

    ctx = multiprocessing.get_context('fork')  # fork is safe + fast on Linux
    with ctx.Pool(processes=num_workers) as pool:
        for i, (idx, status, entry, err_msg) in enumerate(
            pool.imap_unordered(_extract_one_video, work_items, chunksize=4)
        ):
            if status == 'success':
                manifest.append(entry)
                success += 1
            elif status == 'skipped':
                manifest.append(entry)
            elif status == 'failed':
                failed_indices.add(idx)
                failed += 1
                if err_msg and len(first_errors) < 5:
                    first_errors.append(err_msg)

            if (i + 1) % 100 == 0:
                print(f"  [{split_name}] {i+1}/{len(work_items)} "
                      f"| new={success} failed={failed}")

            if success > 0 and success % 500 == 0:
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f)
                with open(failed_path, 'w') as f:
                    json.dump(list(failed_indices), f)

    # Final save
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    with open(failed_path, 'w') as f:
        json.dump(list(failed_indices), f)

    if first_errors:
        print(f"\n  Sample failure reasons (first {len(first_errors)}):")
        for e in first_errors:
            print(f"    - {e}")

    total = already_done + success
    print(f"\n  {split_name} Summary: {total:,} extracted "
          f"({already_done:,} cached, {success:,} new, {failed:,} failed)")

    return split_dir, manifest_path


def extract_all_features(train_df, val_df, val_dir, feature_dir, use_cache=True):
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
        try:
            data = torch.load(pt_path, map_location='cpu', weights_only=True)
            # Defensive shape check — ensures audio is always (1, 128, 63)
            audio = data['audio']
            if audio.shape != (1, 128, 63):
                audio = torch.nn.functional.pad(audio, (0, max(0, 63 - audio.shape[2])))
                audio = audio[:, :, :63]
            return {
                'video': data['video'],
                'audio': audio,
                'labels': data['labels'],
                'type': entry['type'],
                'file': entry['file'],
                'fake_segments': entry.get('fake_segments', []),
                'total_frames':  entry.get('total_frames', 0)
            }
        except Exception as e:
            # Return zero tensors so the DataLoader doesn't crash on a bad .pt file
            print(f"Warning: failed to load {pt_path}: {e}")
            return {
                'video':  torch.zeros(50, 3, 224, 224),
                'audio':  torch.zeros(1, 128, 63),
                'labels': torch.full((2,), -1.0),  # sentinel — filtered in train/val loops
                'type':   entry.get('type', 'real'),
                'file':   entry.get('file', ''),
                'fake_segments': entry.get('fake_segments', []),
                'total_frames':  entry.get('total_frames', 0)
            }


def create_dataloaders(train_dir, train_manifest, val_dir, val_manifest,
                       batch_size=16, num_workers=None):
    """Create DataLoaders with lazy-loading datasets."""
    if num_workers is None:
        num_workers = min(8, max(2, multiprocessing.cpu_count() // 4))
    train_loader = torch.utils.data.DataLoader(
        AVDataset(train_dir, train_manifest),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Fix 10: no-op overhead on CPU-only machines
    )

    val_loader = torch.utils.data.DataLoader(
        AVDataset(val_dir, val_manifest),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Fix 10
    )
    
    return train_loader, val_loader