"""
Data utilities for AV Deepfake Detection
"""

import sys
sys.path.insert(0, '/content/drive/MyDrive/Colab Notebooks/Deepfake')

import os
import json
import pickle
import random
import numpy as np
import pandas as pd
import cv2
import librosa
import torch
from sklearn.model_selection import train_test_split

from config import VAL_DIR, FEATURES_TRAIN_PATH, FEATURES_VAL_PATH, FEATURE_CONFIG


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


def sample_videos(df, samples_per_type, val_split=0.2, seed=42):
    """Sample balanced dataset with train/val split"""
    set_seeds(seed)
    
    df_with_audio = df[df['audio_frames'] > 0].copy()
    print(f"\nVideos with audio: {len(df_with_audio):,}")
    
    samples = []
    for mod_type, count in samples_per_type.items():
        subset = df_with_audio[df_with_audio['modify_type'] == mod_type]
        if len(subset) >= count:
            sampled = subset.sample(count, random_state=seed)
            samples.append(sampled)
            print(f"✓ {mod_type:20s}: {count} samples")
        else:
            print(f"⚠ {mod_type:20s}: only {len(subset)} available, taking all")
            samples.append(subset)
    
    mini_df = pd.concat(samples).reset_index(drop=True)
    
    train_df, val_df = train_test_split(
        mini_df,
        test_size=val_split,
        stratify=mini_df['modify_type'],
        random_state=seed
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print(f"Split Summary:")
    print(f"  Train: {len(train_df)} videos")
    print(f"  Val:   {len(val_df)} videos")
    
    return train_df, val_df


def extract_av_features(video_path, fake_segments=None, total_frames=0, cfg=FEATURE_CONFIG):
    """Extract synchronized audio and video features from video"""
    # Determine which 2-second window to extract
    if fake_segments and len(fake_segments) > 0:
        start_sec = fake_segments[0][0]
    else:
        total_sec = total_frames / cfg['fps']
        start_sec = max(0, (total_sec / 2) - (cfg['duration'] / 2))
    
    # Video extraction
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ✗ Cannot open video: {video_path}")
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
    
    # Pad if needed
    if len(frames) < cfg['num_frames']:
        while len(frames) < cfg['num_frames']:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3)))
    
    video_tensor = torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
    
    # Audio extraction
    try:
        y, sr = librosa.load(
            video_path,
            sr=cfg['sr'],
            offset=start_sec,
            duration=cfg['duration']
        )
        
        if len(y) < cfg['audio_samples']:
            y = np.pad(y, (0, cfg['audio_samples'] - len(y)))
        else:
            y = y[:cfg['audio_samples']]
        
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=cfg['sr'], n_mels=128, n_fft=2048, hop_length=512
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        audio_tensor = torch.FloatTensor(mel_db).unsqueeze(0)
        
    except Exception as e:
        print(f"    ⚠ Audio extraction failed: {e}")
        audio_tensor = torch.zeros(1, 128, 87)
    
    return video_tensor, audio_tensor


def process_split(split_df, split_name, val_dir=VAL_DIR):
    """Process entire split and extract features"""
    print(f"\nProcessing {split_name} set ({len(split_df)} videos)...")
    features = []
    
    for idx, row in split_df.iterrows():
        print(f"\n[{split_name} {idx+1}/{len(split_df)}] {row['modify_type'].upper()}")
        video_path = os.path.join(val_dir, row['file'])
        
        if not os.path.exists(video_path):
            print(f"    ✗ File not found: {video_path}")
            continue
        
        video_tensor, audio_tensor = extract_av_features(
            video_path=video_path,
            fake_segments=row.get('fake_segments'),
            total_frames=row['video_frames']
        )
        
        if video_tensor is None:
            print(f"    ✗ Feature extraction failed")
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
        
        features.append({
            'video': video_tensor,
            'audio': audio_tensor,
            'labels': torch.FloatTensor([audio_label, video_label]),
            'type': mt,
            'file': row['file'],
            'speaker': row['file'].split('/')[1] if '/' in row['file'] else 'unknown'
        })
        
        print(f"    ✓ Video: {video_tensor.shape}, Audio: {audio_tensor.shape}")
    
    return features


def extract_all_features(train_df, val_df, val_dir=VAL_DIR, 
                         cache_train_path=FEATURES_TRAIN_PATH,
                         cache_val_path=FEATURES_VAL_PATH,
                         use_cache=True):
    """Extract features for both splits with caching"""
    if use_cache and os.path.exists(cache_train_path) and os.path.exists(cache_val_path):
        print("=" * 60)
        print("Found cached features. Loading...")
        print("=" * 60)
        with open(cache_train_path, 'rb') as f:
            train_features = pickle.load(f)
        with open(cache_val_path, 'rb') as f:
            val_features = pickle.load(f)
        print(f"✓ Loaded {len(train_features)} train, {len(val_features)} val features")
        return train_features, val_features
    
    print("=" * 60)
    print("SECTION: Feature Extraction")
    print("=" * 60)
    
    train_features = process_split(train_df, "TRAIN", val_dir)
    val_features = process_split(val_df, "VAL", val_dir)
    
    print(f"\n{'='*60}")
    print("Extraction Summary:")
    print(f"  Train: {len(train_features)}/{len(train_df)} successful")
    print(f"  Val:   {len(val_features)}/{len(val_df)} successful")
    
    # Cache features
    os.makedirs(os.path.dirname(cache_train_path), exist_ok=True)
    with open(cache_train_path, 'wb') as f:
        pickle.dump(train_features, f)
    with open(cache_val_path, 'wb') as f:
        pickle.dump(val_features, f)
    print(f"\n✓ Features cached")
    
    return train_features, val_features


class AVDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for AV features"""
    def __init__(self, features):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item = self.features[idx]
        return {
            'video': item['video'],
            'audio': item['audio'],
            'labels': item['labels'],
            'type': item['type'],
            'file': item['file']
        }


def create_dataloaders(train_features, val_features, batch_size=3):
    """Create DataLoaders for train and val sets"""
    train_loader = torch.utils.data.DataLoader(
        AVDataset(train_features),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        AVDataset(val_features),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader
