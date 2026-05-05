"""
Script to regenerate train_manifest.json and val_manifest.json
following the same logic as main.py when no feature extraction is needed.
"""

import os
import json
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = os.environ.get('DATA_DIR', '/Users/jasmi/Desktop/AV-Deepfake1M')
VAL_DIR = os.path.join(DATA_DIR, 'extracted_val', 'val')
METADATA_DIR = os.path.join(DATA_DIR, 'extracted_val')
FEATURES_DIR = os.path.join(os.path.dirname(VAL_DIR), 'checkpoints', 'features')

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def find_metadata():
    """Find val_metadata.json in likely locations"""
    search_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'val_metadata.json'),
        '/Users/jasmi/Desktop/AV-Deepfake1M/Try/val_metadata.json',
        '/Users/jasmi/Desktop/AV-Deepfake1M/extracted_val/val_metadata.json',
    ]
    for path in search_paths:
        if os.path.exists(path):
            print(f"Found metadata at: {path}")
            return path
    raise FileNotFoundError("Could not find val_metadata.json")

def load_metadata():
    """Load and return metadata DataFrame"""
    metadata_path = find_metadata()

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    df = pd.DataFrame(metadata)
    print(f"Loaded {len(df):,} entries from metadata")
    print(f"Modification types:")
    print(df['modify_type'].value_counts())
    return df

def sample_videos(df, samples_per_type, val_split=0.2, seed=42, use_all=False):
    """Sample balanced dataset with train/val split BY SPEAKER - same as data_utils.py"""
    set_seeds(seed)

    df_valid = df[(df['audio_frames'] > 0) & (df['video_frames'] > 0)].copy()
    dropped = len(df) - len(df_valid)
    print(f"\nVideos with both audio and video: {len(df_valid):,} (dropped {dropped:,})")

    if use_all:
        mini_df = df_valid.reset_index(drop=True)
        print(f"\nUsing ALL {len(mini_df):,} videos:")
        for mod_type in mini_df['modify_type'].unique():
            count = len(mini_df[mini_df['modify_type'] == mod_type])
            print(f"  {mod_type:20s}: {count:,} videos")
    else:
        samples = []
        for mod_type, count in samples_per_type.items():
            subset = df_valid[df_valid['modify_type'] == mod_type]
            if len(subset) >= count:
                sampled = subset.sample(count, random_state=seed)
                samples.append(sampled)
                print(f"  {mod_type:20s}: {count} samples")
            else:
                print(f"  {mod_type:20s}: only {len(subset)} available, taking all")
                samples.append(subset)
        mini_df = pd.concat(samples).reset_index(drop=True)

    mini_df['speaker'] = mini_df['file'].apply(
        lambda f: f.split('/')[1] if '/' in f and len(f.split('/')) > 1 else 'unknown'
    )

    n_speakers = mini_df['speaker'].nunique()
    print(f"\nUnique speakers: {n_speakers:,}")

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
    print(f"  Val: {len(val_df):,} videos from {val_speakers:,} speakers")
    print(f"  Speaker overlap: {len(overlap)} (should be 0)")

    print(f"\n  Train distribution:")
    for t in sorted(train_df['modify_type'].unique()):
        print(f"    {t:20s}: {len(train_df[train_df['modify_type'] == t]):,}")
    print(f"  Val distribution:")
    for t in sorted(val_df['modify_type'].unique()):
        print(f"    {t:20s}: {len(val_df[val_df['modify_type'] == t]):,}")

    return train_df, val_df

def create_manifest(split_df, split_name, feature_dir):
    """Create manifest entries for a split following the structure from data_utils.py"""
    split_dir = os.path.join(feature_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    print(f"  Created directory: {split_dir}")

    manifest = []
    for idx, row in split_df.iterrows():
        entry = {
            'idx': idx,
            'file': row['file'],
            'type': row['modify_type'],
            'speaker': row['file'].split('/')[1] if '/' in row['file'] else 'unknown',
            'pt_file': f'{idx}.pt',
            'fake_segments': row.get('fake_segments', []),
            'total_frames': int(row['video_frames'])
        }
        manifest.append(entry)

    manifest_path = os.path.join(feature_dir, f'{split_name}_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCreated {split_name}_manifest.json with {len(manifest)} entries at:")
    print(f"  {manifest_path}")
    return manifest_path

def main():
    print("=" * 60)
    print("REGENERATING TRAIN AND VAL MANIFESTS")
    print("=" * 60)

    val_split = 0.2
    seed = 42

    print(f"\nConfiguration:")
    print(f"  use_all_data: True (all videos from selected speakers)")
    print(f"  val_split: {val_split}")
    print(f"  seed: {seed}")

    df = load_metadata()

    train_df, val_df = sample_videos(
        df,
        samples_per_type={},  # not used when use_all=True
        val_split=val_split,
        seed=seed,
        use_all=True
    )

    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR, exist_ok=True)
        print(f"\nCreated features directory: {FEATURES_DIR}")

    print(f"\n{'='*60}")
    print("CREATING MANIFEST FILES")
    print(f"{'='*60}")

    train_manifest_path = create_manifest(train_df, 'train', FEATURES_DIR)
    val_manifest_path = create_manifest(val_df, 'val', FEATURES_DIR)

    print(f"\n{'='*60}")
    print("MANIFEST GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Train manifest: {train_manifest_path}")
    print(f"  Val manifest: {val_manifest_path}")

    print("\nTo use these manifests, ensure your feature files (.pt) are at:")
    print(f"  {os.path.join(FEATURES_DIR, 'train')}")
    print(f"  {os.path.join(FEATURES_DIR, 'val')}")

if __name__ == "__main__":
    main()