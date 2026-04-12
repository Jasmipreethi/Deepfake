"""
Test Dataset Generator — AV Deepfake Detection

Randomly samples videos from the extracted_val/ folder and organises
them into a test/ directory with real/ and fake/ subfolders, ready
for use with evaluate_models.py and inference.py.

Usage:
    # Basic — 100 videos balanced across all 4 types
    python create_test_dataset.py \\
        --val_dir /path/to/extracted_val/ \\
        --output_dir ./test/

    # Custom count (25 per type = 100 total)
    python create_test_dataset.py \\
        --val_dir /path/to/extracted_val/ \\
        --output_dir ./test/ \\
        --per_type 25

    # Larger test set
    python create_test_dataset.py \\
        --val_dir /path/to/extracted_val/ \\
        --output_dir ./test/ \\
        --per_type 50

    # Custom metadata path if not in default location
    python create_test_dataset.py \\
        --val_dir /path/to/extracted_val/ \\
        --metadata /path/to/val_metadata.json \\
        --output_dir ./test/

Output structure:
    test/
        real/
            id01234_clip001_real.mp4
            id05678_clip002_real.mp4
            ...
        fake/
            id01234_clip003_audio_modified.mp4
            id05678_clip004_visual_modified.mp4
            ...
        test_manifest.json    ← metadata for every sampled video

Requirements:
    pip install pandas tqdm
"""

import os
import sys
import json
import shutil
import random
import argparse
from collections import defaultdict

try:
    import pandas as pd
    from tqdm import tqdm
    from sklearn.model_selection import GroupShuffleSplit
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas', 'tqdm', 'scikit-learn'])
    import pandas as pd
    from tqdm import tqdm
    from sklearn.model_selection import GroupShuffleSplit


# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# Which modify_types go into real/ vs fake/
REAL_TYPES = {'real'}
FAKE_TYPES = {'audio_modified', 'visual_modified', 'both_modified'}

# Human-readable short labels for filenames
TYPE_LABEL = {
    'real':             'real',
    'audio_modified':   'audio_fake',
    'visual_modified':  'video_fake',
    'both_modified':    'both_fake',
}


# ─────────────────────────────────────────────────────────────────────────────
# METADATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def find_metadata(val_dir):
    """Search common locations for val_metadata.json."""
    candidates = [
        os.path.join(val_dir, 'val_metadata.json'),
        os.path.join(val_dir, '..', 'val_metadata.json'),
        os.path.join(os.path.dirname(val_dir.rstrip('/')), 'val_metadata.json'),
        'val_metadata.json',
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.exists(path):
            return path
    return None


def load_metadata(val_dir, metadata_path=None):
    """Load metadata and filter to videos that exist on disk."""
    if metadata_path is None:
        metadata_path = find_metadata(val_dir)

    if metadata_path is None:
        print("ERROR: val_metadata.json not found.")
        print("Searched in and around the val_dir. Use --metadata to specify the path.")
        sys.exit(1)

    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"  Total entries: {len(df):,}")

    # Filter to videos that actually exist on disk
    print("  Checking which videos exist on disk...")
    df['full_path'] = df['file'].apply(lambda f: os.path.join(val_dir, f))
    df['exists'] = df['full_path'].apply(os.path.exists)

    missing = (~df['exists']).sum()
    df = df[df['exists']].reset_index(drop=True)

    print(f"  Found on disk: {len(df):,}  (missing: {missing:,})")
    print(f"\n  Type distribution:")
    for t, count in df['modify_type'].value_counts().items():
        print(f"    {t:20s}: {count:,}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN/VAL SPLIT RECREATION (must match data_utils.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def get_val_speakers(metadata_path, train_seed=42, val_split=0.2, use_all=False, samples_per_type=None):
    """
    Recreate the exact train/val speaker split used during training.

    Uses the same logic as data_utils.sample_videos():
    1. Load full metadata
    2. Filter to videos with both audio and video
    3. Optionally subsample by type (if use_all=False)
    4. GroupShuffleSplit by speaker with seed=42

    Returns: (set of val speaker IDs, set of train speaker IDs)
    """
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    full_df = pd.DataFrame(data)

    # Same filter as data_utils.py
    df_valid = full_df[(full_df['audio_frames'] > 0) & (full_df['video_frames'] > 0)].copy()
    print(f"\n  Recreating training split (seed={train_seed})...")
    print(f"  Videos with both audio+video: {len(df_valid):,}")

    if use_all:
        mini_df = df_valid.reset_index(drop=True)
    else:
        # Subset mode — same as training
        if samples_per_type is None:
            samples_per_type = {
                'real': 40, 'both_modified': 40,
                'audio_modified': 40, 'visual_modified': 40
            }
        random.seed(train_seed)
        samples = []
        for mod_type, count in samples_per_type.items():
            subset = df_valid[df_valid['modify_type'] == mod_type]
            if len(subset) >= count:
                samples.append(subset.sample(count, random_state=train_seed))
            else:
                samples.append(subset)
        mini_df = pd.concat(samples).reset_index(drop=True)

    # Extract speaker IDs (same logic as data_utils.py)
    mini_df['speaker'] = mini_df['file'].apply(
        lambda f: f.split('/')[1] if '/' in f and len(f.split('/')) > 1 else 'unknown'
    )

    # Same GroupShuffleSplit as training
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=train_seed)
    train_idx, val_idx = next(gss.split(mini_df, groups=mini_df['speaker']))

    train_speakers = set(mini_df.iloc[train_idx]['speaker'].unique())
    val_speakers = set(mini_df.iloc[val_idx]['speaker'].unique())

    print(f"  Train speakers: {len(train_speakers):,}")
    print(f"  Val speakers:   {len(val_speakers):,}")
    print(f"  Overlap:        {len(train_speakers & val_speakers)} (should be 0)")

    return val_speakers, train_speakers


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLING (val-speakers only)
# ─────────────────────────────────────────────────────────────────────────────

def sample_videos(df, per_type=25, seed=42, val_speakers=None):
    """
    Sample per_type videos from each modify_type.

    If val_speakers is provided, ONLY samples from those speakers
    to prevent data leakage from the training set.

    Ensures:
    - Equal representation of all 4 manipulation types
    - No overlap with training speakers
    - Reproducible via seed
    """
    random.seed(seed)

    # Filter to val speakers only
    if val_speakers is not None:
        df['speaker'] = df['file'].apply(
            lambda f: f.split('/')[1] if '/' in f and len(f.split('/')) > 1 else 'unknown'
        )
        before = len(df)
        df = df[df['speaker'].isin(val_speakers)].reset_index(drop=True)
        print(f"\n  Filtered to val speakers only: {len(df):,} videos (excluded {before - len(df):,} training videos)")
    else:
        print("\n  WARNING: No val_speakers provided — sampling from ALL videos (risk of data leakage!)")

    all_types = ['real', 'audio_modified', 'visual_modified', 'both_modified']
    sampled = []

    print(f"\nSampling {per_type} videos per type (val speakers only):")
    for mod_type in all_types:
        subset = df[df['modify_type'] == mod_type]
        available = len(subset)

        if available == 0:
            print(f"  ⚠ {mod_type:20s}: not found in val split — skipping")
            continue

        if available < per_type:
            print(f"  ⚠ {mod_type:20s}: only {available} available in val split — taking all")
            chosen = subset
        else:
            chosen = subset.sample(per_type, random_state=seed)
            print(f"  ✓ {mod_type:20s}: {per_type} sampled from {available:,} (val speakers)")

        sampled.append(chosen)

    result = pd.concat(sampled).reset_index(drop=True)

    real_count = len(result[result['modify_type'] == 'real'])
    fake_count = len(result[result['modify_type'] != 'real'])

    print(f"\n  Total sampled: {len(result)}")
    print(f"    Real:  {real_count}  (1 type × {per_type})")
    print(f"    Fake:  {fake_count}  (3 types × {per_type})")
    print(f"\n  All test videos are from val speakers — zero leakage from training.")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# COPY FILES
# ─────────────────────────────────────────────────────────────────────────────

def build_dest_filename(row, idx):
    """
    Build a descriptive destination filename.
    Format: {speaker}_{idx:04d}_{type_label}.mp4
    """
    # Extract speaker from file path e.g. vox_celeb_2/id01234/clip/00001/video.mp4
    parts = row['file'].replace('\\', '/').split('/')
    speaker = parts[1] if len(parts) > 1 else 'unknown'

    # Original filename without extension
    orig_stem = os.path.splitext(os.path.basename(row['file']))[0]

    label = TYPE_LABEL.get(row['modify_type'], row['modify_type'])
    return f"{speaker}_{idx:04d}_{label}.mp4"


def copy_videos(sampled_df, output_dir, use_symlinks=False):
    """
    Copy (or symlink) sampled videos into test/real/ and test/fake/.

    Returns list of manifest entries.
    """
    real_dir = os.path.join(output_dir, 'real')
    fake_dir = os.path.join(output_dir, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    manifest = []
    errors   = []

    action = "Symlinking" if use_symlinks else "Copying"
    print(f"\n{action} videos to {output_dir}...")

    for idx, row in tqdm(sampled_df.iterrows(),
                         total=len(sampled_df), unit="video"):
        src  = row['full_path']
        dest_name = build_dest_filename(row, idx)
        is_real = row['modify_type'] in REAL_TYPES
        dest_dir = real_dir if is_real else fake_dir
        dest = os.path.join(dest_dir, dest_name)

        try:
            if use_symlinks:
                if os.path.exists(dest):
                    os.remove(dest)
                os.symlink(os.path.abspath(src), dest)
            else:
                shutil.copy2(src, dest)

            manifest.append({
                'dest_file':     os.path.join('real' if is_real else 'fake', dest_name),
                'source_file':   row['file'],
                'modify_type':   row['modify_type'],
                'true_label':    1 if is_real else 0,
                'true_verdict':  'REAL' if is_real else 'FAKE',
                'speaker':       row['file'].replace('\\','/').split('/')[1]
                                 if '/' in row['file'] else 'unknown',
                'audio_frames':  int(row.get('audio_frames', 0)),
                'video_frames':  int(row.get('video_frames', 0)),
                'fake_segments': row.get('fake_segments', []),
            })

        except Exception as e:
            errors.append((src, str(e)))
            tqdm.write(f"  ⚠ Failed: {os.path.basename(src)} — {e}")

    if errors:
        print(f"\n  ⚠ {len(errors)} files failed to copy:")
        for src, err in errors[:5]:
            print(f"    {os.path.basename(src)}: {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors)-5} more")

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(manifest, output_dir):
    """Print a summary of what was created."""
    real_dir = os.path.join(output_dir, 'real')
    fake_dir = os.path.join(output_dir, 'fake')

    n_real = len([m for m in manifest if m['true_label'] == 1])
    n_fake = len([m for m in manifest if m['true_label'] == 0])

    type_counts = defaultdict(int)
    for m in manifest:
        type_counts[m['modify_type']] += 1

    print(f"\n{'='*55}")
    print(f"TEST DATASET CREATED")
    print(f"{'='*55}")
    print(f"  Location:   {os.path.abspath(output_dir)}")
    print(f"  Total:      {len(manifest)} videos")
    print(f"    real/     {n_real} videos")
    print(f"    fake/     {n_fake} videos")
    print(f"\n  Breakdown by manipulation type:")
    for t in ['real', 'audio_modified', 'visual_modified', 'both_modified']:
        if type_counts[t]:
            print(f"    {t:20s}: {type_counts[t]}")
    print(f"\n  Manifest:   {os.path.join(output_dir, 'test_manifest.json')}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Generate a balanced test dataset from extracted_val/',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument('--val_dir',    required=True,
                   help='Path to extracted_val/ folder containing video files')
    p.add_argument('--output_dir', default='./test',
                   help='Where to create the test dataset (default: ./test)')
    p.add_argument('--metadata',   default=None,
                   help='Path to val_metadata.json (auto-detected if not set)')
    p.add_argument('--per_type',   type=int, default=25,
                   help='Videos per manipulation type (default: 25 → 100 total)')
    p.add_argument('--seed',       type=int, default=42,
                   help='Random seed for reproducibility (default: 42)')
    # Symlinks = shortcuts that point to the original file instead of copying it.
    # Saves disk space (~0 bytes vs full copy) but breaks if original files are deleted.
    # Use on the server where data stays around. Use copy (default) for portable test sets.
    p.add_argument('--symlinks',   action='store_true',
                   help='Use symlinks instead of copying (saves disk space, '
                        'requires val_dir to remain accessible)')
    p.add_argument('--use_all',    action='store_true',
                   help='Set if training used use_all_data=True (affects split recreation)')
    p.add_argument('--train_seed', type=int, default=42,
                   help='Seed used during training (default: 42)')
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*55}")
    print("TEST DATASET GENERATOR (val-speakers only)")
    print(f"{'='*55}")
    print(f"  val_dir:    {args.val_dir}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  per_type:   {args.per_type}  ({args.per_type * 4} total videos)")
    print(f"  seed:       {args.seed}")
    print(f"  train_seed: {args.train_seed}")
    print(f"  use_all:    {args.use_all}")
    print(f"  mode:       {'symlinks' if args.symlinks else 'copy'}")

    # Check val_dir exists
    if not os.path.exists(args.val_dir):
        print(f"\nERROR: val_dir not found: {args.val_dir}")
        sys.exit(1)

    # Check output_dir is empty or doesn't exist
    if os.path.exists(args.output_dir):
        existing = sum(
            len(files) for _, _, files in os.walk(args.output_dir)
        )
        if existing > 0:
            print(f"\n  output_dir already exists with {existing} files: {args.output_dir}")
            ans = input("  Overwrite? (y/n): ").strip().lower()
            if ans != 'y':
                print("Aborted.")
                sys.exit(0)
            shutil.rmtree(args.output_dir)

    # Load metadata
    print(f"\n{'─'*55}")
    print("LOADING METADATA")
    print(f"{'─'*55}")
    df = load_metadata(args.val_dir, args.metadata)

    # Recreate training split to identify val speakers
    print(f"\n{'─'*55}")
    print("RECREATING TRAIN/VAL SPLIT")
    print(f"{'─'*55}")
    metadata_path = args.metadata or find_metadata(args.val_dir)
    val_speakers, train_speakers = get_val_speakers(
        metadata_path,
        train_seed=args.train_seed,
        use_all=args.use_all
    )

    # Sample — only from val speakers
    print(f"\n{'─'*55}")
    print("SAMPLING (val speakers only)")
    print(f"{'─'*55}")
    sampled = sample_videos(df, per_type=args.per_type, seed=args.seed,
                           val_speakers=val_speakers)

    # Copy
    print(f"\n{'─'*55}")
    print("COPYING FILES")
    print(f"{'─'*55}")
    manifest = copy_videos(sampled, args.output_dir, use_symlinks=args.symlinks)

    # Save manifest
    manifest_path = os.path.join(args.output_dir, 'test_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Also save the speaker lists for reference
    split_info = {
        'train_seed': args.train_seed,
        'use_all': args.use_all,
        'n_train_speakers': len(train_speakers),
        'n_val_speakers': len(val_speakers),
        'val_speakers': sorted(list(val_speakers)),
        'train_speakers': sorted(list(train_speakers)),
    }
    split_path = os.path.join(args.output_dir, 'split_info.json')
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    # Summary
    print_summary(manifest, args.output_dir)
    print(f"  Split info: {split_path}")

    print(f"\n{'─'*55}")
    print("NEXT STEPS")
    print(f"{'─'*55}")
    print(f"""
  Test a single model:
    python inference.py \\
        --model best_model.pth \\
        --video_dir {args.output_dir}/ \\
        --output results.csv

  Compare two models:
    python evaluate_models.py \\
        --model1 run1_best_model.pth \\
        --model2 run2_best_model.pth \\
        --video_dir {args.output_dir}/ \\
        --output_dir eval_results/
""")


if __name__ == '__main__':
    main()