"""
Analyze train_manifest.json and val_manifest.json
to verify the dataset split used for training.
"""

import os
import json
import argparse
from collections import Counter

DEFAULT_MANIFEST_DIR = '/Users/jasmi/Desktop/AV-Deepfake1M/extracted_val/checkpoints/features'

def load_manifest(manifest_path):
    """Load manifest JSON file."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded: {manifest_path} ({len(data):,} entries)")
    return data

def analyze_manifest(manifest, name):
    """Analyze a manifest and print statistics."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {name}")
    print(f"{'='*60}")

    total = len(manifest)
    print(f"\nTotal entries: {total:,}")

    type_counts = Counter(entry['type'] for entry in manifest)
    print(f"\nModification Type Distribution:")
    for t, count in sorted(type_counts.items()):
        pct = count / total * 100
        print(f"  {t:20s}: {count:6,} ({pct:.1f}%)")

    speaker_counts = Counter(entry['speaker'] for entry in manifest)
    print(f"\nSpeaker Statistics:")
    print(f"  Unique speakers: {len(speaker_counts):,}")
    print(f"  Videos per speaker:")
    print(f"    Mean: {sum(speaker_counts.values()) / len(speaker_counts):.1f}")
    print(f"    Min: {min(speaker_counts.values())}")
    print(f"    Max: {max(speaker_counts.values())}")

    videos_with_fake = sum(1 for e in manifest if e.get('fake_segments'))
    print(f"\nVideos with fake_segments: {videos_with_fake:,} ({videos_with_fake/total*100:.1f}%)")

    return {
        'total': total,
        'types': dict(type_counts),
        'n_speakers': len(speaker_counts),
        'speaker_counts': dict(speaker_counts)
    }

def compare_splits(train_manifest, val_manifest):
    """Check for speaker overlap between train and val."""
    print(f"\n{'='*60}")
    print("SPEAKER OVERLAP CHECK")
    print(f"{'='*60}")

    train_speakers = set(e['speaker'] for e in train_manifest)
    val_speakers = set(e['speaker'] for e in val_manifest)

    overlap = train_speakers & val_speakers
    print(f"  Train speakers: {len(train_speakers):,}")
    print(f"  Val speakers: {len(val_speakers):,}")
    print(f"  Overlap: {len(overlap):,}")
    if overlap:
        print(f"  ⚠ WARNING: {len(overlap)} speakers appear in BOTH splits!")
    else:
        print(f"  ✓ No speaker overlap (correct)")

def main():
    parser = argparse.ArgumentParser(description='Analyze train/val manifests')
    parser.add_argument('--manifest_dir', default=DEFAULT_MANIFEST_DIR,
                        help='Directory containing manifest files')
    args = parser.parse_args()

    train_path = os.path.join(args.manifest_dir, 'train_manifest.json')
    val_path = os.path.join(args.manifest_dir, 'val_manifest.json')

    print("MANIFEST ANALYZER")
    print(f"Manifest directory: {args.manifest_dir}")

    train_data = load_manifest(train_path)
    val_data = load_manifest(val_path)

    train_stats = analyze_manifest(train_data, "TRAIN")
    val_stats = analyze_manifest(val_data, "VALIDATION")

    compare_splits(train_data, val_data)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Train: {train_stats['total']:,} videos, {train_stats['n_speakers']:,} speakers")
    print(f"  Val:   {val_stats['total']:,} videos, {val_stats['n_speakers']:,} speakers")
    print(f"  Total: {train_stats['total'] + val_stats['total']:,} videos")

if __name__ == "__main__":
    main()