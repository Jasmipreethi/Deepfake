"""
Metadata Analysis Script for AV-Deepfake1M++ Dataset

Generates visualizations and statistics about the dataset before training.
Run this after downloading to understand the data distribution.

Usage:
    python analyze_data.py
    python analyze_data.py --metadata_path /path/to/val_metadata.json
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# Try loading .env for DATA_DIR
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()


def load_metadata(metadata_path):
    """Load and parse metadata JSON."""
    print(f"Loading: {metadata_path}")
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df):,} entries\n")
    return df


def extract_speaker_ids(df):
    """Extract speaker IDs from file paths."""
    def get_speaker(path):
        parts = path.split('/')
        return parts[1] if len(parts) > 1 else 'unknown'
    
    df['speaker'] = df['file'].apply(get_speaker)
    return df


def print_basic_stats(df):
    """Print basic dataset statistics."""
    print("=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal videos: {len(df):,}")
    print(f"Total speakers: {df['speaker'].nunique():,}")
    
    # Modification types
    print(f"\nModification Type Distribution:")
    type_counts = df['modify_type'].value_counts()
    for t, count in type_counts.items():
        pct = count / len(df) * 100
        print(f"  {t:20s}: {count:6,} ({pct:.1f}%)")
    
    # Frame statistics
    print(f"\nVideo Frames:")
    print(f"  Mean:   {df['video_frames'].mean():.0f}")
    print(f"  Median: {df['video_frames'].median():.0f}")
    print(f"  Min:    {df['video_frames'].min()}")
    print(f"  Max:    {df['video_frames'].max()}")
    
    if 'audio_frames' in df.columns:
        print(f"\nAudio Frames:")
        print(f"  Mean:   {df['audio_frames'].mean():.0f}")
        print(f"  Median: {df['audio_frames'].median():.0f}")
        print(f"  Min:    {df['audio_frames'].min()}")
        print(f"  Max:    {df['audio_frames'].max()}")
        
        zero_audio = (df['audio_frames'] == 0).sum()
        if zero_audio > 0:
            print(f"  WARNING: {zero_audio:,} videos have 0 audio frames!")


def print_speaker_stats(df):
    """Print speaker-level statistics."""
    print("\n" + "=" * 60)
    print("SPEAKER STATISTICS")
    print("=" * 60)
    
    speaker_counts = df['speaker'].value_counts()
    
    print(f"\nVideos per speaker:")
    print(f"  Mean:   {speaker_counts.mean():.1f}")
    print(f"  Median: {speaker_counts.median():.0f}")
    print(f"  Min:    {speaker_counts.min()} ({speaker_counts.idxmin()})")
    print(f"  Max:    {speaker_counts.max()} ({speaker_counts.idxmax()})")
    
    # Speaker type distribution
    print(f"\nSpeakers by modification type:")
    for mt in df['modify_type'].unique():
        speakers = df[df['modify_type'] == mt]['speaker'].nunique()
        print(f"  {mt:20s}: {speakers:,} speakers")
    
    # How many speakers have all 4 types?
    speaker_types = df.groupby('speaker')['modify_type'].nunique()
    print(f"\nSpeakers with all 4 modification types: {(speaker_types == 4).sum():,}")
    print(f"Speakers with < 4 modification types:  {(speaker_types < 4).sum():,}")


def print_fake_segment_stats(df):
    """Analyze fake segment durations."""
    print("\n" + "=" * 60)
    print("FAKE SEGMENT STATISTICS")
    print("=" * 60)
    
    fake_df = df[df['modify_type'] != 'real']
    
    if 'fake_segments' not in fake_df.columns:
        print("  No fake_segments column found")
        return
    
    durations = []
    segment_counts = []
    
    for _, row in fake_df.iterrows():
        segs = row.get('fake_segments')
        if segs and isinstance(segs, list):
            segment_counts.append(len(segs))
            for seg in segs:
                if isinstance(seg, list) and len(seg) == 2:
                    durations.append(seg[1] - seg[0])
    
    if durations:
        print(f"\nFake videos analyzed: {len(fake_df):,}")
        print(f"Total fake segments:  {len(durations):,}")
        print(f"\nSegments per video:")
        print(f"  Mean: {np.mean(segment_counts):.1f}")
        print(f"  Max:  {max(segment_counts)}")
        print(f"\nSegment duration (seconds):")
        print(f"  Mean:   {np.mean(durations):.2f}s")
        print(f"  Median: {np.median(durations):.2f}s")
        print(f"  Min:    {min(durations):.2f}s")
        print(f"  Max:    {max(durations):.2f}s")


def plot_distributions(df, output_dir):
    """Generate distribution plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AV-Deepfake1M++ Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Modification type distribution
    ax = axes[0, 0]
    type_counts = df['modify_type'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    bars = ax.bar(range(len(type_counts)), type_counts.values, color=colors)
    ax.set_xticks(range(len(type_counts)))
    ax.set_xticklabels(type_counts.index, rotation=30, ha='right', fontsize=9)
    ax.set_title('Videos by Modification Type')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, type_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    
    # 2. Videos per speaker distribution
    ax = axes[0, 1]
    speaker_counts = df['speaker'].value_counts()
    ax.hist(speaker_counts.values, bins=50, color='#9b59b6', edgecolor='white', alpha=0.8)
    ax.set_title('Videos per Speaker Distribution')
    ax.set_xlabel('Number of videos')
    ax.set_ylabel('Number of speakers')
    ax.axvline(speaker_counts.mean(), color='red', linestyle='--', 
               label=f'Mean: {speaker_counts.mean():.0f}')
    ax.legend()
    
    # 3. Video frame count distribution
    ax = axes[1, 0]
    ax.hist(df['video_frames'].values, bins=50, color='#1abc9c', edgecolor='white', alpha=0.8)
    ax.set_title('Video Frame Count Distribution')
    ax.set_xlabel('Number of frames')
    ax.set_ylabel('Count')
    ax.axvline(df['video_frames'].mean(), color='red', linestyle='--',
               label=f'Mean: {df["video_frames"].mean():.0f}')
    ax.legend()
    
    # 4. Type distribution per speaker (stacked)
    ax = axes[1, 1]
    speaker_type_counts = df.groupby('speaker')['modify_type'].nunique()
    type_dist = speaker_type_counts.value_counts().sort_index()
    ax.bar(type_dist.index, type_dist.values, color='#e67e22', edgecolor='white')
    ax.set_title('Modification Types per Speaker')
    ax.set_xlabel('Number of modification types')
    ax.set_ylabel('Number of speakers')
    ax.set_xticks(type_dist.index)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'dataset_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {plot_path}")
    
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="Analyze AV-Deepfake1M++ metadata")
    parser.add_argument("--metadata_path", type=str, default=None,
                       help="Path to val_metadata.json")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save plots")
    args = parser.parse_args()
    
    # Find metadata
    data_dir = os.environ.get('DATA_DIR', '/content/drive/MyDrive/val')
    
    if args.metadata_path:
        metadata_path = args.metadata_path
    else:
        # Search common locations
        candidates = [
            os.path.join(data_dir, 'val_metadata.json'),
            os.path.join(data_dir, 'extracted_val', 'val_metadata.json'),
            'val_metadata.json',
        ]
        metadata_path = None
        for path in candidates:
            if os.path.exists(path):
                metadata_path = path
                break
        
        if not metadata_path:
            print("ERROR: val_metadata.json not found!")
            print(f"Searched: {candidates}")
            print("Use --metadata_path to specify the location.")
            return
    
    # Output directory
    output_dir = args.output_dir or os.path.join(os.path.dirname(metadata_path), 'analysis')
    
    # Run analysis
    df = load_metadata(metadata_path)
    df = extract_speaker_ids(df)
    
    print_basic_stats(df)
    print_speaker_stats(df)
    print_fake_segment_stats(df)
    
    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    plot_distributions(df, output_dir)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
