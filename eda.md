# Exploratory Data Analysis (EDA) - AV-Deepfake1M++ Pipeline

This document describes the Exploratory Data Analysis steps and data cleaning procedures performed in the AV-Deepfake1M++ detection pipeline before model training.

---

## EDA Steps in the Pipeline

### 1. Basic Metadata Loading

**Location:** `data_utils.py` - `load_metadata()` (lines 34-60)

The pipeline loads the JSON metadata file and converts it to a pandas DataFrame, then prints the distribution of modification types:

```
Modification types:
  real             20,220
  visual_modified  19,099
  both_modified    19,069
  audio_modified   18,938
```

### 2. Sample & Stratified Split

**Location:** `data_utils.py` - `sample_videos()` (lines 79-159)

The pipeline:
- Counts videos with both audio and video frames
- Reports per-type distribution in train/val splits
- Reports unique speaker count (1,835 speakers)
- Verifies speaker-based split produces zero overlap

---

## Data Cleaning Steps

### 1. Filter Invalid Videos

**Location:** `data_utils.py` (line 94)

```python
df_valid = df[(df['audio_frames'] > 0) & (df['video_frames'] > 0)].copy()
```

**Actions:**
- **Removes videos with zero audio frames** (211 videos flagged in analysis)
- **Removes videos with zero video frames**

**Impact:**
- Original dataset: 77,326 videos
- After cleaning: Videos with valid audio AND video (exact count depends on intersection of zero audio and zero video)

### 2. Speaker-Based Train/Val Split

**Location:** `data_utils.py` (lines 128-129)

```python
gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
train_idx, val_idx = next(gss.split(mini_df, groups=mini_df['speaker']))
```

**Actions:**
- **Prevents identity leakage** — no speaker appears in both train and val splits
- Uses `GroupShuffleSplit` from sklearn to enforce speaker-grouped splitting

**Verification:**
- Reports train speakers, val speakers, and overlap count
- Confirms overlap = 0

### 3. Failed Extraction Handling

**Location:** `data_utils.py` - `process_split_to_disk()` (lines 427-546)

**Actions:**
- Tracks failed videos in `*_failed.json` manifest
- Skips corrupted/unreadable video files
- Saves progress incrementally (every 500 successes)
- Caps processing at 95% completion to account for minor failures

### 4. Runtime Fallbacks

**Location:** `data_utils.py` (lines 280-283, 627-637)

| Failure Scenario | Fallback Behavior |
|------------------|-------------------|
| Audio extraction fails | Returns zero tensor `(1, 128, 63)` |
| Video frame read fails | Returns `(None, None)` — skipped |
| Bad/corrupted `.pt` file | Returns zero tensors with sentinel labels `[-1, -1]` |
| Audio shape mismatch | Pads/truncates to fixed shape `(1, 128, 63)` |

### 5. Defensive Shape Checking

**Location:** `data_utils.py` (lines 613-616)

```python
audio = data['audio']
if audio.shape != (1, 128, 63):
    audio = torch.nn.functional.pad(audio, (0, max(0, 63 - audio.shape[2])))
    audio = audio[:, :, :63]
```

**Actions:**
- Ensures all audio tensors conform to expected shape
- Prevents shape mismatches during batch collation

---

## Dataset Statistics (from val_metadata.json)

| Metric | Value |
|--------|-------|
| Total videos | 77,326 |
| Total speakers | 1,835 |
| Videos per speaker (mean) | 42.1 |
| Videos per speaker (median) | 20 |
| Videos per speaker (min) | 1 (speaker id08916) |
| Videos per speaker (max) | 1,417 (speaker id02760) |

### Modification Type Distribution

| Modification Type | Count | Percentage |
|-------------------|-------|------------|
| real | 20,220 | 26.1% |
| visual_modified | 19,099 | 24.7% |
| both_modified | 19,069 | 24.7% |
| audio_modified | 18,938 | 24.5% |

### Video Characteristics

| Metric | Value |
|--------|-------|
| Video frame count (mean) | 239 |
| Video frame count (median) | 192 |
| Video frame count (min) | 63 |
| Video frame count (max) | 3,810 |

### Audio Characteristics

| Metric | Value |
|--------|-------|
| Audio frame count (mean) | 184,984 |
| Audio frame count (median) | 126,528 |
| Audio frame count (min) | 0 |
| Audio frame count (max) | 6,719,488 |
| Videos with zero audio frames | 211 |

### Speaker Diversity

| Metric | Value |
|--------|-------|
| Speakers with all 4 modification types | 1,336 |
| Speakers with < 4 modification types | 499 |
| Unique speakers in real | 1,731 |
| Unique speakers in visual_modified | 1,707 |
| Unique speakers in both_modified | 1,522 |
| Unique speakers in audio_modified | 1,498 |

### Fake Segment Analysis

| Metric | Value |
|--------|-------|
| Total fake videos | 57,106 |
| Total fake segments | 77,679 |
| Segments per video (mean) | 1.4 |
| Segments per video (max) | 4 |
| Segment duration (mean) | 0.33 seconds |
| Segment duration (median) | 0.30 seconds |
| Segment duration (min) | 0.02 seconds |
| Segment duration (max) | 8.10 seconds |

---

## Real vs. Fake Proportion

| Category | Count | Percentage |
|----------|-------|------------|
| Real videos | 20,220 | 26.1% |
| Fake videos (any modification) | 57,106 | 73.9% |

---

## Key Observations

1. **Balanced modification types:** Near-uniform distribution across all four types (~24.5–26.1% each)

2. **High speaker diversity:** 1,835 speakers with mean 42.1 videos per speaker enables learning individual speaking patterns

3. **Sparse fake segments:** Mean of 1.4 segments per fake video (max 4) indicates localized modifications

4. **Short fake segments:** Mean duration 0.33s (median 0.30s) — modifications are brief and targeted

5. **Audio quality concern:** 211 videos (0.27%) have zero audio frames requiring special handling

6. **Speaker coverage:** 72.8% of speakers (1,336/1,835) have samples across all 4 modification types

---

## Standalone vs. Pipeline EDA

> **Note:** The comprehensive visualizations generated by `analyze_data.py` are **run separately** (manually) and are **not part of `main.py`**. The training pipeline only performs minimal EDA — loading and printing counts. The full analysis (violin plots, KDE, pie charts, fake segment analysis) must be executed independently before running the pipeline.

### Files Generated by Standalone EDA

| File | Description |
|------|-------------|
| `videos_per_speaker_distribution.png` | Violin + boxplot of videos per speaker |
| `video_frames_distribution.png` | KDE + boxplot of video frame counts |
| `modification_type_distribution.png` | Donut chart of modification types |
| `fake_segment_analysis.png` | Dual plot: duration histogram + segments per video |

---

## Pipeline Flow Summary

```
main.py
  │
  ├─ load_metadata()          → Load JSON, print type counts
  ├─ sample_videos()          → Clean data, speaker-split
  │    ├─ Filter: audio_frames > 0 AND video_frames > 0
  │    └─ GroupShuffleSplit by speaker
  │
  ├─ extract_all_features()   → Parallel feature extraction
  │    ├─ Skips already-processed videos
  │    ├─ Tracks failures in *_failed.json
  │    └─ Incremental saves every 500 successes
  │
  ├─ create_dataloaders()     → Lazy-loading Dataset
  │
  └─ train_model()            → Training begins
```

---

## Recommendations for Enhanced EDA in Pipeline

To integrate comprehensive EDA into `main.py` before training:

1. Call `plot_distributions()` from `analyze_data.py` after `load_metadata()`
2. Add data quality reports (duplicate check, missing values, outlier detection)
3. Log class balance metrics to W&B
4. Add correlation analysis between video frames and audio duration