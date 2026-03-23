# AV-Deepfake Detection — Run Logs

A record of all commands run, their outputs, and what was learned. Use this as a reference when re-running on a new instance.

---

## Environment

| Detail | Value |
|---|---|
| Platform | Google Colab |
| Working directory | `/content/drive/MyDrive/Colab Notebooks/Deepfake/` |
| Data directory | `/content/drive/MyDrive/val/` |
| Metadata | `/content/drive/MyDrive/val/val_metadata.json` |
| Videos | `/content/drive/MyDrive/val/extracted_val/val/` |

---

## Run 1 — Test Dataset Generation

**Date:** 23 Mar 2026

**Command:**
```bash
python create_test_data.py \
    --val_dir /content/drive/MyDrive/val/extracted_val/val \
    --metadata /content/drive/MyDrive/val/val_metadata.json \
    --output_dir ./test/
```

**Output:**
```
=======================================================
TEST DATASET GENERATOR
=======================================================
  val_dir:    /content/drive/MyDrive/val/extracted_val/val
  output_dir: ./test/
  per_type:   25  (100 total videos)
  seed:       42
  mode:       copy

LOADING METADATA
  Loading metadata from: /content/drive/MyDrive/val/val_metadata.json
  Total entries: 77,326
  Checking which videos exist on disk...
  Found on disk: 68,851  (missing: 8,475)
  Type distribution:
    real                : 18,037
    visual_modified     : 17,020
    both_modified       : 16,946
    audio_modified      : 16,848

SAMPLING
  ✓ real                : 25 sampled from 18,037
  ✓ audio_modified      : 25 sampled from 16,848
  ✓ visual_modified     : 25 sampled from 17,020
  ✓ both_modified       : 25 sampled from 16,946
  Total sampled: 100
    Real:  25  (1 type × 25)
    Fake:  75  (3 types × 25)

COPYING FILES
  100%|████████████████████████████| 100/100 [00:08<00:00, 11.98video/s]

TEST DATASET CREATED
  Location:   /content/drive/MyDrive/Colab Notebooks/Deepfake/test
  Total:      100 videos
    real/     25 videos
    fake/     75 videos
  Breakdown by manipulation type:
    real                : 25
    audio_modified      : 25
    visual_modified     : 25
    both_modified       : 25
  Manifest:   ./test/test_manifest.json
```

**What was learned:**
- `val_metadata.json` lives one level above `extracted_val/` — always pass `--metadata` explicitly
- 8,475 videos listed in metadata were missing from disk (likely corrupted or skipped during extraction)
- 68,851 / 77,326 = 89% of videos successfully on disk — healthy
- Dataset is well-balanced: ~17–18K per type
- 100 videos copied in 8 seconds — fast even on Google Drive

**Test dataset structure created:**
```
test/
    real/       ← 25 videos  (modify_type = real)
    fake/       ← 75 videos  (25 audio_modified + 25 visual_modified + 25 both_modified)
    test_manifest.json
```

**Important note on balance:**
The test set has 25 real and 75 fake (3:1 ratio). This mirrors the dataset's structure — there are 3 fake modification types vs 1 real type. Use **AUC** as the primary metric when evaluating, not accuracy, since accuracy is misleading with imbalanced sets.

---

## Testing Guidelines

### Which script to use?

| Situation | Script | Notes |
|---|---|---|
| Test a **single video** | `inference.py --video` | No ground truth needed |
| Test a **single folder** of videos | `inference.py --video_dir` | No ground truth needed |
| **Measure accuracy** of one model | `evaluate_models.py` (same model twice) | Needs real/fake folders |
| **Compare two models** | `evaluate_models.py` | Needs real/fake folders |

### `inference.py` vs `evaluate_models.py` — key difference

**`inference.py`** — run one model, get predictions. No accuracy metrics. Works on any video with no ground truth needed.
```
one model → raw predictions (score + verdict) → CSV
```

**`evaluate_models.py`** — compare two models against known labels. Computes AUC, accuracy, confusion matrix, plots.
```
two models + ground truth folders → metrics + plots + CSV
```

> **Note:** The folder names `real/` and `fake/` do **not** affect what the model predicts — the model only sees raw pixels and audio. The folders exist purely so `evaluate_models.py` knows the ground truth label to compute accuracy after prediction.

---

## Testing Commands

### Single video — any model
```bash
python inference.py \
    --model /content/drive/MyDrive/best_model_1.pth \
    --video ./test/real/id01234_0001_real.mp4
```

### Single folder — get predictions for all videos in it
```bash
# Real videos only
python inference.py \
    --model /content/drive/MyDrive/best_model_1.pth \
    --video_dir ./test/real/ \
    --output results_real.csv

# Fake videos only
python inference.py \
    --model /content/drive/MyDrive/best_model_1.pth \
    --video_dir ./test/fake/ \
    --output results_fake.csv
```

### Single model accuracy — full metrics on the test set
```bash
# Pass the same model as both --model1 and --model2
python evaluate_models.py \
    --model1 /content/drive/MyDrive/best_model_1.pth \
    --model2 /content/drive/MyDrive/best_model_1.pth \
    --name1 "Model 1" \
    --name2 "Model 1" \
    --video_dir ./test/ \
    --output_dir eval_results_model1/
```

### Compare two models — side-by-side metrics and plots
```bash
python evaluate_models.py \
    --model1 /content/drive/MyDrive/best_model_1.pth \
    --model2 /content/drive/MyDrive/best_model_2.pth \
    --name1 "Run 1 (10 epochs)" \
    --name2 "Run 2 (20 epochs)" \
    --video_dir ./test/ \
    --output_dir eval_results/
```

### Known issue — inference.py finds 0 videos from test/ root
`inference.py` does not recurse into subfolders. Always point it at `test/real/` or `test/fake/` directly, not `test/`. `evaluate_models.py` handles subfolders automatically.

```bash
# Wrong — finds 0 videos
python inference.py --video_dir ./test/

# Correct — point at subfolder
python inference.py --video_dir ./test/real/
```

---

## Regenerating the Test Dataset

**Different random seed (different 100 videos):**
```bash
python create_test_data.py \
    --val_dir /content/drive/MyDrive/val/extracted_val/val \
    --metadata /content/drive/MyDrive/val/val_metadata.json \
    --output_dir ./test_seed2/ \
    --seed 123
```

**Larger test set (200 videos, 50 per type):**
```bash
python create_test_data.py \
    --val_dir /content/drive/MyDrive/val/extracted_val/val \
    --metadata /content/drive/MyDrive/val/val_metadata.json \
    --output_dir ./test_large/ \
    --per_type 50
```

---

## Dataset Facts (from metadata)

| Fact | Value |
|---|---|
| Total entries in metadata | 77,326 |
| Found on disk | 68,851 (89%) |
| Missing from disk | 8,475 (11%) |
| real videos | 18,037 |
| visual_modified videos | 17,020 |
| both_modified videos | 16,946 |
| audio_modified videos | 16,848 |
| Metadata location | `/content/drive/MyDrive/val/val_metadata.json` |
| Video location | `/content/drive/MyDrive/val/extracted_val/val/` |