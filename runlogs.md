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

## Next Steps — Model Evaluation

Once you have `best_model.pth` files from your training runs:

**Single model on test set:**
```bash
python inference.py \
    --model best_model.pth \
    --video_dir ./test/ \
    --output results.csv
```

**Compare two models:**
```bash
python evaluate_models.py \
    --model1 run1_best_model.pth \
    --model2 run2_best_model.pth \
    --name1 "Run 1 (10 epochs)" \
    --name2 "Run 2 (20 epochs)" \
    --video_dir ./test/ \
    --output_dir eval_results/
```

**Regenerate with a different random seed (different 100 videos):**
```bash
python create_test_data.py \
    --val_dir /content/drive/MyDrive/val/extracted_val/val \
    --metadata /content/drive/MyDrive/val/val_metadata.json \
    --output_dir ./test_seed2/ \
    --seed 123
```

**Larger test set (200 videos):**
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