# AV-Deepfake Detection Pipeline — Complete Reference

## Overview

This pipeline detects deepfake videos using **both audio and video** modalities. It trains a model to answer three questions per video:
1. Is the **audio** real or fake?
2. Is the **video** real or fake?
3. Is the **whole thing** authentic? (joint prediction)

Scores are between 0 and 1 — **1.0 = real, 0.0 = fake**.

---

## File Map

| File | Role |
|---|---|
| `config.py` | All paths, hyperparameters, and feature settings |
| `data_utils.py` | Metadata loading, speaker-based splitting, parallel feature extraction, dataset class |
| `audio.py` | Audio encoder (ResNet18 on mel-spectrograms) |
| `video.py` | Video encoder (ResNet3D-18 on frame sequences) |
| `cross_modal.py` | Fusion module (combines audio + video → predictions) |
| `train_utils.py` | Training loop, validation, loss functions, optimizer, W&B logging |
| `checkpoint_utils.py` | Save/load checkpoints for resumable training |
| `download_data.py` | Download val data from Hugging Face + extract zips |
| `main.py` | Entry point — ties everything together |
| `inference.py` | Standalone inference on new videos (no training dependencies) |
| `analyze_data.py` | Dataset analysis and distribution plots |
| `create_test_data.py` | Sample test videos from extracted_val/ for model evaluation |
| `evaluate_models.py` | Compare two trained models — metrics, plots, CSV results |

---

## Pipeline Flow (Step by Step)

### Step 1 — Load Metadata
```
main.py  →  data_utils.load_metadata(METADATA_DIR)
```
Reads `val_metadata.json`. Each entry has:
- `file` — relative video path (e.g. `vox_celeb_2/id01234/clip.mp4`)
- `modify_type` — `real`, `audio_modified`, `visual_modified`, or `both_modified`
- `audio_frames` / `video_frames` — frame counts
- `fake_segments` — `[[start_sec, end_sec], ...]` timestamps of manipulated regions

### Step 2 — Speaker-Based Sampling & Split
```
main.py  →  data_utils.sample_videos()
```

```
All videos → Filter (audio_frames > 0 AND video_frames > 0)
           → Extract speaker IDs from file paths
           → GroupShuffleSplit by speaker (80/20)
           → Train set (speakers A, B, C...)
           → Val set   (speakers X, Y, Z...) — ZERO speaker overlap
```

**Why speaker-based?** If the same speaker appears in both train and val, the model cheats by recognising the face/voice instead of detecting manipulation. Speaker-based split forces it to learn actual deepfake artifacts.

### Step 3 — Feature Extraction (28 CPU cores in parallel)
```
main.py  →  data_utils.extract_all_features()  →  process_split_to_disk()
```

For each video, a **2-second window** is extracted and saved as a `.pt` file:

| | How it works | Output shape |
|---|---|---|
| **Window** | Fake video: start of first fake segment. Real video: middle 2 seconds | — |
| **Video** | 50 frames (2s × 25fps) → resize 224×224 → augment → ImageNet normalise | `(50, 3, 224, 224)` |
| **Audio** | Load at 16kHz → mel-spectrogram (`n_mels` bins) → dB → normalise | `(1, n_mels, target_t)` |

**Labels:**

| Type | Audio | Video | Joint |
|---|---|---|---|
| `real` | 1 | 1 | 1 |
| `audio_modified` | 0 | 1 | 0 |
| `visual_modified` | 1 | 0 | 0 |
| `both_modified` | 0 | 0 | 0 |

**Parallelism:** 28 workers (`cpu_count - 4`) extract videos simultaneously using `multiprocessing.Pool` with fork context. Each worker re-seeds its RNG using the video index for independent augmentation.

**Resume safety:** Manifest JSON checkpointed every 500 videos. Failed files tracked in `failed.json` and skipped on restart. Pool submits work in batches of 50 — a single worker crash only loses that batch.

### Step 4 — Model Architecture
```
main.py  →  AVDeepfakeDetector()
```

```
Video (B, 50, 3, 224, 224) → ResNet3D-18 (Kinetics pretrained) → 256-d
Audio (B, 1, 128, 63)      → resize (224,224) → ResNet18 (ImageNet pretrained) → 256-d
                                                                ↓
                              [CLS], video_proj(512), audio_proj(512) + pos_embed
                                            ↓
                              Transformer Encoder (2 layers, 8 heads, GELU, pre-norm)
                                            ↓
                              [CLS] output (512-d)
                              ├→ Audio Head → sigmoid → audio_pred
                              ├→ Video Head → sigmoid → video_pred
                              └→ Joint Head → sigmoid → joint_pred
```

All architectural dimensions (`num_heads`, `num_layers`, `ff_multiplier`, `encoder_fc_dim`) are set in `config.py` — no hardcodes in model files.

**Fusion auto-selection:**
- GPU → `TransformerFusion` (cross-modal self-attention)
- CPU → `PretrainedFusion` (MLP, faster without GPU)

### Step 5 — Training
```
main.py  →  train_utils.train_model()
```

**Two-phase training:**

| Phase | Epochs | What trains | Why |
|---|---|---|---|
| **Phase 1 (Frozen)** | `1 → freeze_epochs` | Only fusion module | Fusion starts random — noisy gradients would corrupt pretrained encoder features |
| **Phase 2 (Fine-tune)** | `freeze_epochs → epochs` | Everything | Encoders adapt to deepfake patterns at 10× lower LR than fusion |

`freeze_epochs` and `patience` are auto-computed:
```python
freeze_epochs = max(1, round(epochs * 0.25))   # 25% of total
patience      = max(5, round(epochs * 0.30))   # 30% of total
```
Override by setting explicitly in `config.py`.

**Loss function:**
```
Loss = FocalLoss(audio_pred, audio_label)
     + FocalLoss(video_pred, video_label)
     + joint_loss_weight × FocalLoss(joint_pred, joint_label)
```

Focal Loss `(1-p)^gamma × BCE` downweights easy examples (obvious fakes/reals) and focuses training on hard ambiguous cases (subtle manipulations). `joint_loss_weight=2.0` by default — joint head is the primary detection target.

**Other details:**
- **Optimizer:** AdamW (fusion: `learning_rate=1e-4`, encoders: `encoder_lr=1e-5`, `weight_decay=1e-4`)
- **Scheduler:** ReduceLROnPlateau — halves LR after `scheduler_patience=5` epochs of no val AUC improvement
- **Gradient clipping:** `grad_clip=1.0` — prevents exploding gradients from 3D convolutions
- **DataParallel:** automatically uses all available GPUs — batch size scales with GPU count
- **Progress bars:** tqdm shows live loss, running average, and gradient norm per batch
- **Checkpoints:** saved every epoch — fully resumable (model + optimizer + scheduler + RNG states)

### Step 6 — Evaluation
```
main.py  →  evaluate_model()
```

Loads `best_model.pth` (best val joint AUC) and runs inference using `eval_n_windows=3` windows averaged per video for robustness. Produces:
1. **Scatter plot** — audio_pred vs video_pred, coloured by manipulation type
2. **Prediction distribution** — histogram of joint predictions (bimodal = confident model)
3. **Confusion matrix** — true/false positives and negatives
4. **AUC scores** — audio, video, and joint
5. **Per-type accuracy** — breakdown by `real`, `audio_modified`, `visual_modified`, `both_modified`
6. `eval_results.csv` — every prediction saved for offline analysis

---

## W&B Monitoring Guide

The pipeline logs rich metrics to Weights & Biases. Here is what each graph tells you and how to interpret it.

### Training Dynamics

| W&B Metric | What it measures | How to interpret |
|---|---|---|
| `train/loss` vs `val/loss` | Loss per epoch for train and val | Growing gap = overfitting. Should converge together. |
| `val/overfit_gap` | `train_loss - val_loss` | > 0.1 = overfitting. Negative = underfitting. Ideal: near 0. |
| `train/grad_norm` | Average gradient magnitude per epoch | > 5 = exploding gradients (increase `grad_clip`). Near 0 = vanishing (LR too low). Healthy range: 0.1–2.0. |
| `learning_rate` | Current LR over time | Drops show when ReduceLROnPlateau fired. Each drop = model near a plateau. |
| `epoch_time_s` | Seconds per epoch | Useful for estimating total training time. |
| `phase` | `frozen` or `finetune` | Shows the phase transition point — expect a temporary loss spike here as encoders unfreeze. |

### Detection Quality

| W&B Metric | What it measures | How to interpret |
|---|---|---|
| `val/auc_joint` | Primary metric — overall deepfake detection | > 0.9 = excellent, 0.7–0.9 = good, 0.5–0.7 = poor, 0.5 = random chance. |
| `val/auc_audio` | Audio-only detection | Compares to video — large gap means one modality is ignored. |
| `val/auc_video` | Video-only detection | Should improve alongside audio for balanced learning. |
| `val/accuracy_joint` | Accuracy at `eval_threshold=0.5` | More interpretable than AUC but threshold-dependent. AUC is more reliable. |
| `val/auc_gap` | `|audio_auc - video_auc|` | > 0.1 = model leaning on one modality. Ideally < 0.05. |
| `val/confidence` | Mean `|pred - 0.5| × 2` | 0 = always predicting exactly 0.5 (not learning). 1 = always decisive. Should increase over training. |

### Dataset & Per-Type Insights

| W&B Graph | What it shows | How to interpret |
|---|---|---|
| `val/auc_type/real` | AUC for correctly identifying real videos | Low = model has false positive problem (flagging real videos as fake). |
| `val/auc_type/audio_modified` | AUC for detecting audio-only fakes | Low = model struggles with audio manipulation specifically. Improve audio encoder or augmentation. |
| `val/auc_type/visual_modified` | AUC for detecting video-only fakes | Low = model struggles with visual manipulation. ResNet3D features may need more fine-tuning. |
| `val/auc_type/both_modified` | AUC for detecting both-modality fakes | Usually highest — both modalities are fake so it's the easiest case. |
| `val/accuracy_type/*` | Per-type accuracy alongside AUC | Cross-reference with AUC — if accuracy is high but AUC is low, the threshold may need adjustment. |
| `val/per_type_auc` bar chart | All four types side-by-side | Quickly shows which type is hardest. The shortest bar is your weakest area. |
| `val/prediction_scatter` | Audio score vs video score, coloured by type | 4 distinct clusters = model correctly distinguishes all manipulation types. Overlapping clusters = confusion between types. This is the most informative single graph. |

### Prediction Distributions

| W&B Graph | What it shows | How to interpret |
|---|---|---|
| `val/predictions/joint` histogram | Distribution of joint predictions | **Bimodal** (peaks near 0 and 1) = confident, well-trained model. **Unimodal at 0.5** = model is uncertain and not learning. Should become more bimodal as training progresses. |
| `val/predictions/audio` histogram | Distribution of audio predictions | Same interpretation as joint. Should have peaks near 0 (for fakes) and 1 (for reals). |
| `val/predictions/video` histogram | Distribution of video predictions | Same. Compare to audio histogram — if one is flatter, that modality needs more work. |

### Model Diagnostics

| W&B Graph | What it shows | How to interpret |
|---|---|---|
| `val/confusion_matrix` | True/false positives and negatives | High false negatives = missing deepfakes (dangerous). High false positives = over-flagging real videos. Adjust `eval_threshold` to trade off between the two. |
| `val/roc_curve` (every 2 epochs) | Performance across all thresholds | Area under curve = AUC. A curve hugging the top-left corner is ideal. If the curve barely rises above the diagonal, the model is near random. |
| `val/training_health` table | All key metrics with plain-English status | Single table per epoch showing Good/Low/Balanced/Overfitting/OK for each metric. Check this first for a quick health check. |

### Hardware

| W&B Metric | What it shows | How to interpret |
|---|---|---|
| `gpu/memory_allocated_gb` | VRAM actively used | If increasing epoch-over-epoch = memory leak. Stable = healthy. |
| `gpu/memory_reserved_gb` | VRAM held by PyTorch allocator | Higher than allocated is normal (PyTorch caches). If reserved ≈ total VRAM = OOM risk. |

---

## Reading a Training Run — Checklist

After each run, check these in order:

1. **Did `val/auc_joint` improve?** If stuck at 0.5 after 3+ epochs, something is wrong (check data loading, labels, loss).
2. **Is `val/overfit_gap` growing?** If train loss drops but val loss rises, reduce `batch_size`, increase `dropout`, or add more data.
3. **Is `val/auc_gap` large?** If > 0.1, one modality is being ignored. Check augmentation balance and encoder learning rates.
4. **Are predictions bimodal?** Open `val/predictions/joint` histogram. Flat distribution = model not converging.
5. **Which type has lowest AUC?** Open `val/per_type_auc` bar chart. The lowest bar tells you where to focus improvement.
6. **Is the scatter plot showing 4 clusters?** Open `val/prediction_scatter`. If clusters overlap, the model can't distinguish between types.
7. **Did grad norm explode?** If `train/grad_norm` spikes above 5, reduce `grad_clip` or `learning_rate`.

---

## Config Quick Reference

All values configurable in `config.py` — no hardcodes in model files.

### Model Architecture

| Setting | Default | Effect |
|---|---|---|
| `feature_dim` | 256 | Encoder output size — larger = more expressive but slower |
| `hidden_dim` | 512 | Fusion module width — 2× feature_dim is standard |
| `dropout` | 0.4 | Applied to all encoders and fusion — increase if overfitting |
| `encoder_fc_dim` | 512 | Intermediate FC inside encoder heads |
| `num_heads` | 8 | Transformer attention heads — must divide `hidden_dim` evenly |
| `num_layers` | 2 | Transformer encoder depth — more layers = more capacity |
| `ff_multiplier` | 4 | Feedforward dim = `hidden_dim × ff_multiplier` |
| `n_mels` | 128 | Mel frequency bins — more = finer frequency resolution |
| `top_db` | 80 | Audio dynamic range in dB — lower = more compression |

### Training

| Setting | Default | Effect |
|---|---|---|
| `epochs` | 10 | Max training epochs |
| `freeze_epochs` | auto (25%) | Epochs with frozen encoders — set `None` for formula |
| `patience` | auto (30%) | Early stopping patience — set `None` for formula |
| `batch_size` | 8 | Per-GPU batch size — DataParallel scales this × num_gpus |
| `focal_gamma` | 2.0 | Focus on hard examples — 0 = standard BCE |
| `focal_alpha` | 0.25 | Class balance weight |
| `joint_loss_weight` | 2.0 | Relative weight of joint head in total loss |
| `learning_rate` | 1e-4 | Fusion module learning rate |
| `encoder_lr` | 1e-5 | Encoder learning rate (10× lower to preserve pretrained features) |
| `weight_decay` | 1e-4 | L2 regularisation |
| `grad_clip` | 1.0 | Max gradient norm before clipping |
| `scheduler_factor` | 0.5 | LR multiplier when plateau detected |
| `scheduler_patience` | 5 | Epochs before LR reduction |

### Evaluation

| Setting | Default | Effect |
|---|---|---|
| `eval_threshold` | 0.5 | Decision boundary for real/fake verdict |
| `eval_n_windows` | 3 | Windows averaged per video — more = more accurate, slower |

### Augmentation

| Setting | Default | Effect |
|---|---|---|
| `aug_freq_mask` | 20 | SpecAugment max frequency mask (mel bins) |
| `aug_time_mask` | 15 | SpecAugment max time mask (frames) |
| `aug_brightness` | 0.2 | Video brightness jitter ± range |
| `aug_contrast_min` | 0.8 | Video contrast jitter min multiplier |
| `aug_contrast_max` | 1.2 | Video contrast jitter max multiplier |

### Feature Extraction

| Setting | Default | Effect |
|---|---|---|
| `sr` | 16000 | Audio sample rate (Hz) |
| `fps` | 25 | Video frames per second |
| `num_frames` | 50 | Frames per clip (2s × 25fps) |
| `img_size` | 224 | Frame resize resolution |
| `n_fft` | 1024 | FFT window size |
| `hop_length` | 512 | FFT hop length — `target_t` is derived automatically |

---

## How to Run

```bash
# Full pipeline (download → extract → train → evaluate)
python main.py --fresh

# Resume training from last checkpoint
python main.py

# Without W&B logging
python main.py --no_wandb

# Force a specific fusion type
python main.py --fusion_type transformer

# Run W&B hyperparameter sweep (features must be extracted first)
python main.py --sweep --sweep_count 20

# Analyse dataset before training
python analyze_data.py --metadata_path /path/to/val_metadata.json

# Generate a 100-video test dataset from extracted_val/
# NOTE: always pass --metadata explicitly — val_metadata.json lives one level
#       above extracted_val/, not inside it
python create_test_data.py \
    --val_dir /content/drive/MyDrive/val/extracted_val/val \
    --metadata /content/drive/MyDrive/val/val_metadata.json \
    --output_dir ./test/

# Larger test set (200 videos, 50 per type)
python create_test_data.py \
    --val_dir /content/drive/MyDrive/val/extracted_val/val \
    --metadata /content/drive/MyDrive/val/val_metadata.json \
    --output_dir ./test_large/ \
    --per_type 50

# Run inference on a single video
python inference.py --model checkpoints/best_model.pth --video test.mp4

# Run inference on a test folder
python inference.py --model checkpoints/best_model.pth \
    --video_dir ./test/ --output results.csv

# Compare two trained models
python evaluate_models.py \
    --model1 run1_best_model.pth --name1 "Run 1" \
    --model2 run2_best_model.pth --name2 "Run 2" \
    --video_dir ./test/ --output_dir eval_results/
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **GPU VRAM** | 8 GB | 24 GB |
| **RAM** | 32 GB | 64 GB |
| **CPU Cores** | 8 | 32+ |
| **Storage** | 600 GB SSD | 1 TB NVMe |

**GPU usage:** Both GPUs used via DataParallel during training. Feature extraction is CPU-only (OpenCV bottleneck).

**CPU usage:** 28 cores during feature extraction (`cpu_count - 4`). 8 cores for DataLoader workers during training.

---

## Feature Storage Structure

```
checkpoints/
  features/
    train/
      0.pt, 1.pt, 2.pt, ...     ← one per video (~30MB each)
    val/
      0.pt, 1.pt, ...
    train_manifest.json          ← metadata index (file, type, speaker)
    val_manifest.json
    train_failed.json            ← indices of corrupted/missing files
    val_failed.json
  best_model.pth                 ← best val joint AUC checkpoint
  training_checkpoint.pth        ← latest epoch checkpoint (for resuming)
  wandb_run_id.txt               ← W&B run ID for resuming logged runs
  results/
    final_results.png
    eval_results.csv
    training_curves.png
    training_history.json
    pipeline_log.txt
```

---

## Dataset Facts

| Fact | Value |
|---|---|
| Total entries in val_metadata.json | 77,326 |
| Found on disk (after extraction) | 68,851 (89%) |
| Missing from disk | 8,475 (11% — corrupted or skipped during extraction) |
| real videos | 18,037 |
| visual_modified videos | 17,020 |
| both_modified videos | 16,946 |
| audio_modified videos | 16,848 |
| Metadata location | One level above `extracted_val/` — **not** inside it |
| Video location | `extracted_val/val/vox_celeb_2/...` |

> **Key gotcha:** `val_metadata.json` is at `DATA_DIR/val_metadata.json`, but videos are at `DATA_DIR/extracted_val/val/`. Always pass `--metadata` explicitly when using `create_test_data.py`.

---

## Common Issues and Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| `CUDA out of memory` | `batch_size` too large for VRAM | Reduce `batch_size` in `config.py`. With DataParallel it scales × num_gpus. |
| `val/auc_joint` stuck at 0.5 | Labels wrong or data not loading | Check `fake_segments` collation, sentinel label filtering, and per-type label map. |
| `each element in batch should be of equal size` | Variable `fake_segments` length | Ensure `av_collate_fn` is being used in DataLoaders. |
| All predictions cluster near 0.5 | Model not learning | Check `focal_gamma` — try reducing to 1.0. Check `learning_rate`. |
| `overfit_gap` growing fast | Overfitting | Increase `dropout`, reduce `epochs`, increase augmentation strength. |
| `auc_gap` > 0.1 | One modality ignored | Lower `encoder_lr` for the stronger modality or increase augmentation on the weaker one. |
| Feature extraction all failing | Wrong `VAL_DIR` path | Check `.env` — `VAL_DIR` must point to folder containing `vox_celeb_2/...` subfolders. |
| Pool crashes mid-extraction | OOM or corrupted video | Work is batched in groups of 50 — crash only loses that batch. Restart to continue. |
| `create_test_data.py` — 0 videos found on disk | Wrong `--val_dir` or metadata path mismatch | Pass `--metadata` explicitly. `val_metadata.json` is one level above `extracted_val/`. Use `--val_dir extracted_val/val/` not `extracted_val/`. |
| `val_metadata.json` not found | Script searching wrong location | Always use `--metadata /path/to/val_metadata.json` explicitly with `create_test_data.py`. |

---

## Changes: Val-Speaker-Only Test Set

### Problem
`create_test_data.py` sampled test videos from **all** speakers randomly, including speakers the model trained on. This creates data leakage — inflated metrics that don't reflect real-world performance.

### Solution — `create_test_data.py`
- **New `get_val_speakers()`** — recreates the exact `GroupShuffleSplit(seed=42)` from `data_utils.py` to identify which speakers were in the val split during training
- **`sample_videos()`** now accepts `val_speakers` and only samples from those speakers
- **Saves `split_info.json`** — contains full train/val speaker lists for audit
- **New CLI flags:** `--use_all` (matches training config), `--train_seed` (default 42)

### Why it works
The split is deterministic: same `val_metadata.json` + same `seed=42` + same `GroupShuffleSplit` = exact same speaker assignments every time. No manifests from the server needed.

### Usage
```bash
# Default (subset mode, matches config.py defaults)
python create_test_data.py \
    --val_dir /path/to/extracted_val \
    --metadata /path/to/val_metadata.json \
    --output_dir ./test/

# If training used use_all_data=True
python create_test_data.py --use_all \
    --val_dir /path/to/extracted_val \
    --output_dir ./test/
```

---

## Changes: Web Interface

### Overview
A simplified web application for deepfake detection — three self-contained tabs with no external JS/CSS dependencies. Built on Flask + single-page HTML with SQLite history. All ML logic is imported from `inference.py` — no model architecture duplication.

### Pages

| Page | Features |
|---|---|
| **Analyze** | Drag-and-drop video upload, model selector dropdown with per-model metadata, real-time verdict + audio/video/joint score display, reset for next upload |
| **Compare** | Upload one video, run through both models side-by-side with per-model verdicts, three score breakdowns each, and agree/disagree summary line |
| **History** | SQLite-backed table of all past analyses (file, verdict tag, joint score, model), per-entry delete, bulk clear all |

### Backend (`web/app.py`)
- Imports `load_model` and `predict_video` from `inference.py` — no duplicate architecture
- In-memory model cache — loads each model once, shares across requests
- Model paths hardcoded to `logs/logs_2/best_model.pth` (Model 2) and `logs/logs_3/best_model.pth` (Model 3)
- Multi-window inference (3 windows per video, averaged for robustness)
- SQLite history database (`history.db`) — auto-saves every analysis with file, scores, verdict, model, timestamp
- Uploaded videos cleaned up immediately after processing

### API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/analyze` | POST | Upload + classify a single video |
| `/api/compare` | POST | Compare both models on one video |
| `/api/models` | GET | List available models with metadata and device info |
| `/api/history` | GET/DELETE | Fetch or clear analysis history |
| `/api/history/<id>` | DELETE | Delete a single history entry |

### Usage
```bash
cd web
pip install flask torch torchvision torchaudio opencv-python-headless

python app.py
# Open http://localhost:5000
```