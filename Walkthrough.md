# AV-Deepfake Detection — Detailed Code Walkthrough

A step-by-step explanation of how the entire pipeline works and why each design choice was made.

---

## File Overview

| File | Purpose |
|------|---------|
| `config.py` | Centralized configuration — paths, hyperparameters, dataclass configs |
| `data_utils.py` | Data loading, speaker-based splitting, feature extraction, lazy-loading dataset |
| `audio.py` | Audio encoder models (simple, improved, pretrained ResNet18) |
| `video.py` | Video encoder models (simple, improved, pretrained ResNet3D-18) |
| `cross_modal.py` | Cross-modal fusion modules (MLP, attention, transformer) |
| `train_utils.py` | Training loop, FocalLoss, optimizer/scheduler setup, evaluation metrics |
| `checkpoint_utils.py` | Checkpoint saving/loading for resumable training |
| `main.py` | Full pipeline orchestration — data loading → training → evaluation |
| `inference.py` | Standalone inference on trained models (no training dependencies) |
| `compare_models.py` | Multi-model comparison on test data with comprehensive metrics/plots |
| `create_test_data.py` | Generate leak-free test dataset from validation speakers only |
| `evaluate_models.py` | **DEPRECATED** — superseded by `compare_models.py` |
| `web/app.py` | Flask web server ("DeepScan") for deepfake detection via browser |
| `download_data.py` | Download AV-Deepfake1M++ from Hugging Face |

---

## Step 1: Configuration (`config.py`)

All paths and hyperparameters are centralized in dataclass configs:

```python
DATA_DIR = os.environ.get('DATA_DIR', '.')
VAL_DIR = os.path.join(DATA_DIR, 'extracted_val', 'val')
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', './checkpoints')
```

> **Why env vars?** Makes the pipeline portable — same code works on Colab, VPS, or local by just changing `.env`.

**Dataclass configs:**
```python
MODEL_CONFIG = ModelConfig(feature_dim=256, hidden_dim=512, dropout=0.4)
TRAIN_CONFIG = TrainConfig(epochs=10, batch_size=8, freeze_epochs=None, patience=None, ...)
OPTIM_CONFIG = OptimConfig(learning_rate=1e-4, encoder_lr=1e-5, weight_decay=1e-4)
```

**Feature extraction constants:**
```python
FEATURE_CONFIG = {
    'sr': 16000,      # audio sample rate
    'fps': 25,        # video FPS
    'duration': 2.0,  # clip duration in seconds
    'num_frames': 50, # 2s × 25fps
    'img_size': 224,  # ResNet input
    'audio_samples': 32000, # 2s × 16000Hz
    'n_fft': 1024, hop_length': 512,
    'target_t': 63,   # derived: fixed mel time dimension
}
```

**Key hyperparameters:**
| Setting | Value | Why |
|---|---|---|
| `feature_dim` | 256 | Encoder output size — balanced between expressiveness and overfitting |
| `hidden_dim` | 512 | Fusion layer width — 2× feature_dim gives enough capacity |
| `dropout` | 0.4 | Higher than typical (0.2-0.3) because the dataset is large and we want generalization |
| `freeze_epochs` | auto (25% of epochs) | Set to `None` for formula `max(1, round(epochs * 0.25))`, or set explicitly |
| `patience` | auto (30% of epochs) | Set to `None` for formula `max(5, round(epochs * 0.30))`, or set explicitly |

---

## Step 2: Data Loading (`data_utils.py → load_metadata()`)

```python
df = load_metadata(VAL_DIR)
```

Reads `val_metadata.json` — one entry per video:
```json
{
  "file": "source/id00015/clip_001.mp4",
  "modify_type": "audio_modified",
  "audio_frames": 32000,
  "video_frames": 50,
  "fake_segments": [[1.2, 3.4]]
}
```

> **Why use val as both train and val?** The AV-Deepfake1M++ dataset is huge. We use only the validation split (~77K videos) and create our own train/val split from it. The full train set (1M+ videos) would require far more storage.

---

## Step 3: Speaker-Based Split (`data_utils.py → sample_videos()`)

```python
train_df, val_df = sample_videos(df, samples_per_type, val_split=0.2)
```

Extracts **speaker IDs** from file paths (e.g. `source/id00015/clip.mp4` → `id00015`), then splits using `GroupShuffleSplit`:

```
Train speakers: {A, B, C, D, E} ← 80% of speakers
Val speakers: {X, Y, Z} ← 20% of speakers
Zero overlap ✓
```

> **Why speaker-based, not random?** Random splits cause **identity leakage** — the model learns faces/voices instead of manipulation artifacts. With speaker-based splitting, the model has never seen the val speakers during training, so it must detect actual deepfake artifacts.

---

## Step 4: Feature Extraction (`data_utils.py → extract_av_features()`)

For each video, a **2-second window** is extracted:

### Window Selection
```python
if fake_segments:  # manipulated video
    start_sec = fake_segments[0][0]  # start of first fake segment
else:  # real video
    total_sec = total_frames / cfg['fps']
    start_sec = max(0, (total_sec / 2) - (cfg['duration'] / 2))  # middle 2 seconds
```

> **Why 2 seconds?** Most deepfake artifacts are subtle and localized. 2 seconds captures enough temporal context (50 frames) without requiring massive tensors.

> **Why start at fake segments?** Ensures the model sees the actual manipulated region, not a potentially unmodified part of the video.

### Video Features
```python
for _ in range(cfg['num_frames']):  # 50 frames
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (cfg['img_size'], cfg['img_size']))
    frame = frame / 255.0  # normalize to [0,1]
    frames.append(frame)
# ImageNet normalisation: (frames - mean) / std
video_tensor = torch.FloatTensor(frames_arr).permute(0, 3, 1, 2)
# Output: (50, 3, 224, 224)
```

> **Why 224×224?** Standard ResNet input size. The pretrained weights expect this resolution.

> **Why 50 frames?** 2 seconds × 25 fps = 50 frames. Captures motion patterns that deepfakes often get wrong (temporal inconsistencies).

> **Why ImageNet normalization?** Matches the distribution the pretrained ResNet was trained on.

### Audio Features
```python
waveform, orig_sr = torchaudio.load(video_path)  # load from video file
if orig_sr != cfg['sr']:
    resampler = torchaudio.transforms.Resample(orig_sr, cfg['sr'])
    waveform = resampler(waveform)
waveform = waveform[:, start_sample:end_sample]  # slice 2 seconds
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=cfg['sr'], n_mels=128, n_fft=cfg['n_fft'], hop_length=cfg['hop_length']
)(waveform)
audio_tensor = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_spec)
# Per-sample normalisation: (audio - mean) / (std + 1e-6)
# Output: (1, 128, 63) — fixed time dimension via padding/trimming
```

> **Why torchaudio over librosa?** Librosa uses PySoundFile/audioread which caused crashes on corrupted MP4s. Torchaudio uses FFmpeg directly — more robust, no Python-level crashes.

> **Why mel-spectrogram?** Converts raw audio waveform into a 2D time-frequency image that a CNN can process. The mel scale emphasizes frequencies humans are sensitive to, which is where deepfake artifacts tend to appear (unnatural voice harmonics).

> **Why 128 mel bins?** Standard for speech tasks — captures enough frequency detail without being excessive.

### Augmentation (`spec_augment()`)
Applied to training videos only:
- Random frequency masking (zero out a band of mel bins)
- Random time masking (zero out a band of time steps)
- Random horizontal flip on video frames
- Random brightness/contrast jitter

---

## Step 5: Disk Storage (`data_utils.py → process_split_to_disk()`)

Each video's features are saved as individual `.pt` files:
```python
torch.save({
    'video': video_tensor,   # (50, 3, 224, 224)
    'audio': audio_tensor,   # (1, 128, 63)
    'labels': torch.FloatTensor([audio_label, video_label])
}, f'{idx}.pt')
```

> **Why individual files, not one big pickle?** Loading 77K videos into RAM would need ~2.3TB. Individual files allow **lazy-loading** — the DataLoader reads one file per sample, keeping RAM at ~2-3GB regardless of dataset size.

> **Why .pt format?** PyTorch native — fastest serialization for tensors.

### Resume & Failed Tracking
```python
# Skip if already extracted (check manifest + .pt file existence)
if idx in failed_indices: skip  # corrupted files tracked in failed.json
if os.path.exists(pt_path): skip  # already done
```

> **Why track failures?** Corrupted MP4s (e.g. `moov atom not found`) would be retried on every restart. The `failed.json` file remembers them so they're skipped instantly.

### Lazy-Loading Dataset (`AVDataset`)
```python
class AVDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        data = torch.load(pt_path, map_location='cpu', weights_only=True)
        return {
            'video': data['video'],
            'audio': data['audio'],
            'labels': data['labels'],
            'type': entry['type'],
            'file': entry['file'],
            'fake_segments': entry.get('fake_segments', []),
        }
```
One `.pt` file loaded per sample — RAM stays constant regardless of dataset size.

---

## Step 6: Audio Encoder (`audio.py → PretrainedAudioEncoder`)

```python
backbone = resnet18(pretrained=True)
backbone.conv1 = Conv2d(1, 64, 7, stride=2, padding=3)   # 1-channel input
backbone.fc = Linear(512, 256)                             # project to 256-d
```

The mel-spectrogram `(1, 128, T)` is **resized to (1, 224, 224)** and fed through ResNet18 as a grayscale image:

```
Mel-spec (1, 128, T) → resize (1, 224, 224) → ResNet18 → 256-d vector
```

> **Why ResNet18 for audio?** Mel-spectrograms are 2D images — CNNs designed for images work naturally. ResNet18 is small enough to not overfit, but powerful enough to learn frequency patterns.

> **Why pretrained on ImageNet?** Even though ImageNet has cats and dogs (not audio), the low-level learned features (edges, textures, patterns) transfer surprisingly well to spectrogram analysis. This is well-established in audio ML research.

> **Why change conv1 to 1 channel?** ResNet18 expects 3-channel RGB, but spectrograms are single-channel. We keep the rest of the pretrained weights.

---

## Step 7: Video Encoder (`video.py → PretrainedVideoEncoder`)

```python
backbone = r3d_18(pretrained=True)    # ResNet3D-18, pretrained on Kinetics-400
backbone.fc = Linear(512, 256)         # project to 256-d
```

Video tensor is permuted from `(B, T, C, H, W)` → `(B, C, T, H, W)` for 3D convolutions:

```
Frames (B, 50, 3, 224, 224) → permute (B, 3, 50, 224, 224) → ResNet3D-18 → 256-d vector
```

> **Why 3D convolutions?** Unlike 2D CNNs that process frames independently, 3D convolutions process **space and time jointly**. This captures temporal artifacts — like unnatural eye blinking, lip-sync mismatches, or jittery motion — that are hallmarks of deepfakes.

> **Why pretrained on Kinetics-400?** Kinetics is an action recognition dataset with 400K videos. The pretrained model already understands human motion, facial movement, and temporal patterns — exactly what we need for detecting manipulation.

---

## Step 8: Fusion Module (`cross_modal.py`)

After encoding, we have two 256-d vectors per video. The fusion module combines them:

### MLP Fusion (CPU default)
```
concat(video_256d, audio_256d) → 512-d → ReLU → Dropout → 512-d → ReLU → Dropout
    ├→ Audio classifier (512 → 1 → sigmoid)
    ├→ Video classifier (512 → 1 → sigmoid)
    └→ Joint classifier (512 → 1 → sigmoid)
```

> **Why MLP?** Simple and fast. The concatenation lets the MLP learn correlations between audio and video features (e.g. "audio sounds fake AND video looks fake → both_modified").

### Transformer Fusion (GPU default)
```
[CLS], video_proj(256→512), audio_proj(256→512) + pos_embeddings
    → 2-layer Transformer Encoder (8 heads, GELU, pre-norm)
    → [CLS] output (512-d)
        ├→ Audio classifier → sigmoid
        ├→ Video classifier → sigmoid
        └→ Joint classifier → sigmoid
```

> **Why Transformer?** Multi-head self-attention allows the video and audio tokens to directly **attend to each other**. The model can learn to ask: "Does this audio match this face?" — a question that cross-modal attention is uniquely suited for.

> **Why [CLS] token?** Borrowed from BERT — a learnable token that aggregates information from both modalities. The classifier reads from this single token, which has attended to both audio and video.

> **Why auto-select?** Transformers are matrix-multiplication heavy — very fast on GPU, slow on CPU. The MLP gives comparable results on CPU without the overhead.

---

## Step 9: Training (`train_utils.py → train_model()`)

### Loss Function
```python
loss = BCE(audio_pred, audio_label)      # is audio real?
     + BCE(video_pred, video_label)      # is video real?
     + 2× BCE(joint_pred, joint_label)   # is overall video real?
```

> **Why three separate losses?** Each head specializes in one detection task. The joint head is weighted 2× because it's the primary detection target — "is this video a deepfake?"

> **Why Focal Loss?** `(1-p)^gamma × BCE` downweights easy examples so training focuses on hard, ambiguous cases — subtle manipulations near the decision boundary. `gamma=2.0` by default. When `gamma=0` it reduces to standard BCE, subsuming label smoothing.

### Two-Phase Training

**Phase 1 (Epochs 1-8): Frozen encoders**
```python
for param in model.video_encoder.parameters():
    param.requires_grad = False   # ← no gradients, no updates
```

> **Why freeze?** The fusion module starts with random weights. Its gradients would be noisy and destructive to the carefully pretrained encoder features. Freezing lets the fusion learn stable representations first.

**Phase 2 (Epochs 9-50): Fine-tune all**
```python
optimizer = AdamW([
    {'params': video_encoder, 'lr': 1e-5},    # 10× lower
    {'params': audio_encoder, 'lr': 1e-5},    # 10× lower
    {'params': fusion_module,  'lr': 1e-4},   # normal
])
```

> **Why lower LR for encoders?** They already have good features from pretraining. Small updates let them adapt to deepfake-specific patterns without "catastrophic forgetting" of their general knowledge.

### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

> **Why?** Prevents exploding gradients — especially important with 3D convolutions on 50-frame inputs, which can produce very large gradient magnitudes.

### Scheduler
```python
ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
```

> **Why?** If validation loss doesn't improve for 5 epochs, halves the learning rate. This lets the model "settle into" better minima as training progresses.

### Early Stopping
```python
if patience_counter >= 15:
    break   # stop training
```

> **Why 15?** Generous enough to allow the model to recover after an LR reduction, but stops wasting compute if it's truly converged.

---

## Step 10: Evaluation (`main.py → evaluate_model()`)

Loads the best model (by joint AUC) and produces:

1. **Scatter plot** — audio_pred vs video_pred, colored by type → should show 4 clusters
2. **Per-type accuracy** — bar chart for audio/video/joint per manipulation type
3. **Prediction distribution** — histogram of joint predictions → should be bimodal (peaks near 0 and 1)
4. **Confusion matrix** — 2×2 for joint fake/real classification
5. **AUC scores** — for audio, video, and joint predictions

> **Why AUC over accuracy?** AUC measures ranking quality across all thresholds. Accuracy depends on a threshold choice and can be misleading with imbalanced classes. AUC tells you: "If I pick a random real and a random fake video, how often does the model rank them correctly?"

---

## Step 11: Standalone Testing (`create_test_data.py` + `evaluate_models.py`)

After training, use these two scripts to test and compare models without the training pipeline.

### Generate a leak-free test dataset
```bash
python create_test_data.py \
    --val_dir /path/to/extracted_val \
    --metadata /path/to/val_metadata.json \
    --output_dir ./test/
```

The script **recreates the exact train/val split** from training (`GroupShuffleSplit(seed=42)`) and only samples from val speakers. This guarantees zero data leakage.

Output:
```
test/
    real/    ← 25 videos (val speakers only)
    fake/    ← 75 videos (val speakers only)
    test_manifest.json
    split_info.json   ← full train/val speaker lists for audit
```

> **Why val-speakers only?** If test videos include training speakers, the model recognises faces/voices it trained on, inflating metrics. By restricting to val speakers, we test only on identities the model has never seen.

> **Why is this reproducible without saving manifests?** The split is deterministic: same `val_metadata.json` + same `seed=42` + same `GroupShuffleSplit` = exact same speaker assignments every time.

### Compare two models
```bash
python evaluate_models.py \
    --model1 run1_best_model.pth \
    --model2 run2_best_model.pth \
    --video_dir ./test/ \
    --output_dir eval_results/
```

Produces:
- `model_comparison.png` — 6-panel figure (metrics bar chart, ROC curves, score distributions, confusion matrices, audio vs video scatter)
- `training_history.png` — AUC and loss curves from checkpoint history
- `model1_predictions.csv` / `model2_predictions.csv` — every prediction with scores
- `metrics_summary.json` — all metrics in JSON

> **Key insight:** `best_model.pth` is for testing. `training_checkpoint.pth` is for resuming training. Always use `best_model.pth` for accuracy evaluation.

---

## Step 12: Web Interface (`web/app.py`)

A full-featured web app ("DeepScan") for deepfake detection with three pages:

```bash
cd web
pip install flask flask-cors reportlab matplotlib
export MODEL1_PATH=/path/to/best_model_1.pth
export MODEL2_PATH=/path/to/best_model_2.pth
python app.py
# Open http://localhost:5000
```

### Features
- **Analyse** — single or batch video upload with adjustable threshold, model selector, and window count
- **Compare** — run one video through both models side-by-side
- **History** — SQLite-backed dashboard of all past analyses with PDF report download

### How it works
1. User uploads video(s) via drag & drop
2. Flask saves to temp, runs multi-window inference on selected model
3. Returns per-window scores, mel spectrograms (base64), and modality-based explanation
4. Frontend renders score bars, window timeline, mel images, and verdict
5. Analysis saved to SQLite — downloadable as PDF report
6. Temp files deleted immediately after processing

### Architecture
```
Browser → POST /api/analyse (multipart, single or batch)
    → Flask saves temp → multi-window inference → per-window scores
    → Mel spectrogram rendering → explanation generation
    → Save to SQLite → JSON response → Frontend renders results

Browser → POST /api/compare (one video, two models)
    → Run both models → side-by-side JSON → Frontend renders comparison

Browser → GET /api/report/<id>
    → Fetch from SQLite → Generate PDF → Download
```

> **Why self-contained?** The backend duplicates model architecture classes instead of importing from `inference.py`, avoiding path issues and making the web app independently deployable.