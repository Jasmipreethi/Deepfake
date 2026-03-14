# AV-Deepfake Detection — Detailed Code Walkthrough

A step-by-step explanation of how the entire pipeline works and why each design choice was made.

---

## Step 1: Configuration (`config.py`)

All paths and hyperparameters are centralized here.

**Paths read from environment variables:**
```python
DATA_DIR = os.environ.get('DATA_DIR', '/content/drive/MyDrive/val')
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', '/content/drive/MyDrive/checkpoints')
```

> **Why env vars?** Makes the pipeline portable — same code works on Colab, VPS, or local by just changing `.env`.

**Key hyperparameters:**
| Setting | Value | Why |
|---|---|---|
| `feature_dim` | 256 | Encoder output size — balanced between expressiveness and overfitting |
| `hidden_dim` | 512 | Fusion layer width — 2× feature_dim gives enough capacity |
| `dropout` | 0.4 | Higher than typical (0.2-0.3) because the dataset is large and we want generalization |
| `freeze_epochs` | 8 | Enough time for fusion to stabilize before unfreezing encoders |
| `patience` | 15 | Generous, since AUC can plateau then improve when LR drops |

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
Train speakers: {A, B, C, D, E}    ← 80% of speakers
Val speakers:   {X, Y, Z}          ← 20% of speakers
                Zero overlap ✓
```

> **Why speaker-based, not random?** Random splits cause **identity leakage** — the model learns faces/voices instead of manipulation artifacts. With speaker-based splitting, the model has never seen the val speakers during training, so it must detect actual deepfake artifacts.

---

## Step 4: Feature Extraction (`data_utils.py → extract_av_features()`)

For each video, a **2-second window** is extracted:

### Window Selection
```python
if fake_segments:           # manipulated video
    start_sec = fake_segments[0][0]    # start of first fake segment
else:                        # real video
    start_sec = total_duration / 2 - 1 # middle 2 seconds
```

> **Why 2 seconds?** Most deepfake artifacts are subtle and localized. 2 seconds captures enough temporal context (50 frames) without requiring massive tensors.

> **Why start at fake segments?** Ensures the model sees the actual manipulated region, not a potentially unmodified part of the video.

### Video Features
```python
frames = []
for i in range(num_frames):              # 50 frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start + i)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (224, 224)) # resize for ResNet
    frame = frame[:, :, ::-1] / 255.0     # BGR→RGB, normalize to [0,1]
    frames.append(frame)
# Output: (50, 3, 224, 224)
```

> **Why 224×224?** Standard ResNet input size. The pretrained weights expect this resolution.

> **Why 50 frames?** 2 seconds × 25 fps = 50 frames. Captures motion patterns that deepfakes often get wrong (temporal inconsistencies).

### Audio Features
```python
waveform, sr = torchaudio.load(video_path)    # load from the video file
waveform = resample(waveform, sr, 16000)       # resample to 16kHz
waveform = waveform[:, start:end]              # slice 2 seconds
mel_spec = MelSpectrogram(n_mels=128, n_fft=1024, hop_length=512)(waveform)
audio = AmplitudeToDB()(mel_spec)
# Output: (1, 128, T)
```

> **Why torchaudio over librosa?** Librosa uses PySoundFile/audioread which caused crashes on corrupted MP4s. Torchaudio uses FFmpeg directly — more robust, no Python-level crashes.

> **Why mel-spectrogram?** Converts raw audio waveform into a 2D time-frequency image that a CNN can process. The mel scale emphasizes frequencies humans are sensitive to, which is where deepfake artifacts tend to appear (unnatural voice harmonics).

> **Why 128 mel bins?** Standard for speech tasks — captures enough frequency detail without being excessive.

---

## Step 5: Disk Storage (`data_utils.py → process_split_to_disk()`)

Each video's features are saved as individual `.pt` files:
```python
torch.save({
    'video': video_tensor,    # (50, 3, 224, 224)
    'audio': audio_tensor,    # (1, 128, T)
    'labels': [audio_label, video_label]
}, f'{idx}.pt')
```

> **Why individual files, not one big pickle?** Loading 77K videos into RAM would need ~2.3TB. Individual files allow **lazy-loading** — the DataLoader reads one file per sample, keeping RAM at ~2-3GB regardless of dataset size.

> **Why .pt format?** PyTorch native — fastest serialization for tensors.

### Resume & Failed Tracking
```python
# Skip if already extracted
if idx in failed_indices:  skip   # corrupted files
if os.path.exists(pt_path):  skip  # already done
```

> **Why track failures?** Corrupted MP4s (e.g. `moov atom not found`) would be retried on every restart. The `failed.json` file remembers them so they're skipped instantly.

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

> **Why label smoothing (5%)?** Instead of hard labels (0 or 1), uses soft labels (0.025 or 0.975). Prevents the model from being overconfident, which improves generalization — especially important with subtle deepfakes.

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
