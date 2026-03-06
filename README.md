# Deepfake Detection (AV-Deepfake1M++)

Deepfake detection using audio-video cross-modal fusion with deep learning.

## Pipeline Flow

```mermaid
flowchart TD
    A["1. Load Metadata\nval_metadata.json"] --> B["2. Speaker-Based Split\n80/20 train/val\nzero speaker overlap"]
    B --> C["3. Extract Features\nVideo: 50 frames → (50,3,224,224)\nAudio: mel-spectrogram → (1,128,T)"]
    C --> D["4. Save to Disk\nIndividual .pt files\nresumable extraction"]
    D --> E["5. Train Model\nPhase 1: Frozen encoders\nPhase 2: Fine-tune all"]
    E --> F["6. Evaluate\nAUC, Accuracy, Confusion Matrix"]
```

## Model Architecture

```mermaid
flowchart LR
    V["Video\n(B,50,3,224,224)"] --> VE["ResNet3D-18\n(pretrained)"]
    A["Audio mel-spec\n(B,1,128,T)"] --> AE["ResNet18\n(pretrained)"]
    VE --> |"256-d"| CAT["Concat → Fusion MLP\n512-d"]
    AE --> |"256-d"| CAT
    CAT --> AH["Audio Head → σ"]
    CAT --> VH["Video Head → σ"]
    CAT --> JH["Joint Head → σ"]
```

## File Structure

```
├── config.py           # Configuration settings
├── audio.py            # Audio encoder (ResNet18)
├── video.py            # Video encoder (ResNet3D-18)
├── cross_modal.py      # Cross-modal fusion models
├── data_utils.py       # Data loading, speaker split, feature extraction
├── train_utils.py      # Training loop, loss, optimizer
├── checkpoint_utils.py # Checkpoint management
└── main.py             # Main execution script
```

## Usage

```bash
# Full dataset run
python main.py

# Start fresh (clear old checkpoints/features)
python main.py --fresh

# Without W&B logging
python main.py --no_wandb

# Custom encoder/fusion
python main.py --encoder_type pretrained --fusion_type attention --epochs 30
```
