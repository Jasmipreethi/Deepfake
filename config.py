"""
Configuration settings for AV Deepfake Detection Pipeline
"""

import os

# =============================================================================
# PATHS (configurable via .env or environment variables)
# =============================================================================

# Defaults are for Google Colab — override in .env for VPS/local
DATA_DIR = os.environ.get('DATA_DIR', '/content/drive/MyDrive/val')
VAL_DIR = os.path.join(DATA_DIR, 'extracted_val')
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', '/content/drive/MyDrive/checkpoints')

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Checkpoint file paths
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'training_checkpoint.pth')
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
FEATURES_DIR = os.path.join(CHECKPOINT_DIR, 'features')
WANDB_ID_PATH = os.path.join(CHECKPOINT_DIR, 'wandb_run_id.txt')
RESULTS_DIR = os.path.join(CHECKPOINT_DIR, 'results')

# Create directories
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    feature_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.4

@dataclass
class TrainConfig:
    project_name: str = "av-deepfake-detection"
    run_name: str = "full-val-modular"
    architecture: str = "PretrainedResNet3D_ResNet18"
    dataset: str = "AVDeepfake1M++"
    use_all_data: bool = False
    samples_per_type: dict = field(default_factory=lambda: {
        "real": 40,
        "both_modified": 40,
        "audio_modified": 40,
        "visual_modified": 40
    })
    batch_size: int = 16
    epochs: int = 50
    freeze_epochs: int = 8
    patience: int = 15
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    val_split: float = 0.2
    checkpoint_freq: int = 1
    resume: bool = True

@dataclass
class OptimConfig:
    learning_rate: float = 1e-4
    encoder_lr: float = 1e-5
    weight_decay: float = 1e-4

MODEL_CONFIG = ModelConfig()
TRAIN_CONFIG = TrainConfig()
OPTIM_CONFIG = OptimConfig()

# =============================================================================
# FEATURE EXTRACTION CONFIGURATION
# =============================================================================

FEATURE_CONFIG = {
    'sr': 16000,           # Audio sample rate
    'fps': 25,             # Video FPS
    'duration': 2.0,       # Clip duration in seconds
    'num_frames': 50,      # 2s * 25fps
    'img_size': 224,       # Video frame size
    'audio_samples': 32000 # 2s * 16000Hz
}