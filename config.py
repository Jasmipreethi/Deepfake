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
WAND_ID_PATH = os.path.join(CHECKPOINT_DIR, 'wandb_run_id.txt')

# Create directories
os.makedirs(FEATURES_DIR, exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    "feature_dim": 256,
    "hidden_dim": 512,
    "dropout": 0.4,
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAIN_CONFIG = {
    "project_name": "av-deepfake-detection",
    "run_name": "full-val-modular",
    "architecture": "PretrainedResNet3D_ResNet18",
    "dataset": "AVDeepfake1M++",
    "use_all_data": True,  # True = use all videos; False = use samples_per_type subset
    "samples_per_type": {   # Only used when use_all_data is False
        "real": 5,
        "both_modified": 5,
        "audio_modified": 5,
        "visual_modified": 5
    },
    "batch_size": 16,
    "epochs": 50,
    "freeze_epochs": 8,
    "patience": 15,
    "grad_clip": 1.0,
    "label_smoothing": 0.05,
    "val_split": 0.2,
    "checkpoint_freq": 1,
    "resume": True
}

# =============================================================================
# OPTIMIZER CONFIGURATION
# =============================================================================

OPTIM_CONFIG = {
    "learning_rate": 1e-4,
    "encoder_lr": 1e-5,
    "weight_decay": 1e-4,
}

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