# This is where the main project and code is stored.

"""
Main execution script for AV Deepfake Detection Pipeline
"""
import os
import shutil
import sys
sys.path.insert(0, '/content/drive/MyDrive/Colab Notebooks/Deepfake')

import argparse
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np

# Import configuration
from config import (
    VAL_DIR, CHECKPOINT_DIR, CHECKPOINT_PATH, BEST_MODEL_PATH,
    FEATURES_DIR, WAND_ID_PATH,
    MODEL_CONFIG, TRAIN_CONFIG, OPTIM_CONFIG
)

# Import modular components
from audio import get_audio_encoder
from video import get_video_encoder
from cross_modal import get_fusion_module
from data_utils import (
    set_seeds, load_metadata, sample_videos, 
    extract_all_features, create_dataloaders
)
from train_utils import train_model, calculate_auc
from checkpoint_utils import CheckpointManager

# =============================================================================
# FULL MODEL (Combines all modules)
# =============================================================================

class AVDeepfakeDetector(nn.Module):
    """
    Complete AV Deepfake Detector
    Combines: Audio Encoder + Video Encoder + Cross-Modal Fusion
    """
    def __init__(self, encoder_type='pretrained', fusion_type='pretrained',
                 feature_dim=256, hidden_dim=512, dropout=0.4):
        super().__init__()
        
        # Modular encoders
        self.video_encoder = get_video_encoder(encoder_type, feature_dim)
        self.audio_encoder = get_audio_encoder(encoder_type, feature_dim)
        
        # Cross-modal fusion
        self.fusion_module = get_fusion_module(
            fusion_type, feature_dim, hidden_dim, dropout
        )
    
    def forward(self, video, audio):
        """
        Forward pass through all modules
        
        Args:
            video: (B, T, C, H, W) - video frames
            audio: (B, 1, 128, 87) - audio spectrogram
        
        Returns:
            dict with predictions and features
        """
        # Encode each modality
        video_feat = self.video_encoder(video)
        audio_feat = self.audio_encoder(audio)
        
        # Cross-modal fusion and prediction
        return self.fusion_module(video_feat, audio_feat)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AV Deepfake Detection')
    
    parser.add_argument(
        '--encoder_type', 
        type=str, 
        default='pretrained',
        choices=['simple', 'improved', 'pretrained'],
        help='Encoder architecture for both audio and video'
    )
    
    parser.add_argument(
        '--fusion_type',
        type=str,
        default='pretrained',
        choices=['simple', 'improved', 'pretrained', 'attention'],
        help='Cross-modal fusion method'
    )
    
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh (ignore existing checkpoints and features)'
    )
    
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    
    parser.add_argument(
        '--feature_dim',
        type=int,
        default=None,
        help='Override feature dimension'
    )
    
    return parser.parse_args()


# =============================================================================
# W&B SETUP
# =============================================================================

def setup_wandb(checkpoint_manager, config, disable=False):
    """Initialize W&B with resume support"""
    if disable:
        return None
    
    run_id = checkpoint_manager.get_wandb_run_id()
    
    if run_id:
        try:
            wandb.init(
                project=config['project_name'],
                id=run_id,
                resume="must",
                config=config
            )
            print(f"  ✓ Resumed W&B run: {run_id}")
            return wandb.run
        except Exception as e:
            print(f"  ⚠ Could not resume W&B: {e}")
    
    # Start new run
    wandb.init(
        project=config['project_name'],
        name=f"{config['run_name']}-{config['encoder_type']}-{config['fusion_type']}",
        config=config,
        tags=[config['encoder_type'], config['fusion_type'], f"{config['samples']}-samples"]
    )
    print(f"  ✓ Started new W&B run: {wandb.run.id}")
    return wandb.run


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, train_loader, val_loader, device):
    """Run final evaluation and create visualization"""
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    model.eval()
    
    # Collect all predictions
    all_results = []
    
    for split_name, loader in [("train", train_loader), ("val", val_loader)]:
        with torch.no_grad():
            for batch in loader:
                outputs = model(batch['video'].to(device), batch['audio'].to(device))
                
                for i in range(len(batch['file'])):
                    all_results.append({
                        'split': split_name,
                        'file': batch['file'][i],
                        'type': batch['type'][i],
                        'audio_gt': batch['labels'][i, 0].item(),
                        'video_gt': batch['labels'][i, 1].item(),
                        'audio_pred': outputs['audio_pred'][i].item(),
                        'video_pred': outputs['video_pred'][i].item(),
                        'joint_pred': outputs['joint_pred'][i].item()
                    })
    
    df = pd.DataFrame(all_results)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Prediction space scatter plot
    ax1 = axes[0, 0]
    colors = {
        'real': '#2ecc71',
        'audio_modified': '#3498db',
        'visual_modified': '#e67e22',
        'both_modified': '#e74c3c'
    }
    
    for split in ['train', 'val']:
        for t in df['type'].unique():
            subset = df[(df['type'] == t) & (df['split'] == split)]
            ax1.scatter(
                subset['audio_pred'], 
                subset['video_pred'],
                c=colors.get(t, 'gray'),
                marker='s' if split == 'val' else 'o',
                s=150 if split == 'val' else 80,
                alpha=0.8 if split == 'val' else 0.5,
                edgecolors='black' if split == 'val' else 'none'
            )
    
    ax1.axhline(0.5, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(0.5, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Audio Prediction (0=Fake, 1=Real)')
    ax1.set_ylabel('Video Prediction (0=Fake, 1=Real)')
    ax1.set_title('Audio-Video Prediction Space')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # 2. Per-type accuracy
    ax2 = axes[0, 1]
    stats = []
    
    for t in df['type'].unique():
        subset = df[df['type'] == t]
        for m in ['audio', 'video', 'joint']:
            if m == 'joint':
                y_true = ((subset['audio_gt'] == 0) | (subset['video_gt'] == 0)).astype(int)
            else:
                y_true = subset[f'{m}_gt']
            y_pred = (subset[f'{m}_pred'] > 0.5).astype(int)
            acc = accuracy_score(y_true, y_pred)
            stats.append({'type': t, 'mod': m, 'acc': acc})
    
    df_stats = pd.DataFrame(stats).pivot(index='type', columns='mod', values='acc')
    df_stats.plot(kind='bar', ax=ax2, color=['#3498db', '#2ecc71', '#e67e22'])
    ax2.set_title('Per-Type Accuracy')
    ax2.set_ylim(0, 1.1)
    ax2.axhline(0.5, color='r', linestyle='--', alpha=0.5)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Prediction distribution
    ax3 = axes[1, 0]
    val_df = df[df['split'] == 'val']
    ax3.hist(val_df['joint_pred'], bins=15, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(0.5, color='r', linestyle='--')
    ax3.set_xlabel('Joint Prediction')
    ax3.set_title('Validation Prediction Distribution')
    
    # 4. Confusion matrix
    ax4 = axes[1, 1]
    y_true = ((val_df['audio_gt'] == 0) | (val_df['video_gt'] == 0)).astype(int)
    y_pred = (val_df['joint_pred'] > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    im = ax4.imshow(cm, cmap='Blues')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Pred Fake', 'Pred Real'])
    ax4.set_yticklabels(['Actual Fake', 'Actual Real'])
    
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=14, fontweight='bold')
    ax4.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'final_results.png'), dpi=150)
    plt.show()
    
    # Print metrics
    print(f"\nValidation Metrics:")
    for mod in ['audio', 'video', 'joint']:
        if mod == 'joint':
            y_true = ((val_df['audio_gt'] == 0) | (val_df['video_gt'] == 0)).astype(int)
        else:
            y_true = val_df[f'{mod}_gt']
        y_pred = val_df[f'{mod}_pred']
        auc = calculate_auc(y_true, y_pred)
        acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
        print(f"  {mod.upper()}: AUC={auc:.3f}, Acc={acc:.3f}")
    
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    args = parse_args()
    
    # Merge configs
    config = {**TRAIN_CONFIG, **OPTIM_CONFIG, **MODEL_CONFIG}
    config['encoder_type'] = args.encoder_type
    config['fusion_type'] = args.fusion_type
    
    if args.epochs:
        config['epochs'] = args.epochs
    if args.feature_dim:
        config['feature_dim'] = args.feature_dim
    
    # Set seeds
    set_seeds(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(CHECKPOINT_PATH, BEST_MODEL_PATH, WAND_ID_PATH)
    
    # Handle --fresh flag
    if args.fresh:
        print("Starting fresh - removing existing checkpoints and features...")
        checkpoint_manager.clean_checkpoints()
        if os.path.exists(FEATURES_DIR):
            shutil.rmtree(FEATURES_DIR)
            os.makedirs(FEATURES_DIR, exist_ok=True)
            print(f"Cleared {FEATURES_DIR}")
    
    # Load data
    print("\n" + "=" * 60)
    print("DATA LOADING")
    print("=" * 60)
    
    df = load_metadata(VAL_DIR)
    train_df, val_df = sample_videos(
        df, 
        config['samples_per_type'],
        config['val_split'],
        use_all=config.get('use_all_data', False)
    )
    
    # Check existing features and ask user whether to extract more
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION STATUS")
    print("=" * 60)
    
    train_manifest_path = os.path.join(FEATURES_DIR, 'train_manifest.json')
    val_manifest_path = os.path.join(FEATURES_DIR, 'val_manifest.json')
    
    existing_train = 0
    existing_val = 0
    if os.path.exists(train_manifest_path):
        with open(train_manifest_path, 'r') as f:
            import json
            existing_train = len(json.load(f))
    if os.path.exists(val_manifest_path):
        with open(val_manifest_path, 'r') as f:
            import json
            existing_val = len(json.load(f))
    
    print(f"  Train: {existing_train:,} / {len(train_df):,} extracted")
    print(f"  Val:   {existing_val:,} / {len(val_df):,} extracted")
    
    if existing_train < len(train_df) or existing_val < len(val_df):
        user_input = input("\nContinue extracting features? (y = extract more, n = skip to training with existing): ").strip().lower()
        
        if user_input == 'y':
            train_dir, train_manifest, val_dir_feat, val_manifest = extract_all_features(
                train_df, val_df, VAL_DIR, FEATURES_DIR,
                use_cache=not args.fresh
            )
        else:
            print("Skipping extraction — using existing features for training.")
            train_dir = os.path.join(FEATURES_DIR, 'train')
            val_dir_feat = os.path.join(FEATURES_DIR, 'val')
            train_manifest = train_manifest_path
            val_manifest = val_manifest_path
    else:
        print("All features already extracted!")
        train_dir = os.path.join(FEATURES_DIR, 'train')
        val_dir_feat = os.path.join(FEATURES_DIR, 'val')
        train_manifest = train_manifest_path
        val_manifest = val_manifest_path
    
    # Create dataloaders (lazy-loading from disk)
    train_loader, val_loader = create_dataloaders(
        train_dir, train_manifest,
        val_dir_feat, val_manifest,
        config['batch_size']
    )
    
    # Initialize model (combining all modules)
    print("\n" + "=" * 60)
    print(f"MODEL INITIALIZATION")
    print(f"  Encoder: {args.encoder_type}")
    print(f"  Fusion:  {args.fusion_type}")
    print("=" * 60)
    
    model = AVDeepfakeDetector(
        encoder_type=args.encoder_type,
        fusion_type=args.fusion_type,
        feature_dim=config['feature_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters ({total_params * 4 / 1024 / 1024:.2f} MB)")
    
    # Test forward pass
    with torch.no_grad():
        test_v = torch.randn(2, 50, 3, 224, 224).to(device)
        test_a = torch.randn(2, 1, 128, 87).to(device)
        test_out = model(test_v, test_a)
        print(f"✓ Test forward pass: A={test_out['audio_pred'].shape}, "
              f"V={test_out['video_pred'].shape}, J={test_out['joint_pred'].shape}")
    
    # Setup W&B
    print("\n" + "=" * 60)
    print("W&B SETUP")
    print("=" * 60)
    wandb_run = setup_wandb(checkpoint_manager, config, disable=args.no_wandb)
    
    if wandb_run:
        wandb.watch(model, log="parameters", log_freq=100)
    
    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_manager=checkpoint_manager,
        wandb_run=wandb_run
    )
    
    # Load best model for evaluation
    print("\nLoading best model for final evaluation...")
    best_checkpoint = checkpoint_manager.load_best_model(model, device)
    
    # Evaluate
    results_df = evaluate_model(model, train_loader, val_loader, device)
    
    # Log final results
    if wandb_run:
        wandb.summary.update({
            'best_epoch': best_checkpoint['epoch'],
            'best_auc': best_checkpoint['best_val_auc'],
            'total_epochs': len(history['train_loss'])
        })
        wandb.log({"final_analysis": wandb.Image(os.path.join(CHECKPOINT_DIR, 'final_results.png'))})
        print(f"\nView run at: {wandb.run.url}")
        wandb.finish()
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
