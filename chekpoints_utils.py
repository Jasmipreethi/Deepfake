"""
Checkpoint management utilities for resumable training
"""

import os
import torch
import random
import numpy as np


class CheckpointManager:
    """Manages saving and loading of training checkpoints"""
    
    def __init__(self, checkpoint_path, best_model_path, wandb_id_path=None):
        self.checkpoint_path = checkpoint_path
        self.best_model_path = best_model_path
        self.wandb_id_path = wandb_id_path
    
    def checkpoint_exists(self):
        """Check if a checkpoint exists"""
        return os.path.exists(self.checkpoint_path)
    
    def save_checkpoint(self, epoch, model, optimizer, scheduler, history,
                       best_val_auc, patience_counter, is_best=False,
                       wandb_run_id=None):
        """Save complete training state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': history,
            'best_val_auc': best_val_auc,
            'patience_counter': patience_counter,
            'random_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            },
            'wandb_run_id': wandb_run_id
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_path)
        
        # Save best model separately
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"  ✓ New best model saved (AUC: {best_val_auc:.3f})")
        
        # Save W&B run ID
        if wandb_run_id and self.wandb_id_path:
            with open(self.wandb_id_path, 'w') as f:
                f.write(wandb_run_id)
        
        return checkpoint
    
    def load_checkpoint(self, model, device='cpu'):
        """Load checkpoint if it exists"""
        if not self.checkpoint_exists():
            print("No checkpoint found. Starting from scratch.")
            return None
        
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore random states
        random.setstate(checkpoint['random_state']['python'])
        np.random.set_state(checkpoint['random_state']['numpy'])
        torch.set_rng_state(checkpoint['random_state']['torch'])
        if checkpoint['random_state']['torch_cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['random_state']['torch_cuda'])
        
        print(f"  ✓ Resumed from epoch {checkpoint['epoch']+1}")
        print(f"  ✓ Best AUC so far: {checkpoint['best_val_auc']:.3f}")
        
        return checkpoint
    
    def load_best_model(self, model, device='cpu'):
        """Load the best saved model"""
        if not os.path.exists(self.best_model_path):
            raise FileNotFoundError(f"No best model found at {self.best_model_path}")
        
        checkpoint = torch.load(self.best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    
    def get_wandb_run_id(self):
        """Get W&B run ID from file if it exists"""
        if self.wandb_id_path and os.path.exists(self.wandb_id_path):
            with open(self.wandb_id_path, 'r') as f:
                return f.read().strip()
        return None
    
    def clean_checkpoints(self):
        """Remove all checkpoint files"""
        for path in [self.checkpoint_path, self.best_model_path, self.wandb_id_path]:
            if path and os.path.exists(path):
                os.remove(path)
                print(f"Removed {path}")