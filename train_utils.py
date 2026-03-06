"""
Training utilities for AV Deepfake Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

from sklearn.metrics import roc_auc_score


class LabelSmoothingBCE(nn.Module):
    """Binary Cross Entropy with label smoothing"""
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy(pred, target)


def get_loss_functions(label_smoothing=0.05):
    """Get training and validation loss functions"""
    return LabelSmoothingBCE(smoothing=label_smoothing), nn.BCELoss()


def get_optimizer(model, learning_rate=1e-4, encoder_lr=1e-5, 
                  weight_decay=1e-4, freeze_encoders=True):
    """Create optimizer with optional encoder freezing"""
    if freeze_encoders:
        # Freeze encoder parameters
        for param in model.video_encoder.parameters():
            param.requires_grad = False
        for param in model.audio_encoder.parameters():
            param.requires_grad = False
        
        optimizer = optim.AdamW(
            model.fusion_module.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        trainable = sum(p.numel() for p in model.fusion_module.parameters() if p.requires_grad)
        print(f"✓ Phase 1: Frozen encoders, {trainable:,} params trainable")
    else:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        
        optimizer = optim.AdamW([
            {'params': model.video_encoder.parameters(), 'lr': encoder_lr},
            {'params': model.audio_encoder.parameters(), 'lr': encoder_lr},
            {'params': model.fusion_module.parameters(), 'lr': learning_rate}
        ], weight_decay=weight_decay)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Phase 2: Fine-tuning all, {trainable:,} params trainable")
    
    return optimizer


def get_scheduler(optimizer, mode='min', factor=0.5, patience=5):
    """Get learning rate scheduler"""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience
    )


def calculate_auc(y_true, y_pred):
    """Calculate AUC score safely"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(np.unique(y_true)) < 2:
        return 0.5
    
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return 0.5


def train_epoch(model, train_loader, criterion, criterion_hard, optimizer, 
                device, grad_clip=1.0):
    """Run one training epoch"""
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        labels = batch['labels'].to(device)
        
        audio_labels = labels[:, 0:1]
        video_labels = labels[:, 1:2]
        joint_labels = ((audio_labels == 1) & (video_labels == 1)).float()
        
        optimizer.zero_grad()
        outputs = model(video, audio)
        
        loss = (criterion(outputs['audio_pred'], audio_labels) +
                criterion(outputs['video_pred'], video_labels) +
                2.0 * criterion_hard(outputs['joint_pred'], joint_labels))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion_hard, device):
    """Run validation"""
    model.eval()
    total_loss = 0.0
    
    all_preds = {'audio': [], 'video': [], 'joint': []}
    all_labels = {'audio': [], 'video': [], 'joint': []}
    all_types = []
    
    with torch.no_grad():
        for batch in val_loader:
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['labels'].to(device)
            
            audio_labels = labels[:, 0:1]
            video_labels = labels[:, 1:2]
            joint_labels = ((audio_labels == 1) & (video_labels == 1)).float()
            
            outputs = model(video, audio)
            
            loss = (criterion_hard(outputs['audio_pred'], audio_labels) +
                    criterion_hard(outputs['video_pred'], video_labels) +
                    2.0 * criterion_hard(outputs['joint_pred'], joint_labels))
            
            total_loss += loss.item()
            
            all_preds['audio'].extend(outputs['audio_pred'].cpu().numpy())
            all_preds['video'].extend(outputs['video_pred'].cpu().numpy())
            all_preds['joint'].extend(outputs['joint_pred'].cpu().numpy())
            all_labels['audio'].extend(audio_labels.cpu().numpy())
            all_labels['video'].extend(video_labels.cpu().numpy())
            all_labels['joint'].extend(joint_labels.cpu().numpy())
            all_types.extend(batch['type'])
    
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, all_preds, all_labels, all_types


def train_model(model, train_loader, val_loader, config, device, 
                checkpoint_manager=None, wandb_run=None):
    """Full training loop with checkpointing and logging"""
    criterion, criterion_hard = get_loss_functions(config.get('label_smoothing', 0.05))
    
    # Determine starting point
    start_epoch = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_auc_audio': [], 'val_auc_video': [], 'val_auc_joint': [],
        'learning_rate': [], 'epoch_time': []
    }
    best_val_auc = 0.0
    patience_counter = 0
    
    # Resume from checkpoint if available
    if checkpoint_manager and config.get('resume', True):
        checkpoint = checkpoint_manager.load_checkpoint(model)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint['history']
            best_val_auc = checkpoint['best_val_auc']
            patience_counter = checkpoint['patience_counter']
    
    # Initial optimizer (frozen)
    freeze_encoders = (start_epoch < config.get('freeze_epochs', 8))
    optimizer = get_optimizer(
        model,
        learning_rate=config.get('learning_rate', 1e-4),
        encoder_lr=config.get('encoder_lr', 1e-5),
        weight_decay=config.get('weight_decay', 1e-4),
        freeze_encoders=freeze_encoders
    )
    scheduler = get_scheduler(optimizer)
    
    # Load optimizer state if resuming in same phase
    if checkpoint_manager and checkpoint_manager.checkpoint_exists() and config.get('resume', True):
        checkpoint = torch.load(checkpoint_manager.checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get('optimizer_state_dict') and \
           (checkpoint['epoch'] < config.get('freeze_epochs', 8)) == freeze_encoders:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  ✓ Restored optimizer state")
        if checkpoint.get('scheduler_state_dict') and \
           (checkpoint['epoch'] < config.get('freeze_epochs', 8)) == freeze_encoders:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"\nStarting from epoch {start_epoch + 1}/{config['epochs']}")
    print(f"Best AUC so far: {best_val_auc:.3f}")
    
    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.time()
        
        # Phase transition
        if epoch == config.get('freeze_epochs', 8):
            print(f"\n>>> Phase transition: Unfreezing encoders at epoch {epoch + 1}")
            optimizer = get_optimizer(
                model,
                learning_rate=config.get('learning_rate', 1e-4),
                encoder_lr=config.get('encoder_lr', 1e-5),
                weight_decay=config.get('weight_decay', 1e-4),
                freeze_encoders=False
            )
            scheduler = get_scheduler(optimizer)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, criterion_hard, optimizer,
            device, config.get('grad_clip', 1.0)
        )
        
        # Validate
        val_loss, val_preds, val_labels, val_types = validate(
            model, val_loader, criterion_hard, device
        )
        
        # Calculate metrics
        val_auc_audio = calculate_auc(val_labels['audio'], val_preds['audio'])
        val_auc_video = calculate_auc(val_labels['video'], val_preds['video'])
        val_auc_joint = calculate_auc(val_labels['joint'], val_preds['joint'])
        
        # Update history
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc_audio'].append(val_auc_audio)
        history['val_auc_video'].append(val_auc_video)
        history['val_auc_joint'].append(val_auc_joint)
        history['learning_rate'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        
        # Logging
        log_dict = {
            'epoch': epoch,
            'phase': 'frozen' if epoch < config.get('freeze_epochs', 8) else 'finetune',
            'train/loss': train_loss,
            'val/loss': val_loss,
            'val/auc_joint': val_auc_joint,
            'val/auc_audio': val_auc_audio,
            'val/auc_video': val_auc_video,
            'val/auc_gap': abs(val_auc_audio - val_auc_video),
            'learning_rate': current_lr,
            'epoch_time': epoch_time
        }
        
        if wandb_run:
            wandb_run.log(log_dict)
        
        # Console output
        phase = "[F]" if epoch < config.get('freeze_epochs', 8) else "[U]"
        if (epoch + 1) % 2 == 0 or epoch == start_epoch:
            print(f"Ep {epoch+1:2d}/{config['epochs']} {phase} | "
                  f"Loss: T{train_loss:.3f}/V{val_loss:.3f} | "
                  f"AUC: J{val_auc_joint:.2f} A{val_auc_audio:.2f} V{val_auc_video:.2f} | "
                  f"Time: {epoch_time:.1f}s")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Checkpointing
        is_best = val_auc_joint > best_val_auc
        if is_best:
            best_val_auc = val_auc_joint
            patience_counter = 0
        else:
            patience_counter += 1
        
        if checkpoint_manager and (epoch + 1) % config.get('checkpoint_freq', 1) == 0:
            checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                best_val_auc=best_val_auc,
                patience_counter=patience_counter,
                is_best=is_best,
                wandb_run_id=wandb_run.id if wandb_run else None
            )
        
        # Early stopping
        if patience_counter >= config.get('patience', 12):
            print(f"\n⏹️ Early stopping at epoch {epoch+1} (best AUC: {best_val_auc:.3f})")
            break
    
    # Final save
    if checkpoint_manager:
        checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            best_val_auc=best_val_auc,
            patience_counter=patience_counter,
            is_best=False,
            wandb_run_id=wandb_run.id if wandb_run else None
        )
    
    print("\n✅ Training complete")
    return history