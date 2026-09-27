"""
Training utilities for AV Deepfake Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, accuracy_score


class FocalLoss(nn.Module):
    """Focal Loss — replaces both LabelSmoothingBCE and hard BCELoss.

    Downweights easy examples via (1-p)^gamma so training focuses on
    hard, ambiguous samples (subtle manipulations). Subsumes label
    smoothing: when gamma=0 and alpha=0.5 it reduces to standard BCE.

    Args:
        gamma: focus parameter. 0 = BCE, 2 = standard Focal Loss.
        alpha: class balance weight for the positive (real) class.
               1-alpha is applied to the negative (fake) class.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        # Clamp to avoid log(0)
        pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)

        # Per-sample BCE
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

        # Focal weight: (1-p)^gamma for positives, p^gamma for negatives
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


def get_loss_functions(focal_gamma=2.0, focal_alpha=0.25):
    """Get loss functions for training and validation.

    Both use FocalLoss — validation uses gamma=0 (standard BCE behaviour)
    so val loss remains comparable across runs and easy to interpret.
    """
    train_criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    val_criterion   = FocalLoss(gamma=0.0, alpha=0.5)   # equivalent to BCE
    return train_criterion, val_criterion


def _unwrap(model):
    """Unwrap DataParallel to access submodules directly.
    
    DataParallel wraps the model so model.video_encoder becomes
    model.module.video_encoder. This helper handles both cases.
    """
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def get_optimizer(model, learning_rate=1e-4, encoder_lr=1e-5,
                  weight_decay=1e-4, freeze_encoders=True):
    """Create optimizer with optional encoder freezing"""
    core = _unwrap(model)  # handles DataParallel transparently

    if freeze_encoders:
        # Freeze encoder parameters
        for param in core.video_encoder.parameters():
            param.requires_grad = False
        for param in core.audio_encoder.parameters():
            param.requires_grad = False

        optimizer = optim.AdamW(
            core.fusion_module.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        trainable = sum(p.numel() for p in core.fusion_module.parameters() if p.requires_grad)
        print(f"✓ Phase 1: Frozen encoders, {trainable:,} params trainable")
    else:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW([
            {'params': core.video_encoder.parameters(), 'lr': encoder_lr},
            {'params': core.audio_encoder.parameters(), 'lr': encoder_lr},
            {'params': core.fusion_module.parameters(), 'lr': learning_rate}
        ], weight_decay=weight_decay)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Phase 2: Fine-tuning all, {trainable:,} params trainable")

    return optimizer


def get_scheduler(optimizer, mode='max', factor=0.5, patience=5):
    """Get learning rate scheduler"""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience
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
                device, grad_clip=1.0, epoch=None, total_epochs=None):
    """Run one training epoch"""
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    desc = f"Epoch {epoch}/{total_epochs} [Train]" if epoch else "Train"
    pbar = tqdm(train_loader, desc=desc, leave=False, dynamic_ncols=True, unit="batch")

    for batch in pbar:
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        labels = batch['labels'].to(device)

        # Skip corrupted samples (sentinel labels = [-1, -1])
        valid_mask = (labels[:, 0] >= 0)
        if valid_mask.sum() == 0:
            continue
        video  = video[valid_mask]
        audio  = audio[valid_mask]
        labels = labels[valid_mask]

        audio_labels = labels[:, 0:1]
        video_labels = labels[:, 1:2]
        joint_labels = ((audio_labels == 1) & (video_labels == 1)).float()

        optimizer.zero_grad()
        outputs = model(video, audio)
        
        loss = (criterion(outputs['audio_pred'], audio_labels) +
                criterion(outputs['video_pred'], video_labels) +
                2.0 * criterion(outputs['joint_pred'], joint_labels)) # Changed from criterion_hard to criterion for label smoothing
        
        loss.backward()

        # Compute grad norm before clipping for logging
        grad_norm = sum(
            p.grad.norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        total_grad_norm += grad_norm

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'avg':  f'{total_loss / num_batches:.4f}',
                          'gnorm': f'{grad_norm:.2f}'})

    pbar.close()
    avg_grad_norm = total_grad_norm / max(num_batches, 1)
    return total_loss / max(num_batches, 1), avg_grad_norm


def validate(model, val_loader, criterion_hard, device, epoch=None, total_epochs=None):
    """Run validation"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds = {'audio': [], 'video': [], 'joint': []}
    all_labels = {'audio': [], 'video': [], 'joint': []}
    all_types = []

    desc = f"Epoch {epoch}/{total_epochs} [Val]  " if epoch else "Val"
    pbar = tqdm(val_loader, desc=desc, leave=False, dynamic_ncols=True, unit="batch")

    with torch.no_grad():
        for batch in pbar:
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['labels'].to(device)

            # Skip corrupted samples
            valid_mask = (labels[:, 0] >= 0)
            if valid_mask.sum() == 0:
                continue
            video  = video[valid_mask]
            audio  = audio[valid_mask]
            labels = labels[valid_mask]

            audio_labels = labels[:, 0:1]
            video_labels = labels[:, 1:2]
            joint_labels = ((audio_labels == 1) & (video_labels == 1)).float()

            outputs = model(video, audio)
            
            loss = (criterion_hard(outputs['audio_pred'], audio_labels) +
                    criterion_hard(outputs['video_pred'], video_labels) +
                    2.0 * criterion_hard(outputs['joint_pred'], joint_labels))
            
            total_loss += loss.item()
            num_batches += 1

            all_preds['audio'].extend(outputs['audio_pred'].cpu().numpy())
            all_preds['video'].extend(outputs['video_pred'].cpu().numpy())
            all_preds['joint'].extend(outputs['joint_pred'].cpu().numpy())
            all_labels['audio'].extend(audio_labels.cpu().numpy())
            all_labels['video'].extend(video_labels.cpu().numpy())
            all_labels['joint'].extend(joint_labels.cpu().numpy())
            all_types.extend(batch['type'])

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    pbar.close()
    avg_loss = total_loss / max(num_batches, 1)

    return avg_loss, all_preds, all_labels, all_types


def train_model(model, train_loader, val_loader, config, device, 
                checkpoint_manager=None, wandb_run=None):
    """Full training loop with checkpointing and logging"""
    criterion, criterion_hard = get_loss_functions(
        focal_gamma=config.get('focal_gamma', 2.0),
        focal_alpha=config.get('focal_alpha', 0.25)
    )
    
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
    # Fix 8: load checkpoint once and reuse the same dict for both model
    #        weights and optimizer/scheduler state — avoids loading the file twice.
    loaded_checkpoint = None
    if checkpoint_manager and config.get('resume', True):
        loaded_checkpoint = checkpoint_manager.load_checkpoint(model)
        if loaded_checkpoint:
            start_epoch = loaded_checkpoint['epoch'] + 1
            history = loaded_checkpoint['history']
            best_val_auc = loaded_checkpoint['best_val_auc']
            patience_counter = loaded_checkpoint['patience_counter']

    # Resolve freeze_epochs — formula: max(1, round(epochs * 0.25))
    # Override by setting freeze_epochs explicitly in config
    freeze_epochs = config.get('freeze_epochs') or max(1, round(config['epochs'] * 0.25))
    config['freeze_epochs'] = freeze_epochs  # store resolved value for phase transition logic

    # Resolve patience — formula: max(5, round(epochs * 0.30))
    # Override by setting patience explicitly in config
    patience = config.get('patience') or max(5, round(config['epochs'] * 0.30))
    config['patience'] = patience
    print(f"  freeze_epochs={freeze_epochs}, patience={patience} "
          f"(auto-computed from epochs={config['epochs']})")

    # Initial optimizer (frozen or unfrozen depending on resume point)
    freeze_encoders = (start_epoch < freeze_epochs)
    optimizer = get_optimizer(
        model,
        learning_rate=config.get('learning_rate', 1e-4),
        encoder_lr=config.get('encoder_lr', 1e-5),
        weight_decay=config.get('weight_decay', 1e-4),
        freeze_encoders=freeze_encoders
    )
    scheduler = get_scheduler(
        optimizer,
        factor=config.get('scheduler_factor', 0.5),
        patience=config.get('scheduler_patience', 5)
    )

    # Restore optimizer/scheduler state from the already-loaded checkpoint dict
    if loaded_checkpoint is not None:
        resumed_in_same_phase = (
            (loaded_checkpoint['epoch'] < config.get('freeze_epochs', 8)) == freeze_encoders
        )
        if resumed_in_same_phase:
            if loaded_checkpoint.get('optimizer_state_dict'):
                optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
                print("  ✓ Restored optimizer state")
            if loaded_checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(loaded_checkpoint['scheduler_state_dict'])
    
    print(f"\nStarting from epoch {start_epoch + 1}/{config['epochs']}")
    print(f"Best AUC so far: {best_val_auc:.3f}")
    
    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.time()
        
        # Phase transition
        if epoch == config.get('freeze_epochs', 8):
            print(f"\n>>> Phase transition: Unfreezing encoders at epoch {epoch + 1}")
            optimizer = get_optimizer(
                model,  # _unwrap() inside get_optimizer handles DataParallel
                learning_rate=config.get('learning_rate', 1e-4),
                encoder_lr=config.get('encoder_lr', 1e-5),
                weight_decay=config.get('weight_decay', 1e-4),
                freeze_encoders=False
            )
            scheduler = get_scheduler(
        optimizer,
        factor=config.get('scheduler_factor', 0.5),
        patience=config.get('scheduler_patience', 5)
    )
        
        # Train
        train_loss, avg_grad_norm = train_epoch(
            model, train_loader, criterion, criterion_hard, optimizer,
            device, config.get('grad_clip', 1.0),
            epoch=epoch + 1, total_epochs=config['epochs']
        )

        # Validate
        val_loss, val_preds, val_labels, val_types = validate(
            model, val_loader, criterion_hard, device,
            epoch=epoch + 1, total_epochs=config['epochs']
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
        
        # -------------------------------------------------------
        # METRICS COMPUTATION
        # -------------------------------------------------------
        def _fl(lst):
            return [float(np.array(x).flat[0]) for x in lst]

        joint_gt_f   = _fl(val_labels['joint'])
        audio_gt_f   = _fl(val_labels['audio'])
        video_gt_f   = _fl(val_labels['video'])
        joint_pred_f = _fl(val_preds['joint'])
        audio_pred_f = _fl(val_preds['audio'])
        video_pred_f = _fl(val_preds['video'])

        joint_gt_bin   = [int(x > 0.5) for x in joint_gt_f]
        joint_pred_bin = [int(x > 0.5) for x in joint_pred_f]
        val_acc_joint  = accuracy_score(joint_gt_bin, joint_pred_bin)

        per_type_auc = {}
        per_type_acc = {}
        for mod_type in set(val_types):
            mask = [i for i, t in enumerate(val_types) if t == mod_type]
            if len(mask) < 2:
                continue
            gt_t   = [joint_gt_f[i]   for i in mask]
            pred_t = [joint_pred_f[i] for i in mask]
            pred_b = [int(x > 0.5) for x in pred_t]
            gt_b   = [int(x > 0.5) for x in gt_t]
            per_type_auc[mod_type] = calculate_auc(gt_t, pred_t)
            per_type_acc[mod_type] = accuracy_score(gt_b, pred_b)

        overfit_gap = train_loss - val_loss
        confidence  = float(np.mean([abs(p - 0.5) * 2 for p in joint_pred_f]))

        # Base log dict
        log_dict = {
            'epoch':               epoch,
            'phase':               'frozen' if epoch < config.get('freeze_epochs', 8) else 'finetune',
            'train/loss':          train_loss,
            'val/loss':            val_loss,
            'train/grad_norm':     avg_grad_norm,
            'learning_rate':       current_lr,
            'epoch_time_s':        epoch_time,
            'val/auc_joint':       val_auc_joint,
            'val/auc_audio':       val_auc_audio,
            'val/auc_video':       val_auc_video,
            'val/accuracy_joint':  val_acc_joint,
            'val/auc_gap':         abs(val_auc_audio - val_auc_video),
            'val/overfit_gap':     overfit_gap,
            'val/confidence':      confidence,
        }

        for mt, auc in per_type_auc.items():
            log_dict[f'val/auc_type/{mt}']      = auc
        for mt, acc in per_type_acc.items():
            log_dict[f'val/accuracy_type/{mt}'] = acc

        if wandb_run:
            import wandb as _wandb

            # 1. Prediction distributions
            log_dict['val/predictions/joint'] = _wandb.Histogram(joint_pred_f)
            log_dict['val/predictions/audio'] = _wandb.Histogram(audio_pred_f)
            log_dict['val/predictions/video'] = _wandb.Histogram(video_pred_f)

            # 2. Confusion matrix
            log_dict['val/confusion_matrix'] = _wandb.plot.confusion_matrix(
                y_true=joint_gt_bin, preds=joint_pred_bin,
                class_names=['Fake', 'Real']
            )

            # 3. ROC curve every 2 epochs
            if (epoch + 1) % 2 == 0:
                log_dict['val/roc_curve'] = _wandb.plot.roc_curve(
                    y_true=joint_gt_f,
                    y_probas=[[1 - p, p] for p in joint_pred_f],
                    labels=['Fake', 'Real']
                )

            # 4. Audio vs video scatter coloured by type
            scatter_data = _wandb.Table(
                columns=['audio_score', 'video_score', 'joint_score', 'type', 'correct'],
                data=[
                    [audio_pred_f[i], video_pred_f[i], joint_pred_f[i],
                     val_types[i],
                     'correct' if joint_pred_bin[i] == joint_gt_bin[i] else 'wrong']
                    for i in range(len(val_types))
                ]
            )
            log_dict['val/prediction_scatter'] = _wandb.plot.scatter(
                scatter_data, 'audio_score', 'video_score',
                title='Audio vs Video Predictions by Manipulation Type'
            )

            # 5. Per-type AUC bar chart
            type_rows = [[mt, per_type_auc.get(mt, 0), per_type_acc.get(mt, 0)]
                         for mt in ['real','audio_modified','visual_modified','both_modified']
                         if mt in per_type_auc]
            type_table = _wandb.Table(columns=['type', 'auc', 'accuracy'], data=type_rows)
            log_dict['val/per_type_auc'] = _wandb.plot.bar(
                type_table, 'type', 'auc', title='AUC by Manipulation Type'
            )

            # 6. Training health summary
            health_rows = [
                ['Joint AUC',    f'{val_auc_joint:.3f}',
                 'Good' if val_auc_joint > 0.7 else 'Low'],
                ['Audio AUC',   f'{val_auc_audio:.3f}',
                 'Good' if val_auc_audio > 0.7 else 'Low'],
                ['Video AUC',   f'{val_auc_video:.3f}',
                 'Good' if val_auc_video > 0.7 else 'Low'],
                ['Accuracy',    f'{val_acc_joint:.3f}',
                 'Good' if val_acc_joint > 0.7 else 'Low'],
                ['AUC Gap',     f'{abs(val_auc_audio-val_auc_video):.3f}',
                 'Balanced' if abs(val_auc_audio-val_auc_video) < 0.1 else 'Imbalanced'],
                ['Overfit Gap', f'{overfit_gap:.3f}',
                 'Overfitting' if overfit_gap > 0.1 else 'OK'],
                ['Confidence',  f'{confidence:.3f}',
                 'Confident' if confidence > 0.4 else 'Uncertain'],
                ['Grad Norm',   f'{avg_grad_norm:.3f}',
                 'Exploding' if avg_grad_norm > 5 else 'OK'],
            ]
            log_dict['val/training_health'] = _wandb.Table(
                columns=['metric', 'value', 'status'], data=health_rows
            )

            # 7. GPU memory
            if torch.cuda.is_available():
                log_dict['gpu/memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
                log_dict['gpu/memory_reserved_gb']  = torch.cuda.memory_reserved()  / 1e9

            wandb_run.log(log_dict)
        
        # Console output
        phase = "[F]" if epoch < config.get('freeze_epochs', 8) else "[U]"
        if (epoch + 1) % 2 == 0 or epoch == start_epoch:
            print(f"Ep {epoch+1:2d}/{config['epochs']} {phase} | "
                  f"Loss: T{train_loss:.3f}/V{val_loss:.3f} | "
                  f"AUC: J{val_auc_joint:.2f} A{val_auc_audio:.2f} V{val_auc_video:.2f} | "
                  f"Time: {epoch_time:.1f}s")
        
        # Scheduler step
        scheduler.step(val_auc_joint)
        
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
    
    # Final save — use last completed epoch or start_epoch-1 if loop never ran
    final_epoch = locals().get('epoch', max(0, start_epoch - 1))
    if checkpoint_manager:
        checkpoint_manager.save_checkpoint(
            epoch=final_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            best_val_auc=best_val_auc,
            patience_counter=patience_counter,
            is_best=False,
            wandb_run_id=wandb_run.id if wandb_run else None
        )
    
    print("\nTraining complete")
    return history