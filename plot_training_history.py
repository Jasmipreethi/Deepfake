"""
Plot training history from per-epoch metrics JSON (output.txt).

Reads Model 4/5 training metrics from logs/logs_5/output.txt and generates
a three-panel figure: training/validation loss, validation AUC per head,
and learning rate schedule.

Usage:
    python plot_training_history.py                          # uses logs/logs_5/output.txt
    python plot_training_history.py <json_path>              # custom path
    python plot_training_history.py <json_path> <output_dir>  # custom output

Output:
    figures/training_history_model4.png
"""

import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#1e293b',
    'axes.labelcolor': '#1e293b',
    'text.color': '#1e293b',
    'xtick.color': '#334155',
    'ytick.color': '#334155',
    'grid.color': '#e2e8f0',
    'grid.alpha': 0.6,
    'axes.grid': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 100,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = {
    'train_loss': '#e74c3c',
    'val_loss':   '#3498db',
    'joint_auc':  '#e74c3c',
    'audio_auc':  '#3498db',
    'video_auc':  '#2ecc71',
    'lr':         '#9b59b6',
    'grid':       '#e2e8f0',
    'phase_line': '#f39c12',
}


def load_history(json_path):
    with open(json_path) as f:
        return json.load(f)


def plot_training_history(history, output_dir, run_label="Model 4/5"):
    epochs = list(range(1, len(history['train_loss']) + 1))
    freeze_epochs = 2  # first 2 epochs were frozen; epoch 3+ fine-tuned

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Training History — {run_label}', fontsize=14, fontweight='bold')

    # ── Panel 1: Loss ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], color=COLORS['train_loss'],
            marker='o', linewidth=2, markersize=6, label='Train Loss (Focal, γ=2.0)')
    ax.plot(epochs, history['val_loss'], color=COLORS['val_loss'],
            marker='s', linewidth=2, markersize=6, label='Val Loss (BCE)')
    ax.axvline(x=freeze_epochs + 0.5, color=COLORS['phase_line'],
               linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend(fontsize=9)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])

    # label "Frozen" and "Fine-tune" at top
    ymax = ax.get_ylim()[1]
    ax.text(1.5, ymax * 0.95, 'Frozen', ha='center', fontsize=8,
            color=COLORS['phase_line'], style='italic')
    ax.text(4, ymax * 0.95, 'Fine-tune', ha='center', fontsize=8,
            color=COLORS['phase_line'], style='italic')

    # ── Panel 2: Validation AUC ────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history['val_auc_joint'], color=COLORS['joint_auc'],
            marker='o', linewidth=2.5, markersize=7, label='Joint AUC')
    ax.plot(epochs, history['val_auc_audio'], color=COLORS['audio_auc'],
            marker='s', linewidth=2, markersize=6, label='Audio AUC')
    ax.plot(epochs, history['val_auc_video'], color=COLORS['video_auc'],
            marker='^', linewidth=2, markersize=6, label='Video AUC')
    ax.axvline(x=freeze_epochs + 0.5, color=COLORS['phase_line'],
               linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('Validation AUC')
    ax.legend(fontsize=9)
    ax.set_xticks(epochs)
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])

    # ── Panel 3: Learning Rate ─────────────────────────────────
    ax = axes[2]
    ax.plot(epochs, history['learning_rate'], color=COLORS['lr'],
            marker='D', linewidth=2, markersize=6)
    ax.axvline(x=freeze_epochs + 0.5, color=COLORS['phase_line'],
               linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0e'))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'training_history_model4.png')
    plt.savefig(out_path)
    plt.close()
    print(f'Saved: {out_path}')

    # ── Summary stats ──────────────────────────────────────────
    print(f'\n  Epochs completed: {len(epochs)}')
    print(f'  Final train loss: {history["train_loss"][-1]:.4f}')
    print(f'  Final val loss:   {history["val_loss"][-1]:.4f}')
    print(f'  Final joint AUC:  {history["val_auc_joint"][-1]:.4f}')
    print(f'  Final audio AUC:  {history["val_auc_audio"][-1]:.4f}')
    print(f'  Final video AUC:  {history["val_auc_video"][-1]:.4f}')
    print(f'  Total epoch time: {sum(history["epoch_time"]):.0f}s '
          f'({sum(history["epoch_time"]) / 3600:.1f}h)')


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'logs/logs_5/output.txt'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'figures'

    if not os.path.exists(json_path):
        print(f'ERROR: {json_path} not found')
        sys.exit(1)

    run_label = 'Model 4/5'
    if 'logs_2' in json_path:
        run_label = 'Model 2'
    elif 'logs_3' in json_path:
        run_label = 'Model 3'

    history = load_history(json_path)
    plot_training_history(history, output_dir, run_label)


if __name__ == '__main__':
    main()
