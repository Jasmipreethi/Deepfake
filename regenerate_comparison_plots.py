"""Regenerate model_comparison.png and training_history.png from existing CSVs with white theme."""
import os, sys, json, csv, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch

warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#1e293b', 'axes.labelcolor': '#1e293b',
    'text.color': '#1e293b', 'xtick.color': '#334155', 'ytick.color': '#334155',
    'grid.color': '#e2e8f0', 'grid.alpha': 0.6, 'axes.grid': True,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 100, 'savefig.dpi': 150, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

PALETTE = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']

CSV_FILES = {
    'Model 2 (5ep)': 'comparison_results/Model_2_(5ep,_best)_predictions.csv',
    'Model 3 (3ep)': 'comparison_results/Model_3_(3ep)_predictions.csv',
    'Model 4 (5ep)': 'comparison_results/Model_4_(5ep)_predictions.csv',
}


def load_history(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return ckpt.get('history', {})


def build_results():
    results = []
    cp_map = {
        'Model 2 (5ep)': 'logs/logs_2/best_model.pth',
        'Model 3 (3ep)': 'logs/logs_3/best_model.pth',
        'Model 4 (5ep)': 'logs/logs_4/best_model.pth',
    }

    for name, csv_path in CSV_FILES.items():
        df = pd.read_csv(csv_path)
        # Compute metrics
        y_true = df['true_label'].values
        y_pred = df['joint_score'].values
        y_bin  = (y_pred >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

        tp = int(((y_bin == 1) & (y_true == 1)).sum())
        tn = int(((y_bin == 0) & (y_true == 0)).sum())
        fp = int(((y_bin == 1) & (y_true == 0)).sum())
        fn = int(((y_bin == 0) & (y_true == 1)).sum())

        per_type = {}
        for mt in df['modify_type'].unique():
            sub = df[df['modify_type'] == mt]
            if len(sub) < 2:
                continue
            try:
                per_type[mt] = {
                    'auc': roc_auc_score(sub['true_label'].values, sub['joint_score'].values),
                    'acc': (sub['correct'] == True).mean(),
                }
            except ValueError:
                per_type[mt] = {'auc': 1.0 if (sub['correct'] == True).all() else 0.0,
                                'acc': (sub['correct'] == True).mean()}

        metrics = {
            'auc': auc, 'accuracy': accuracy_score(y_true, y_bin),
            'precision': precision_score(y_true, y_bin, zero_division=0),
            'recall': recall_score(y_true, y_bin, zero_division=0),
            'f1': f1_score(y_true, y_bin, zero_division=0),
            'fpr': fpr, 'tpr': tpr, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'per_type': per_type,
        }

        info = {}
        if name in cp_map and os.path.exists(cp_map[name]):
            info['history'] = load_history(cp_map[name])

        results.append({'name': name, 'df': df, 'metrics': metrics, 'info': info})
        print(f'  {name}: AUC={auc:.4f} Acc={metrics["accuracy"]:.3f}')
    return results


def plot_all(results, output_dir, threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    n = len(results)
    names   = [r['name'] for r in results]
    metrics = [r['metrics'] for r in results]
    colors  = PALETTE[:n]

    fig = plt.figure(figsize=(22, 24))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35,
                            top=0.93, bottom=0.04, left=0.06, right=0.97)

    # Summary bar charts
    ax0 = fig.add_subplot(gs[0, :])
    mk = ['auc', 'accuracy', 'precision', 'recall', 'f1']
    ml = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(mk)); w = 0.8 / n
    for i, (m, name, color) in enumerate(zip(metrics, names, colors)):
        vals = [m[k] for k in mk]
        bars = ax0.bar(x + i*w - (n-1)*w/2, vals, w, label=name, color=color, alpha=0.85)
        for b, v in zip(bars, vals):
            ax0.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=7)
    ax0.set_xticks(x); ax0.set_xticklabels(ml, fontsize=10)
    ax0.set_ylim(0, 1.15)
    ax0.set_title('Overall Metrics Comparison', fontsize=11, fontweight='bold')
    ax0.legend(fontsize=9)
    ax0.axhline(0.5, color='#95a5a6', linestyle='--', linewidth=0.8)
    ax0.grid(axis='y', alpha=0.4, color='#e2e8f0')

    # ROC curves
    ax_roc = fig.add_subplot(gs[1, 0])
    ax_roc.plot([0,1],[0,1], '--', color='#bdc3c7', linewidth=1)
    for m, name, color in zip(metrics, names, colors):
        ax_roc.plot(m['fpr'], m['tpr'], color=color, linewidth=2,
                    label=f'{name}  AUC={m["auc"]:.3f}')
        ax_roc.fill_between(m['fpr'], m['tpr'], alpha=0.05, color=color)
    ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves', fontsize=11, fontweight='bold')
    ax_roc.legend(fontsize=7); ax_roc.grid(alpha=0.4, color='#e2e8f0')

    # Score distributions
    ax_dist = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 1, 30)
    for r, color in zip(results, colors):
        df = r['df']
        ax_dist.hist(df[df['true_label']==1]['joint_score'], bins=bins, alpha=0.4, color='#2ecc71')
        ax_dist.hist(df[df['true_label']==0]['joint_score'], bins=bins, alpha=0.4, color=color, label=r['name'])
    ax_dist.axvline(threshold, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.5)
    ax_dist.set_xlabel('Joint Score (0=fake, 1=real)'); ax_dist.set_ylabel('Count')
    ax_dist.set_title('Score Distributions', fontsize=11, fontweight='bold')
    ax_dist.legend(fontsize=7); ax_dist.grid(alpha=0.4, color='#e2e8f0')

    # Confusion matrix table
    ax_cm = fig.add_subplot(gs[1, 2]); ax_cm.axis('off')
    ax_cm.set_title('Confusion Matrix Summary', fontsize=11, fontweight='bold')
    tbl = ax_cm.table(cellText=[
        [r['name'], str(r['metrics']['tp']), str(r['metrics']['tn']),
         str(r['metrics']['fp']), str(r['metrics']['fn']), f'{r["metrics"]["accuracy"]:.1%}']
        for r in results
    ], colLabels=['Model', 'TP', 'TN', 'FP', 'FN', 'Acc'], cellLoc='center', loc='center',
       bbox=[0, 0.1, 1, 0.8])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor('white' if row > 0 else '#f0f4f8')
        cell.set_edgecolor('#d1d8e0')
        cell.set_text_props(color='#1e293b' if row > 0 else '#334155')

    # Per-type AUC
    ax_type = fig.add_subplot(gs[2, :])
    type_order = ['real', 'audio_modified', 'visual_modified', 'both_modified']
    x2 = np.arange(len(type_order))
    for i, (r, color) in enumerate(zip(results, colors)):
        pt = r['metrics']['per_type']
        vals = [pt.get(t, {}).get('auc', 0) for t in type_order]
        bars = ax_type.bar(x2 + i*w - (n-1)*w/2, vals, w, label=r['name'], color=color, alpha=0.85)
        for b, v in zip(bars, vals):
            if v > 0:
                ax_type.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{v:.3f}',
                             ha='center', va='bottom', fontsize=7)
    ax_type.set_xticks(x2)
    ax_type.set_xticklabels(['Real', 'Audio Modified', 'Visual Modified', 'Both Modified'], fontsize=10)
    ax_type.set_ylim(0, 1.15); ax_type.set_ylabel('AUC')
    ax_type.set_title('AUC by Manipulation Type', fontsize=11, fontweight='bold')
    ax_type.legend(fontsize=9)
    ax_type.axhline(0.5, color='#95a5a6', linestyle='--', linewidth=0.8)
    ax_type.grid(axis='y', alpha=0.4, color='#e2e8f0')

    # Audio vs Video scatter
    for idx, (r, color) in enumerate(zip(results[:3], colors)):
        ax_s = fig.add_subplot(gs[3, idx])
        df = r['df']
        for vtype, marker, tc in [
            ('real', 'o', '#2ecc71'), ('audio_modified', 's', '#f39c12'),
            ('visual_modified', '^', '#9b59b6'), ('both_modified', 'D', '#e74c3c'),
        ]:
            sub = df[df['modify_type'] == vtype]
            if len(sub):
                ax_s.scatter(sub['audio_score'], sub['video_score'],
                             c=tc, marker=marker, alpha=0.6, s=20, label=vtype.replace('_', ' '))
        ax_s.axvline(0.5, color='#95a5a6', linestyle='--', linewidth=0.8)
        ax_s.axhline(0.5, color='#95a5a6', linestyle='--', linewidth=0.8)
        ax_s.set_xlim(-0.02, 1.02); ax_s.set_ylim(-0.02, 1.02)
        ax_s.set_xlabel('Audio Score'); ax_s.set_ylabel('Video Score')
        ax_s.set_title(f'Audio vs Video — {r["name"]}', fontsize=11, fontweight='bold')
        ax_s.legend(fontsize=6, loc='lower right')
        ax_s.grid(alpha=0.3, color='#e2e8f0')

    fig.text(0.5, 0.97, 'Model Accuracy Comparison — AV Deepfake Detection',
             ha='center', va='top', fontsize=16, fontweight='bold', color='#1e293b')
    out = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(out)
    plt.close()
    print(f'\n  Saved: {out}')


def plot_training_history(results, output_dir):
    has_history = any(r['info'].get('history', {}).get('val_auc_joint') for r in results)
    if not has_history:
        print("  No training history — skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = PALETTE[:len(results)]
    markers = ['o', '^', 's', 'D']

    for r, color, marker in zip(results, colors, markers):
        h = r['info'].get('history', {})
        name = r['name']
        if h.get('val_auc_joint'):
            eps = range(1, len(h['val_auc_joint']) + 1)
            axes[0].plot(eps, h['val_auc_joint'], color=color, linewidth=2,
                         marker=marker, markersize=5, label=name)
            best = max(h['val_auc_joint'])
            best_ep = h['val_auc_joint'].index(best) + 1
            axes[0].axvline(best_ep, color=color, linestyle='--', alpha=0.3, linewidth=1)
        if h.get('train_loss'):
            eps = range(1, len(h['train_loss']) + 1)
            axes[1].plot(eps, h['train_loss'], color=color, linewidth=2,
                         marker=marker, markersize=5, label=f'{name} train', linestyle='-')
        if h.get('val_loss'):
            eps = range(1, len(h['val_loss']) + 1)
            axes[1].plot(eps, h['val_loss'], color=color, linewidth=1.5,
                         marker=marker, markersize=4, label=f'{name} val', linestyle='--', alpha=0.7)

    for ax, title, ylabel in [(axes[0], 'Validation AUC over Epochs', 'Joint AUC'),
                               (axes[1], 'Loss over Epochs', 'Loss')]:
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.4, color='#e2e8f0')
        if 'AUC' in ylabel:
            ax.set_ylim(0, 1.05)

    fig.suptitle('Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(output_dir, 'training_history.png')
    plt.savefig(out)
    plt.close()
    print(f'  Saved: {out}')


def main():
    results = build_results()
    if not results:
        print('ERROR: no results')
        return
    plot_all(results, 'comparison_results')
    plot_training_history(results, 'comparison_results')
    print('\nDone.')


if __name__ == '__main__':
    main()
