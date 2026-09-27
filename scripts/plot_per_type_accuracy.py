"""
Plot per-type accuracy grouped bar chart from model prediction CSVs.

Reads prediction CSV files from comparison_results/ and generates a grouped
bar chart showing test-set accuracy by manipulation type for each model.

Usage:
    python plot_per_type_accuracy.py
    python plot_per_type_accuracy.py <csv_glob_pattern> <output_dir>

Output:
    figures/per_type_accuracy_bar_chart.png
"""

import os
import sys
import csv
import glob
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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

TYPE_ORDER = ['real', 'audio_modified', 'visual_modified', 'both_modified']
TYPE_LABELS = ['Real', 'Audio Mod.', 'Visual Mod.', 'Both Mod.']
MODEL_COLORS = ['#3498db', '#2ecc71', '#f39c12']
MODEL_NAMES = ['Model 2 (5ep)', 'Model 3 (3ep)', 'Model 4 (5ep)']


def compute_per_type_accuracy(csv_path):
    """Return {modify_type: accuracy_float} from a prediction CSV."""
    correct = defaultdict(lambda: {'correct': 0, 'total': 0})
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            mt = row['modify_type']
            correct[mt]['total'] += 1
            if row['correct'] == 'True':
                correct[mt]['correct'] += 1

    return {
        mt: (correct[mt]['correct'] / correct[mt]['total'] * 100)
        for mt in TYPE_ORDER if correct[mt]['total'] > 0
    }


def plot_per_type_accuracy(data, output_path):
    """
    data: dict of model_name -> {type: accuracy_pct}
    """
    n_types = len(TYPE_ORDER)
    n_models = len(data)
    x = np.arange(n_types)
    width = 0.22
    offset = np.linspace(-(n_models - 1) * width / 2,
                          (n_models - 1) * width / 2,
                          n_models)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Test Set Accuracy by Manipulation Type', fontsize=14,
                 fontweight='bold')

    for i, (name, acc_dict) in enumerate(data.items()):
        accs = [acc_dict.get(t, 0) for t in TYPE_ORDER]
        bars = ax.bar(x + offset[i], accs, width,
                      label=name, color=MODEL_COLORS[i % len(MODEL_COLORS)],
                      edgecolor='white', linewidth=0.5)

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{acc:.0f}%', ha='center', va='bottom', fontsize=10,
                    fontweight='bold')

    ax.set_xlabel('Manipulation Type', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(TYPE_LABELS, fontsize=11)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, color='#e2e8f0')

    # annotate 100% line for reference
    ax.axhline(y=100, color='#95a5a6', linestyle=':', linewidth=1, alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f'Saved: {output_path}')

    # Print per-type summary
    for name, acc_dict in data.items():
        print(f'\n  {name}:')
        for t in TYPE_ORDER:
            if t in acc_dict:
                print(f'    {t:20s}: {acc_dict[t]:.0f}%')


def main():
    csv_glob = sys.argv[1] if len(sys.argv) > 1 else 'comparison_results/Model_*_predictions.csv'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'figures'
    out_path = os.path.join(out_dir, 'per_type_accuracy_bar_chart.png')

    csv_files = sorted(glob.glob(csv_glob))
    if not csv_files:
        print(f'ERROR: no files matching {csv_glob}')
        sys.exit(1)

    data = {}
    for path in csv_files:
        base = os.path.splitext(os.path.basename(path))[0]
        # Normalize model names
        if 'Model_2' in base and 'best' in base:
            label = 'Model 2 (5ep, best)'
        elif 'Model_2' in base:
            label = 'Model 2 (5ep)'
        elif 'Model_3' in base:
            label = 'Model 3 (3ep)'
        elif 'Model_4' in base:
            label = 'Model 4 (5ep)'
        elif 'Model_5' in base:
            label = 'Model 5 (5ep)'
        else:
            label = base

        acc = compute_per_type_accuracy(path)
        if acc:
            data[label] = acc
            print(f'  Loaded {label}: {len(acc)} types')

    # Use representative files: Model_2_best, Model_3, Model_4
    preferred = {}
    for label, acc in data.items():
        if 'Model 2' in label and 'best' not in label:
            continue  # skip non-best Model 2
        if 'Model 5' in label:
            continue  # Model 5 = Model 4 duplicate
        # rename Model_2 best to just Model 2
        display = label.replace('Model 2 (5ep, best)', 'Model 2 (5ep)')
        preferred[display] = acc

    if not preferred:
        print('ERROR: no models found')
        sys.exit(1)

    # Ensure consistent ordering: Model 2, Model 3, Model 4
    ordered = {}
    for name in ['Model 2 (5ep)', 'Model 3 (3ep)', 'Model 4 (5ep)']:
        if name in preferred:
            ordered[name] = preferred[name]

    plot_per_type_accuracy(ordered, out_path)


if __name__ == '__main__':
    main()
