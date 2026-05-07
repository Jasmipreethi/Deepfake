"""
Plot calibration curves for Model 2 and Model 3 from prediction CSVs.

Generates side-by-side reliability diagrams showing predicted probability
vs. actual fraction of positives for each model. A perfectly calibrated model
follows the diagonal. Score compression (fake scores drifting toward 0.5)
appears as upward deviation from the diagonal.

Usage:
    python plot_calibration_curves.py
"""

import os, sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

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

CSV_PATHS = {
    'Model 2 (5ep)': 'comparison_results/Model_2_(5ep,_best)_predictions.csv',
    'Model 3 (3ep)': 'comparison_results/Model_3_(3ep)_predictions.csv',
    'Model 4 (5ep)': 'comparison_results/Model_4_(5ep)_predictions.csv',
}
COLORS = {'Model 2 (5ep)': '#3498db', 'Model 3 (3ep)': '#2ecc71',
          'Model 4 (5ep)': '#f39c12'}
MARKERS = {'Model 2 (5ep)': 'o', 'Model 3 (3ep)': 's', 'Model 4 (5ep)': '^'}


def compute_calibration_data(csv_path):
    """Return prob_true, prob_pred from a prediction CSV."""
    y_true = []
    y_pred = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            y_true.append(int(row['true_label']))
            y_pred.append(float(row['joint_score']))

    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10,
                                              strategy='uniform')
    return prob_true, prob_pred


def plot_calibration(data_dict, output_path):
    """Plot reliability diagrams for all models side by side."""
    n = len(data_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.suptitle('Reliability Diagrams: Score Calibration', fontsize=14,
                 fontweight='bold')

    for ax, (name, (prob_true, prob_pred)) in zip(axes, data_dict.items()):
        color = COLORS.get(name, '#333')
        marker = MARKERS.get(name, 'o')

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], '--', color='#bdc3c7', linewidth=1.5,
                label='Perfect calibration')

        # Calibration curve
        ax.plot(prob_pred, prob_true, color=color, marker=marker,
                linewidth=2, markersize=8, label=name)
        ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.1,
                        color=color)

        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Actual fraction of positives')
        ax.set_title(name, fontsize=12, fontweight='bold', color=color)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.4, color='#e2e8f0')
        ax.set_aspect('equal')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f'Saved: {output_path}')

    # Print calibration summary
    print()
    for name, (prob_true, prob_pred) in data_dict.items():
        gaps = np.abs(prob_true - prob_pred)
        mean_gap = np.mean(gaps)
        print(f'  {name}: mean calibration gap = {mean_gap:.4f}')


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'figures'
    out_path = os.path.join(out_dir, 'calibration_curves.png')

    data = {}
    for name, path in CSV_PATHS.items():
        if os.path.exists(path):
            prob_true, prob_pred = compute_calibration_data(path)
            data[name] = (prob_true, prob_pred)
            print(f'  Loaded {name}: {len(prob_true)} bins')
        else:
            print(f'  Skipped {name}: file not found')

    if not data:
        print('ERROR: no prediction CSVs found')
        sys.exit(1)

    plot_calibration(data, out_path)


if __name__ == '__main__':
    main()
