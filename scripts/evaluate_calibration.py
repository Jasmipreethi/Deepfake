"""
Model Calibration & Temperature Scaling Evaluator
Calculates Expected Calibration Error (ECE) and applies Temperature Scaling
to optimize decision boundaries for deployment.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def compute_ece(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    
    Args:
        probs: Array of predicted probabilities (0 to 1).
        labels: Array of ground truth binary labels (0 or 1).
        n_bins: Number of probability confidence bins.
    
    Returns:
        ece: Expected Calibration Error value.
        bin_accs: Bin accuracies.
        bin_confs: Bin confidences.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_sizes.append(np.sum(in_bin))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_sizes.append(0)
            
    return ece, bin_accs, bin_confs, bin_sizes


def optimize_temperature(logits, labels, lr=0.01, max_iter=100):
    """
    Optimizes a single temperature parameter T on logits to minimize NLL.
    
    Args:
        logits: Tensor of raw model logits.
        labels: Tensor of ground truth targets.
    
    Returns:
        best_temp: Calibrated Temperature float value.
    """
    temperature = torch.ones(1, requires_grad=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    
    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss
        
    optimizer.step(eval_loss)
    return temperature.item()


def main():
    print("=" * 60)
    print("Model Calibration & Temperature Scaling Assessment")
    print("=" * 60)
    
    results_dir = os.path.join(os.path.dirname(__file__), "..", "data_and_results", "comparison_results")
    
    models = ["Model_2", "Model_3", "Model_4", "Model_5"]
    calibration_report = {}
    
    for model_name in models:
        csv_file = os.path.join(results_dir, f"{model_name}_predictions.csv")
        if not os.path.exists(csv_file):
            # Fallback to tagged filename
            candidates = [f for f in os.listdir(results_dir) if f.startswith(model_name) and f.endswith(".csv")]
            if candidates:
                csv_file = os.path.join(results_dir, candidates[0])
            else:
                continue
                
        df = pd.read_csv(csv_file)
        joint_probs = df['joint_score'].values
        labels = df['true_label'].values
        
        # Raw ECE
        raw_ece, _, _, _ = compute_ece(joint_probs, labels)
        
        # Convert probs to logits for temperature optimization
        clamped_probs = np.clip(joint_probs, 1e-6, 1.0 - 1e-6)
        logits_np = np.log(clamped_probs / (1.0 - clamped_probs))
        
        logits_tensor = torch.tensor(logits_np, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        # Optimize T
        optimal_T = optimize_temperature(logits_tensor, labels_tensor)
        
        # Calibrated Probs
        calibrated_logits = logits_np / optimal_T
        calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
        calibrated_ece, _, _, _ = compute_ece(calibrated_probs, labels)
        
        # Accuracy at threshold 0.5 before & after
        raw_acc = np.mean((joint_probs >= 0.5) == labels)
        calibrated_acc = np.mean((calibrated_probs >= 0.5) == labels)
        
        calibration_report[model_name] = {
            'raw_ece': float(raw_ece),
            'optimal_temperature': float(optimal_T),
            'calibrated_ece': float(calibrated_ece),
            'raw_acc_at_0.5': float(raw_acc),
            'calibrated_acc_at_0.5': float(calibrated_acc)
        }
        
        print(f"[{model_name}]")
        print(f"  - Raw ECE:             {raw_ece:.4f}")
        print(f"  - Optimal Temperature: {optimal_T:.4f}")
        print(f"  - Calibrated ECE:      {calibrated_ece:.4f} ({(raw_ece - calibrated_ece)/raw_ece*100:.1f}% reduction)")
        print(f"  - Raw Acc (0.5):       {raw_acc*100:.1f}%")
        print(f"  - Calibrated Acc(0.5): {calibrated_acc*100:.1f}%\n")
        
    out_json = os.path.join(results_dir, "calibration_metrics.json")
    with open(out_json, "w") as f:
        json.dump(calibration_report, f, indent=2)
    print(f"✓ Saved calibration analysis to: {out_json}")

if __name__ == "__main__":
    main()
