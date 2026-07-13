"""
scripts/analyze_exp1.py

Extracts calibration and performance metrics for the curriculum sweep.
Models:
- exp003 (V0 baseline, N=0)
- exp005 (V2, N=20)
- exp006 (V3, N=40)
- exp007 (V4, N=60)
- exp004 (V1, N=91)
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.dataset import IPIDataset
from src.models.student import load_student_checkpoint
from src.adaptive.failure_analysis import run_inference
from src.utils.reproducibility import get_device
from src.evaluation.calibration import compute_ece
from torch.utils.data import DataLoader

def compute_metrics_at_threshold(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return f1

def analyze_checkpoint(checkpoint_path, dataloader, device, y_true):
    print(f"Loading {checkpoint_path}...")
    model, _ = load_student_checkpoint(checkpoint_path)
    model = model.to(device)
    preds = run_inference(model, dataloader, device)
    y_prob = np.array(preds["predicted_prob"].tolist())
    
    # Defaults
    f1_50 = compute_metrics_at_threshold(y_true, y_prob, 0.5)
    
    # Best F1
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [compute_metrics_at_threshold(y_true, y_prob, t) for t in thresholds]
    best_f1 = max(f1s)
    
    # AUROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # ECE
    ece = compute_ece(y_true, y_prob)
    
    return {
        "F1 @ 0.5": f1_50,
        "Best F1": best_f1,
        "AUROC": roc_auc,
        "ECE": ece
    }

def main():
    device = get_device()
    
    val_path = _PROJECT_ROOT / "data/processed/dataset_v0/validation.csv"
    df = pd.read_csv(val_path)
    y_true = df["label"].values.astype(int)
    
    print("Loading tokenizer...")
    # Load any tokenizer to instantiate dataset
    _, tokenizer = load_student_checkpoint(_PROJECT_ROOT / "outputs/mac_experiments/exp003/checkpoint/best")
    dataset = IPIDataset(val_path, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    experiments = [
        ("N=0 (Baseline)", "outputs/mac_experiments/exp003/checkpoint/best"),
        ("N=20", "outputs/mac_experiments/exp005/checkpoint/best"),
        ("N=40", "outputs/mac_experiments/exp006/checkpoint/best"),
        ("N=60", "outputs/mac_experiments/exp007/checkpoint/best"),
        ("N=91", "outputs/mac_experiments/exp004/checkpoint/best"),
    ]
    
    results = {}
    for name, path in experiments:
        full_path = _PROJECT_ROOT / path
        if full_path.exists():
            metrics = analyze_checkpoint(full_path, dataloader, device, y_true)
            results[name] = metrics
        else:
            print(f"Skipping {name}, checkpoint not found.")
            
    print("\n=== CURRICULUM SWEEP RESULTS ===")
    print(f"{'Curriculum':<15} | {'F1 @ 0.5':<10} | {'Best F1':<10} | {'AUROC':<10} | {'ECE':<10}")
    print("-" * 65)
    for name, m in results.items():
        print(f"{name:<15} | {m['F1 @ 0.5']:.4f}     | {m['Best F1']:.4f}    | {m['AUROC']:.4f}    | {m['ECE']:.4f}")

if __name__ == "__main__":
    main()
