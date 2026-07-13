"""
scripts/run_exp4_temperature_scaling.py

Experiment 4: Temperature Scaling
Fits a single temperature parameter to the V1 (N=91) model's validation logits
to improve calibration without changing model weights.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.dataset import IPIDataset
from src.models.student import load_student_checkpoint
from src.adaptive.failure_analysis import run_inference
from src.utils.reproducibility import get_device
from src.evaluation.calibration import compute_ece
from torch.utils.data import DataLoader

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1

def main():
    device = get_device()
    
    val_path = _PROJECT_ROOT / "data/processed/dataset_v0/validation.csv"
    df = pd.read_csv(val_path)
    y_true = df["label"].values.astype(int)
    
    print("Loading V1 Model...")
    checkpoint_path = _PROJECT_ROOT / "outputs/mac_experiments/exp004/checkpoint/best"
    model, tokenizer = load_student_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    
    dataset = IPIDataset(val_path, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    print("Extracting logits...")
    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            
    logits = torch.cat(all_logits, dim=0)
    labels = torch.tensor(y_true, dtype=torch.long)
    
    # Pre-scaling metrics
    probs_pre = F.softmax(logits, dim=1)[:, 1].numpy()
    ece_pre = compute_ece(y_true, probs_pre)
    f1_pre = compute_metrics(y_true, probs_pre, 0.5)[3]
    
    print(f"\nBefore Temperature Scaling:")
    print(f"ECE:      {ece_pre:.4f}")
    print(f"F1 @ 0.5: {f1_pre:.4f}")
    
    # Fit Temperature
    print("\nFitting Temperature Parameter...")
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    # Define a custom module so we can pass it to LBFGS easily
    class TemperatureScaler(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        def forward(self, logits):
            return logits / self.temperature
            
    scaler = TemperatureScaler()
    optimizer = LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    nll_loss = nn.CrossEntropyLoss()
    
    def eval():
        optimizer.zero_grad()
        loss = nll_loss(scaler(logits), labels)
        loss.backward()
        return loss
        
    optimizer.step(eval)
    
    optimal_t = scaler.temperature.item()
    print(f"Optimal Temperature: {optimal_t:.4f}")
    
    # Post-scaling metrics
    scaled_logits = scaler(logits).detach()
    probs_post = F.softmax(scaled_logits, dim=1)[:, 1].numpy()
    ece_post = compute_ece(y_true, probs_post)
    f1_post = compute_metrics(y_true, probs_post, 0.5)[3]
    
    print(f"\nAfter Temperature Scaling (T={optimal_t:.4f}):")
    print(f"ECE:      {ece_post:.4f}")
    print(f"F1 @ 0.5: {f1_post:.4f}")
    
    # Find optimal threshold for T-scaled model
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [compute_metrics(y_true, probs_post, t)[3] for t in thresholds]
    best_f1_post = max(f1s)
    best_t_idx = np.argmax(f1s)
    best_thresh = thresholds[best_t_idx]
    
    print(f"Best F1:  {best_f1_post:.4f} @ {best_thresh:.2f}")
    
    import json
    results = {
        "pre": {"ece": ece_pre, "f1_05": f1_pre},
        "post": {"temperature": optimal_t, "ece": ece_post, "f1_05": f1_post, "best_f1": best_f1_post, "best_threshold": best_thresh}
    }
    
    out_dir = _PROJECT_ROOT / "reports" / "ablation_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "exp4_temperature_scaling.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
