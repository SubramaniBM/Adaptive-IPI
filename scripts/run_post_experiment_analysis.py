"""
scripts/run_post_experiment_analysis.py

Comprehensive post-experiment analysis of the V0 and V1 student models
to determine the root cause of the AUROC vs F1 divergence.

Generates:
1. Raw predictions CSV
2. Probability distribution histograms
3. Threshold sweep metrics
4. ROC and PR curves
5. Calibration analysis (Reliability diagram, ECE, Brier)
"""

import sys
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    brier_score_loss,
    confusion_matrix
)
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.dataset import IPIDataset
from src.models.student import load_student_checkpoint
from src.adaptive.failure_analysis import run_inference
from src.utils.reproducibility import get_device
from src.evaluation.calibration import compute_ece


def compute_metrics_at_threshold(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    # Zero_division=0 to prevent warnings when no positive predictions
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1


def main():
    device = get_device()
    reports_dir = _PROJECT_ROOT / "reports" / "post_experiment_analysis"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Load Models and Run Inference
    # ---------------------------------------------------------
    print("Loading V0 model...")
    v0_checkpoint = _PROJECT_ROOT / "outputs/mac_experiments/exp008/checkpoint/best"
    v0_model, tokenizer = load_student_checkpoint(v0_checkpoint)
    v0_model = v0_model.to(device)
    
    print("Loading V1 model...")
    v1_checkpoint = _PROJECT_ROOT / "outputs/mac_experiments/exp009/checkpoint/best"
    v1_model, _ = load_student_checkpoint(v1_checkpoint)
    v1_model = v1_model.to(device)
    
    val_path = _PROJECT_ROOT / "data/processed/dataset_v0/validation.csv"
    print(f"Loading dataset from {val_path}...")
    df = pd.read_csv(val_path)
    dataset = IPIDataset(val_path, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    print("Running inference for V0...")
    v0_preds = run_inference(v0_model, dataloader, device)
    print("Running inference for V1...")
    v1_preds = run_inference(v1_model, dataloader, device)
    
    y_true = df["label"].values.astype(int)
    v0_prob = np.array(v0_preds["predicted_prob"].tolist())
    v1_prob = np.array(v1_preds["predicted_prob"].tolist())
    
    # ---------------------------------------------------------
    # Step 0: Validate Evaluation Pipeline
    # ---------------------------------------------------------
    print("\n--- STEP 0: VALIDATE PIPELINE ---")
    v0_acc, v0_prec, v0_rec, v0_f1 = compute_metrics_at_threshold(y_true, v0_prob, 0.5)
    v1_acc, v1_prec, v1_rec, v1_f1 = compute_metrics_at_threshold(y_true, v1_prob, 0.5)
    
    print(f"V0 @ 0.5 -> Acc: {v0_acc:.4f}, Prec: {v0_prec:.4f}, Rec: {v0_rec:.4f}, F1: {v0_f1:.4f}")
    print(f"V1 @ 0.5 -> Acc: {v1_acc:.4f}, Prec: {v1_prec:.4f}, Rec: {v1_rec:.4f}, F1: {v1_f1:.4f}")
    print("---------------------------------")
    
    # ---------------------------------------------------------
    # 2. Save Raw Predictions
    # ---------------------------------------------------------
    raw_df = pd.DataFrame({
        "sample_id": df["id"] if "id" in df.columns else df.index,
        "ground_truth": y_true,
        "prob_v0": v0_prob,
        "pred_v0": (v0_prob >= 0.5).astype(int),
        "prob_v1": v1_prob,
        "pred_v1": (v1_prob >= 0.5).astype(int)
    })
    raw_df.to_csv(reports_dir / "analysis_raw_predictions.csv", index=False)
    print(f"Saved raw predictions to {reports_dir / 'analysis_raw_predictions.csv'}")
    
    # ---------------------------------------------------------
    # 1. Probability Distribution Analysis
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(v0_prob[y_true == 0], color='blue', alpha=0.5, label='Negatives', bins=50)
    sns.histplot(v0_prob[y_true == 1], color='red', alpha=0.5, label='Positives', bins=50)
    plt.title('V0 Probability Distribution')
    plt.xlabel('Predicted Probability')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(v1_prob[y_true == 0], color='blue', alpha=0.5, label='Negatives', bins=50)
    sns.histplot(v1_prob[y_true == 1], color='red', alpha=0.5, label='Positives', bins=50)
    plt.title('V1 Probability Distribution')
    plt.xlabel('Predicted Probability')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(reports_dir / "prob_distribution.png", dpi=150)
    plt.close()
    
    # ---------------------------------------------------------
    # 2. Threshold Sweep
    # ---------------------------------------------------------
    thresholds = np.linspace(0.01, 0.99, 99)
    
    v0_metrics = {"acc": [], "prec": [], "rec": [], "f1": []}
    v1_metrics = {"acc": [], "prec": [], "rec": [], "f1": []}
    
    for t in thresholds:
        a0, p0, r0, f0 = compute_metrics_at_threshold(y_true, v0_prob, t)
        v0_metrics["acc"].append(a0); v0_metrics["prec"].append(p0)
        v0_metrics["rec"].append(r0); v0_metrics["f1"].append(f0)
        
        a1, p1, r1, f1 = compute_metrics_at_threshold(y_true, v1_prob, t)
        v1_metrics["acc"].append(a1); v1_metrics["prec"].append(p1)
        v1_metrics["rec"].append(r1); v1_metrics["f1"].append(f1)
        
    v0_best_idx = np.argmax(v0_metrics["f1"])
    v1_best_idx = np.argmax(v1_metrics["f1"])
    
    print("\nBest F1")
    print("V0")
    print(f"Threshold: {thresholds[v0_best_idx]:.2f}")
    print(f"Accuracy:  {v0_metrics['acc'][v0_best_idx]:.4f}")
    print(f"Precision: {v0_metrics['prec'][v0_best_idx]:.4f}")
    print(f"Recall:    {v0_metrics['rec'][v0_best_idx]:.4f}")
    print(f"F1:        {v0_metrics['f1'][v0_best_idx]:.4f}")
    
    print("\nV1")
    print(f"Threshold: {thresholds[v1_best_idx]:.2f}")
    print(f"Accuracy:  {v1_metrics['acc'][v1_best_idx]:.4f}")
    print(f"Precision: {v1_metrics['prec'][v1_best_idx]:.4f}")
    print(f"Recall:    {v1_metrics['rec'][v1_best_idx]:.4f}")
    print(f"F1:        {v1_metrics['f1'][v1_best_idx]:.4f}")
    print("--------------------------\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(thresholds, v0_metrics["f1"], label="V0", color="blue")
    axes[0, 0].plot(thresholds, v1_metrics["f1"], label="V1", color="orange")
    axes[0, 0].set_title("F1 vs Threshold"); axes[0, 0].legend()
    
    axes[0, 1].plot(thresholds, v0_metrics["acc"], label="V0", color="blue")
    axes[0, 1].plot(thresholds, v1_metrics["acc"], label="V1", color="orange")
    axes[0, 1].set_title("Accuracy vs Threshold"); axes[0, 1].legend()
    
    axes[1, 0].plot(thresholds, v0_metrics["prec"], label="V0", color="blue")
    axes[1, 0].plot(thresholds, v1_metrics["prec"], label="V1", color="orange")
    axes[1, 0].set_title("Precision vs Threshold"); axes[1, 0].legend()
    
    axes[1, 1].plot(thresholds, v0_metrics["rec"], label="V0", color="blue")
    axes[1, 1].plot(thresholds, v1_metrics["rec"], label="V1", color="orange")
    axes[1, 1].set_title("Recall vs Threshold"); axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(reports_dir / "threshold_sweep.png", dpi=150)
    plt.close()
    
    # ---------------------------------------------------------
    # 3. ROC Analysis
    # ---------------------------------------------------------
    fpr0, tpr0, _ = roc_curve(y_true, v0_prob)
    roc_auc0 = auc(fpr0, tpr0)
    
    fpr1, tpr1, _ = roc_curve(y_true, v1_prob)
    roc_auc1 = auc(fpr1, tpr1)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr0, tpr0, color='blue', lw=2, label=f'V0 (AUROC = {roc_auc0:.3f})')
    plt.plot(fpr1, tpr1, color='orange', lw=2, label=f'V1 (AUROC = {roc_auc1:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(reports_dir / "roc_curve.png", dpi=150)
    plt.close()
    
    # ---------------------------------------------------------
    # 4. Precision-Recall Curve
    # ---------------------------------------------------------
    precision0, recall0, _ = precision_recall_curve(y_true, v0_prob)
    pr_auc0 = auc(recall0, precision0)
    
    precision1, recall1, _ = precision_recall_curve(y_true, v1_prob)
    pr_auc1 = auc(recall1, precision1)
    
    plt.figure(figsize=(6, 6))
    plt.plot(recall0, precision0, color='blue', lw=2, label=f'V0 (AUPRC = {pr_auc0:.3f})')
    plt.plot(recall1, precision1, color='orange', lw=2, label=f'V1 (AUPRC = {pr_auc1:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(reports_dir / "pr_curve.png", dpi=150)
    plt.close()
    
    # ---------------------------------------------------------
    # 5. Calibration Analysis
    # ---------------------------------------------------------
    v0_ece = compute_ece(y_true, v0_prob)
    v1_ece = compute_ece(y_true, v1_prob)
    
    v0_brier = brier_score_loss(y_true, v0_prob)
    v1_brier = brier_score_loss(y_true, v1_prob)
    
    from sklearn.calibration import calibration_curve
    prob_true0, prob_pred0 = calibration_curve(y_true, v0_prob, n_bins=10)
    prob_true1, prob_pred1 = calibration_curve(y_true, v1_prob, n_bins=10)
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred0, prob_true0, "s-", label=f"V0 (ECE: {v0_ece:.3f})")
    plt.plot(prob_pred1, prob_true1, "s-", label=f"V1 (ECE: {v1_ece:.3f})")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title("Reliability Diagram")
    plt.legend(loc="lower right")
    plt.savefig(reports_dir / "reliability_diagram.png", dpi=150)
    plt.close()
    
    # ---------------------------------------------------------
    # 6. Confusion Matrices
    # ---------------------------------------------------------
    def plot_cm(cm, ax, title):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"], ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    cm_v0_50 = confusion_matrix(y_true, (v0_prob >= 0.5).astype(int))
    cm_v1_50 = confusion_matrix(y_true, (v1_prob >= 0.5).astype(int))
    cm_v0_best = confusion_matrix(y_true, (v0_prob >= thresholds[v0_best_idx]).astype(int))
    cm_v1_best = confusion_matrix(y_true, (v1_prob >= thresholds[v1_best_idx]).astype(int))
    
    plot_cm(cm_v0_50, axes[0, 0], "V0 @ 0.5")
    plot_cm(cm_v1_50, axes[0, 1], "V1 @ 0.5")
    plot_cm(cm_v0_best, axes[1, 0], f"V0 @ {thresholds[v0_best_idx]:.2f}")
    plot_cm(cm_v1_best, axes[1, 1], f"V1 @ {thresholds[v1_best_idx]:.2f}")
    
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrices.png", dpi=150)
    plt.close()
    
    # ---------------------------------------------------------
    # 7. Error Analysis (False Negatives)
    # ---------------------------------------------------------
    # Focus on threshold 0.5 to understand the original metric drop
    v0_fn = (y_true == 1) & (v0_prob < 0.5)
    v1_fn = (y_true == 1) & (v1_prob < 0.5)
    
    newly_corrected = v0_fn & ~v1_fn
    newly_broken = ~v0_fn & v1_fn
    
    print(f"\n--- ERROR ANALYSIS (@ 0.5) ---")
    print(f"V0 False Negatives: {v0_fn.sum()}")
    print(f"V1 False Negatives: {v1_fn.sum()}")
    print(f"Newly Corrected by V1: {newly_corrected.sum()}")
    print(f"Newly Broken by V1:    {newly_broken.sum()}")
    
    # Save a JSON with the summary stats to populate the markdown report
    summary = {
        "v0": {
            "default": {"acc": v0_acc, "prec": v0_prec, "rec": v0_rec, "f1": v0_f1},
            "best": {
                "threshold": thresholds[v0_best_idx],
                "acc": v0_metrics["acc"][v0_best_idx],
                "prec": v0_metrics["prec"][v0_best_idx],
                "rec": v0_metrics["rec"][v0_best_idx],
                "f1": v0_metrics["f1"][v0_best_idx]
            },
            "roc_auc": roc_auc0,
            "pr_auc": pr_auc0,
            "ece": v0_ece,
            "brier": v0_brier,
            "fn_count": int(v0_fn.sum())
        },
        "v1": {
            "default": {"acc": v1_acc, "prec": v1_prec, "rec": v1_rec, "f1": v1_f1},
            "best": {
                "threshold": thresholds[v1_best_idx],
                "acc": v1_metrics["acc"][v1_best_idx],
                "prec": v1_metrics["prec"][v1_best_idx],
                "rec": v1_metrics["rec"][v1_best_idx],
                "f1": v1_metrics["f1"][v1_best_idx]
            },
            "roc_auc": roc_auc1,
            "pr_auc": pr_auc1,
            "ece": v1_ece,
            "brier": v1_brier,
            "fn_count": int(v1_fn.sum())
        },
        "error_analysis": {
            "newly_corrected": int(newly_corrected.sum()),
            "newly_broken": int(newly_broken.sum())
        }
    }
    
    with open(reports_dir / "analysis_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nPost-experiment analysis complete! Check reports/post_experiment_analysis/")


if __name__ == "__main__":
    main()
