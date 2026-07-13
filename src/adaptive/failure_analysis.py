"""
src/adaptive/failure_analysis.py

Inference utilities for the Adaptive-IPI evaluation pipeline.

Provides ``run_inference`` which runs a student model on a DataLoader
and returns per-sample predictions and probabilities in a DataFrame.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import json
from pathlib import Path

from src.utils.logging import get_logger


def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """Run inference and return per-sample predictions.

    Args:
        model: Trained student model.
        dataloader: DataLoader for inference.
        device: Device to run on.

    Returns:
        DataFrame with columns: predicted_label, predicted_prob.
    """
    model.eval()
    all_preds = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            # Probability of the positive class (attack)
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            
            # Compute loss
            labels = batch["labels"].to(device)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            total_samples += labels.size(0)

    df = pd.DataFrame({
        "predicted_label": all_preds,
        "predicted_prob": all_probs,
    })
    df.attrs["eval_loss"] = total_loss / max(total_samples, 1)
    return df


def identify_failures(eval_df: pd.DataFrame, predictions_df: pd.DataFrame, confidence_threshold: float) -> list[dict]:
    """Identify failures from predictions."""
    failures = []
    
    # Merge or align
    for i, row in eval_df.iterrows():
        pred_row = predictions_df.iloc[i]
        true_label = row["label"]
        pred_label = pred_row["predicted_label"]
        prob = pred_row["predicted_prob"]
        
        if true_label != pred_label:
            failure = {
                "sample_id": row.get("id", f"sample_{i}"),
                "text": row.get("text", row.get("context", "")),
                "true_label": int(true_label),
                "predicted_label": int(pred_label),
                "probability": float(prob),
                "attack_family": row.get("attack_family", "N/A"),
                "attack_position": row.get("attack_position", "N/A"),
            }
            failures.append(failure)
            
    return failures


def generate_failure_report(failures: list[dict], output_dir: Path) -> dict:
    """Generate a failure report and save to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "total_failures": len(failures),
        "by_failure_type": {
            "false_positives": sum(1 for f in failures if f["predicted_label"] == 1),
            "false_negatives": sum(1 for f in failures if f["predicted_label"] == 0),
        }
    }
    
    with open(output_dir / "failure_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    with open(output_dir / "failures.jsonl", "w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(failure) + "\n")
            
    return report
