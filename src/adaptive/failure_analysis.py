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

from src.utils.logging import get_logger

logger = get_logger(__name__)


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

    return pd.DataFrame({
        "predicted_label": all_preds,
        "predicted_prob": all_probs,
    })
