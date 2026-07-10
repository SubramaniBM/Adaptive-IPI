"""
src/evaluation/evaluator.py

Evaluation runner for the Adaptive-IPI project.

Orchestrates model evaluation: runs inference, computes metrics,
generates calibration plots, and saves all results to the experiment
directory (Change #5: save everything).
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.adaptive.failure_analysis import run_inference
from src.core.enums import Label
from src.core.types import EvaluationResult
from src.evaluation.calibration import (
    compute_ece,
    plot_confidence_histogram,
    plot_reliability_diagram,
)
from src.evaluation.metrics import compute_metrics, format_metrics_report
from src.utils.io import write_csv, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Quick evaluation function for use during training.

    Returns a flat dict of scalar metrics suitable for the training
    engine's callback system.

    Args:
        model: Trained student model.
        dataloader: Evaluation DataLoader.
        device: Device to run on.

    Returns:
        Dict of metric name → value.
    """
    predictions_df = run_inference(model, dataloader, device)

    # Collect ground-truth labels from the dataloader's dataset
    dataset = dataloader.dataset
    y_true = np.array(dataset.labels)
    y_pred = np.array(predictions_df["predicted_label"].tolist())
    y_prob = np.array(predictions_df["predicted_prob"].tolist())

    result = compute_metrics(y_true, y_pred, y_prob)
    result.ece = compute_ece(y_true, y_prob)

    return {
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "auroc": result.auroc,
        "auprc": result.auprc,
        "ece": result.ece,
    }


def run_full_evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    eval_df: pd.DataFrame,
    output_dir: Union[str, Path],
    tag: str = "",
) -> EvaluationResult:
    """Run comprehensive evaluation and save all artifacts.

    Saves (Change #5):
        - metrics.json
        - predictions.csv
        - confusion_matrix.png
        - reliability_diagram.png
        - confidence_histogram.png
        - evaluation_report.txt

    Args:
        model: Trained student model.
        dataloader: Evaluation DataLoader.
        device: Device to run on.
        eval_df: DataFrame with ground-truth data (for metadata columns).
        output_dir: Directory to save all evaluation artifacts.
        tag: Optional tag for figure titles (e.g., "pre-curriculum").

    Returns:
        EvaluationResult with all computed metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    predictions_df = run_inference(model, dataloader, device)

    y_true = eval_df["label"].values.astype(int)
    y_pred = np.array(predictions_df["predicted_label"].tolist())
    y_prob = np.array(predictions_df["predicted_prob"].tolist())

    # Compute standard metrics with per-category breakdowns
    attack_types = eval_df["attack_type"].values if "attack_type" in eval_df.columns else None
    context_types = eval_df["context_type"].values if "context_type" in eval_df.columns else None

    result = compute_metrics(
        y_true, y_pred, y_prob,
        attack_types=attack_types,
        context_types=context_types,
    )

    # Compute calibration (Change #6)
    result.ece = compute_ece(y_true, y_prob)

    # Reconstruct original DataFrame with predictions
    pred_df = eval_df.copy()
    pred_df["predicted_label"] = y_pred
    pred_df["predicted_prob"] = y_prob
    if "text" in pred_df.columns and "email_text" not in pred_df.columns:
        pred_df["email_text"] = pred_df["text"]

    # Use PredictionRecorder to save all evaluation artifacts
    from src.evaluation.prediction_recorder import PredictionRecorder
    recorder = PredictionRecorder(output_dir)
    recorder.record_all(result, pred_df)

    # Save report text for human readability
    report = format_metrics_report(result)
    (output_dir / "evaluation_report.txt").write_text(report, encoding="utf-8")
    logger.info(f"\n{report}")

    # Generate visualizations
    title_prefix = f"{tag} — " if tag else ""

    plot_reliability_diagram(
        y_true, y_prob,
        title=f"{title_prefix}Reliability Diagram",
        output_path=output_dir / "reliability_diagram.png",
    )

    plot_confidence_histogram(
        y_true, y_prob,
        title=f"{title_prefix}Confidence Distribution",
        output_path=output_dir / "confidence_histogram.png",
    )

    # Confusion matrix plot
    _plot_confusion_matrix(
        result.confusion_matrix,
        title=f"{title_prefix}Confusion Matrix",
        output_path=output_dir / "confusion_matrix.png",
    )

    logger.info(f"Full evaluation saved to {output_dir}")
    return result


def _plot_confusion_matrix(
    cm: list[list[int]],
    title: str = "Confusion Matrix",
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot and save a confusion matrix.

    Args:
        cm: 2x2 confusion matrix as nested list.
        title: Plot title.
        output_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
