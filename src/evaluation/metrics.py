"""
src/evaluation/metrics.py

Standard evaluation metrics for binary IPI classification.

Computes accuracy, precision, recall, F1, AUROC, AUPRC, and
per-category breakdowns.
"""

from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.core.types import EvaluationResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    attack_types: Optional[np.ndarray] = None,
    context_types: Optional[np.ndarray] = None,
) -> EvaluationResult:
    """Compute comprehensive evaluation metrics.

    Args:
        y_true: Ground-truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        y_prob: Predicted probabilities for the positive class, shape (N,).
        attack_types: Optional array of attack type strings for per-type breakdown.
        context_types: Optional array of context type strings for per-type breakdown.

    Returns:
        EvaluationResult dataclass with all computed metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Handle single-class edge cases gracefully
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        if y_true[0] == 1:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    result = EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        confusion_matrix=cm.tolist(),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )

    # AUROC (requires both classes present)
    if len(np.unique(y_true)) > 1:
        result.auroc = float(roc_auc_score(y_true, y_prob))
        result.auprc = float(average_precision_score(y_true, y_prob))
    else:
        logger.warning("Only one class present — AUROC/AUPRC set to 0.0")

    # Per-attack-type breakdown
    if attack_types is not None:
        result.per_attack_type = _per_category_metrics(
            y_true, y_pred, y_prob, attack_types
        )

    # Per-context-type breakdown
    if context_types is not None:
        result.per_context_type = _per_category_metrics(
            y_true, y_pred, y_prob, context_types
        )

    return result


def _per_category_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    categories: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by category.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.
        categories: Category labels for each sample.

    Returns:
        Dict mapping category name to metric dict.
    """
    results: dict[str, dict[str, float]] = {}
    unique_cats = np.unique(categories)

    for cat in unique_cats:
        mask = categories == cat
        if mask.sum() == 0:
            continue

        cat_true = y_true[mask]
        cat_pred = y_pred[mask]
        cat_prob = y_prob[mask]

        metrics: dict[str, float] = {
            "count": int(mask.sum()),
            "accuracy": float(accuracy_score(cat_true, cat_pred)),
            "precision": float(precision_score(cat_true, cat_pred, zero_division=0)),
            "recall": float(recall_score(cat_true, cat_pred, zero_division=0)),
            "f1": float(f1_score(cat_true, cat_pred, zero_division=0)),
        }

        if len(np.unique(cat_true)) > 1:
            metrics["auroc"] = float(roc_auc_score(cat_true, cat_prob))

        results[str(cat)] = metrics

    return results


def format_metrics_report(result: EvaluationResult) -> str:
    """Format evaluation metrics as a human-readable report string.

    Args:
        result: EvaluationResult dataclass.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 60,
        "  EVALUATION REPORT",
        "=" * 60,
        "",
        f"  Accuracy  : {result.accuracy:.4f}",
        f"  Precision : {result.precision:.4f}",
        f"  Recall    : {result.recall:.4f}",
        f"  F1        : {result.f1:.4f}",
        f"  AUROC     : {result.auroc:.4f}",
        f"  AUPRC     : {result.auprc:.4f}",
        f"  ECE       : {result.ece:.4f}",
        "",
        "  Confusion Matrix:",
        f"    {result.confusion_matrix}",
    ]

    if result.per_attack_type:
        lines += ["", "  Per Attack Type:"]
        for cat, metrics in sorted(result.per_attack_type.items()):
            lines.append(
                f"    {cat:<35} F1={metrics.get('f1', 0):.4f}  "
                f"n={metrics.get('count', 0)}"
            )

    if result.per_context_type:
        lines += ["", "  Per Context Type:"]
        for cat, metrics in sorted(result.per_context_type.items()):
            lines.append(
                f"    {cat:<35} F1={metrics.get('f1', 0):.4f}  "
                f"n={metrics.get('count', 0)}"
            )

    lines += ["", "=" * 60]
    return "\n".join(lines)
