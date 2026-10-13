"""
v2/src/evaluation/span_metrics.py

Evaluation metrics for token-level classification and character-level span localization.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


def compute_token_level_metrics(
    all_preds: List[List[int]],
    all_labels: List[List[int]],
) -> Dict[str, float]:
    """
    Computes token-level Precision, Recall, F1 for the injection class (non-O).
    Labels of -100 are ignored.
    """
    tp = 0
    fp = 0
    fn = 0
    total_tokens = 0

    for preds, labels in zip(all_preds, all_labels):
        for p, l in zip(preds, labels):
            if l == -100:
                continue
            total_tokens += 1
            is_pred_attack = p in (1, 2)
            is_true_attack = l in (1, 2)

            if is_pred_attack and is_true_attack:
                tp += 1
            elif is_pred_attack and not is_true_attack:
                fp += 1
            elif not is_pred_attack and is_true_attack:
                fn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "token_precision": prec,
        "token_recall": rec,
        "token_f1": f1,
        "token_tp": tp,
        "token_fp": fp,
        "token_fn": fn,
        "total_evaluated_tokens": total_tokens,
    }


def compute_span_character_iou(
    pred_span: Tuple[int, int],
    gt_span: Tuple[int, int],
) -> float:
    """Computes Jaccard / Intersection over Union for character spans."""
    p_start, p_end = pred_span
    g_start, g_end = gt_span

    inter_start = max(p_start, g_start)
    inter_end = min(p_end, g_end)

    if inter_start >= inter_end:
        return 0.0

    intersection = inter_end - inter_start
    union = (p_end - p_start) + (g_end - g_start) - intersection
    return intersection / union if union > 0 else 0.0


def compute_document_level_metrics(
    y_true_binary: List[int],
    y_pred_binary: List[int],
) -> Dict[str, float]:
    """
    Computes document-level classification metrics:
    y = 1 if document contains injection, 0 if clean benign.
    """
    tp = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 1 and yp == 0)

    acc = (tp + tn) / max(len(y_true_binary), 1)
    prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / max(prec + rec, 1e-8) if (prec + rec) > 0 else 0.0
    fpr = fp / max(fp + tn, 1) if (fp + tn) > 0 else 0.0

    return {
        "doc_accuracy": acc,
        "doc_precision": prec,
        "doc_recall": rec,
        "doc_f1": f1,
        "doc_fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
