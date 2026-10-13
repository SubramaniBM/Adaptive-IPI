"""
v2/src/adversarial/hard_negative_miner.py

Uncertainty & Hard-Negative Mining Engine for Closed-Loop Adversarial Training.
Evaluates model confidence, predictive entropy, and decision margins to isolate:
1. False Negatives (Successful adversarial evasions)
2. False Positives (False alarms on legitimate imperative emails)
3. Boundary Edge Cases (High token entropy / tight prediction margins)
"""

import math
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def calculate_span_iou(pred_start: int, pred_end: int, gt_start: int, gt_end: int) -> float:
    """Calculates character-level Intersection-over-Union."""
    if pred_start == -1 or gt_start == -1:
        return 0.0
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    if inter_end <= inter_start:
        return 0.0
    inter = inter_end - inter_start
    union = (pred_end - pred_start) + (gt_end - gt_start) - inter
    return inter / union if union > 0 else 0.0


class HardNegativeMiner:
    """Evaluates candidate text streams and extracts the hardest adversarial evasions and false alarms."""

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        threshold: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold

    def evaluate_candidates(
        self,
        candidates: List[Dict[str, Any]],
        batch_size: int = 16,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Runs candidate samples through the model and computes hardness metrics.
        Returns:
            scored_candidates: list of candidates with computed hardness scores and evaluation results.
            metrics: summary metrics (ASR, FPR, Mean IoU, Mean Entropy).
        """
        self.model.eval()
        scored = []
        attacks_total = 0
        attacks_evaded = 0
        benign_total = 0
        benign_false_positives = 0
        total_iou = 0.0
        all_entropies = []

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            texts = [c["text"] for c in batch]

            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            offset_mappings = encodings.pop("offset_mapping").cpu().numpy()
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [B, L, 3]
                probs = F.softmax(logits, dim=-1)  # [B, L, 3]

            probs_np = probs.cpu().numpy()
            att_np = attention_mask.cpu().numpy()

            for j, cand in enumerate(batch):
                seq_len = int(att_np[j].sum())
                cand_probs = probs_np[j, :seq_len]  # [L, 3]
                offsets = offset_mappings[j, :seq_len]

                # Calculate predictive token entropy: H = -sum(p * log(p))
                p_safe = np.clip(cand_probs, 1e-9, 1.0)
                entropy_per_token = -np.sum(p_safe * np.log(p_safe), axis=-1)  # [L]
                mean_entropy = float(np.mean(entropy_per_token[1:-1])) if seq_len > 2 else 0.0
                all_entropies.append(mean_entropy)

                # Extract predicted spans
                pred_spans = []
                in_span = False
                span_s, span_e = -1, -1

                for t_idx in range(1, seq_len - 1):
                    start_ch, end_ch = offsets[t_idx]
                    if start_ch == end_ch:
                        continue

                    # Probability of IPI = P(B-IPI) + P(I-IPI)
                    p_ipi = cand_probs[t_idx, 1] + cand_probs[t_idx, 2]

                    if p_ipi >= self.threshold:
                        if not in_span:
                            in_span = True
                            span_s = start_ch
                        span_e = end_ch
                    else:
                        if in_span:
                            pred_spans.append((span_s, span_e))
                            in_span = False
                if in_span:
                    pred_spans.append((span_s, span_e))

                # Primary predicted span
                if pred_spans:
                    primary_pred = pred_spans[0]
                else:
                    primary_pred = (-1, -1)

                is_attack = cand["label"] == 1
                gt_start = cand.get("span_start", -1)
                gt_end = cand.get("span_end", -1)

                if is_attack:
                    attacks_total += 1
                    iou = calculate_span_iou(primary_pred[0], primary_pred[1], gt_start, gt_end)
                    total_iou += iou

                    # Evasion condition: model missed span completely or IoU < 0.5
                    evaded = iou < 0.5
                    if evaded:
                        attacks_evaded += 1
                        # High hardness for evasions
                        hardness = 1.0 + (1.0 - iou) + (0.5 * mean_entropy)
                    else:
                        # Hardness proportional to boundary uncertainty
                        hardness = (1.0 - iou) + (0.3 * mean_entropy)

                    cand_result = dict(cand)
                    cand_result.update({
                        "pred_span": primary_pred,
                        "iou": iou,
                        "evaded": evaded,
                        "false_alarm": False,
                        "mean_entropy": mean_entropy,
                        "hardness": float(hardness),
                    })
                    scored.append(cand_result)

                else:
                    benign_total += 1
                    # Benign sample: any predicted span is a False Positive
                    false_alarm = len(pred_spans) > 0
                    if false_alarm:
                        benign_false_positives += 1
                        # Maximum hardness for false alarms
                        max_p_ipi = float(np.max(cand_probs[1:-1, 1] + cand_probs[1:-1, 2]))
                        hardness = 2.0 + max_p_ipi + (0.5 * mean_entropy)
                    else:
                        # Near-misses have higher p_ipi
                        max_p_ipi = float(np.max(cand_probs[1:-1, 1] + cand_probs[1:-1, 2])) if seq_len > 2 else 0.0
                        hardness = max_p_ipi + (0.2 * mean_entropy)

                    cand_result = dict(cand)
                    cand_result.update({
                        "pred_span": primary_pred,
                        "iou": 0.0,
                        "evaded": False,
                        "false_alarm": false_alarm,
                        "mean_entropy": mean_entropy,
                        "hardness": float(hardness),
                    })
                    scored.append(cand_result)

        # Compute summary metrics
        asr = (attacks_evaded / attacks_total * 100.0) if attacks_total > 0 else 0.0
        fpr = (benign_false_positives / benign_total * 100.0) if benign_total > 0 else 0.0
        mean_iou = (total_iou / attacks_total * 100.0) if attacks_total > 0 else 0.0
        avg_entropy = float(np.mean(all_entropies)) if all_entropies else 0.0

        summary_metrics = {
            "attack_success_rate": asr,
            "false_positive_rate": fpr,
            "mean_iou": mean_iou,
            "mean_entropy": avg_entropy,
            "total_attacks": attacks_total,
            "attacks_evaded": attacks_evaded,
            "total_benign": benign_total,
            "benign_false_alarms": benign_false_positives,
        }

        # Sort candidates descending by hardness
        scored.sort(key=lambda x: x["hardness"], reverse=True)
        return scored, summary_metrics

    def mine_hard_negatives(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int = 150,
        batch_size: int = 16,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Extracts top_k hardest samples (both evasions and false alarms)."""
        scored, metrics = self.evaluate_candidates(candidates, batch_size=batch_size)
        return scored[:top_k], metrics
