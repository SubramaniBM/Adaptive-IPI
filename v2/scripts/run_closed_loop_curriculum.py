"""
v2/scripts/run_closed_loop_curriculum.py

Closed-Loop Adaptive Adversarial Hard-Negative Curriculum Orchestrator.
Realizes the true "Adaptive-IPI" vision:
Iterative closed-loop between an automated Red-Teaming Adversary and
an Uncertainty/Entropy Hard-Negative Miner, driving Attack Success Rate (ASR)
to near-zero while keeping Benign False Positive Rate (FPR) minimal.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from v2.src.adversarial.advanced_red_teamer import AdvancedRedTeamer
from v2.src.adversarial.hard_negative_miner import HardNegativeMiner
from v2.src.datasets.span_dataset import FastSpanIPIDataset, DynamicSpanCollator


def evaluate_model_on_suite(
    model,
    tokenizer,
    device: torch.device,
    eval_suite: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluates detector on a test suite using HardNegativeMiner metrics."""
    miner = HardNegativeMiner(model, tokenizer, device, threshold=threshold)
    _, metrics = miner.evaluate_candidates(eval_suite, batch_size=16)
    return metrics


def fine_tune_round(
    model,
    tokenizer,
    train_df: pd.DataFrame,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 3e-5,
) -> float:
    """Performs rapid adaptive fine-tuning on harvested hard negatives."""
    dataset = FastSpanIPIDataset(train_df, tokenizer, max_length=256)
    collator = DynamicSpanCollator(
        pad_token_id=tokenizer.pad_token_id or 0,
        pad_label_id=-100,
        max_length=256,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    total_loss = 0.0

    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss


def main():
    print("=" * 70)
    print("  ADAPTIVE-IPI: CLOSED-LOOP ADVERSARIAL CURRICULUM ORCHESTRATOR  ")
    print("=" * 70)

    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Compute Device: Apple Silicon MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Compute Device: NVIDIA CUDA")
    else:
        device = torch.device("cpu")
        print("Compute Device: Standard CPU")

    # 2. Paths
    checkpoints_dir = _PROJECT_ROOT / "v2" / "checkpoints"
    base_ckpt = checkpoints_dir / "span_modernbert_hardened" / "best"
    if not base_ckpt.exists():
        base_ckpt = checkpoints_dir / "span_modernbert_robust" / "best"
    if not base_ckpt.exists():
        raise FileNotFoundError(f"Base checkpoint not found at {base_ckpt}")

    output_ckpt_dir = checkpoints_dir / "adaptive_curriculum"
    output_ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_dir = _PROJECT_ROOT / "v2" / "data" / "processed"

    # Load replay training buffer from train_hardened.csv or train_robust.csv
    replay_file = data_dir / "train_hardened.csv"
    if not replay_file.exists():
        replay_file = data_dir / "train_robust.csv"
    replay_df = pd.read_csv(replay_file).sample(n=min(200, len(pd.read_csv(replay_file))), random_state=42)

    # 3. Load Tokenizer & Base Model
    print(f"\nLoading baseline model from: {base_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(str(base_ckpt))
    model = AutoModelForTokenClassification.from_pretrained(str(base_ckpt))
    model.to(device)

    # 4. Generate Constant Held-Out Evaluation Suite (120 samples)
    print("\nSynthesizing held-out adversarial evaluation benchmark (120 samples)...")
    red_teamer = AdvancedRedTeamer()
    eval_suite = red_teamer.generate_adversarial_suite(num_attacks=60, num_hard_benign=60)

    # 5. Round 0 Evaluation
    print("\n[ROUND 0] Evaluating Baseline Hardened Model against Red-Teaming Suite...")
    r0_metrics = evaluate_model_on_suite(model, tokenizer, device, eval_suite)
    print(f"  -> Attack Success Rate (ASR) : {r0_metrics['attack_success_rate']:.2f}% (lower is better)")
    print(f"  -> Benign False Alarm (FPR)  : {r0_metrics['false_positive_rate']:.2f}% (lower is better)")
    print(f"  -> Mean Span IoU             : {r0_metrics['mean_iou']:.2f}% (higher is better)")
    print(f"  -> Mean Predictive Entropy   : {r0_metrics['mean_entropy']:.4f}")

    convergence_history = [{
        "round": 0,
        "name": "Baseline (Static Hardened)",
        "asr": r0_metrics["attack_success_rate"],
        "fpr": r0_metrics["false_positive_rate"],
        "mean_iou": r0_metrics["mean_iou"],
        "entropy": r0_metrics["mean_entropy"],
    }]

    # 6. Curriculum Iterations
    num_rounds = 2
    for r in range(1, num_rounds + 1):
        print(f"\n" + "-" * 70)
        print(f"  STARTING ADAPTIVE CURRICULUM ROUND {r}/{num_rounds}")
        print("-" * 70)

        # A. Red-Teamer generates candidate stream
        print(f"[Round {r}] Generating candidate adversarial pool (250 samples)...")
        candidate_pool = red_teamer.generate_adversarial_suite(
            num_attacks=125,
            num_hard_benign=125,
        )

        # B. HardNegativeMiner isolates hard edge cases
        print(f"[Round {r}] Mining hardest evasions and high-entropy edge cases...")
        miner = HardNegativeMiner(model, tokenizer, device, threshold=0.5)
        hard_samples, pool_metrics = miner.mine_hard_negatives(candidate_pool, top_k=100)
        print(f"  -> Top 100 Hard samples mined (Pool ASR before adaptation: {pool_metrics['attack_success_rate']:.1f}%)")

        # C. Construct curriculum training dataset
        mined_rows = []
        for s in hard_samples:
            mined_rows.append({
                "context": s["text"],
                "label": s["label"],
                "attack_instruction": s["payload"] if s["label"] == 1 else "",
            })
        mined_df = pd.DataFrame(mined_rows)

        # Mix with replay buffer (to preserve general language ability)
        curriculum_df = pd.concat([mined_df, replay_df[["context", "label", "attack_instruction"]]], ignore_index=True)
        curriculum_df = curriculum_df.sample(frac=1.0, random_state=42 + r).reset_index(drop=True)
        curriculum_csv = data_dir / f"curriculum_round_{r}.csv"
        curriculum_df.to_csv(curriculum_csv, index=False)
        print(f"  -> Created adaptive training split: {len(curriculum_df)} samples ({curriculum_csv.name})")

        # D. Fine-tune on hard curriculum
        lr = 3e-5 if r == 1 else 2e-5
        print(f"[Round {r}] Fine-tuning model on hard curriculum (lr={lr}, 1 epoch)...")
        train_loss = fine_tune_round(
            model=model,
            tokenizer=tokenizer,
            train_df=curriculum_df,
            device=device,
            epochs=1,
            batch_size=16,
            lr=lr,
        )
        print(f"  -> Training Loss: {train_loss:.4f}")

        # E. Save Round Checkpoint
        r_save_dir = output_ckpt_dir / f"round_{r}"
        r_save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(r_save_dir))
        tokenizer.save_pretrained(str(r_save_dir))
        print(f"  -> Saved round checkpoint: {r_save_dir.relative_to(_PROJECT_ROOT)}")

        # F. Re-evaluate on the constant held-out test suite
        print(f"[Round {r}] Evaluating updated model on held-out benchmark suite...")
        r_metrics = evaluate_model_on_suite(model, tokenizer, device, eval_suite)
        print(f"  -> Attack Success Rate (ASR) : {r_metrics['attack_success_rate']:.2f}% (lower is better)")
        print(f"  -> Benign False Alarm (FPR)  : {r_metrics['false_positive_rate']:.2f}% (lower is better)")
        print(f"  -> Mean Span IoU             : {r_metrics['mean_iou']:.2f}% (higher is better)")
        print(f"  -> Mean Predictive Entropy   : {r_metrics['mean_entropy']:.4f}")

        convergence_history.append({
            "round": r,
            "name": f"Adaptive Round {r}",
            "asr": r_metrics["attack_success_rate"],
            "fpr": r_metrics["false_positive_rate"],
            "mean_iou": r_metrics["mean_iou"],
            "entropy": r_metrics["mean_entropy"],
        })

    # 7. Copy Best Model to final checkpoint directory
    final_best_dir = checkpoints_dir / "span_modernbert_adaptive" / "best"
    final_best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_best_dir))
    tokenizer.save_pretrained(str(final_best_dir))
    print(f"\nFinal Best Adaptive Model saved to: {final_best_dir.relative_to(_PROJECT_ROOT)}")

    # 8. Save Convergence Metrics JSON
    results_path = data_dir / "curriculum_convergence.json"
    with open(results_path, "w") as f:
        json.dump(convergence_history, f, indent=2)
    print(f"Convergence history written to: {results_path.relative_to(_PROJECT_ROOT)}")

    # 9. Print Final Summary Table
    print("\n" + "=" * 70)
    print("        ADAPTIVE-IPI CURRICULUM CONVERGENCE RESULTS        ")
    print("=" * 70)
    print(f"{'Round':<18} | {'ASR (%)':<10} | {'IoU (%)':<10} | {'Benign FPR (%)':<16} | {'Entropy':<10}")
    print("-" * 70)
    for entry in convergence_history:
        print(
            f"{entry['name']:<18} | "
            f"{entry['asr']:>9.2f}% | "
            f"{entry['mean_iou']:>9.2f}% | "
            f"{entry['fpr']:>15.2f}% | "
            f"{entry['entropy']:>9.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
