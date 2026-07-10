"""
scripts/run_kd_dry_run.py

Task 6: Knowledge Distillation Dry Run
───────────────────────────────────────
Runs ONE optimization step using:
- REAL teacher outputs from teacher_dry_run.jsonl
- ModernBERT student model
- KD loss function

Verifies: KD loss, CE loss, combined loss.

Must be run AFTER run_teacher_dry_run.py.

Usage:
    python scripts/run_kd_dry_run.py

Outputs: reports/kd_dry_run.json
"""

import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    from src.utils.logging import setup_logging, get_logger
    setup_logging()
    logger = get_logger(__name__)

    logger.info("━" * 60)
    logger.info("KD DRY RUN — Single Optimization Step")
    logger.info("━" * 60)

    import torch
    from src.utils.reproducibility import set_seed
    set_seed()
    device = torch.device("cpu")

    reports_dir = _PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Verify teacher dry run output exists ──────────────────
    teacher_path = reports_dir / "teacher_dry_run.jsonl"
    if not teacher_path.exists():
        logger.error(f"Teacher dry run output not found: {teacher_path}")
        logger.error("Run scripts/run_teacher_dry_run.py first.")
        sys.exit(1)

    from src.utils.io import read_jsonl
    teacher_annotations = read_jsonl(teacher_path)
    annotated_ids = {a["id"] for a in teacher_annotations}
    logger.info(f"Loaded {len(teacher_annotations)} REAL teacher annotations")

    # ── Step 2: Load training data (only samples that have annotations) ──
    from src.utils.io import read_csv
    import pandas as pd

    dataset_dir = _PROJECT_ROOT / "data" / "processed" / "dataset_v0"
    train_df = read_csv(dataset_dir / "train.csv")

    # Filter to only annotated samples
    train_df = train_df[train_df["id"].isin(annotated_ids)].reset_index(drop=True)
    if len(train_df) == 0:
        logger.error("No matching samples between train.csv and teacher annotations.")
        sys.exit(1)

    # Write temp file for IPIDataset
    smoke_dir = _PROJECT_ROOT / "outputs" / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(smoke_dir / "kd_smoke.csv", index=False)

    logger.info(f"Using {len(train_df)} samples with real teacher annotations")

    # ── Step 3: Load model ────────────────────────────────────────────
    from src.models.student import create_student
    from src.utils.config import load_config

    student_config = load_config(_PROJECT_ROOT / "configs" / "student.yaml")
    model, tokenizer = create_student(
        model_id=student_config.get("model_id", "answerdotai/ModernBERT-base")
    )
    model.to(device)
    model.train()

    # ── Step 4: Create dataset WITH real teacher annotations ──────────
    from src.datasets.dataset import IPIDataset
    from torch.utils.data import DataLoader

    train_dataset = IPIDataset(
        data_path=smoke_dir / "kd_smoke.csv",
        tokenizer=tokenizer,
        max_length=student_config.get("max_length", 512),
        teacher_annotations_path=teacher_path,
    )

    train_loader = DataLoader(train_dataset, batch_size=len(train_df), shuffle=False)

    # ── Step 5: Create KD loss function ───────────────────────────────
    from src.training.losses import DistillationLoss, CrossEntropyLoss

    # Use reasonable defaults for the dry run
    kd_temperature = 4.0
    kd_alpha = 0.7

    kd_loss_fn = DistillationLoss(temperature=kd_temperature, alpha=kd_alpha)
    ce_loss_fn = CrossEntropyLoss()

    # ── Step 6: Single forward + backward pass ────────────────────────
    start_time = time.time()
    batch = next(iter(train_loader))

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # CE loss (baseline)
    ce_loss = ce_loss_fn(student_logits=logits, labels=labels)

    # KD loss (requires teacher probs)
    kd_result = {}
    if "teacher_probs" in batch:
        teacher_probs = batch["teacher_probs"].to(device)
        kd_loss = kd_loss_fn(
            student_logits=logits,
            labels=labels,
            teacher_probs=teacher_probs,
        )
        kd_loss.backward()

        # Verify gradients
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)

        kd_result = {
            "ce_loss": round(ce_loss.item(), 6),
            "kd_loss": round(kd_loss.item(), 6),
            "temperature": kd_temperature,
            "alpha": kd_alpha,
            "gradients_computed": has_grads,
            "teacher_probs_present": True,
        }
        logger.info(f"  CE loss:  {ce_loss.item():.6f}")
        logger.info(f"  KD loss:  {kd_loss.item():.6f}")
        logger.info(f"  Gradients: {'✓' if has_grads else '✗'}")
    else:
        # Teacher probs not merged — likely annotation format issue
        ce_loss.backward()
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)

        kd_result = {
            "ce_loss": round(ce_loss.item(), 6),
            "kd_loss": None,
            "temperature": kd_temperature,
            "alpha": kd_alpha,
            "gradients_computed": has_grads,
            "teacher_probs_present": False,
            "warning": "Teacher probs not merged into batch. Check annotation format.",
        }
        logger.warning("Teacher probs not found in batch. KD loss not computed.")

    elapsed = time.time() - start_time

    # ── Build report ──────────────────────────────────────────────────
    report = {
        "task": "KD Dry Run",
        "elapsed_seconds": round(elapsed, 2),
        "samples_used": len(train_df),
        "student_model": student_config.get("model_id"),
        "results": kd_result,
        "passed": kd_result.get("gradients_computed", False)
                  and kd_result.get("teacher_probs_present", False),
    }

    report_path = reports_dir / "kd_dry_run.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    if report["passed"]:
        logger.info("━" * 60)
        logger.info("✅ KD DRY RUN PASSED — Real teacher→student pipeline verified.")
        logger.info(f"  Report: {report_path}")
        logger.info("━" * 60)
    else:
        logger.error("━" * 60)
        logger.error("❌ KD DRY RUN FAILED — See report for details.")
        logger.error("━" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
