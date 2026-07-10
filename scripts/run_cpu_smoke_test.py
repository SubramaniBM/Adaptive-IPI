"""
scripts/run_cpu_smoke_test.py

Task 2: CPU Smoke Test
──────────────────────
Runs a miniature end-to-end training experiment on CPU using ~100 samples.
Verifies: dataset loading, tokenizer, forward pass, loss, backprop,
checkpoint saving, and evaluation loop.

All overrides are passed via CLI — production configs are NEVER modified.

Usage:
    python scripts/run_cpu_smoke_test.py
    python scripts/run_cpu_smoke_test.py --epochs 1 --batch-size 2 --samples 100

Outputs: reports/cpu_smoke_test.md
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU Smoke Test")
    parser.add_argument("--config", type=Path,
                        default=_PROJECT_ROOT / "configs" / "student.yaml")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Override num_epochs for smoke test (default: 1)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Override batch_size for smoke test (default: 2)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of training samples to use (default: 100)")
    parser.add_argument("--eval-samples", type=int, default=20,
                        help="Number of eval samples to use (default: 20)")
    args = parser.parse_args()

    from src.utils.logging import setup_logging, get_logger
    setup_logging()
    logger = get_logger(__name__)

    results = {
        "dataset_loads": False,
        "tokenizer_loads": False,
        "forward_pass": False,
        "loss_computes": False,
        "backpropagation": False,
        "checkpoint_saving": False,
        "evaluation_loop": False,
    }
    errors = []
    start_time = time.time()

    try:
        import torch
        from src.utils.config import load_config
        from src.utils.reproducibility import set_seed

        set_seed()
        device = torch.device("cpu")

        # Load student config from YAML (read-only), then apply CLI overrides IN MEMORY
        student_config = load_config(args.config)
        student_config["num_epochs"] = args.epochs
        student_config["batch_size"] = args.batch_size

        logger.info("━" * 60)
        logger.info("CPU SMOKE TEST")
        logger.info(f"  Model: {student_config['model_id']}")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Training samples: {args.samples}")
        logger.info(f"  Eval samples: {args.eval_samples}")
        logger.info("━" * 60)

        # ── Step 1: Dataset Loading ───────────────────────────────────
        logger.info("Step 1: Loading dataset...")
        from src.utils.io import read_csv
        import pandas as pd

        dataset_dir = _PROJECT_ROOT / "data" / "processed" / "dataset_v0"
        train_df = read_csv(dataset_dir / "train.csv")
        val_df = read_csv(dataset_dir / "validation.csv")

        # Subsample
        train_df = train_df.head(args.samples)
        val_df = val_df.head(args.eval_samples)

        # Write temp subsampled CSVs for IPIDataset
        smoke_dir = _PROJECT_ROOT / "outputs" / "smoke_test"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(smoke_dir / "train_smoke.csv", index=False)
        val_df.to_csv(smoke_dir / "val_smoke.csv", index=False)

        results["dataset_loads"] = True
        logger.info(f"  ✓ Dataset loaded: {len(train_df)} train, {len(val_df)} val")

        # ── Step 2: Tokenizer & Model Loading ─────────────────────────
        logger.info("Step 2: Loading tokenizer and model...")
        from src.models.student import create_student

        model, tokenizer = create_student(
            model_id=student_config.get("model_id", "answerdotai/ModernBERT-base")
        )
        results["tokenizer_loads"] = True
        logger.info("  ✓ Tokenizer and model loaded")

        # ── Step 3: Create datasets and dataloaders ───────────────────
        logger.info("Step 3: Creating datasets...")
        from src.datasets.dataset import IPIDataset
        from torch.utils.data import DataLoader

        train_dataset = IPIDataset(
            data_path=smoke_dir / "train_smoke.csv",
            tokenizer=tokenizer,
            max_length=student_config.get("max_length", 512),
        )
        val_dataset = IPIDataset(
            data_path=smoke_dir / "val_smoke.csv",
            tokenizer=tokenizer,
            max_length=student_config.get("max_length", 512),
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # ── Step 4: Forward pass ──────────────────────────────────────
        logger.info("Step 4: Testing forward pass...")
        model.to(device)
        model.train()

        batch = next(iter(train_loader))
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        assert logits.shape[-1] == 2, f"Expected 2 logits, got {logits.shape[-1]}"
        results["forward_pass"] = True
        logger.info(f"  ✓ Forward pass succeeded. Logits shape: {logits.shape}")

        # ── Step 5: Loss computation ──────────────────────────────────
        logger.info("Step 5: Testing loss computation...")
        from src.training.losses import create_loss_function

        loss_fn = create_loss_function({"loss_type": "ce"})
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(student_logits=outputs.logits, labels=labels)
        assert loss.item() > 0, "Loss should be positive"
        results["loss_computes"] = True
        logger.info(f"  ✓ Loss computed: {loss.item():.4f}")

        # ── Step 6: Backpropagation ───────────────────────────────────
        logger.info("Step 6: Testing backpropagation...")
        loss.backward()

        # Verify gradients exist
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grads, "No gradients computed"
        results["backpropagation"] = True
        logger.info("  ✓ Backpropagation succeeded. Gradients computed.")

        # ── Step 7: Checkpoint saving ─────────────────────────────────
        logger.info("Step 7: Testing checkpoint saving...")
        from src.models.student import save_student_checkpoint

        ckpt_dir = smoke_dir / "checkpoint_smoke"
        save_student_checkpoint(model, tokenizer, ckpt_dir)
        assert (ckpt_dir / "config.json").exists(), "Checkpoint config missing"
        results["checkpoint_saving"] = True
        logger.info(f"  ✓ Checkpoint saved to {ckpt_dir}")

        # ── Step 8: Evaluation loop ───────────────────────────────────
        logger.info("Step 8: Testing evaluation loop...")
        from src.evaluation.evaluator import evaluate_model

        model.eval()
        eval_metrics = evaluate_model(model, val_loader, device)
        assert "f1" in eval_metrics, "Missing F1 in eval metrics"
        assert "accuracy" in eval_metrics, "Missing accuracy in eval metrics"
        results["evaluation_loop"] = True
        logger.info(f"  ✓ Evaluation loop succeeded. Metrics: {eval_metrics}")

    except Exception as e:
        errors.append(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        logger.error(f"Smoke test encountered error: {e}")

    elapsed = time.time() - start_time
    all_passed = all(results.values())

    # ── Generate report ───────────────────────────────────────────────
    reports_dir = _PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# CPU Smoke Test Report\n",
        f"**Status**: {'✅ ALL PASSED' if all_passed else '❌ SOME CHECKS FAILED'}\n",
        f"**Elapsed**: {elapsed:.1f}s\n",
        "",
        "| Check | Result |",
        "| :--- | :--- |",
    ]
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        lines.append(f"| {check} | {status} |")

    if errors:
        lines.append("\n## Errors\n")
        for err in errors:
            lines.append(f"```\n{err}\n```\n")

    report_path = reports_dir / "cpu_smoke_test.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved to {report_path}")

    if all_passed:
        logger.info("━" * 60)
        logger.info("✅ CPU SMOKE TEST PASSED — All checks green.")
        logger.info("━" * 60)
    else:
        logger.error("━" * 60)
        logger.error("❌ CPU SMOKE TEST FAILED — See report for details.")
        logger.error("━" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
