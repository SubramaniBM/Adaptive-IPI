"""
scripts/run_teacher_dry_run.py

Task 5: Teacher Annotation Dry Run
────────────────────────────────────
Runs the REAL teacher annotation pipeline on exactly 5 samples.
Uses the actual Transformers backend with a small local model
to verify the full pipeline: dataset → teacher → annotation → saved JSON.

Usage:
    python scripts/run_teacher_dry_run.py
    python scripts/run_teacher_dry_run.py --model Qwen/Qwen2.5-0.5B-Instruct --samples 5

Outputs: reports/teacher_dry_run.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher Annotation Dry Run")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Teacher model for dry run. Use a small model for CPU testing.",
    )
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of samples to annotate (default: 5)")
    parser.add_argument("--config", type=Path,
                        default=_PROJECT_ROOT / "configs" / "teacher.yaml")
    args = parser.parse_args()

    from src.utils.logging import setup_logging, get_logger
    setup_logging()
    logger = get_logger(__name__)

    logger.info("━" * 60)
    logger.info("TEACHER ANNOTATION DRY RUN")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Samples: {args.samples}")
    logger.info("━" * 60)

    from src.utils.config import load_config
    from src.utils.io import read_csv
    from src.utils.reproducibility import set_seed

    set_seed()

    # Load real config, override model for CPU feasibility
    teacher_config = load_config(args.config)
    teacher_config["model_id"] = args.model
    teacher_config["backend"] = "transformers"  # CPU-compatible
    teacher_config["backend_kwargs"] = {}        # Clear vLLM-specific kwargs
    teacher_config["batch_size"] = 1
    teacher_config["max_tokens"] = 512

    # Load REAL training data
    dataset_dir = _PROJECT_ROOT / "data" / "processed" / "dataset_v0"
    train_df = read_csv(dataset_dir / "train.csv")

    # Take exactly N samples (mix of benign and attack)
    benign = train_df[train_df["label"] == 0].head(args.samples // 2 + 1)
    attack = train_df[train_df["label"] == 1].head(args.samples // 2 + 1)
    import pandas as pd
    sample_df = pd.concat([benign, attack]).head(args.samples).reset_index(drop=True)

    # Bridge column: annotator expects 'text', CSV has 'context'
    if "text" not in sample_df.columns and "context" in sample_df.columns:
        sample_df = sample_df.rename(columns={"context": "text"})

    logger.info(f"Selected {len(sample_df)} samples ({sample_df['label'].value_counts().to_dict()})")

    # Run REAL annotation pipeline
    reports_dir = _PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "teacher_dry_run.jsonl"

    # Clean any prior run
    if output_path.exists():
        output_path.unlink()

    from src.teacher.annotator import create_annotator

    start_time = time.time()
    annotator = create_annotator(teacher_config, output_path)
    annotations = annotator.annotate(sample_df, resume=False)
    elapsed = time.time() - start_time

    # Build structured report
    report = {
        "model": args.model,
        "backend": "transformers",
        "samples_requested": args.samples,
        "samples_annotated": len(annotations),
        "elapsed_seconds": round(elapsed, 1),
        "annotations": [],
    }

    import dataclasses
    for ann in annotations:
        ann_dict = dataclasses.asdict(ann)
        report["annotations"].append(ann_dict)

    # Verify required fields
    checks = {
        "has_predictions": all(
            a["teacher_prediction"] in (0, 1) for a in report["annotations"]
        ),
        "has_probabilities": all(
            isinstance(a["teacher_probs"], list) and len(a["teacher_probs"]) == 2
            for a in report["annotations"]
        ),
        "has_reasoning": all(
            isinstance(a["teacher_reasoning"], str) and len(a["teacher_reasoning"]) > 0
            for a in report["annotations"]
        ),
        "valid_json_format": len(report["annotations"]) > 0,
    }
    report["checks"] = checks
    report["passed"] = all(checks.values()) and len(annotations) == args.samples

    # Save report
    report_path = reports_dir / "teacher_dry_run.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    if report["passed"]:
        logger.info("━" * 60)
        logger.info("✅ TEACHER DRY RUN PASSED")
        logger.info(f"  {len(annotations)} annotations saved to {output_path}")
        logger.info(f"  Report saved to {report_path}")
        logger.info("━" * 60)
    else:
        logger.error("━" * 60)
        logger.error("❌ TEACHER DRY RUN FAILED — See report for details.")
        logger.error("━" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
