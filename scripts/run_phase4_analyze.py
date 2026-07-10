"""
scripts/run_phase4_analyze.py

Phase 4: Validation & Failure Analysis
───────────────────────────────────────
Load the trained student checkpoint, run inference on the validation
set, identify failures, and generate a failure report.

Usage:
    python scripts/run_phase4_analyze.py --checkpoint experiments/exp001/checkpoint/best
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from torch.utils.data import DataLoader

from src.adaptive.failure_analysis import (
    generate_failure_report,
    identify_failures,
    run_inference,
)
from src.datasets.dataset import IPIDataset
from src.models.student import load_student_checkpoint
from src.utils.config import load_config
from src.utils.experiment import get_dataset_version_dir, init_experiment
from src.utils.io import read_csv
from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import get_device, set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: Failure Analysis"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to student model checkpoint directory.",
    )
    parser.add_argument(
        "--data-config", type=Path,
        default=_PROJECT_ROOT / "configs" / "data.yaml",
    )
    parser.add_argument(
        "--adaptive-config", type=Path,
        default=_PROJECT_ROOT / "configs" / "adaptive.yaml",
    )
    parser.add_argument(
        "--dataset-version", type=int, default=0,
    )
    parser.add_argument(
        "--split", type=str, default="validation",
        help="Which split to analyse: 'validation' or 'test'.",
    )
    args = parser.parse_args()

    # Setup
    setup_logging(log_file=_PROJECT_ROOT / "logs" / "phase4.log")
    data_config = load_config(args.data_config)
    adaptive_config = load_config(args.adaptive_config)
    set_seed(data_config.get("seed", 42))
    device = get_device()

    logger.info("━" * 60)
    logger.info("PHASE 4: Failure Analysis")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Split: {args.split}")
    logger.info("━" * 60)

    # Load model
    model, tokenizer = load_student_checkpoint(args.checkpoint)
    model = model.to(device)

    # Load data
    processed_dir = _PROJECT_ROOT / data_config.get("processed_dir", "data/processed")
    dataset_dir = get_dataset_version_dir(processed_dir, args.dataset_version)
    split_path = dataset_dir / f"{args.split}.csv"

    if not split_path.exists():
        logger.error(f"Split file not found: {split_path}")
        sys.exit(1)

    eval_df = read_csv(split_path)
    eval_dataset = IPIDataset(split_path, tokenizer, max_length=512)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    # Run inference
    predictions_df = run_inference(model, eval_dataloader, device)

    # Identify failures
    confidence_threshold = adaptive_config.get("failure_analysis", {}).get(
        "confidence_threshold", 0.7
    )
    failures = identify_failures(eval_df, predictions_df, confidence_threshold)

    # Generate report
    experiments_dir = _PROJECT_ROOT / "experiments"
    exp_dir = init_experiment(
        experiments_dir,
        config={"data": data_config, "adaptive": adaptive_config},
        phase="phase4_analyze",
        description=f"Failure analysis on {args.split} split",
    )

    report = generate_failure_report(failures, exp_dir)

    logger.info("━" * 60)
    logger.info(f"Phase 4 complete. Report: {exp_dir}")
    logger.info(f"  Total failures: {report['total_failures']}")
    for ft, count in report.get("by_failure_type", {}).items():
        logger.info(f"    {ft}: {count:,}")
    logger.info("━" * 60)


if __name__ == "__main__":
    main()
