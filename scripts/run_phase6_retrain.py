"""
scripts/run_phase6_retrain.py

Phase 6: Retraining with Augmented Data
────────────────────────────────────────
Retrain the student model using the augmented dataset
(original + generated hard negatives).

Usage:
    python scripts/run_phase6_retrain.py --dataset-version 1
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_model
from src.training.train import run_training
from src.utils.config import load_config
from src.utils.experiment import get_dataset_version_dir, init_experiment
from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6: Retraining with Augmented Data"
    )
    parser.add_argument(
        "--student-config", type=Path,
        default=_PROJECT_ROOT / "configs" / "student.yaml",
    )
    parser.add_argument(
        "--distill-config", type=Path,
        default=_PROJECT_ROOT / "configs" / "distillation.yaml",
    )
    parser.add_argument(
        "--data-config", type=Path,
        default=_PROJECT_ROOT / "configs" / "data.yaml",
    )
    parser.add_argument(
        "--dataset-version", type=int, required=True,
        help="Augmented dataset version to train on.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    # Setup
    setup_logging(log_file=_PROJECT_ROOT / "logs" / "phase6.log")
    student_config = load_config(args.student_config)
    distill_config = load_config(args.distill_config)
    data_config = load_config(args.data_config)
    
    data_config["seed"] = args.seed
    student_config["seed"] = args.seed
    set_seed(args.seed)

    logger.info("━" * 60)
    logger.info("PHASE 6: Retraining with Augmented Data")
    logger.info(f"  Dataset version: {args.dataset_version}")
    logger.info(f"  Seed: {args.seed}")
    logger.info("━" * 60)

    # Paths
    processed_dir = _PROJECT_ROOT / data_config.get("processed_dir", "data/processed")
    dataset_dir = get_dataset_version_dir(processed_dir, args.dataset_version)
    experiments_dir = _PROJECT_ROOT / student_config.get("experiments_dir", "experiments")

    train_path = dataset_dir / "train.csv"
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error(f"Run Phase 5 first to create dataset_v{args.dataset_version}.")
        sys.exit(1)

    eval_path = dataset_dir / "validation.csv"
    if not eval_path.exists():
        eval_path = dataset_dir / "test.csv"
    if not eval_path.exists():
        eval_path = None

    # Teacher annotations path (reuse from Phase 2 + new annotations for generated data)
    teacher_path = None
    if distill_config.get("loss_type") == "kd":
        teacher_path = _PROJECT_ROOT / data_config.get("teacher_annotations_path", "data/processed/teacher_annotations.jsonl")
        if not teacher_path.exists():
            logger.warning("Teacher annotations not found — falling back to CE loss.")
            distill_config["loss_type"] = "ce"
            teacher_path = None

    # Init experiment
    full_config = {
        "student": student_config,
        "distillation": distill_config,
        "data": data_config,
        "dataset_version": args.dataset_version,
    }
    exp_dir = init_experiment(
        experiments_dir,
        config=full_config,
        phase="phase6_retrain",
        description=f"Retraining on augmented dataset_v{args.dataset_version}",
    )

    # Run training
    final_state = run_training(
        train_data_path=train_path,
        eval_data_path=eval_path,
        student_config=student_config,
        distillation_config=distill_config,
        experiment_dir=exp_dir,
        teacher_annotations_path=teacher_path,
        eval_fn=evaluate_model,
    )

    logger.info("━" * 60)
    logger.info(f"Phase 6 complete. Experiment: {exp_dir}")
    logger.info("━" * 60)


if __name__ == "__main__":
    main()
