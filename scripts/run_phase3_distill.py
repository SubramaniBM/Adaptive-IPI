"""
scripts/run_phase3_distill.py

Phase 3: Initial Knowledge Distillation
────────────────────────────────────────
Train the ModernBERT student model using teacher soft labels.

Usage:
    python scripts/run_phase3_distill.py
    python scripts/run_phase3_distill.py --loss-type ce    # Baseline (hard labels only)
    python scripts/run_phase3_distill.py --loss-type kd    # Knowledge distillation
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_model
from src.training.train import run_training
from src.utils.config import load_config, load_configs
from src.utils.experiment import get_dataset_version_dir, init_experiment
from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: Initial Knowledge Distillation"
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
        "--loss-type", type=str, default=None,
        help="Override loss type: 'ce' or 'kd'.",
    )
    parser.add_argument(
        "--dataset-version", type=int, default=0,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run even if already completed.",
    )
    args = parser.parse_args()

    # Setup
    setup_logging(log_file=_PROJECT_ROOT / "logs" / "phase3.log")
    student_config = load_config(args.student_config)
    distill_config = load_config(args.distill_config)
    data_config = load_config(args.data_config)
    set_seed(data_config.get("seed", 42))

    # Override loss type if specified
    if args.loss_type:
        distill_config["loss_type"] = args.loss_type

    logger.info("━" * 60)
    logger.info("PHASE 3: Initial Knowledge Distillation")
    logger.info(f"  Loss type: {distill_config.get('loss_type', 'ce')}")
    logger.info(f"  Dataset version: {args.dataset_version}")
    logger.info("━" * 60)

    # Paths
    processed_dir = _PROJECT_ROOT / data_config.get("processed_dir", "data/processed")
    dataset_dir = get_dataset_version_dir(processed_dir, args.dataset_version)
    experiments_dir = _PROJECT_ROOT / student_config.get("experiments_dir", "experiments")

    train_path = dataset_dir / "train.csv"
    # Use validation split if available, otherwise test
    eval_path = dataset_dir / "validation.csv"
    if not eval_path.exists():
        eval_path = dataset_dir / "test.csv"
    if not eval_path.exists():
        eval_path = None

    # Teacher annotations (for KD)
    teacher_path = None
    if distill_config.get("loss_type") in ["kd", "cw"]:
        teacher_path = _PROJECT_ROOT / data_config.get(
            "teacher_annotations_path",
            "data/processed/teacher_annotations.jsonl",
        )
        if not teacher_path.exists():
            logger.error(f"Teacher annotations not found: {teacher_path}")
            logger.error("Run Phase 2 first, or use --loss-type ce for baseline.")
            sys.exit(1)

    # Check for existing experiments to resume or skip
    exp_dirs = sorted(experiments_dir.glob("exp*"))
    exp_dir = None
    resume_from = None

    if exp_dirs:
        latest_exp = exp_dirs[-1]
        metadata_path = latest_exp / "metadata.json"
        is_same_version = False
        if metadata_path.exists():
            import json
            try:
                with open(metadata_path) as f:
                    meta = json.load(f)
                    if meta.get("config", {}).get("dataset_version") == args.dataset_version and meta.get("phase") == "phase3_distill":
                        is_same_version = True
            except:
                pass

        if is_same_version and not args.force:
            if (latest_exp / "checkpoint" / "final").exists():
                logger.info(f"Phase 3 already completed for dataset version {args.dataset_version} in {latest_exp}. Skipping.")
                sys.exit(0)
            elif (latest_exp / "checkpoint" / "latest").exists():
                logger.info(f"Found incomplete Phase 3 in {latest_exp}. Resuming...")
                exp_dir = latest_exp
                resume_from = latest_exp / "checkpoint" / "latest"

    if exp_dir is None:
        # Init experiment (Change #5)
        full_config = {
            "student": student_config,
            "distillation": distill_config,
            "data": data_config,
            "dataset_version": args.dataset_version,
        }
        exp_dir = init_experiment(
            experiments_dir,
            config=full_config,
            phase="phase3_distill",
            description=f"Initial distillation with loss={distill_config.get('loss_type')}",
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
    logger.info(f"Phase 3 complete. Experiment: {exp_dir}")
    logger.info("━" * 60)


if __name__ == "__main__":
    main()
