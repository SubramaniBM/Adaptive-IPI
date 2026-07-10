"""
scripts/run_phase2_teacher.py

Phase 2: Teacher Annotation
────────────────────────────
Run the teacher model (Qwen3-32B-Instruct) on the training set
to produce rich annotations: prediction, probabilities, entropy,
and reasoning.

Usage:
    python scripts/run_phase2_teacher.py
    python scripts/run_phase2_teacher.py --backend vllm
    python scripts/run_phase2_teacher.py --resume
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.teacher.annotator import create_annotator
from src.utils.config import load_config, load_configs
from src.utils.experiment import get_dataset_version_dir
from src.utils.io import read_csv
from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Teacher Annotation"
    )
    parser.add_argument(
        "--config", type=Path, default=_PROJECT_ROOT / "configs" / "teacher.yaml",
        help="Path to teacher configuration YAML.",
    )
    parser.add_argument(
        "--data-config", type=Path, default=_PROJECT_ROOT / "configs" / "data.yaml",
        help="Path to data configuration YAML.",
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        help="Override inference backend (vllm, transformers, api).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing annotations.",
    )
    parser.add_argument(
        "--dataset-version", type=int, default=0,
        help="Dataset version to annotate.",
    )
    args = parser.parse_args()

    # Setup
    setup_logging(log_file=_PROJECT_ROOT / "logs" / "phase2.log")
    teacher_config = load_config(args.config)
    data_config = load_config(args.data_config)
    set_seed(data_config.get("seed", 42))

    # Override backend if specified
    if args.backend:
        teacher_config["backend"] = args.backend

    logger.info("━" * 60)
    logger.info("PHASE 2: Teacher Annotation")
    logger.info(f"  Model: {teacher_config['model_id']}")
    logger.info(f"  Backend: {teacher_config['backend']}")
    logger.info("━" * 60)

    # Load training data
    processed_dir = _PROJECT_ROOT / data_config.get("processed_dir", "data/processed")
    dataset_dir = get_dataset_version_dir(processed_dir, args.dataset_version)
    train_path = dataset_dir / "train.csv"

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Run Phase 1 first: python scripts/run_phase1_preprocess.py")
        sys.exit(1)

    train_df = read_csv(train_path)
    logger.info(f"Loaded {len(train_df):,} training samples")

    # Create annotator and run
    output_path = _PROJECT_ROOT / teacher_config.get(
        "annotations_path", "data/processed/teacher_annotations.jsonl"
    )
    annotator = create_annotator(teacher_config, output_path)
    annotations = annotator.annotate(train_df, resume=args.resume)

    logger.info("━" * 60)
    logger.info(f"Phase 2 complete. {len(annotations):,} annotations saved to {output_path}")
    logger.info("━" * 60)


if __name__ == "__main__":
    main()
