"""
scripts/run_phase7_evaluate.py

Phase 7: Final Evaluation
─────────────────────────
Run comprehensive evaluation on the test set and generate
all artifacts: metrics, predictions, confusion matrix,
calibration plots, and comparative analysis.

Usage:
    python scripts/run_phase7_evaluate.py --checkpoint experiments/exp003/checkpoint/best
    python scripts/run_phase7_evaluate.py --checkpoint experiments/exp003/checkpoint/best --compare experiments/exp001
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from torch.utils.data import DataLoader

from src.datasets.dataset import IPIDataset
from src.evaluation.analysis import compare_experiments, plot_metric_comparison
from src.evaluation.evaluator import run_full_evaluation
from src.models.student import load_student_checkpoint
from src.utils.config import load_config
from src.utils.experiment import get_dataset_version_dir, init_experiment
from src.utils.io import read_csv
from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import get_device, set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7: Final Evaluation"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to student model checkpoint.",
    )
    parser.add_argument(
        "--data-config", type=Path,
        default=_PROJECT_ROOT / "configs" / "data.yaml",
    )
    parser.add_argument(
        "--eval-config", type=Path,
        default=_PROJECT_ROOT / "configs" / "evaluation.yaml",
    )
    parser.add_argument(
        "--dataset-version", type=int, default=0,
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Which split to evaluate on.",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Tag for figure titles (e.g., 'post-curriculum').",
    )
    parser.add_argument(
        "--compare", type=Path, nargs="*", default=None,
        help="Experiment directories to compare against.",
    )
    args = parser.parse_args()

    # Setup
    setup_logging(log_file=_PROJECT_ROOT / "logs" / "phase7.log")
    data_config = load_config(args.data_config)
    eval_config = load_config(args.eval_config)
    set_seed(data_config.get("seed", 42))
    device = get_device()

    logger.info("━" * 60)
    logger.info("PHASE 7: Final Evaluation")
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

    # Init experiment
    experiments_dir = _PROJECT_ROOT / "experiments"
    exp_dir = init_experiment(
        experiments_dir,
        config={"evaluation": eval_config, "data": data_config},
        phase="phase7_evaluate",
        description=f"Evaluation on {args.split} split, checkpoint={args.checkpoint}",
    )

    # Run full evaluation
    result = run_full_evaluation(
        model=model,
        dataloader=eval_dataloader,
        device=device,
        eval_df=eval_df,
        output_dir=exp_dir,
        tag=args.tag,
    )

    # Comparative analysis
    if args.compare:
        compare_dirs = [str(d) for d in args.compare] + [str(exp_dir)]
        compare_labels = [d.name for d in args.compare] + [exp_dir.name]

        figures_dir = _PROJECT_ROOT / eval_config.get("figures_dir", "figures")
        comparison_df = compare_experiments(compare_dirs, compare_labels, figures_dir)

        plot_metric_comparison(
            comparison_df,
            metrics=["accuracy", "precision", "recall", "f1", "auroc", "ece"],
            title="Pre-Curriculum vs Post-Curriculum",
            output_path=figures_dir / "comparison.png",
        )

    logger.info("━" * 60)
    logger.info(f"Phase 7 complete. Results: {exp_dir}")
    logger.info("━" * 60)


if __name__ == "__main__":
    main()
