"""
scripts/run_phase1_preprocess.py

Phase 1: Dataset Preprocessing
───────────────────────────────
Load BIPIA, convert to unified schema, clean, and save
to a versioned dataset directory (dataset_v0).

Usage:
    python scripts/run_phase1_preprocess.py
    python scripts/run_phase1_preprocess.py --config configs/data.yaml
    python scripts/run_phase1_preprocess.py --bipia-dir /path/to/bipia
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.dataset_builder import DatasetBuilder
from src.datasets.preprocessing import preprocess_and_save
from src.utils.config import load_config
from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: Dataset Preprocessing"
    )
    parser.add_argument(
        "--config", type=Path, default=_PROJECT_ROOT / "configs" / "data.yaml",
        help="Path to data configuration YAML.",
    )
    parser.add_argument(
        "--bipia-dir", type=Path, default=None,
        help="Override: path to cloned BIPIA repository.",
    )
    args = parser.parse_args()

    # Setup
    setup_logging(log_file=_PROJECT_ROOT / "logs" / "phase1.log")
    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    logger.info("━" * 60)
    logger.info("PHASE 1: Dataset Preprocessing")
    logger.info("━" * 60)

    # Build dataset deterministically
    bipia_dir = args.bipia_dir or _PROJECT_ROOT / config.get("bipia", {}).get("raw_dir", "data/raw/bipia")
    builder = DatasetBuilder(
        task=config.get("task", "email"),
        balance=config.get("balance", False),
        seed=seed,
        bipia_root=str(bipia_dir),
        target_attack_samples=config.get("target_attack_samples", None)
    )
    # Provide generated_benign_path so we don't drop our 789 generated intents
    gen_path = _PROJECT_ROOT / "data" / "generated" / "benign_intents.jsonl"
    df = builder.build(generated_benign_path=str(gen_path) if gen_path.exists() else None)

    if len(df) == 0:
        logger.error("No samples loaded. Check BIPIA installation.")
        sys.exit(1)

    # Preprocess and save to versioned directory
    processed_dir = _PROJECT_ROOT / config.get("processed_dir", "data/processed")
    version = config.get("preprocessing", {}).get("initial_dataset_version", 0)

    output_dir = preprocess_and_save(df, processed_dir, version=version)
    
    # Save deterministic dataset construction reports
    builder.save_reports(output_dir, df)

    logger.info("━" * 60)
    logger.info(f"Phase 1 complete. Output: {output_dir}")
    logger.info("━" * 60)


if __name__ == "__main__":
    main()
