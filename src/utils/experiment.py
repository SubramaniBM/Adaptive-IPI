"""
src/utils/experiment.py

Experiment directory management for the Adaptive-IPI project.

Implements Change #5: every experiment automatically saves its full
configuration, metrics, predictions, and checkpoints in a numbered
directory under ``experiments/``. Experiments are never overwritten.

Directory structure per experiment:

    experiments/
    └── exp001/
        ├── experiment.yaml      # Full config snapshot
        ├── metrics.json         # Evaluation metrics
        ├── predictions.csv      # Per-sample predictions
        ├── confusion_matrix.png # Confusion matrix plot
        └── checkpoint/          # Model checkpoint directory
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from src.utils.config import save_config
from src.utils.io import write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_next_experiment_dir(
    experiments_root: Union[str, Path],
    prefix: str = "exp",
) -> Path:
    """Determine the next available experiment directory.

    Scans existing directories matching the pattern ``{prefix}NNN``
    and returns the next sequential directory path.

    Args:
        experiments_root: Root directory containing experiment subdirs.
        prefix: Prefix for experiment directories.

    Returns:
        Path to the new experiment directory (not yet created).
    """
    experiments_root = Path(experiments_root)
    experiments_root.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_num = 0

    for child in experiments_root.iterdir():
        if child.is_dir():
            match = pattern.match(child.name)
            if match:
                max_num = max(max_num, int(match.group(1)))

    next_num = max_num + 1
    return experiments_root / f"{prefix}{next_num:03d}"


def init_experiment(
    experiments_root: Union[str, Path],
    config: dict[str, Any],
    phase: str,
    description: str = "",
    prefix: str = "exp",
) -> Path:
    """Create and initialise a new experiment directory.

    Creates the directory, saves the config snapshot and metadata.

    Args:
        experiments_root: Root directory for all experiments.
        config: Full configuration dictionary to snapshot.
        phase: Pipeline phase identifier (e.g., "phase3_distill").
        description: Free-text description of this experiment.
        prefix: Prefix for experiment directory names.

    Returns:
        Path to the created experiment directory.
    """
    exp_dir = get_next_experiment_dir(experiments_root, prefix)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    save_config(config, exp_dir / "experiment.yaml")

    # Save metadata
    metadata = {
        "experiment_id": exp_dir.name,
        "phase": phase,
        "description": description,
        "created_at": datetime.now().isoformat(),
    }
    write_json(metadata, exp_dir / "metadata.json")

    logger.info(f"Initialised experiment directory: {exp_dir}")
    return exp_dir


def save_metrics(exp_dir: Union[str, Path], metrics: dict[str, Any]) -> None:
    """Save evaluation metrics to an experiment directory.

    Args:
        exp_dir: Experiment directory path.
        metrics: Dictionary of metric names to values.
    """
    write_json(metrics, Path(exp_dir) / "metrics.json")
    logger.info(f"Saved metrics to {exp_dir}/metrics.json")


def get_checkpoint_dir(exp_dir: Union[str, Path]) -> Path:
    """Return the checkpoint subdirectory within an experiment.

    Creates the directory if it does not exist.

    Args:
        exp_dir: Experiment directory path.

    Returns:
        Path to the checkpoint subdirectory.
    """
    checkpoint_dir = Path(exp_dir) / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_dataset_version_dir(
    processed_root: Union[str, Path],
    version: int,
) -> Path:
    """Return the path to a versioned dataset directory.

    Implements Change #4: immutable versioned datasets.

    Args:
        processed_root: Root directory for processed data.
        version: Dataset version number (0, 1, 2, ...).

    Returns:
        Path to the versioned dataset directory.
    """
    from src.core.constants import DATASET_VERSION_PREFIX

    return Path(processed_root) / f"{DATASET_VERSION_PREFIX}{version}"


def get_latest_dataset_version(processed_root: Union[str, Path]) -> int:
    """Find the latest versioned dataset in the processed directory.

    Args:
        processed_root: Root directory containing versioned datasets.

    Returns:
        The highest version number found, or -1 if none exist.
    """
    from src.core.constants import DATASET_VERSION_PREFIX

    processed_root = Path(processed_root)
    if not processed_root.exists():
        return -1

    pattern = re.compile(rf"^{re.escape(DATASET_VERSION_PREFIX)}(\d+)$")
    max_version = -1

    for child in processed_root.iterdir():
        if child.is_dir():
            match = pattern.match(child.name)
            if match:
                max_version = max(max_version, int(match.group(1)))

    return max_version
