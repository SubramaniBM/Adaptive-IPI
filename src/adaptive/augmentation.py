"""
src/adaptive/augmentation.py

Merge generated hard-negative examples into the training data.

Creates a new versioned dataset (Change #4) by combining the
original training data with verified generated examples.
The original dataset is never modified.
"""

from pathlib import Path
from typing import Union

import pandas as pd

from src.core.constants import SOURCE_GENERATED
from src.core.enums import AttackType, Split
from src.datasets.preprocessing import preprocess_and_save, validate_schema
from src.utils.experiment import get_dataset_version_dir, get_latest_dataset_version
from src.utils.io import read_csv, write_csv
from src.utils.logging import get_logger

logger = get_logger(__name__)


def augment_training_data(
    generated_df: pd.DataFrame,
    processed_root: Union[str, Path],
    source_version: int,
) -> Path:
    """Merge generated examples into training data and create a new version.

    Creates dataset_v{N+1} containing:
        - All original training data from dataset_v{source_version}
        - Verified generated hard-negative examples
        - Original validation and test data (unchanged)

    The source version is never modified (Change #4: immutable datasets).

    Args:
        generated_df: Verified generated examples DataFrame.
        processed_root: Root directory for processed data.
        source_version: Version number of the source dataset.

    Returns:
        Path to the new versioned dataset directory.
    """
    processed_root = Path(processed_root)

    # Load source dataset
    source_dir = get_dataset_version_dir(processed_root, source_version)
    source_train = read_csv(source_dir / f"{Split.TRAIN.value}.csv")

    logger.info(f"Augmenting dataset_v{source_version}")
    logger.info(f"  Original train size: {len(source_train):,}")
    logger.info(f"  Generated examples:  {len(generated_df):,}")

    # Prepare generated examples
    generated_df = generated_df.copy()
    generated_df["split"] = Split.TRAIN.value
    if "source" not in generated_df.columns:
        generated_df["source"] = SOURCE_GENERATED
    if "id" not in generated_df.columns:
        generated_df["id"] = [
            f"{SOURCE_GENERATED}_{i:06d}" for i in range(len(generated_df))
        ]

    # Merge
    augmented_train = pd.concat([source_train, generated_df], ignore_index=True)

    # Re-assign IDs for the merged set
    augmented_train["id"] = [
        f"{row['source']}_{i:06d}" for i, row in augmented_train.iterrows()
    ]

    logger.info(f"  Augmented train size: {len(augmented_train):,}")

    # Create new version
    new_version = source_version + 1
    new_dir = get_dataset_version_dir(processed_root, new_version)
    new_dir.mkdir(parents=True, exist_ok=True)

    # Save augmented train
    write_csv(augmented_train, new_dir / f"{Split.TRAIN.value}.csv")

    # Copy validation and test from source (unchanged)
    for split in [Split.VALIDATION, Split.TEST]:
        split_path = source_dir / f"{split.value}.csv"
        if split_path.exists():
            split_df = read_csv(split_path)
            write_csv(split_df, new_dir / f"{split.value}.csv")
            logger.info(f"  Copied {split.value}: {len(split_df):,} rows")

    # Save unified
    all_parts = [augmented_train]
    for split in [Split.VALIDATION, Split.TEST]:
        split_path = new_dir / f"{split.value}.csv"
        if split_path.exists():
            all_parts.append(read_csv(split_path))

    unified = pd.concat(all_parts, ignore_index=True)
    write_csv(unified, new_dir / "unified.csv")

    logger.info(f"Created dataset_v{new_version} at {new_dir}")
    return new_dir
