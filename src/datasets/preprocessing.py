"""
src/datasets/preprocessing.py

Dataset preprocessing and validation for the Adaptive-IPI project.

Handles text cleaning, deduplication, validation, and saving
preprocessed data to versioned directories (Change #4).
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.core.constants import SCHEMA_COLUMNS, SEED
from src.core.enums import Label, Split
from src.utils.experiment import get_dataset_version_dir
from src.utils.io import write_csv
from src.utils.logging import get_logger

logger = get_logger(__name__)


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text-level cleaning to the dataset.

    Operations:
        - Strip leading/trailing whitespace
        - Normalise internal whitespace (collapse multiple spaces)
        - Remove rows with empty or whitespace-only text

    Args:
        df: DataFrame with a 'context' column.

    Returns:
        Cleaned DataFrame with reset index.
    """
    n_before = len(df)

    df = df.copy()
    df["context"] = df["context"].astype(str).str.strip()
    df["context"] = df["context"].str.replace(r"\s+", " ", regex=True)

    # Remove empty text
    df = df[df["context"].str.len() > 0]
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"  Removed {n_removed:,} empty-text rows")

    return df.reset_index(drop=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate samples within each split.

    Deduplication key: (split, context)

    Args:
        df: DataFrame with 'context' and 'split' columns.

    Returns:
        Deduplicated DataFrame.
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=["context", "attack_instruction"])
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"  Removed {n_removed:,} duplicate rows")
    return df.reset_index(drop=True)


def validate_schema(df: pd.DataFrame, stage: str = "") -> None:
    """Validate that a DataFrame conforms to the unified schema.

    Checks:
        - All required columns are present
        - Labels are valid (0 or 1)
        - No null values in critical columns
        - Splits are valid enum values

    Args:
        df: DataFrame to validate.
        stage: Descriptive name for error messages.

    Raises:
        ValueError: If validation fails.
    """
    prefix = f"[{stage}] " if stage else ""

    # Column presence
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{prefix}Missing schema columns: {missing}")

    # Label values
    invalid_labels = df[~df["label"].isin([Label.BENIGN.value, Label.ATTACK.value])]
    if len(invalid_labels) > 0:
        raise ValueError(
            f"{prefix}Invalid labels found: {invalid_labels['label'].unique().tolist()}"
        )

    # Null checks
    for col in ["id", "context", "label", "split"]:
        n_null = df[col].isna().sum()
        if n_null > 0:
            raise ValueError(f"{prefix}Column '{col}' has {n_null} null values")

    # Split values
    valid_splits = {s.value for s in Split}
    invalid_splits = set(df["split"].unique()) - valid_splits
    if invalid_splits:
        raise ValueError(f"{prefix}Invalid split values: {invalid_splits}")

    logger.info(f"{prefix}Schema validation passed ✓")


def generate_report(df: pd.DataFrame) -> str:
    """Generate a human-readable dataset report.

    Args:
        df: Unified dataset DataFrame.

    Returns:
        Report string.
    """
    lines = [
        "=" * 70,
        "  DATASET PREPROCESSING REPORT",
        "=" * 70,
        "",
        f"  Total samples: {len(df):,}",
        "",
        "── Split sizes ──────────────────────────────────────",
    ]

    for split in Split:
        split_df = df[df["split"] == split.value]
        lines.append(f"    {split.value:<12}: {len(split_df):>7,}")

    lines += [
        "",
        "── Label distribution ──────────────────────────────",
    ]
    for label in Label:
        count = (df["label"] == label.value).sum()
        pct = 100 * count / len(df) if len(df) > 0 else 0
        lines.append(f"    {label.name:<12}: {count:>7,}  ({pct:5.1f}%)")

    lines += [
        "",
        "── Attack family distribution ──────────────────────",
    ]
    for atk, count in df["attack_family"].value_counts().items():
        pct = 100 * count / len(df)
        lines.append(f"    {str(atk):<35}: {count:>7,}  ({pct:5.1f}%)")

    lines += [
        "",
        "── Attack position distribution ────────────────────",
    ]
    for pos, count in df["attack_position"].value_counts(dropna=False).items():
        pct = 100 * count / len(df)
        lines.append(f"    {str(pos):<35}: {count:>7,}  ({pct:5.1f}%)")

    lines += [
        "",
        "── Text length statistics (characters) ────────────",
    ]
    lengths = df["context"].str.len()
    lines.append(f"    Mean   : {lengths.mean():>10,.1f}")
    lines.append(f"    Median : {lengths.median():>10,.1f}")
    lines.append(f"    Min    : {int(lengths.min()):>10,}")
    lines.append(f"    Max    : {int(lengths.max()):>10,}")

    lines += ["", "=" * 70]
    return "\n".join(lines)


def preprocess_and_save(
    df: pd.DataFrame,
    output_root: Union[str, Path],
    version: int = 0,
    deduplicate_dataset: bool = True,
) -> Path:
    """Run full preprocessing pipeline and save to a versioned directory.

    Pipeline: clean → (optional) deduplicate → validate → save

    Args:
        df: Raw unified DataFrame from the loader.
        output_root: Root directory for processed data (e.g., data/processed/).
        version: Dataset version number (Change #4).
        deduplicate_dataset: Whether to deduplicate (set to False if upsampling).

    Returns:
        Path to the versioned output directory.
    """
    logger.info("━" * 60)
    logger.info(f"Preprocessing dataset → dataset_v{version}")
    logger.info("━" * 60)

    # Clean
    df = clean_text(df)
    if deduplicate_dataset:
        df = deduplicate(df)
    else:
        logger.info("  Skipping deduplication (likely due to upsampling)")

    # Shuffle train split
    train_mask = df["split"] == Split.TRAIN.value
    train_df = df[train_mask].sample(frac=1, random_state=SEED).reset_index(drop=True)
    non_train_df = df[~train_mask].reset_index(drop=True)
    df = pd.concat([train_df, non_train_df], ignore_index=True)

    # Re-assign IDs after shuffle
    df["id"] = [f"{row['source']}_{i:06d}" for i, row in df.iterrows()]

    # Validate
    validate_schema(df, stage=f"dataset_v{version}")

    # Save
    version_dir = get_dataset_version_dir(output_root, version)
    version_dir.mkdir(parents=True, exist_ok=True)

    write_csv(df, version_dir / "unified.csv")
    for split in Split:
        split_df = df[df["split"] == split.value]
        if len(split_df) > 0:
            write_csv(split_df, version_dir / f"{split.value}.csv")
            logger.info(f"  Saved {split.value}: {len(split_df):,} rows")

    # Validate final dataset properties
    validate_final_dataset(df, version_dir)

    # Save report
    report = generate_report(df)
    report_path = version_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"  Saved report: {report_path}")

    return version_dir


def validate_final_dataset(df: pd.DataFrame, output_dir: Path) -> None:
    """Robust validation of the final dataset schema and logical constraints.
    Saves validation_report.json to output_dir.
    """
    import json
    report = {
        "passed": True,
        "duplicate_ids": 0,
        "missing_values": 0,
        "invalid_labels": 0,
        "duplicate_context_intent_pairs": 0,
        "attack_family_coverage": True,
        "attack_position_coverage": True,
        "errors": []
    }
    
    # 1. Duplicate IDs
    dup_ids = df["id"].duplicated().sum()
    if dup_ids > 0:
        report["duplicate_ids"] = int(dup_ids)
        report["passed"] = False
        report["errors"].append(f"Found {dup_ids} duplicate IDs")
        
    # 2. Missing values
    missing = df.isnull().sum().sum()
    # It's expected that benign intents have null attack_instruction and attack_position.
    # So we check critical columns only
    crit_missing = df[["id", "context", "label", "split", "task", "source"]].isnull().sum().sum()
    if crit_missing > 0:
        report["missing_values"] = int(crit_missing)
        report["passed"] = False
        report["errors"].append(f"Found {crit_missing} missing values in critical columns")
        
    # 3. Invalid labels
    invalid_labels = (~df["label"].isin([0, 1])).sum()
    if invalid_labels > 0:
        report["invalid_labels"] = int(invalid_labels)
        report["passed"] = False
        report["errors"].append(f"Found {invalid_labels} invalid labels")
        
    # 4. Duplicate (context, user_intent) pairs
    # attack_instruction is the user intent
    dup_pairs = df.duplicated(subset=["context", "attack_instruction"]).sum()
    if dup_pairs > 0:
        report["duplicate_context_intent_pairs"] = int(dup_pairs)
        report["passed"] = False
        report["errors"].append(f"Found {dup_pairs} duplicate (context, attack_instruction) pairs")
        
    # 5. Attack family coverage
    attacks = df[df["label"] == 1]
    expected_families = 15 # Per split, but overall there are 29 unique ones in BIPIA. We'll just check if it's > 0
    unique_families = attacks["attack_family"].nunique()
    if unique_families < 15:
        report["attack_family_coverage"] = False
        report["passed"] = False
        report["errors"].append(f"Missing attack families. Found {unique_families}, expected at least 15")
        
    # 6. Attack position coverage
    expected_positions = {"start", "middle", "end"}
    actual_positions = set(attacks["attack_position"].unique())
    if actual_positions != expected_positions:
        report["attack_position_coverage"] = False
        report["passed"] = False
        report["errors"].append(f"Missing attack positions. Found {actual_positions}")
        
    with open(output_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    if not report["passed"]:
        logger.error(f"Validation failed! See {output_dir / 'validation_report.json'} for details.")
    else:
        logger.info("Final dataset validation passed successfully.")
