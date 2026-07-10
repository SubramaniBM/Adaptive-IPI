"""
src/adaptive/verification.py

Verify generated hard-negative examples before adding them to training.

Ensures generated examples are:
    - Well-formed (valid text, correct schema)
    - Not duplicates of existing training data
    - Correctly labelled (optional teacher re-verification)
    - Of sufficient quality for training

NOTE: The verification criteria are research decisions. This module
provides basic structural checks. The researcher should add
quality-specific verification based on the generation strategy.
"""

from typing import Any

import pandas as pd

from src.core.constants import SCHEMA_COLUMNS
from src.core.enums import Label
from src.utils.logging import get_logger

logger = get_logger(__name__)


def verify_generated_examples(
    generated_df: pd.DataFrame,
    existing_train_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Verify and filter generated hard-negative examples.

    Applies structural checks and deduplication. Quality-specific
    verification should be added by the researcher.

    Args:
        generated_df: DataFrame of generated examples.
        existing_train_df: Current training set (for deduplication).
        config: Verification configuration.

    Returns:
        Verified DataFrame (filtered to valid examples).
    """
    n_initial = len(generated_df)
    logger.info(f"Verifying {n_initial:,} generated examples")

    # Check 1: Schema validation
    required_cols = ["text", "label"]
    missing = [c for c in required_cols if c not in generated_df.columns]
    if missing:
        raise ValueError(f"Generated examples missing columns: {missing}")

    # Check 2: Remove empty/whitespace text
    generated_df = generated_df[generated_df["text"].str.strip().str.len() > 0].copy()
    n_empty = n_initial - len(generated_df)
    if n_empty > 0:
        logger.info(f"  Removed {n_empty:,} empty-text examples")

    # Check 3: Validate labels
    valid_labels = {Label.BENIGN.value, Label.ATTACK.value}
    generated_df = generated_df[generated_df["label"].isin(valid_labels)].copy()
    n_invalid_label = n_initial - n_empty - len(generated_df)
    if n_invalid_label > 0:
        logger.info(f"  Removed {n_invalid_label:,} invalid-label examples")

    # Check 4: Deduplicate against existing training data
    if len(existing_train_df) > 0 and "text" in existing_train_df.columns:
        existing_texts = set(existing_train_df["text"].tolist())
        before_dedup = len(generated_df)
        generated_df = generated_df[~generated_df["text"].isin(existing_texts)].copy()
        n_dup = before_dedup - len(generated_df)
        if n_dup > 0:
            logger.info(f"  Removed {n_dup:,} duplicates of existing training data")

    # Check 5: Minimum text length
    min_length = config.get("min_text_length", 20)
    generated_df = generated_df[generated_df["text"].str.len() >= min_length].copy()

    n_final = len(generated_df)
    logger.info(
        f"Verification complete: {n_final:,}/{n_initial:,} examples passed "
        f"({n_initial - n_final:,} removed)"
    )

    return generated_df.reset_index(drop=True)
