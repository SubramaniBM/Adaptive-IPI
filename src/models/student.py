"""
src/models/student.py

ModernBERT student model for binary IPI classification.

Provides a thin wrapper and factory function for creating the student
model and tokenizer. Uses HuggingFace's AutoModelForSequenceClassification
with ModernBERT-base as the backbone.

Key notes:
    - ModernBERT supports up to 8,192 tokens (vs. BERT's 512)
    - ModernBERT does NOT use token_type_ids
    - num_labels=2 for binary classification
"""

from pathlib import Path
from typing import Optional, Union

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.core.constants import NUM_LABELS, STUDENT_MODEL_ID
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_student(
    model_id: str = STUDENT_MODEL_ID,
    num_labels: int = NUM_LABELS,
    **kwargs,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Create a ModernBERT student model and tokenizer.

    Args:
        model_id: HuggingFace model identifier. Defaults to ModernBERT-base.
        num_labels: Number of output classes. Defaults to 2 (binary).
        **kwargs: Additional arguments passed to ``from_pretrained``.

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading student model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        **kwargs,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    return model, tokenizer


def load_student_checkpoint(
    checkpoint_path: Union[str, Path],
    model_id: str = STUDENT_MODEL_ID,
    num_labels: int = NUM_LABELS,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a student model from a saved checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        model_id: Original model identifier (for tokenizer).
        num_labels: Number of output classes.

    Returns:
        Tuple of (model, tokenizer).
    """
    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Loading student checkpoint: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=num_labels,
    )

    return model, tokenizer


def save_student_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Union[str, Path],
) -> None:
    """Save the student model and tokenizer to a directory.

    Args:
        model: The student model.
        tokenizer: The tokenizer.
        output_dir: Directory to save to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved student checkpoint: {output_dir}")
