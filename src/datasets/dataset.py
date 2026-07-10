"""
src/datasets/dataset.py

PyTorch Dataset class for the Adaptive-IPI project.

Provides an ``IPIDataset`` that wraps preprocessed CSV files
and handles tokenization for ModernBERT. Supports optional
teacher soft-label columns for knowledge distillation.
"""

from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.core.enums import Label
from src.utils.io import read_csv
from src.utils.logging import get_logger

logger = get_logger(__name__)


class IPIDataset(Dataset):
    """PyTorch Dataset for IPI detection.

    Loads a preprocessed CSV file and tokenizes text on-the-fly.
    Supports optional soft-label columns for knowledge distillation.

    Attributes:
        df: Underlying DataFrame.
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum token sequence length.
        has_soft_labels: Whether teacher soft labels are available.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        teacher_annotations_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            data_path: Path to a preprocessed CSV file with unified schema.
            tokenizer: HuggingFace tokenizer (e.g., ModernBERT tokenizer).
            max_length: Maximum number of tokens per sample.
            teacher_annotations_path: Optional path to teacher annotations
                JSONL file. If provided, soft labels are merged by sample ID.
        """
        self.df = read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_soft_labels = False

        # Bridge column naming: frozen CSV uses 'context', model code expects 'text'
        if "text" not in self.df.columns and "context" in self.df.columns:
            self.df = self.df.rename(columns={"context": "text"})

        # Merge teacher annotations if available
        if teacher_annotations_path is not None:
            self._merge_teacher_annotations(teacher_annotations_path)

        logger.info(
            f"Loaded IPIDataset: {len(self.df):,} samples, "
            f"max_length={max_length}, soft_labels={self.has_soft_labels}"
        )

    def _merge_teacher_annotations(self, annotations_path: Union[str, Path]) -> None:
        """Merge teacher annotations into the dataset by sample ID.

        Args:
            annotations_path: Path to teacher annotations JSONL file.
        """
        from src.utils.io import read_jsonl

        annotations = read_jsonl(annotations_path)
        if not annotations:
            logger.warning(f"No teacher annotations found at {annotations_path}")
            return

        ann_df = pd.DataFrame(annotations)
        required_cols = ["id", "teacher_probs"]
        missing = [c for c in required_cols if c not in ann_df.columns]
        if missing:
            logger.warning(f"Teacher annotations missing columns: {missing}")
            return

        # Merge on ID
        before_len = len(self.df)
        self.df = self.df.merge(ann_df[["id", "teacher_probs"]], on="id", how="left")
        after_len = len(self.df)

        n_matched = self.df["teacher_probs"].notna().sum()
        logger.info(f"  Merged teacher annotations: {n_matched:,}/{after_len:,} samples matched")

        self.has_soft_labels = n_matched > 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single tokenized sample.

        Returns:
            Dictionary with keys:
                - input_ids: Token IDs (LongTensor)
                - attention_mask: Attention mask (LongTensor)
                - labels: Hard label (LongTensor)
                - teacher_probs: Soft label probabilities (FloatTensor),
                    only if teacher annotations are available
        """
        row = self.df.iloc[idx]

        # Tokenize
        encoding = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }

        # Add soft labels if available
        if self.has_soft_labels:
            probs = row.get("teacher_probs")
            if probs is not None and not (isinstance(probs, float) and pd.isna(probs)):
                if isinstance(probs, str):
                    import ast
                    probs = ast.literal_eval(probs)
                item["teacher_probs"] = torch.tensor(probs, dtype=torch.float)

        return item

    @property
    def label_counts(self) -> dict[int, int]:
        """Return the count of each label in the dataset."""
        return self.df["label"].value_counts().to_dict()

    @property
    def texts(self) -> list[str]:
        """Return all text samples as a list."""
        return self.df["text"].tolist()

    @property
    def labels(self) -> list[int]:
        """Return all labels as a list."""
        return self.df["label"].tolist()
