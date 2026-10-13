"""
v2/src/datasets/span_dataset.py

High-performance token-level dataset with in-memory pre-tokenization
and length-bucketed dynamic padding (multiples of 64) optimized for Apple Silicon MPS.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


class FastSpanIPIDataset(Dataset):
    """
    In-memory pre-tokenized dataset for high-throughput training.
    Labels:
      -100: Special tokens / Ignored tokens
         0: O (Benign / Outside)
         1: B-INJECTION (Beginning of injection span)
         2: I-INJECTION (Inside injection span)
    """

    LABEL_O = 0
    LABEL_B = 1
    LABEL_I = 2

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 256,
    ):
        self.features: List[Dict[str, Any]] = []
        self._prepare(df, tokenizer, max_length)

    def _prepare(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
    ) -> None:
        for idx in range(len(df)):
            row = df.iloc[idx]
            text = str(row["context"])
            is_attack = int(row["label"]) == 1
            attack_instruction = str(row.get("attack_instruction", ""))

            span: Optional[Tuple[int, int]] = None
            if is_attack and attack_instruction and attack_instruction in text:
                s = text.find(attack_instruction)
                e = s + len(attack_instruction)
                span = (s, e)

            encoding = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_offsets_mapping=True,
            )

            input_ids = encoding["input_ids"]
            offset_mapping = encoding["offset_mapping"]
            seq_len = len(input_ids)

            labels = []
            in_span = False

            for i in range(seq_len):
                tok_start, tok_end = offset_mapping[i]
                if tok_start == 0 and tok_end == 0:
                    labels.append(-100)
                    continue

                if span is not None:
                    span_start, span_end = span
                    if max(tok_start, span_start) < min(tok_end, span_end):
                        if not in_span:
                            labels.append(self.LABEL_B)
                            in_span = True
                        else:
                            labels.append(self.LABEL_I)
                        continue

                in_span = False
                labels.append(self.LABEL_O)

            self.features.append({
                "input_ids": input_ids,
                "attention_mask": [1] * seq_len,
                "labels": labels,
            })

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.features[idx]


class DynamicSpanCollator:
    """
    Pads each batch to a multiple of 64 (e.g. 64, 128, 192, 256).
    Prevents Apple Silicon Metal (MPS) shader recompilations while saving massive compute.
    """

    def __init__(self, pad_token_id: int, pad_label_id: int = -100, max_length: int = 256):
        self.pad_token_id = pad_token_id
        self.pad_label_id = pad_label_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raw_max = max(len(item["input_ids"]) for item in batch)
        # Length bucketing: round up to nearest multiple of 64
        bucket_len = min(self.max_length, max(64, ((raw_max + 63) // 64) * 64))

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in batch:
            input_ids = item["input_ids"][:bucket_len]
            labels = item["labels"][:bucket_len]
            pad_len = max(0, bucket_len - len(input_ids))

            padded_input_ids = input_ids + [self.pad_token_id] * pad_len
            padded_mask = [1] * len(input_ids) + [0] * pad_len
            padded_labels = labels + [self.pad_label_id] * pad_len

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_mask)
            batch_labels.append(padded_labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
