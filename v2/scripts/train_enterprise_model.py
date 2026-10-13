"""
v2/scripts/train_enterprise_model.py

Trains ModernBERT on the Enterprise Structured & Multilingual Curriculum.
Initializes from `v2/checkpoints/span_modernbert_adaptive/best` and hardens
against JSON, CSV, Code docstrings, and Multilingual (ES, FR, DE, ZH, RU) vectors.
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from v2.src.datasets.span_dataset import FastSpanIPIDataset, DynamicSpanCollator


def main():
    print("=" * 70)
    print("  ADAPTIVE-IPI: TRAINING ENTERPRISE STRUCTURED & MULTILINGUAL MODEL  ")
    print("=" * 70)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device: NVIDIA CUDA")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # Paths
    base_ckpt = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_adaptive" / "best"
    if not base_ckpt.exists():
        base_ckpt = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_hardened" / "best"

    save_dir = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_enterprise" / "best"
    save_dir.mkdir(parents=True, exist_ok=True)

    data_path = _PROJECT_ROOT / "v2" / "data" / "processed" / "train_enterprise.csv"
    train_df = pd.read_csv(data_path)
    print(f"Loaded {len(train_df)} training samples from {data_path.name}")

    print(f"Loading checkpoint from {base_ckpt.relative_to(_PROJECT_ROOT)}...")
    tokenizer = AutoTokenizer.from_pretrained(str(base_ckpt))
    model = AutoModelForTokenClassification.from_pretrained(str(base_ckpt))
    model.to(device)

    dataset = FastSpanIPIDataset(train_df, tokenizer, max_length=256)
    collator = DynamicSpanCollator(
        pad_token_id=tokenizer.pad_token_id or 0,
        pad_label_id=-100,
        max_length=256,
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collator,
    )

    epochs = 2
    lr = 3e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"\nStarting fine-tuning: {epochs} epochs, {len(loader)} steps per epoch...")
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            if (step + 1) % 25 == 0 or (step + 1) == len(loader):
                cur_loss = total_loss / (step + 1)
                print(f"Epoch {epoch}/{epochs} | Step {step+1}/{len(loader)} | Loss: {cur_loss:.4f}")

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch} completed in {elapsed:.1f}s. Average Loss: {total_loss / len(loader):.4f}")

    print(f"\nTraining completed in {time.time() - total_start:.1f}s.")
    print(f"Saving enterprise model checkpoint to: {save_dir.relative_to(_PROJECT_ROOT)}")
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print("Checkpoint saved successfully.")


if __name__ == "__main__":
    main()
