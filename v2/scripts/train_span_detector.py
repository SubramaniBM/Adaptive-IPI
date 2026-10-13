"""
v2/scripts/train_span_detector.py

High-speed training script for ModernBERT Span Extraction using dynamic batch padding.
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
import pandas as pd
import time

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from v2.src.datasets.span_dataset import FastSpanIPIDataset, DynamicSpanCollator
from v2.src.evaluation.span_metrics import compute_token_level_metrics


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            labs = labels.cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labs)

    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = compute_token_level_metrics(all_preds, all_labels)
    metrics["val_loss"] = avg_loss
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ModernBERT Span Detector (Fast Dynamic)")
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "v2" / "configs" / "span_detector.yaml",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train-file", type=str, default="train.csv", help="Name of CSV in data output_dir")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory to save checkpoints")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    seed = config["training"]["seed"]
    max_length = config["model"]["max_length"]
    model_id = config["model"]["model_id"]
    save_dir = args.save_dir or (_PROJECT_ROOT / config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"🚀 Training on device: {device} (Batch size: {batch_size}, Dynamic Batch Padding)")

    data_dir = _PROJECT_ROOT / config["data"]["output_dir"]
    train_df = pd.read_csv(data_dir / args.train_file)
    val_df = pd.read_csv(data_dir / "validation.csv")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Pre-tokenizing datasets in memory...")
    t0 = time.time()
    train_dataset = FastSpanIPIDataset(train_df, tokenizer, max_length=max_length)
    val_dataset = FastSpanIPIDataset(val_df, tokenizer, max_length=max_length)
    print(f"Dataset pre-tokenization completed in {time.time() - t0:.2f}s!")

    collator = DynamicSpanCollator(
        pad_token_id=tokenizer.pad_token_id,
        pad_label_id=-100,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collator,
    )

    id2label = {int(k): v for k, v in config["labels"]["id2label"].items()}
    label2id = {k: int(v) for k, v in config["labels"]["label2id"].items()}

    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    print(f"Starting optimized training: {epochs} epochs, {total_steps} total steps...")

    train_start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
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
            scheduler.step()

            train_loss += loss.item()

            if (step + 1) % 25 == 0 or (step + 1) == len(train_loader):
                cur_loss = train_loss / (step + 1)
                elapsed = time.time() - epoch_start
                rate = (step + 1) / max(elapsed, 0.1)
                print(f"Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)} | Loss: {cur_loss:.4f} | Speed: {rate:.1f} steps/s")

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} training completed in {epoch_time:.1f}s. Evaluating validation set...")

        val_metrics = evaluate(model, val_loader, device)
        print(
            f"=== Epoch {epoch} Evaluation === "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Token F1: {val_metrics['token_f1']*100:.2f}% | "
            f"Token Precision: {val_metrics['token_precision']*100:.2f}% | "
            f"Token Recall: {val_metrics['token_recall']*100:.2f}%"
        )

        if val_metrics["token_f1"] > best_val_f1:
            best_val_f1 = val_metrics["token_f1"]
            best_dir = save_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  --> Saved new best checkpoint to {best_dir} (Token F1: {best_val_f1*100:.2f}%)")

    total_time = time.time() - train_start
    print(f"\n🎉 Training complete in {total_time:.1f}s! Best Validation Token F1: {best_val_f1*100:.2f}%")


if __name__ == "__main__":
    main()
