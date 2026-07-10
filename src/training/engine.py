"""
src/training/engine.py

Custom training engine for the Adaptive-IPI project.

Implements a from-scratch training loop with full control over:
    - Loss computation (CE / KD / composite)
    - Gradient accumulation
    - Learning rate scheduling
    - Callback system (logging, best-model, early stopping)
    - Evaluation during training

This replaces HuggingFace Trainer (Change #1) to support the
pipeline's non-standard requirements: baseline CE, KD,
adaptive retraining, and curriculum scheduling.

The engine is ~250 lines and gives complete control.
"""

from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.student import save_student_checkpoint
from src.training.callbacks import (
    BestModelCallback,
    EarlyStoppingCallback,
    TrainingCallback,
    TrainingState,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TrainingEngine:
    """Custom training engine for the IPI detection student model.

    Provides a clean training loop with support for knowledge
    distillation, callbacks, and checkpoint management.

    Attributes:
        model: The student model.
        tokenizer: The tokenizer (saved with checkpoints).
        loss_fn: Loss function module (CE or KD).
        device: Training device.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        loss_fn: nn.Module,
        device: torch.device,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        callbacks: Optional[list[TrainingCallback]] = None,
    ) -> None:
        """Initialise the training engine.

        Args:
            model: Student model to train.
            tokenizer: Tokenizer for checkpoint saving.
            loss_fn: Loss function (CrossEntropyLoss or DistillationLoss).
            device: Device to train on.
            learning_rate: Peak learning rate.
            weight_decay: Weight decay for AdamW.
            warmup_ratio: Fraction of total steps for linear warmup.
            max_grad_norm: Maximum gradient norm for clipping.
            gradient_accumulation_steps: Steps to accumulate before update.
            callbacks: List of training callbacks.
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.callbacks = callbacks or []

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        eval_fn: Optional[callable] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> TrainingState:
        """Run the training loop.

        Args:
            train_dataloader: DataLoader for training data.
            num_epochs: Number of training epochs.
            eval_dataloader: Optional DataLoader for evaluation.
            eval_fn: Optional function that takes (model, dataloader, device)
                and returns a dict of metrics.
            checkpoint_dir: Optional directory to save checkpoints.

        Returns:
            Final TrainingState.
        """
        # Setup optimizer
        optimizer = self._create_optimizer()
        total_steps = len(train_dataloader) * num_epochs // self.gradient_accumulation_steps
        scheduler = self._create_scheduler(optimizer, total_steps)

        state = TrainingState()
        self._fire_callbacks("on_train_begin", state)

        logger.info("━" * 60)
        logger.info(f"Training for {num_epochs} epochs")
        logger.info(f"  Total steps: {total_steps:,}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info("━" * 60)

        for epoch in range(num_epochs):
            state.epoch = epoch
            self._fire_callbacks("on_epoch_begin", state)

            # Train one epoch
            train_loss = self._train_epoch(
                train_dataloader, optimizer, scheduler, state
            )
            state.train_loss = train_loss
            state.learning_rate = scheduler.get_last_lr()[0]

            # Evaluate
            if eval_dataloader is not None and eval_fn is not None:
                eval_metrics = eval_fn(self.model, eval_dataloader, self.device)
                state.eval_metrics = eval_metrics
            else:
                state.eval_metrics = {}

            self._fire_callbacks("on_epoch_end", state)

            # Save checkpoint if best model callback says so
            if checkpoint_dir:
                for cb in self.callbacks:
                    if isinstance(cb, BestModelCallback) and cb.should_save:
                        save_student_checkpoint(
                            self.model, self.tokenizer,
                            Path(checkpoint_dir) / "best"
                        )

            # Early stopping check
            for cb in self.callbacks:
                if isinstance(cb, EarlyStoppingCallback) and cb.should_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    self._fire_callbacks("on_train_end", state)
                    return state

        # Save final checkpoint
        if checkpoint_dir:
            save_student_checkpoint(
                self.model, self.tokenizer,
                Path(checkpoint_dir) / "final"
            )

        self._fire_callbacks("on_train_end", state)
        return state

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        state: TrainingState,
    ) -> float:
        """Train for one epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(
            dataloader,
            desc=f"Epoch {state.epoch}",
            leave=True,
        )

        for step, batch in enumerate(progress):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            # Compute loss
            loss_kwargs = {"student_logits": logits, "labels": labels}
            if "teacher_probs" in batch:
                loss_kwargs["teacher_probs"] = batch["teacher_probs"].to(self.device)

            loss = self.loss_fn(**loss_kwargs)

            # Scale for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                state.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            progress.set_postfix(loss=f"{loss.item():.4f}")

            self._fire_callbacks("on_step_end", state)

        return total_loss / max(num_batches, 1)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay exclusion for bias/LayerNorm."""
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        param_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(param_groups, lr=self.learning_rate)

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ) -> LambdaLR:
        """Create a linear warmup + linear decay learning rate scheduler."""
        warmup_steps = int(total_steps * self.warmup_ratio)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step)
                / float(max(1, total_steps - warmup_steps)),
            )

        return LambdaLR(optimizer, lr_lambda)

    def _fire_callbacks(self, event: str, state: TrainingState) -> None:
        """Fire a callback event on all registered callbacks."""
        for callback in self.callbacks:
            getattr(callback, event)(state)
