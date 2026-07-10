"""
src/training/train.py

Training orchestration for the Adaptive-IPI project.

High-level functions that wire together the dataset, model, loss,
engine, and callbacks into complete training runs. Called by the
phase scripts (run_phase3_distill.py, run_phase6_retrain.py).
"""

from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch.utils.data import DataLoader

from src.datasets.dataset import IPIDataset
from src.models.student import create_student
from src.training.callbacks import (
    BestModelCallback,
    EarlyStoppingCallback,
    MetricLoggerCallback,
    TrainingState,
)
from src.training.engine import TrainingEngine
from src.training.losses import create_loss_function
from src.utils.config import load_config
from src.utils.experiment import get_checkpoint_dir, init_experiment, save_metrics
from src.utils.logging import get_logger
from src.utils.reproducibility import get_device, set_seed

logger = get_logger(__name__)


def run_training(
    train_data_path: Union[str, Path],
    eval_data_path: Optional[Union[str, Path]],
    student_config: dict[str, Any],
    distillation_config: dict[str, Any],
    experiment_dir: Union[str, Path],
    teacher_annotations_path: Optional[Union[str, Path]] = None,
    eval_fn: Optional[callable] = None,
) -> TrainingState:
    """Execute a complete training run.

    This is the main entry point for both Phase 3 (initial distillation)
    and Phase 6 (retraining with augmented data).

    Args:
        train_data_path: Path to training CSV.
        eval_data_path: Path to evaluation CSV (optional).
        student_config: Student model configuration.
        distillation_config: Loss function configuration.
        experiment_dir: Directory for this experiment's artifacts.
        teacher_annotations_path: Optional path to teacher annotations
            for knowledge distillation.
        eval_fn: Optional evaluation function.

    Returns:
        Final TrainingState with metrics.
    """
    set_seed()
    device = get_device()

    # Create model and tokenizer
    model, tokenizer = create_student(
        model_id=student_config.get("model_id", "answerdotai/ModernBERT-base"),
    )

    # Create datasets
    max_length = student_config.get("max_length", 512)
    train_dataset = IPIDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        teacher_annotations_path=teacher_annotations_path,
    )

    eval_dataset = None
    if eval_data_path:
        eval_dataset = IPIDataset(
            data_path=eval_data_path,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    # Create dataloaders
    batch_size = student_config.get("batch_size", 16)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=student_config.get("num_workers", 0),
        pin_memory=True,
    )

    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=student_config.get("num_workers", 0),
            pin_memory=True,
        )

    # Create loss function
    loss_fn = create_loss_function(distillation_config)

    # Create callbacks
    checkpoint_dir = get_checkpoint_dir(experiment_dir)
    callbacks = [
        MetricLoggerCallback(
            log_file=Path(experiment_dir) / "training_log.jsonl"
        ),
        BestModelCallback(
            monitor=student_config.get("monitor_metric", "f1"),
            mode="max",
        ),
        EarlyStoppingCallback(
            monitor=student_config.get("monitor_metric", "f1"),
            patience=student_config.get("early_stopping_patience", 3),
            mode="max",
        ),
    ]

    # Create and run engine
    engine = TrainingEngine(
        model=model,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        device=device,
        learning_rate=student_config.get("learning_rate", 2e-5),
        weight_decay=student_config.get("weight_decay", 0.01),
        warmup_ratio=student_config.get("warmup_ratio", 0.1),
        max_grad_norm=student_config.get("max_grad_norm", 1.0),
        gradient_accumulation_steps=student_config.get("gradient_accumulation_steps", 1),
        callbacks=callbacks,
    )

    final_state = engine.train(
        train_dataloader=train_dataloader,
        num_epochs=student_config.get("num_epochs", 3),
        eval_dataloader=eval_dataloader,
        eval_fn=eval_fn,
        checkpoint_dir=checkpoint_dir,
    )

    # Save final metrics
    if final_state.eval_metrics:
        save_metrics(experiment_dir, final_state.eval_metrics)

    return final_state
