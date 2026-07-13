"""
src/training/callbacks.py

Training callbacks for the Adaptive-IPI training engine.

Provides hooks into the training loop for logging, metric tracking,
and early stopping. Callbacks are called by the TrainingEngine
at predefined points in the training loop.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingState:
    """Snapshot of the current training state, passed to callbacks.

    Attributes:
        epoch: Current epoch (0-indexed).
        global_step: Total training steps so far.
        train_loss: Average training loss for the current epoch.
        eval_metrics: Evaluation metrics dict (if eval was run).
        learning_rate: Current learning rate.
    """

    epoch: int = 0
    global_step: int = 0
    train_loss: float = 0.0
    eval_metrics: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0


class TrainingCallback:
    """Base class for training callbacks.

    Override the methods you need. All methods are no-ops by default.
    """

    def on_train_begin(self, state: TrainingState) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, state: TrainingState) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, state: TrainingState) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, state: TrainingState) -> None:
        """Called at the end of each epoch (after evaluation)."""
        pass

    def on_step_end(self, state: TrainingState) -> None:
        """Called after each training step."""
        pass


class MetricLoggerCallback(TrainingCallback):
    """Logs training metrics to console and optionally to a JSON file.

    Saves a running log of per-epoch metrics for later analysis.
    """

    def __init__(self, log_file: Optional[Union[str, Path]] = None) -> None:
        """Initialise the metric logger.

        Args:
            log_file: Optional path to save metrics as JSONL.
        """
        self.log_file = Path(log_file) if log_file else None
        self.history: list[dict[str, Any]] = []

    def on_epoch_end(self, state: TrainingState) -> None:
        """Log metrics at the end of each epoch."""
        record = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "train_loss": state.train_loss,
            "learning_rate": state.learning_rate,
            **state.eval_metrics,
        }
        self.history.append(record)

        # Console log
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in state.eval_metrics.items())
        logger.info(
            f"Epoch {state.epoch} | "
            f"train_loss={state.train_loss:.4f} | "
            f"lr={state.learning_rate:.2e} | "
            f"{metrics_str}"
        )

        # File log
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


class BestModelCallback(TrainingCallback):
    """Tracks the best model based on a monitored metric.

    Sets a ``should_save`` flag when a new best is found.
    The training engine checks this flag to save the checkpoint.

    Attributes:
        best_value: Best metric value seen so far.
        best_epoch: Epoch where the best value was observed.
        should_save: True if a new best was found this epoch.
    """

    def __init__(
        self,
        monitor: str = "f1",
        mode: str = "max",
    ) -> None:
        """Initialise the best-model tracker.

        Args:
            monitor: Name of the metric to monitor.
            mode: "max" (higher is better) or "min" (lower is better).
        """
        self.monitor = monitor
        self.mode = mode
        self.best_value: float = float("-inf") if mode == "max" else float("inf")
        self.best_epoch: int = -1
        self.best_metrics: dict[str, float] = {}
        self.should_save: bool = False

    def on_epoch_end(self, state: TrainingState) -> None:
        """Check if the current epoch produced a new best metric."""
        current = state.eval_metrics.get(self.monitor)
        if current is None:
            return

        self.should_save = False

        if self.mode == "max" and current > self.best_value:
            self.best_value = current
            self.best_epoch = state.epoch
            self.best_metrics = state.eval_metrics.copy()
            self.should_save = True
            logger.info(f"  ★ New best {self.monitor}: {current:.4f} (epoch {state.epoch})")

        elif self.mode == "min" and current < self.best_value:
            self.best_value = current
            self.best_epoch = state.epoch
            self.best_metrics = state.eval_metrics.copy()
            self.should_save = True
            logger.info(f"  ★ New best {self.monitor}: {current:.4f} (epoch {state.epoch})")


class EarlyStoppingCallback(TrainingCallback):
    """Stop training if a monitored metric does not improve.

    Sets a ``should_stop`` flag that the training engine checks.

    Attributes:
        should_stop: True if training should be stopped.
    """

    def __init__(
        self,
        monitor: str = "f1",
        patience: int = 3,
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        """Initialise early stopping.

        Args:
            monitor: Name of the metric to monitor.
            patience: Number of epochs without improvement before stopping.
            mode: "max" (higher is better) or "min" (lower is better).
            min_delta: Minimum change to qualify as improvement.
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value: float = float("-inf") if mode == "max" else float("inf")
        self.wait: int = 0
        self.should_stop: bool = False

    def on_epoch_end(self, state: TrainingState) -> None:
        """Check if training should stop."""
        current = state.eval_metrics.get(self.monitor)
        if current is None:
            return

        improved = False
        if self.mode == "max":
            improved = current > (self.best_value + self.min_delta)
        else:
            improved = current < (self.best_value - self.min_delta)

        if improved:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping: {self.monitor} did not improve for "
                    f"{self.patience} epochs (best={self.best_value:.4f})"
                )
