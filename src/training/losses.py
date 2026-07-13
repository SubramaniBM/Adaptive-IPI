"""
src/training/losses.py

Loss functions for the Adaptive-IPI training pipeline.

Implements:
    - Standard cross-entropy loss (baseline)
    - Knowledge distillation loss (KL divergence + CE)
    - Composite loss (configurable weighting)
    - Confidence Weighted loss (weighted by teacher confidence)

NOTE: The KD temperature (τ) and alpha (α) weighting are research
decisions. They are exposed as configurable parameters, not hardcoded.
The researcher should determine these values empirically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss for binary classification.

    This is the baseline loss function (Experiment A: hard labels only).
    """

    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            student_logits: Student model output logits, shape (B, num_classes).
            labels: Ground-truth labels, shape (B,).

        Returns:
            Scalar loss tensor.
        """
        return self.ce(student_logits, labels)


class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining KL divergence and cross-entropy.

    Loss = α * KL(softened_student || softened_teacher)
         + (1 - α) * CE(student, hard_labels)

    where T is the temperature for softmax softening and α controls
    the weighting between distillation and hard-label losses.

    NOTE: The values of temperature and alpha are research decisions.
    This class provides the mechanism; the researcher determines the values.
    """

    def __init__(
        self,
        temperature: float,
        alpha: float,
    ) -> None:
        """Initialise the distillation loss.

        Args:
            temperature: Temperature τ for softmax softening.
                Higher values produce softer probability distributions.
                Typical range: 1.0–20.0 (research decision).
            alpha: Weighting between distillation and CE loss.
                α=1.0: pure distillation, α=0.0: pure CE.
                Typical range: 0.1–0.9 (research decision).

        Raises:
            ValueError: If temperature <= 0 or alpha not in [0, 1].
        """
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

        self.temperature = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_probs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the combined distillation loss.

        Args:
            student_logits: Student model output logits, shape (B, num_classes).
            labels: Ground-truth hard labels, shape (B,).
            teacher_probs: Teacher probability distribution, shape (B, num_classes).

        Returns:
            Scalar loss tensor.
        """
        # Soften student logits
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)

        # Soften teacher probabilities
        # Teacher probs are already probabilities, so we apply temperature scaling
        # by converting to logits first, then softening
        teacher_logits = torch.log(teacher_probs.clamp(min=1e-8))
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence loss (scaled by T²)
        distill_loss = self.kl(soft_student, soft_teacher)

        # Hard-label cross-entropy loss
        ce_loss = self.ce(student_logits, labels)

        # Combined loss
        total_loss = self.alpha * distill_loss + (1.0 - self.alpha) * ce_loss

        return total_loss


class ConfidenceWeightedLoss(nn.Module):
    """Cross-entropy loss weighted by teacher confidence.

    Loss = CE(student, hard_labels) * max(teacher_probs)
    """

    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_probs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the confidence-weighted loss.

        Args:
            student_logits: Student model output logits, shape (B, num_classes).
            labels: Ground-truth hard labels, shape (B,).
            teacher_probs: Teacher probability distribution, shape (B, num_classes).

        Returns:
            Scalar loss tensor (batch mean).
        """
        ce_loss = self.ce(student_logits, labels)
        
        # Weight by teacher confidence (max probability)
        weights, _ = torch.max(teacher_probs, dim=-1)
        
        weighted_loss = ce_loss * weights
        return weighted_loss.mean()


def create_loss_function(config: dict) -> nn.Module:
    """Factory function to create a loss function from config.

    Args:
        config: Distillation configuration dictionary containing:
            - loss_type: "ce" or "kd"
            - temperature: (for KD) softmax temperature
            - alpha: (for KD) loss weighting

    Returns:
        Loss function module.

    Raises:
        ValueError: If loss_type is unknown.
        NotImplementedError: If KD parameters are not yet specified.
    """
    loss_type = config.get("loss_type", "ce")

    if loss_type == "ce":
        return CrossEntropyLoss()

    elif loss_type == "kd":
        temperature = config.get("temperature")
        alpha = config.get("alpha")

        if temperature is None or alpha is None:
            raise NotImplementedError(
                "Knowledge distillation temperature (τ) and alpha (α) must be "
                "specified in the distillation config. These are research decisions — "
                "set them in configs/distillation.yaml after empirical determination."
            )

        return DistillationLoss(temperature=temperature, alpha=alpha)

    elif loss_type == "cw":
        return ConfidenceWeightedLoss()

    else:
        raise ValueError(f"Unknown loss type: {loss_type!r}. Supported: 'ce', 'kd', 'cw'")
