"""
src/core/types.py

Data structures (dataclasses) used across the Adaptive-IPI project.

These define the canonical shapes of data flowing between modules.
Using dataclasses rather than raw dicts ensures type safety and
makes the interfaces between modules explicit.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SampleRecord:
    """A single sample in the unified dataset schema.

    Attributes:
        id: Globally unique sample identifier.
        context: The text (benign context or poisoned text).
        attack_instruction: The raw malicious instruction string, or None if benign.
        label: Binary label (0 = benign, 1 = attack).
        attack_family: Category of attack or "benign".
        attack_position: Position of the attack (e.g., "start", "middle", "end") or None.
        split: Dataset split ("train", "test").
        task: BIPIA task origin (e.g., "email").
        source: Dataset source identifier (e.g., "bipia", "generated").
    """

    id: str
    context: str
    attack_instruction: Optional[str]
    label: int
    attack_family: str
    attack_position: Optional[str]
    split: str
    task: str
    source: str


@dataclass
class TeacherAnnotation:
    """Teacher model annotation for a single sample.

    Stores the full output from the teacher model, including
    probability distribution, entropy, and reasoning chain.
    This enables post-hoc analysis of teacher uncertainty
    (Change #7: rich teacher annotations).

    Attributes:
        id: Sample identifier (matches SampleRecord.id).
        teacher_prediction: Teacher's predicted label (0 or 1).
        teacher_probs: Probability distribution [p_benign, p_attack].
        teacher_entropy: Shannon entropy of the probability distribution.
        teacher_reasoning: Teacher's reasoning chain (full text).
    """

    id: str
    teacher_prediction: int
    teacher_probs: list[float]
    teacher_entropy: float
    teacher_reasoning: str


@dataclass
class FailureCase:
    """A sample where the student model failed.

    Used in Phase 4 (failure analysis) and Phase 5 (hard-negative generation).

    Attributes:
        sample: The original sample record.
        predicted_label: Student's predicted label.
        predicted_prob: Student's predicted probability for the positive class.
        true_label: Ground-truth label.
        failure_type: "false_positive", "false_negative", or "low_confidence".
        teacher_annotation: Optional teacher annotation for this sample.
    """

    sample: SampleRecord
    predicted_label: int
    predicted_prob: float
    true_label: int
    failure_type: str
    teacher_annotation: Optional[TeacherAnnotation] = None


@dataclass
class ExperimentConfig:
    """Metadata for a single experiment run.

    Tracks configuration and paths for reproducibility (Change #5).

    Attributes:
        experiment_id: Unique experiment identifier (e.g., "exp001").
        phase: Pipeline phase that produced this experiment.
        config_snapshot: Full configuration dict used for this run.
        output_dir: Directory where all experiment artifacts are saved.
        dataset_version: Which versioned dataset was used.
        description: Free-text description of this experiment.
    """

    experiment_id: str
    phase: str
    config_snapshot: dict[str, Any]
    output_dir: Path
    dataset_version: str
    description: str = ""


@dataclass
class EvaluationResult:
    """Container for evaluation metrics from a single evaluation run.

    Attributes:
        accuracy: Overall accuracy.
        precision: Precision (per-class or macro).
        recall: Recall (per-class or macro).
        f1: F1 score (per-class or macro).
        auroc: Area under ROC curve.
        auprc: Area under precision-recall curve.
        ece: Expected calibration error (Change #6).
        confusion_matrix: 2x2 confusion matrix as nested list.
        tp: True positives count.
        fp: False positives count.
        tn: True negatives count.
        fn: False negatives count.
        per_attack_type: Metrics broken down by attack type.
        per_context_type: Metrics broken down by context type.
    """

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auroc: float = 0.0
    auprc: float = 0.0
    ece: float = 0.0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    confusion_matrix: list[list[int]] = field(default_factory=lambda: [[0, 0], [0, 0]])
    per_attack_type: dict[str, dict[str, float]] = field(default_factory=dict)
    per_context_type: dict[str, dict[str, float]] = field(default_factory=dict)
