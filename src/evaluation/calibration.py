"""
src/evaluation/calibration.py

Calibration metrics and visualizations for the Adaptive-IPI project.

Implements Change #6: ECE, reliability diagrams, and confidence
histograms. These are essential for the paper's claim that the
adaptive curriculum improves calibration.

Reviewers at ACL/EMNLP WILL ask for calibration analysis.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the average absolute difference between predicted
    confidence and actual accuracy, weighted by the number of samples
    in each confidence bin.

    ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Args:
        y_true: Ground-truth binary labels, shape (N,).
        y_prob: Predicted probabilities for the positive class, shape (N,).
        n_bins: Number of equal-width bins for calibration.

    Returns:
        Expected Calibration Error (lower is better).
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]

        # Samples in this bin
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        # Average confidence and accuracy in this bin
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()

        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_reliability_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> dict[str, np.ndarray]:
    """Compute data for a reliability diagram.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        Dictionary with:
            - bin_centers: Center of each bin
            - bin_accuracies: Empirical accuracy in each bin
            - bin_confidences: Mean confidence in each bin
            - bin_counts: Number of samples in each bin
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        n_bin = mask.sum()
        bin_counts.append(n_bin)
        bin_centers.append((lo + hi) / 2)

        if n_bin == 0:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
        else:
            bin_accuracies.append(float(y_true[mask].mean()))
            bin_confidences.append(float(y_prob[mask].mean()))

    return {
        "bin_centers": np.array(bin_centers),
        "bin_accuracies": np.array(bin_accuracies),
        "bin_confidences": np.array(bin_confidences),
        "bin_counts": np.array(bin_counts),
    }


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot a reliability diagram (calibration curve).

    Shows the relationship between predicted confidence and empirical
    accuracy. A perfectly calibrated model follows the diagonal.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins.
        title: Plot title.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    data = compute_reliability_data(y_true, y_prob, n_bins)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Reliability diagram
    ax1.bar(
        data["bin_centers"],
        data["bin_accuracies"],
        width=1.0 / n_bins,
        alpha=0.6,
        edgecolor="black",
        label="Model",
    )
    ax1.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogram of predictions
    ax2.bar(
        data["bin_centers"],
        data["bin_counts"],
        width=1.0 / n_bins,
        alpha=0.6,
        edgecolor="black",
        color="gray",
    )
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved reliability diagram: {output_path}")

    return fig


def plot_confidence_histogram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 50,
    title: str = "Confidence Histogram",
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot a confidence histogram separated by true class.

    Shows the distribution of predicted probabilities for benign
    and attack samples separately.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities.
        n_bins: Number of histogram bins.
        title: Plot title.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    benign_probs = y_prob[y_true == 0]
    attack_probs = y_prob[y_true == 1]

    ax.hist(
        benign_probs, bins=n_bins, alpha=0.6, label="Benign",
        color="steelblue", edgecolor="black", linewidth=0.5,
    )
    ax.hist(
        attack_probs, bins=n_bins, alpha=0.6, label="Attack",
        color="coral", edgecolor="black", linewidth=0.5,
    )

    ax.set_xlabel("Predicted probability (attack)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved confidence histogram: {output_path}")

    return fig
