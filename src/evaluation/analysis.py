"""
src/evaluation/analysis.py

Error analysis and comparative visualization for the Adaptive-IPI project.

Provides tools for comparing pre-curriculum vs. post-curriculum
performance and generating publication-ready figures.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.types import EvaluationResult
from src.utils.io import read_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compare_experiments(
    exp_dirs: list[Union[str, Path]],
    labels: list[str],
    output_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Compare metrics across multiple experiments.

    Args:
        exp_dirs: List of experiment directory paths.
        labels: Human-readable labels for each experiment.
        output_dir: Optional directory to save comparison artifacts.

    Returns:
        DataFrame with one row per experiment and columns for each metric.
    """
    rows = []
    for exp_dir, label in zip(exp_dirs, labels):
        metrics_path = Path(exp_dir) / "metrics.json"
        if not metrics_path.exists():
            logger.warning(f"No metrics.json found in {exp_dir}")
            continue

        metrics = read_json(metrics_path)
        metrics["experiment"] = label
        rows.append(metrics)

    df = pd.DataFrame(rows)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from src.utils.io import write_csv
        write_csv(df, output_dir / "comparison.csv")
        logger.info(f"Saved comparison table: {output_dir / 'comparison.csv'}")

    return df


def plot_metric_comparison(
    comparison_df: pd.DataFrame,
    metrics: list[str],
    title: str = "Experiment Comparison",
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot a grouped bar chart comparing metrics across experiments.

    Args:
        comparison_df: DataFrame from compare_experiments.
        metrics: List of metric column names to plot.
        title: Plot title.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_experiments = len(comparison_df)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_experiments

    for i, (_, row) in enumerate(comparison_df.iterrows()):
        values = [row.get(m, 0) for m in metrics]
        offset = (i - n_experiments / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=row.get("experiment", f"Exp {i}"))

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved comparison plot: {output_path}")

    return fig
